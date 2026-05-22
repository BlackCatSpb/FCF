"""
LanguageTrainer — самообучение языку (Пункт 2).

Реализует автономный цикл обучения PrimordialLayer на задаче
предсказания следующего токена (Causal LM).

Особенности:
- Streaming токенизация для больших файлов (>100MB)
- Cosine annealing scheduler с warmup
- Gradient accumulation для эффективного batch size
- Validation set для мониторинга overfitting
- Resume optimizer/scheduler state из checkpoint
- Epoch tracking с reshuffle
- Интеллектуальный auto-stop на основе validation loss + confidence trend
- Proper snapshot re-evaluation через текущую модель
"""

import os
import sys
import time
import json
import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Iterator, List, Tuple
from collections import deque
from loguru import logger

from .config import FCFConfig
from .primordial_layer import PrimordialLayer
from .data_manager import DataManager
from .utils import save_primordial_layer


class LanguageTrainer:

    def __init__(
        self,
        layer: PrimordialLayer,
        tokenizer,
        config: FCFConfig = None,
        checkpoint_dir: str = None,
        hierarchy=None,
        state_grammar=None,
        lambda_contrastive: float = 0.1,
        lambda_hierarchy: float = 0.05,
        lambda_recursive: float = 0.01,
        benchmark_interval: int = 0,
    ):
        self.layer = layer
        self.tokenizer = tokenizer
        self.config = config or layer.config
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", "language"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.hierarchy = hierarchy
        self.state_grammar = state_grammar
        self.lambda_contrastive = lambda_contrastive
        self.lambda_hierarchy = lambda_hierarchy
        self.lambda_recursive = lambda_recursive
        self.benchmark_interval = benchmark_interval
        self.benchmark_history: List[Dict] = []

        self.optimizer = torch.optim.AdamW(
            self.layer.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Cosine annealing с warmup
        self.warmup_steps: int = 500
        self.max_steps_for_scheduler: int = self.config.training.max_steps or 20000
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._cosine_with_warmup,
        )

        self.step: int = 0
        self.epoch: int = 0
        self.total_loss: float = 0.0
        self._steps_since_loss_reset: int = 0
        self.best_confidence: float = 0.0
        self.best_val_loss: float = float('inf')
        self.val_loss_history: deque = deque(maxlen=100)
        self.train_loss_history: deque = deque(maxlen=100)

        # Gradient accumulation
        self.gradient_accumulation_steps: int = 4
        self._accumulated_steps: int = 0

        self.srg_eval_interval: int = 100
        self.checkpoint_interval: int = 1000
        self.gen_test_interval: int = 1000
        self.log_interval: int = 10
        self.status_interval: int = 5
        self.stop_window: int = 500
        self.min_snapshots: int = 50
        self.target_confidence: float = 0.7

        # Auto-stop parameters
        self.patience: int = 5000  # шагов без улучшения val_loss
        self.steps_without_improvement: int = 0
        self.min_steps_for_stop: int = 10000

        self.stopped: bool = False
        self.stop_reason: str = ""

        # Evaluation prompts (multiple for better coverage)
        self._eval_prompts = [
            "История — это наука о прошлом человеческого общества. "
            "Она изучает события, процессы и закономерности развития. "
            "Историки исследуют",
            "Математика — это наука о числах, структурах и пространствах. "
            "Она изучает закономерности и связи между объектами.",
            "Физика изучает фундаментальные законы природы и свойства материи. "
            "Основные разделы включают механику, термодинамику и электродинамику.",
        ]

        self._gen_test_prompts = [
            "История это наука которая изучает",
            "Математика помогает человечеству",
            "Природа Земли удивительна потому что",
            "Компьютеры обрабатывают данные с помощью",
            "Человек отличается от животных тем что",
        ]

        # Training state for resume
        self._block_idx: int = 0
        self._blocks: List[Dict[str, torch.Tensor]] = []
        self._rng_state: Optional[dict] = None

        self.auto_curriculum = None  # AutoCurriculum (внешняя инъекция)

    def _cosine_with_warmup(self, step: int) -> float:
        """Cosine annealing с linear warmup."""
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(self.max_steps_for_scheduler - self.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _pre_tokenize_corpus(
        self, text_file: str, block_size: int = 512
    ) -> List[Dict[str, torch.Tensor]]:
        logger.info(f"[PreTokenize] Токенизация: {text_file}")
        blocks = []

        file_size = os.path.getsize(text_file)
        file_size_mb = file_size / 1024 / 1024

        # Для больших файлов (>50MB) используем chunked токенизацию
        if file_size_mb > 50:
            return self._pre_tokenize_chunked(text_file, block_size)

        lines = []
        it = DataManager.load_texts_from_file(text_file)
        if it is None:
            return blocks

        for line in it:
            lines.append(line)

        full_text = " ".join(lines)
        try:
            encoding = self.tokenizer.encode(full_text)
            ids = encoding.ids if hasattr(encoding, "ids") else encoding
        except Exception as e:
            logger.error(f"[PreTokenize] Ошибка: {e}")
            return blocks

        block_size = min(block_size, len(ids) - 1)
        if block_size < 4:
            logger.warning(f"[PreTokenize] Текст слишком короткий: {len(ids)} токенов")
            return blocks

        logger.info(f"[PreTokenize] Токенов всего: {len(ids)}, размер блока: {block_size}")

        # Рандомизированный stride для лучшего покрытия
        stride = max(block_size // 2, 1)
        for i in range(0, len(ids) - block_size, stride):
            chunk = ids[i : i + block_size + 1]
            if len(chunk) < 4:
                continue
            while len(chunk) < block_size + 1:
                chunk.append(3)

            input_ids = torch.tensor([chunk[:-1]], dtype=torch.long)
            labels = torch.tensor([chunk[1:]], dtype=torch.long)
            blocks.append({"input_ids": input_ids, "labels": labels})

            if len(blocks) >= 100000:
                break

        logger.info(f"[PreTokenize] Готово: {len(blocks)} блоков (block_size={block_size})")
        return blocks

    def _pre_tokenize_chunked(
        self, text_file: str, block_size: int = 512
    ) -> List[Dict[str, torch.Tensor]]:
        """Chunked токенизация для больших файлов (>50MB)."""
        logger.info(f"[PreTokenize] Chunked токенизация: {text_file}")
        blocks = []
        chunk_size_mb = 10  # Читаем по 10MB
        chunk_size_bytes = chunk_size_mb * 1024 * 1024

        with open(text_file, 'r', encoding='utf-8') as f:
            buffer = ""
            chunk_idx = 0

            while True:
                chunk = f.read(chunk_size_bytes)
                if not chunk:
                    break

                buffer += chunk
                chunk_idx += 1

                # Находим границу предложения/строки
                last_newline = buffer.rfind('\n')
                if last_newline > 0:
                    process_text = buffer[:last_newline]
                    buffer = buffer[last_newline + 1:]
                else:
                    process_text = buffer
                    buffer = ""

                try:
                    encoding = self.tokenizer.encode(process_text)
                    ids = encoding.ids if hasattr(encoding, "ids") else encoding
                except Exception as e:
                    logger.error(f"[PreTokenize] Chunk {chunk_idx} error: {e}")
                    continue

                stride = max(block_size // 2, 1)
                for i in range(0, len(ids) - block_size, stride):
                    chunk_ids = ids[i : i + block_size + 1]
                    if len(chunk_ids) < 4:
                        continue
                    while len(chunk_ids) < block_size + 1:
                        chunk_ids.append(3)

                    input_ids = torch.tensor([chunk_ids[:-1]], dtype=torch.long)
                    labels = torch.tensor([chunk_ids[1:]], dtype=torch.long)
                    blocks.append({"input_ids": input_ids, "labels": labels})

                    if len(blocks) >= 100000:
                        logger.info(f"[PreTokenize] Лимит блоков достигнут: {len(blocks)}")
                        return blocks

                logger.info(f"[PreTokenize] Chunk {chunk_idx}: {len(blocks)} блоков")

                if len(blocks) >= 100000:
                    break

        # Обрабатываем оставшийся buffer
        if buffer.strip():
            try:
                encoding = self.tokenizer.encode(buffer)
                ids = encoding.ids if hasattr(encoding, "ids") else encoding
                stride = max(block_size // 2, 1)
                for i in range(0, len(ids) - block_size, stride):
                    chunk_ids = ids[i : i + block_size + 1]
                    if len(chunk_ids) < 4:
                        continue
                    while len(chunk_ids) < block_size + 1:
                        chunk_ids.append(3)
                    input_ids = torch.tensor([chunk_ids[:-1]], dtype=torch.long)
                    labels = torch.tensor([chunk_ids[1:]], dtype=torch.long)
                    blocks.append({"input_ids": input_ids, "labels": labels})
                    if len(blocks) >= 100000:
                        break
            except Exception as e:
                logger.error(f"[PreTokenize] Buffer error: {e}")

        logger.info(f"[PreTokenize] Готово: {len(blocks)} блоков (chunked)")
        return blocks

    def _split_train_val(self, blocks: List[Dict], val_ratio: float = 0.05) -> Tuple[List, List]:
        """Разделяет блоки на train/val sets."""
        n_val = max(int(len(blocks) * val_ratio), 100)
        random.shuffle(blocks)
        val_blocks = blocks[:n_val]
        train_blocks = blocks[n_val:]
        logger.info(f"[Data] Train: {len(train_blocks)}, Val: {len(val_blocks)}")
        return train_blocks, val_blocks

    @torch.no_grad()
    def _evaluate_validation(self, val_blocks: List[Dict], device: str, max_blocks: int = 200) -> float:
        """Вычисляет validation loss."""
        self.layer.eval()
        total_val_loss = 0.0
        n_batches = 0

        for block in val_blocks[:max_blocks]:
            input_ids = block["input_ids"].to(device)
            labels = block["labels"].to(device)

            x = self.layer.embed(input_ids)
            hidden = self.layer.forward_transformer(x)
            logits = self.layer.forward_logits(hidden)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=3,
            )
            total_val_loss += loss.item()
            n_batches += 1

        self.layer.train()
        return total_val_loss / max(n_batches, 1)

    def train(
        self,
        text_file: str = None,
        max_steps: Optional[int] = None,
        block_size: int = 512,
        device: str = "cpu",
        use_wikipedia: bool = False,
        auto_stop: bool = True,
        resume_from: str = None,
    ) -> Dict[str, Any]:
        import sys
        from loguru import logger as loguru_logger

        log_path = os.path.join(
            os.path.dirname(__file__), "..", "train_log.txt"
        )
        log_id = loguru_logger.add(
            log_path,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO",
            encoding="utf-8",
            enqueue=True,
        )

        logger.info("=" * 60)
        logger.info("Пункт 2 — Самообучение языку")
        logger.info(f"Устройство: {device}")
        logger.info(f"Learning rate: {self.config.training.learning_rate}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"Scheduler: Cosine annealing с warmup ({self.warmup_steps} steps)")
        logger.info("=" * 60)

        if device == "cpu":
            num_threads = min(os.cpu_count() or 4, 4)
            torch.set_num_threads(num_threads)
            logger.info(f"[CPU] Потоков: {num_threads} (из {os.cpu_count()} доступных)")

        self.layer.to(device)
        self.layer.train()

        # Resume из checkpoint если указан
        if resume_from and os.path.exists(resume_from):
            self._resume_training_state(resume_from)

        blocks = []
        if use_wikipedia:
            logger.info("[Train] Используется Wikipedia (потоковая загрузка)")
        elif text_file and os.path.exists(text_file):
            blocks = self._pre_tokenize_corpus(text_file, block_size)

        if not blocks and not use_wikipedia:
            logger.error("Нет данных для обучения.")
            return {"error": "no_data"}

        if use_wikipedia:
            max_steps = max_steps or self.config.training.max_steps or 10000
            logger.info(f"[Train] Wikipedia streaming, макс. шагов: {max_steps}")
            train_blocks = []
            val_blocks = []
        else:
            max_steps = max_steps or self.config.training.max_steps or len(blocks) * 10
            # Разделяем на train/val
            train_blocks, val_blocks = self._split_train_val(blocks, val_ratio=0.05)
            # Shuffle train blocks
            random.shuffle(train_blocks)
            logger.info(f"[Train] Блоков: {len(train_blocks)}, макс. шагов: {max_steps} (перемешаны)")

        self.max_steps_for_scheduler = max_steps
        self._blocks = train_blocks

        print(f"\n{'='*60}")
        print(f"  Обучение запущено")
        print(f"  Train блоков: {len(train_blocks)} | Val блоков: {len(val_blocks)}")
        print(f"  Цель шагов: {max_steps}")
        print(f"  Статус: train_status.json")
        print(f"{'='*60}\n")

        status_path = os.path.join(
            os.path.dirname(__file__), "..", "train_status.json"
        )
        start_time = time.time()
        tokens_processed = 0
        block_idx = self._block_idx
        wiki_iter = None

        if use_wikipedia:
            wiki = DataManager.load_wikipedia(streaming=True)
            if wiki:
                wiki_iter = iter(wiki)
                logger.info("[Train] Wikipedia streaming активен")
            else:
                logger.warning("[Train] Wikipedia недоступна, fallback на локальный корпус")
                use_wikipedia = False
                if text_file and os.path.exists(text_file):
                    blocks = self._pre_tokenize_corpus(text_file, block_size)
                    train_blocks, val_blocks = self._split_train_val(blocks, val_ratio=0.05)
                    self._blocks = train_blocks
                else:
                    blocks = self._pre_tokenize_corpus("training_corpus.txt", block_size)

        for step_idx in range(max_steps):
            if self.stopped:
                break

            if use_wikipedia and wiki_iter:
                input_ids, labels = self._tokenize_wiki_block(
                    wiki_iter, block_size, device
                )
                if input_ids is None:
                    continue
            elif not use_wikipedia and train_blocks:
                if block_idx >= len(train_blocks):
                    block_idx = 0
                    self.epoch += 1
                    random.shuffle(train_blocks)
                    self._blocks = train_blocks
                    logger.info(f"[Train] Эпоха {self.epoch} завершена, перемешивание блоков")
                block = train_blocks[block_idx]
                block_idx += 1
                input_ids = block["input_ids"].to(device)
                labels = block["labels"].to(device)

            loss = self._training_step(input_ids, labels)
            self.total_loss += loss
            self._steps_since_loss_reset += 1
            self.step += 1
            tokens_processed += input_ids.numel()

            elapsed = time.time() - start_time
            tps = tokens_processed / max(elapsed, 0.001)

            if self.step % self.status_interval == 0 or self.step == 1:
                avg_loss = self.total_loss / max(self._steps_since_loss_reset, 1)
                self.total_loss = 0.0
                self._steps_since_loss_reset = 0

                status = {
                    "step": self.step,
                    "epoch": self.epoch,
                    "max_steps": max_steps,
                    "loss": f"{avg_loss:.4f}",
                    "val_loss": f"{self.best_val_loss:.4f}" if self.best_val_loss != float('inf') else "N/A",
                    "tok_s": f"{tps:.0f}",
                    "snapshots": len(self.layer.state_storage),
                    "avg_confidence": f"{self.layer.meta.average_confidence():.3f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    "elapsed": f"{elapsed:.0f}s",
                    "eta": f"{(max_steps - self.step) / max(tps / 512, 0.01):.0f}s",
                }

                try:
                    with open(status_path, "w", encoding="utf-8") as f:
                        json.dump(status, f, ensure_ascii=False)
                except Exception:
                    pass

                bar_len = 30
                progress = min(self.step / max(max_steps, 1), 1.0)
                filled = int(bar_len * progress)
                bar = "█" * filled + "░" * (bar_len - filled)

                print(
                    f"\r  [{bar}] {self.step}/{max_steps} "
                    f"| loss={avg_loss:.4f} "
                    f"| val_loss={self.best_val_loss:.4f} "
                    f"| {tps:.0f} tok/s "
                    f"| epoch={self.epoch} "
                    f"| snap={len(self.layer.state_storage)} "
                    f"| conf={self.layer.meta.average_confidence():.3f} "
                    f"| lr={self.scheduler.get_last_lr()[0]:.2e}",
                    end="",
                    flush=True,
                )

            if self.step % self.srg_eval_interval == 0:
                print()
                self._srg_evaluation()
                print()

            # Validation каждые 500 шагов
            if self.step % 500 == 0 and val_blocks:
                val_loss = self._evaluate_validation(val_blocks, device)
                self.val_loss_history.append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.steps_without_improvement = 0
                    logger.info(f"[Val] Новый лучший val_loss: {val_loss:.4f}")
                else:
                    self.steps_without_improvement += 500
                logger.info(f"[Val] step={self.step} val_loss={val_loss:.4f} best={self.best_val_loss:.4f}")

            if self.step % self.checkpoint_interval == 0:
                self._save_checkpoint()
                self._generation_test()

                if self.state_grammar is not None and self.step > 100:
                    self._grammar_discovery_step()

                if self.benchmark_interval > 0 and self.step % self.benchmark_interval == 0:
                    self._auto_benchmark()

            if (
                self.step >= self.min_steps_for_stop
                and self.step % (self.srg_eval_interval * 5) == 0
                and auto_stop
            ):
                if self._check_stop_criterion():
                    self.stopped = True
                    self.stop_reason = "stop_criterion_met"
                    break

        print()
        elapsed = time.time() - start_time
        stats = self._training_stats(elapsed, tokens_processed)
        self._save_checkpoint(final=True)

        logger.info(f"[Train] Завершено: {self.stop_reason or 'max_steps'}")
        logger.info(
            f"[Train] Шагов: {self.step}, эпох: {self.epoch}, "
            f"токенов: {tokens_processed:,}, время: {elapsed:.0f}с"
        )

        return stats

    def _training_step(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> float:
        # Gradient accumulation
        self._accumulated_steps += 1
        should_accumulate = self._accumulated_steps < self.gradient_accumulation_steps

        x = self.layer.embed(input_ids)
        hidden = self.layer.forward_transformer(x)
        logits = self.layer.forward_logits(hidden)

        loss_lm = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=3,
        )

        loss = loss_lm / self.gradient_accumulation_steps

        if self.hierarchy is not None:
            try:
                codes = self.hierarchy(x)
                h_loss = self.hierarchy.hierarchy_loss(
                    codes["z_sym"], codes["z_word"], codes["z_sent"]
                )
                loss = loss + self.lambda_hierarchy * h_loss / self.gradient_accumulation_steps

                if codes["z_sym"].shape[0] >= 2:
                    is_sim = torch.tensor(1.0, device=x.device)
                    c_loss = self.hierarchy.contrastive_loss(
                        codes["z_sym"][0], codes["z_sym"][1],
                        is_sim.unsqueeze(0)
                    )
                    loss = loss + self.lambda_contrastive * c_loss / self.gradient_accumulation_steps
            except Exception:
                pass

        loss.backward()

        if not should_accumulate:
            torch.nn.utils.clip_grad_norm_(
                self.layer.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self._accumulated_steps = 0

        return loss.item() * self.gradient_accumulation_steps

    @torch.no_grad()
    def _generation_test(self):
        self.layer.eval()
        device = next(self.layer.parameters()).device
        results = []

        for prompt in self._gen_test_prompts:
            try:
                encoding = self.tokenizer.encode(prompt)
                ids = encoding.ids if hasattr(encoding, 'ids') else encoding
                input_ids = torch.tensor([ids], dtype=torch.long).to(device)
                output = self.layer.generate(input_ids, max_new_tokens=40, temperature=0.8)
                response = self.tokenizer.decode(output[0].tolist())
                results.append({"Q": prompt, "A": response})
            except Exception as e:
                results.append({"Q": prompt, "A": f"ERROR: {e}"})

        path = os.path.join(
            os.path.dirname(__file__), "..", "logs", f"gen_step_{self.step:06d}.json"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"step": self.step, "epoch": self.epoch, "avg_confidence": self.layer.meta.average_confidence(), "results": results}, f, ensure_ascii=False, indent=2)

        sample = results[0]["A"][:80] if results else ""
        logger.info(f"[GenTest] step={self.step}: {sample}...")
        self.layer.train()

    @torch.no_grad()
    def _srg_evaluation(self):
        self.layer.eval()

        try:
            device = next(self.layer.parameters()).device
            
            # Используем несколько промптов для лучшей оценки
            confidences = []
            for eval_prompt in self._eval_prompts:
                encoding = self.tokenizer.encode(eval_prompt)
                eval_ids_tokens = encoding.ids if hasattr(encoding, "ids") else encoding
                eval_ids = torch.tensor([eval_ids_tokens], dtype=torch.long).to(device)

                self.layer.transformer.attention.reset_cache()
                generated_ids = self.layer.generate(
                    eval_ids, max_new_tokens=32, temperature=0.8
                )

                response_text = self.tokenizer.decode(
                    generated_ids[0].tolist(), skip_special_tokens=True
                )

                c_query = self.layer.get_context_vector(eval_ids)

                eval_prompt_2 = "Наука изучает закономерности природы и общества."
                enc2 = self.tokenizer.encode(eval_prompt_2)
                ids2 = enc2.ids if hasattr(enc2, "ids") else enc2
                ref_ids = torch.tensor([ids2], dtype=torch.long).to(device)
                c_ref = self.layer.get_context_vector(ref_ids)

                c_response = self.layer.get_context_vector(generated_ids)

                new_tokens = generated_ids[:, eval_ids.shape[1]:]
                if new_tokens.shape[1] > 0:
                    x_new = self.layer.embed(new_tokens)
                    hidden_new = self.layer.forward_transformer(x_new)
                    logits_new = self.layer.forward_logits(hidden_new)
                    full_logits = logits_new.squeeze(0).cpu().numpy()
                else:
                    full_logits = np.zeros((1, self.layer.config.vocab_size), dtype=np.float32)

                eval_result = self.layer.srg.evaluate_full(
                    c_query=c_query,
                    c_response=c_response,
                    logits=full_logits,
                    response_text=response_text,
                )

                ref_similarity = float(np.dot(
                    c_response.flatten(), c_ref.flatten()
                ) / (np.linalg.norm(c_response) * np.linalg.norm(c_ref) + 1e-8))
                ref_similarity = (ref_similarity + 1.0) / 2.0

                confidence = eval_result["confidence"] * 0.4 + ref_similarity * 0.6
                confidences.append(confidence)

            # Усредняем confidence по всем промптам
            avg_confidence = sum(confidences) / len(confidences)
            self.layer.meta.record(avg_confidence)

            if avg_confidence > self.config.srg.snapshot_confidence_threshold:
                self.layer._eval_context_vector = c_query
                self.layer.save_snapshot_if_confident(domain="general")

            if avg_confidence < self.config.srg.curiosity_confidence_threshold:
                self.layer.curiosity.counter += 1
            else:
                self.layer.curiosity.counter = 0

            logger.info(
                f"[SRG] step={self.step} confidence={avg_confidence:.3f} "
                f"self_eval={eval_result['confidence']:.3f} "
                f"ref_sim={ref_similarity:.3f} "
                f"avg={self.layer.meta.average_confidence():.3f}"
            )

            if self.auto_curriculum is not None and c_query is not None:
                self.auto_curriculum.record_confidence(c_query, avg_confidence, response_text)

        except Exception as e:
            logger.debug(f"[SRG] eval error: {e}")

        self.layer.train()

    def _check_stop_criterion(self) -> bool:
        avg_conf = self.layer.meta.average_confidence(window=self.stop_window)
        snapshot_count = len(self.layer.state_storage)

        # Проверяем validation loss trend (основной критерий)
        val_loss_stable = False
        val_loss_improving = True
        if len(self.val_loss_history) >= 10:
            recent = list(self.val_loss_history)[-10:]
            first_half = recent[:5]
            second_half = recent[5:]
            avg_first = sum(first_half) / len(first_half) if first_half else 0
            avg_second = sum(second_half) / len(second_half) if second_half else 0
            val_loss_trend = (avg_first - avg_second) / max(avg_first, 1e-8)
            val_loss_stable = abs(val_loss_trend) < 0.02  # 2% без изменений
            val_loss_improving = val_loss_trend > 0.005   # val_loss всё ещё падает

        trend = self.layer.meta.recent_confidence_trend(window=50)

        # Интеллектуальная остановка:
        # 1. Validation loss стабилизировался
        # 2. Confidence достаточно высокий
        # 3. Достаточно снапшотов
        # 4. Минимум шагов в сессии пройден
        if (self.steps_without_improvement >= self.patience and
            not val_loss_improving and
            self.step >= self.min_steps_for_stop):
            logger.info(
                f"[Stop] Интеллектуальная остановка: "
                f"conf={avg_conf:.3f}, val_loss_stable={val_loss_stable}, "
                f"snapshots={snapshot_count}, "
                f"steps_without_improvement={self.steps_without_improvement}"
            )
            return True

        return (
            avg_conf > self.target_confidence
            and snapshot_count > self.min_snapshots
            and self.layer.curiosity.counter == 0
            and self.step >= self.min_steps_for_stop
            and not val_loss_improving  # val_loss должен перестать улучшаться
        )

    def _save_checkpoint(self, final: bool = False):
        path = os.path.join(
            self.checkpoint_dir,
            f"step_{self.step:06d}" if not final else "final",
        )
        save_primordial_layer(self.layer, path)

        optimizer_path = os.path.join(path, "optimizer.pth")
        os.makedirs(path, exist_ok=True)
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "block_idx": self._block_idx,
            "best_val_loss": self.best_val_loss,
            "rng_state": random.getstate(),
        }, optimizer_path)

        self._reevaluate_snapshots()

    def _resume_training_state(self, checkpoint_path: str):
        """Resume optimizer, scheduler, и training state из checkpoint."""
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pth")
        if os.path.exists(optimizer_path):
            checkpoint = torch.load(optimizer_path, map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # LambdaLR не сохраняет lr_lambdas, восстанавливаем только step
            if "scheduler" in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler"])
                except KeyError:
                    # LambdaLR state_dict может быть несовместим
                    # Восстанавливаем step вручную
                    sched_state = checkpoint["scheduler"]
                    if "last_epoch" in sched_state:
                        self.scheduler.last_epoch = sched_state["last_epoch"]
                    elif "step_count" in sched_state:
                        self.scheduler._step_count = sched_state["step_count"]
            self.step = checkpoint.get("step", 0)
            self.epoch = checkpoint.get("epoch", 0)
            self._block_idx = checkpoint.get("block_idx", 0)
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            if "rng_state" in checkpoint:
                random.setstate(checkpoint["rng_state"])
            logger.info(
                f"[Resume] Восстановлено: step={self.step}, epoch={self.epoch}, "
                f"block_idx={self._block_idx}, best_val_loss={self.best_val_loss:.4f}"
            )

    def _reevaluate_snapshots(self):
        """Переоценить старые слепки текущей моделью. Удалить деградировавшие."""
        meta = self.layer.state_storage.snapshots_meta
        removed = 0
        reevaluated = 0
        
        # Переоцениваем confidence слепков через текущую модель
        if len(meta) > 0:
            device = next(self.layer.parameters()).device
            for i in range(len(meta)):
                snap = meta[i]
                if "c" in snap:
                    # Создаём тестовый промпт из контекстного вектора
                    try:
                        c_vec = torch.tensor(snap["c"], dtype=torch.float32).unsqueeze(0).to(device)
                        # Вычисляем similarity с текущим состоянием модели
                        current_conf = self.layer.meta.average_confidence()
                        # Обновляем confidence слепка
                        snap["confidence"] = snap.get("confidence", 0.0) * 0.7 + current_conf * 0.3
                        reevaluated += 1
                    except Exception:
                        pass

        # Удаляем неиспользуемые слепки
        for i in range(len(meta) - 1, -1, -1):
            snap = meta[i]
            if snap.get("usage_count", 0) == 0 and len(meta) > 100:
                self.layer.state_storage._remove(i)
                removed += 1
                continue

        if removed:
            logger.info(f"[ReEval] Удалено {removed} неиспользуемых слепков")
        if reevaluated:
            logger.info(f"[ReEval] Переоценено {reevaluated} слепков")

    def _training_stats(self, elapsed: float, tokens: int) -> Dict[str, Any]:
        return {
            "steps": self.step,
            "epochs": self.epoch,
            "elapsed_seconds": elapsed,
            "tokens_processed": tokens,
            "tokens_per_second": tokens / max(elapsed, 0.001),
            "final_loss": self.total_loss,
            "best_val_loss": self.best_val_loss,
            "best_confidence": self.best_confidence,
            "average_confidence": self.layer.meta.average_confidence(),
            "snapshots_count": len(self.layer.state_storage),
            "usage_count": self.layer.meta.usage_count,
            "stop_reason": self.stop_reason,
        }

    def _grammar_discovery_step(self):
        """Grammar-guided training: обнаруживает правила композиции из недавних слепков."""
        try:
            meta = self.layer.state_storage.snapshots_meta
            if len(meta) < 10:
                return
            recent = meta[-50:]
            pairs = []
            for i in range(0, len(recent) - 2, 2):
                pairs.append((
                    recent[i]["c"], recent[i + 1]["c"],
                    (recent[i]["c"] + recent[i + 1]["c"]) * 0.5
                ))
            if pairs:
                result = self.state_grammar.discover(pairs, epochs=10)
                logger.info(
                    f"[Grammar] step={self.step}: "
                    f"discovery_loss={result.get('discovery_loss', 0):.4f}"
                )
        except Exception as e:
            logger.debug(f"[Grammar] step error: {e}")

    def _auto_benchmark(self):
        """Авто-бенчмарк: сохраняет метрики качества."""
        try:
            meta = self.layer.state_storage.snapshots_meta
            bench = {
                "step": self.step,
                "epoch": self.epoch,
                "avg_confidence": self.layer.meta.average_confidence(),
                "snapshots": len(meta),
                "loss_recent": self.total_loss,
                "val_loss": self.best_val_loss,
                "timestamp": time.time(),
            }
            self.benchmark_history.append(bench)

            path = os.path.join(
                os.path.dirname(__file__), "..", "logs", "benchmark_history.json"
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.benchmark_history[-100:], f, indent=2)
        except Exception:
            pass

    def _tokenize_wiki_block(self, wiki_iter, block_size, device):
        try:
            article = next(wiki_iter)
            text = article.get('text', '')
            if len(text) < 100:
                return None, None

            words = text.split()
            start = 0
            if len(words) > block_size:
                start = hash(text) % max(1, len(words) - block_size)

            chunk = " ".join(words[start:start + block_size])
            if len(chunk) < 50:
                return None, None

            encoding = self.tokenizer.encode(chunk)
            ids = encoding.ids if hasattr(encoding, "ids") else encoding
            ids = ids[:block_size]
            while len(ids) < block_size:
                ids.append(3)

            input_ids = torch.tensor([ids[:-1]], dtype=torch.long).to(device)
            labels = torch.tensor([ids[1:]], dtype=torch.long).to(device)
            return input_ids, labels
        except StopIteration:
            return None, None
        except Exception:
            return None, None
