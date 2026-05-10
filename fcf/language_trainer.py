"""
LanguageTrainer — самообучение языку (Пункт 2).

Реализует автономный цикл обучения PrimordialLayer на задаче
предсказания следующего токена (Causal LM).

Особенности:
- Предварительная токенизация корпуса в блоки (быстрее цикла)
- Статус-файл train_status.json для мониторинга
- Периодический вывод прогресса
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Iterator, List
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
    ):
        self.layer = layer
        self.tokenizer = tokenizer
        self.config = config or layer.config
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", "language"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.layer.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        self.step: int = 0
        self.total_loss: float = 0.0
        self.best_confidence: float = 0.0

        self.srg_eval_interval: int = 100
        self.checkpoint_interval: int = 1000
        self.log_interval: int = 10
        self.status_interval: int = 5
        self.stop_window: int = 500
        self.min_snapshots: int = 1000
        self.target_confidence: float = 0.7

        self.stopped: bool = False
        self.stop_reason: str = ""

        self._eval_prompt = (
            "История — это наука о прошлом человеческого общества. "
            "Она изучает события, процессы и закономерности развития. "
            "Историки исследуют"
        )

    def _pre_tokenize_corpus(
        self, text_file: str, block_size: int = 512
    ) -> List[Dict[str, torch.Tensor]]:
        logger.info(f"[PreTokenize] Токенизация: {text_file}")
        blocks = []

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

        for i in range(0, len(ids) - block_size, max(block_size // 2, 1)):
            chunk = ids[i : i + block_size + 1]
            if len(chunk) < 4:
                continue
            while len(chunk) < block_size + 1:
                chunk.append(0)

            input_ids = torch.tensor([chunk[:-1]], dtype=torch.long)
            labels = torch.tensor([chunk[1:]], dtype=torch.long)
            blocks.append({"input_ids": input_ids, "labels": labels})

            if len(blocks) >= 10000:
                break

        logger.info(f"[PreTokenize] Готово: {len(blocks)} блоков (block_size={block_size})")
        return blocks

    def train(
        self,
        text_file: str = None,
        max_steps: Optional[int] = None,
        block_size: int = 512,
        device: str = "cpu",
        use_wikipedia: bool = False,
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
        logger.info("=" * 60)

        if device == "cpu":
            num_threads = min(os.cpu_count() or 4, 16)
            torch.set_num_threads(num_threads)
            logger.info(f"[CPU] Потоков: {num_threads}")

        self.layer.to(device)
        self.layer.train()

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
        else:
            max_steps = max_steps or self.config.training.max_steps or len(blocks) * 10
            logger.info(f"[Train] Блоков: {len(blocks)}, макс. шагов: {max_steps}")
        print(f"\n{'='*60}")
        print(f"  Обучение запущено")
        print(f"  Блоков: {len(blocks)} | Цель шагов: {max_steps}")
        print(f"  Статус: train_status.json")
        print(f"{'='*60}\n")

        status_path = os.path.join(
            os.path.dirname(__file__), "..", "train_status.json"
        )
        start_time = time.time()
        tokens_processed = 0
        block_idx = 0
        wiki_iter = None

        if use_wikipedia:
            wiki = DataManager.load_wikipedia(streaming=True)
            if wiki:
                wiki_iter = iter(wiki)
            else:
                logger.error("Wikipedia не загружена")
                return {"error": "no_wikipedia"}

        for step_idx in range(max_steps):
            if self.stopped:
                break

            if use_wikipedia and wiki_iter:
                input_ids, labels = self._tokenize_wiki_block(
                    wiki_iter, block_size, device
                )
                if input_ids is None:
                    continue
            else:
                block = blocks[block_idx % len(blocks)]
                block_idx += 1
                input_ids = block["input_ids"].to(device)
                labels = block["labels"].to(device)

            loss = self._training_step(input_ids, labels)
            self.total_loss += loss
            self.step += 1
            tokens_processed += input_ids.numel()

            elapsed = time.time() - start_time
            tps = tokens_processed / max(elapsed, 0.001)

            if self.step % self.status_interval == 0 or self.step == 1:
                avg_loss = self.total_loss / max(
                    self.step % self.log_interval or 1, 1
                )
                self.total_loss = 0.0

                status = {
                    "step": self.step,
                    "max_steps": max_steps,
                    "loss": f"{avg_loss:.4f}",
                    "tok_s": f"{tps:.0f}",
                    "snapshots": len(self.layer.state_storage),
                    "avg_confidence": f"{self.layer.meta.average_confidence():.3f}",
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
                    f"| {tps:.0f} tok/s "
                    f"| snap={len(self.layer.state_storage)} "
                    f"| conf={self.layer.meta.average_confidence():.3f}",
                    end="",
                    flush=True,
                )

            if self.step % self.srg_eval_interval == 0:
                print()
                self._srg_evaluation()
                print()

            if self.step % self.checkpoint_interval == 0:
                self._save_checkpoint()

            if (
                self.step >= self.stop_window
                and self.step % (self.srg_eval_interval * 5) == 0
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
            f"[Train] Шагов: {self.step}, токенов: {tokens_processed:,}, "
            f"время: {elapsed:.0f}с"
        )

        return stats

    def _training_step(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> float:
        self.optimizer.zero_grad()

        x = self.layer.embed(input_ids)
        hidden = self.layer.forward_transformer(x)
        logits = self.layer.forward_logits(hidden)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.layer.parameters(), max_norm=1.0
        )

        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _srg_evaluation(self):
        self.layer.eval()

        try:
            device = next(self.layer.parameters()).device
            encoding = self.tokenizer.encode(self._eval_prompt)
            eval_ids_tokens = encoding.ids if hasattr(encoding, "ids") else encoding
            eval_ids = torch.tensor([eval_ids_tokens], dtype=torch.long).to(device)

            generated_ids = self.layer.generate(
                eval_ids, max_new_tokens=32, temperature=0.8
            )

            response_text = self.tokenizer.decode(
                generated_ids[0].tolist(), skip_special_tokens=True
            )

            c_query = self.layer.get_context_vector(eval_ids)
            c_response = self.layer.get_context_vector(generated_ids)

            x = self.layer.embed(generated_ids)
            hidden = self.layer.forward_transformer(x)
            logits = self.layer.forward_logits(hidden)
            last_logits = logits[:, -1, :].squeeze(0).cpu().numpy()

            eval_result = self.layer.srg.evaluate_full(
                c_query=c_query,
                c_response=c_response,
                logits=last_logits,
                response_text=response_text,
            )

            confidence = eval_result["confidence"]
            self.layer.meta.record(confidence)

            if confidence > self.config.srg.snapshot_confidence_threshold:
                self.layer._eval_context_vector = c_query
                self.layer.save_snapshot_if_confident(domain="general")

            if confidence < self.config.srg.curiosity_confidence_threshold:
                self.layer.curiosity.counter += 1
            else:
                self.layer.curiosity.counter = 0

            logger.info(
                f"[SRG] step={self.step} confidence={confidence:.3f} "
                f"avg={self.layer.meta.average_confidence():.3f}"
            )

        except Exception as e:
            logger.warning(f"[SRG] Ошибка: {e}")

        self.layer.train()

    def _check_stop_criterion(self) -> bool:
        avg_conf = self.layer.meta.average_confidence(window=self.stop_window)
        snapshot_count = len(self.layer.state_storage)
        return (
            avg_conf > self.target_confidence
            and snapshot_count > self.min_snapshots
            and self.layer.curiosity.counter == 0
        )

    def _save_checkpoint(self, final: bool = False):
        path = os.path.join(
            self.checkpoint_dir,
            f"step_{self.step:06d}" if not final else "final",
        )
        save_primordial_layer(self.layer, path)

    def _training_stats(self, elapsed: float, tokens: int) -> Dict[str, Any]:
        return {
            "steps": self.step,
            "elapsed_seconds": elapsed,
            "tokens_processed": tokens,
            "tokens_per_second": tokens / max(elapsed, 0.001),
            "final_loss": self.total_loss,
            "best_confidence": self.best_confidence,
            "average_confidence": self.layer.meta.average_confidence(),
            "snapshots_count": len(self.layer.state_storage),
            "usage_count": self.layer.meta.usage_count,
            "stop_reason": self.stop_reason,
        }

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
                ids.append(0)

            input_ids = torch.tensor([ids[:-1]], dtype=torch.long).to(device)
            labels = torch.tensor([ids[1:]], dtype=torch.long).to(device)
            return input_ids, labels
        except StopIteration:
            return None, None
        except Exception:
            return None, None
