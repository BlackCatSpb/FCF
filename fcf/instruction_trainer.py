"""
InstructionTrainer — инструктивное дообучение (Пункт 3).

Обучает PrimordialLayer следовать инструкциям на датасете пар
«инструкция → ответ» (Saiga или локальный JSON).

Отличия от Пункта 2 (языковое обучение):
- Данные: размеченные пары вместо сырого текста
- Формат: chat-шаблон <|im_start|> с маскированием user-части
- Learning rate: 1e-5 (на порядок ниже, сохраняет языковые навыки)
- Контрольная выборка инструкций для SRG-оценки
- Буфер pending_clarifications для неудачных примеров
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


CHAT_IM_START = "<|im_start|>"
CHAT_IM_END = "<|im_end|>"


class InstructionTrainer:

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
            os.path.dirname(__file__), "..", "checkpoints", "instruction"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.layer.parameters(),
            lr=self.config.training.lora_learning_rate,
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

        self.pending_clarifications: List[Dict] = []

    def _format_chat(self, instruction: str, output: str) -> str:
        return (
            f"{CHAT_IM_START}user\n{instruction}{CHAT_IM_END}\n"
            f"{CHAT_IM_START}assistant\n{output}{CHAT_IM_END}"
        )

    def _tokenize_with_mask(
        self, instruction: str, output: str, max_len: int = 512
    ) -> tuple:
        full_text = self._format_chat(instruction, output)
        prefix = f"{CHAT_IM_START}user\n{instruction}{CHAT_IM_END}\n{CHAT_IM_START}assistant\n"

        try:
            full_enc = self.tokenizer.encode(full_text)
            full_ids = full_enc.ids if hasattr(full_enc, "ids") else full_enc

            prefix_enc = self.tokenizer.encode(prefix)
            prefix_len = len(
                prefix_enc.ids if hasattr(prefix_enc, "ids") else prefix_enc
            )
        except Exception as e:
            logger.debug(f"[Tokenize] Ошибка: {e}")
            return None, None

        if len(full_ids) < 4 or prefix_len >= len(full_ids):
            return None, None

        full_ids = full_ids[:max_len]
        while len(full_ids) < max_len:
            full_ids.append(0)

        labels = full_ids[1:] + [0]
        labels[: prefix_len - 1] = [-100] * max(0, prefix_len - 1)

        input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long)
        labels = torch.tensor([labels[:-1]], dtype=torch.long)

        return input_ids, labels

    def _load_instructions(
        self, json_path: str
    ) -> List[Dict[str, str]]:
        if not os.path.exists(json_path):
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return []

    def _prepare_instruction_blocks(
        self, pairs: List[Dict[str, str]], max_blocks: int = 5000
    ) -> List[Dict[str, torch.Tensor]]:
        blocks = []
        for pair in pairs:
            instruction = pair.get("instruction", "")
            output = pair.get("output", "")
            if not instruction or not output:
                continue

            input_ids, labels = self._tokenize_with_mask(instruction, output)
            if input_ids is None:
                continue

            blocks.append({"input_ids": input_ids, "labels": labels})
            if len(blocks) >= max_blocks:
                break

        return blocks

    def train(
        self,
        instructions_file: str = None,
        max_steps: Optional[int] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        import sys
        from loguru import logger as loguru_logger

        log_path = os.path.join(
            os.path.dirname(__file__), "..", "train_instruction_log.txt"
        )
        log_id = loguru_logger.add(
            log_path,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO",
            encoding="utf-8",
            enqueue=True,
        )

        logger.info("=" * 60)
        logger.info("Пункт 3 — Инструктивное дообучение")
        logger.info(f"Устройство: {device}")
        logger.info(f"Learning rate: {self.config.training.lora_learning_rate}")
        logger.info("=" * 60)

        if device == "cpu":
            num_threads = min(os.cpu_count() or 4, 4)
            torch.set_num_threads(num_threads)
            logger.info(f"[CPU] Потоков: {num_threads}")

        self.layer.to(device)
        self.layer.train()

        pairs = []
        if instructions_file and os.path.exists(instructions_file):
            pairs = self._load_instructions(instructions_file)

        if not pairs:
            logger.error("Нет данных для обучения.")
            return {"error": "no_data"}

        blocks = self._prepare_instruction_blocks(pairs)
        if not blocks:
            logger.error("Не удалось токенизировать инструкции.")
            return {"error": "tokenize_failed"}

        max_steps = max_steps or self.config.training.max_steps or len(blocks) * 100
        logger.info(f"[Train] Пар: {len(pairs)}, блоков: {len(blocks)}, макс. шагов: {max_steps}")

        print(f"\n{'='*60}")
        print(f"  Инструктивное дообучение запущено")
        print(f"  Инструкций: {len(pairs)} | Блоков: {len(blocks)} | Цель шагов: {max_steps}")
        print(f"  Статус: train_instruction_status.json")
        print(f"{'='*60}\n")

        status_path = os.path.join(
            os.path.dirname(__file__), "..", "train_instruction_status.json"
        )
        start_time = time.time()
        tokens_processed = 0
        block_idx = 0

        for step_idx in range(max_steps):
            if self.stopped:
                break

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
                self._srg_evaluation(blocks, device)
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

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.layer.parameters(), max_norm=1.0
        )

        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _srg_evaluation(self, blocks: list, device: str):
        self.layer.eval()

        try:
            test_block = blocks[0]
            input_ids = test_block["input_ids"].to(device)

            generated_ids = self.layer.generate(
                input_ids, max_new_tokens=64, temperature=0.7
            )

            response_text = self.tokenizer.decode(
                generated_ids[0].tolist(), skip_special_tokens=True
            )

            c_query = self.layer.get_context_vector(input_ids)

            response_ids = generated_ids[:, -64:]
            c_response = self.layer.get_context_vector(
                torch.cat([input_ids, response_ids], dim=1)
            )

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
                self.layer.save_snapshot_if_confident(domain="instruction")

            if confidence < self.config.srg.curiosity_confidence_threshold:
                self.layer.curiosity.counter += 1
            else:
                self.layer.curiosity.counter = 0

            if self.layer.curiosity.should_ask(confidence):
                instruction = "инструктивный запрос"
                question = self.layer.curiosity.generate_clarification(
                    layer=self.layer,
                    tokenizer=self.tokenizer,
                    original_query=instruction,
                    generated_answer=response_text[-200:],
                )
                self.pending_clarifications.append({
                    "instruction": instruction,
                    "answer": response_text[-200:],
                    "question": question,
                })

            logger.info(
                f"[SRG] step={self.step} confidence={confidence:.3f} "
                f"avg={self.layer.meta.average_confidence():.3f} "
                f"ethics={eval_result['ethics_score']:.3f}"
            )

            sample = response_text[:120].replace("\n", " ")
            logger.info(f"[SRG] Ответ: {sample}...")

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
            "pending_clarifications": len(self.pending_clarifications),
            "stop_reason": self.stop_reason,
        }
