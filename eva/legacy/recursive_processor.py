"""
RecursiveProcessor — рекурсивная обработка (Пункт 5).

Перед созданием нового слоя система пробует «додумать» ответ,
пропуская скрытое состояние через последний слой несколько раз.

Протокол:
1. Запрос → все существующие слои → скрытое состояние x
2. x пропускается через последний слой повторно
3. Генерация ответа → SRG-оценка
4. Если confidence > 0.7 — успех, рекурсия остановлена
5. Если confidence низкая — повтор до MAX_DEPTH (5)
6. Если после исчерпания лимита confidence низкая → сигнал к кристаллизации

Защита:
- MAX_RECURSION_DEPTH = 5
- Таймаут на запрос (секунды)
- Адаптивное демпфирование скрытого состояния
"""

import time
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from loguru import logger


class RecursiveProcessor:

    def __init__(
        self,
        max_depth: int = 5,
        confidence_threshold: float = 0.7,
        damping: float = 0.85,
        timeout_seconds: float = 30.0,
    ):
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold
        self.damping = damping
        self.timeout_seconds = timeout_seconds

        self.recursion_history: List[Dict[str, Any]] = []
        self.failed_queries: List[Dict[str, Any]] = []
        self.recursion_exhausted_count: int = 0

    def process(
        self,
        layer,
        input_ids: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        start_time = time.time()

        x = layer.embed(input_ids)
        hidden = layer.forward_transformer(x)

        best_confidence = 0.0
        best_response = ""
        best_hidden = hidden

        for depth in range(self.max_depth + 1):
            if time.time() - start_time > self.timeout_seconds:
                logger.warning(f"[Recursion] Таймаут на глубине {depth}")
                break

            if depth > 0:
                hidden = hidden * self.damping + layer.forward_transformer(
                    layer.forward_transformer(hidden)
                ) * (1.0 - self.damping)

            logits = layer.forward_logits(hidden)
            response_ids = self._sample_from_logits(
                layer, logits, input_ids, max_new_tokens
            )
            response_text = tokenizer.decode(
                response_ids[0].tolist(), skip_special_tokens=True
            )

            c_query = layer.get_context_vector(input_ids)
            c_response = layer.get_context_vector(
                torch.cat([input_ids, response_ids[:, -64:]], dim=1)
                if response_ids.shape[1] > 64
                else response_ids
            )

            last_logits = logits[:, -1, :].squeeze(0).cpu().numpy()
            eval_result = layer.srg.evaluate_full(
                c_query=c_query,
                c_response=c_response,
                logits=last_logits,
                response_text=response_text,
            )

            confidence = eval_result["confidence"]

            self.recursion_history.append({
                "depth": depth,
                "confidence": float(confidence),
                "response_len": len(response_text),
            })

            if confidence > best_confidence:
                best_confidence = confidence
                best_response = response_text
                best_hidden = hidden

            if confidence >= self.confidence_threshold:
                logger.info(
                    f"[Recursion] Успех на глубине {depth}: "
                    f"confidence={confidence:.3f}"
                )
                return {
                    "response": response_text,
                    "confidence": float(confidence),
                    "recursion_depth": depth,
                    "recursion_exhausted": False,
                    "history": self.recursion_history,
                }

        logger.warning(
            f"[Recursion] Исчерпан лимит ({self.max_depth}): "
            f"best_confidence={best_confidence:.3f}"
        )

        self.recursion_exhausted_count += 1

        return {
            "response": best_response,
            "confidence": float(best_confidence),
            "recursion_depth": self.max_depth,
            "recursion_exhausted": True,
            "history": self.recursion_history,
            "best_hidden": best_hidden,
        }

    def _sample_from_logits(
        self,
        layer,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / 0.8
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            x = layer.embed(generated)
            hidden = layer.forward_transformer(x)
            logits = layer.forward_logits(hidden)

        return generated

    def should_crystallize(self, threshold: int = 3) -> bool:
        return self.recursion_exhausted_count >= threshold

    def add_failed_query(
        self, query: str, response: str, confidence: float
    ):
        self.failed_queries.append({
            "query": query,
            "response": response,
            "confidence": confidence,
        })
        if len(self.failed_queries) > 100:
            self.failed_queries.pop(0)

    def get_failed_queries(self) -> list:
        return list(self.failed_queries)

    def reset(self):
        self.recursion_history.clear()
        self.recursion_exhausted_count = 0
