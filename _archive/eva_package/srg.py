"""
SemanticRelevanceGate (SRG) — механизм самооценки ответов.

Вычисляет итоговую уверенность по трём компонентам:
1. Семантическое сходство (косинусное) запроса и ответа
2. Энтропийная уверенность (из логитов последнего токена)
3. Этический скор (через EthicsFilter)

Формула:
    confidence = w_sim * similarity + w_ent * entropy_score + w_eth * ethics_score

Если ethics_score < ethics_threshold → ответ безусловно отклоняется (0.0).
"""

import numpy as np
from typing import Optional
from loguru import logger

from .ethics_filter import EthicsFilter


class SemanticRelevanceGate:

    def __init__(
        self,
        w_sim: float = 0.4,
        w_ent: float = 0.3,
        w_eth: float = 0.3,
        ethics_threshold: float = 0.3,
    ):
        self.w_sim = w_sim
        self.w_ent = w_ent
        self.w_eth = w_eth
        self.ethics_threshold = ethics_threshold
        self.ethics_filter = EthicsFilter(ethics_threshold)

    def evaluate(
        self,
        c_query: np.ndarray,
        c_response: np.ndarray,
        logits: Optional[np.ndarray] = None,
        response_text: str = "",
    ) -> float:
        similarity = self._compute_similarity(c_query, c_response)

        entropy_score = 0.5
        if logits is not None and len(logits) > 0:
            entropy_score = self._compute_entropy_score(logits)

        ethics_score, _ = self.ethics_filter.evaluate(response_text)
        if ethics_score < self.ethics_threshold:
            logger.debug(
                f"[SRG] Ответ отклонён этическим фильтром (score={ethics_score:.2f})"
            )
            return 0.0

        confidence = (
            self.w_sim * similarity
            + self.w_ent * entropy_score
            + self.w_eth * ethics_score
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def evaluate_full(
        self,
        c_query: np.ndarray,
        c_response: np.ndarray,
        logits: Optional[np.ndarray] = None,
        response_text: str = "",
    ) -> dict:
        similarity = self._compute_similarity(c_query, c_response)

        entropy_score = 0.5
        if logits is not None and len(logits) > 0:
            entropy_score = self._compute_entropy_score(logits)

        ethics_score, axiom_scores = self.ethics_filter.evaluate(response_text)

        if ethics_score < self.ethics_threshold:
            confidence = 0.0
        else:
            confidence = (
                self.w_sim * similarity
                + self.w_ent * entropy_score
                + self.w_eth * ethics_score
            )
            confidence = float(np.clip(confidence, 0.0, 1.0))

        return {
            "confidence": confidence,
            "similarity": float(similarity),
            "entropy_score": float(entropy_score),
            "ethics_score": float(ethics_score),
            "axiom_scores": axiom_scores,
        }

    def _compute_similarity(
        self, c_query: np.ndarray, c_response: np.ndarray
    ) -> float:
        c_q = c_query.flatten()
        c_r = c_response.flatten()

        dot = np.dot(c_q, c_r)
        norm = np.linalg.norm(c_q) * np.linalg.norm(c_r) + 1e-8
        sim = dot / norm

        return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))

    def _compute_entropy_score(self, logits: np.ndarray) -> float:
        if logits.ndim == 2:
            scores = []
            for t in range(logits.shape[0]):
                logits_t = logits[t].flatten()
                logits_stable = logits_t - np.max(logits_t)
                probs = np.exp(logits_stable) / np.sum(np.exp(logits_stable))
                probs = np.clip(probs, 1e-10, 1.0)
                entropy = -np.sum(probs * np.log2(probs))
                max_entropy = np.log2(len(logits_t))
                max_entropy = max(max_entropy, 1e-10)
                scores.append(float(1.0 - entropy / max_entropy))
            return float(np.mean(scores)) if scores else 0.5

        logits_flat = logits.flatten()
        logits_stable = logits_flat - np.max(logits_flat)
        probs = np.exp(logits_stable) / np.sum(np.exp(logits_stable))
        probs = np.clip(probs, 1e-10, 1.0)

        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(logits_flat))
        max_entropy = max(max_entropy, 1e-10)

        return float(1.0 - entropy / max_entropy)
