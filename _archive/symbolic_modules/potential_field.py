"""
PotentialField — поле потенциалов связей между символами.

Не веса модели, а накопленная статистика co-occurrence
с учётом attention. Аналог "синаптической карты" в мозге.

P[i, j] — потенциал связи символа i → символ j.
Усиливается при совместной активации через attention.
Ослабляется при неиспользовании (decay).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from loguru import logger


class PotentialField(nn.Module):
    """
    Матрица потенциалов [V × V] — сила аффинности между символами.

    Использует экспоненциальное скользящее среднее (EMA) для накопления
    co-occurrence с учётом attention weights.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256):
        super().__init__()

        # Потенциалы: обучаемый параметр + накопленный опыт
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Символьные эмбеддинги (потенциалы символов)
        self.char_potential = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)

        # Матрица аффинности [V × V] — сила связи i→j
        # Инициализация: все связи нейтральны (0.5)
        self.register_buffer("affinity", torch.full((vocab_size, vocab_size), 0.5))

        # Счётчик co-occurrence для взвешенного обновления
        self.register_buffer("co_occurrence_count", torch.zeros(vocab_size, vocab_size, dtype=torch.float64))

        # Гиперпараметры
        self.ema_alpha: float = 0.0005  # tiny EMA: frequent pairs → 0.8, rare → 0.505
        self.decay_rate: float = 0.9999  # распад неиспользуемых связей
        self.global_decay_rate: float = 0.99999  # СВЕРХМЕДЛЕННЫЙ глобальный decay
        self.min_affinity: float = 0.01
        self.max_affinity: float = 10.0

    def strengthen(self, i: int, j: int, attention_weight: float, weight: float = 1.0):
        """Инкремент co-occurrence счётчика. Affinity вычисляется периодически через PMI+count."""
        self.co_occurrence_count[i, j] += weight * (1.0 + attention_weight)
        # Простая формула: count-based с высоким порогом (без насыщения)
        threshold = 100000.0  # FIX: was 500000 — sync with train_to_convergence.py
        raw = float(self.co_occurrence_count[i, j])
        self.affinity[i, j] = 0.5 + 0.5 * min(raw / threshold, 1.0)

    def strengthen_batch(
        self,
        input_ids: torch.Tensor,         # [B, L]
        attention_matrix: torch.Tensor,  # [B, H, L, L]
        confidence: float = 0.5,
    ):
        B, H, L, _ = attention_matrix.shape
        avg_attention = attention_matrix.mean(dim=1)  # [B, L, L]
        weight = min(confidence, 0.95)

        for b in range(B):
            for i in range(L):
                for j in range(i + 1, min(i + 5, L)):  # ТОЛЬКО ближайшие соседи
                    attn_w = float(avg_attention[b, i, j])
                    boost = 0.01 if j == i + 1 else 0.0  # Базовый boost для соседей
                    effective_w = max(attn_w, boost)
                    if effective_w > 0.0001:
                        idx_i = int(input_ids[b, i].item())
                        idx_j = int(input_ids[b, j].item())
                        if idx_i < self.vocab_size and idx_j < self.vocab_size:
                            self.strengthen(idx_i, idx_j, effective_w, weight)

    def weaken_all(self, factor: float = None):
        """Глобальный decay неиспользуемых связей (sleep mode)."""
        rate = factor if factor is not None else self.decay_rate
        self.affinity *= rate
        self.affinity = self.affinity.clamp(self.min_affinity, self.max_affinity)

    def recompute_affinity_hybrid(self):
        """Пересчитать всю матрицу аффинности через PMI+count гибрид."""
        import numpy as np
        count = self.co_occurrence_count.cpu().numpy()
        n = count.shape[0]
        total = count.sum() + 1e-10
        rs = count.sum(axis=1, keepdims=True) + 1e-10
        cs = count.sum(axis=0, keepdims=True) + 1e-10

        pmi = np.maximum(np.log(count * total / (rs * cs)), 0)
        pmi_norm = pmi / (pmi.max() + 1e-10)
        cnt_norm = np.minimum(count / 600000.0, 1.0)

        hybrid = 0.5 + 0.25 * cnt_norm + 0.25 * pmi_norm
        np.fill_diagonal(hybrid, 0.5)

        self.affinity = torch.tensor(hybrid.astype(np.float32))

    def weaken_unused(self, min_usage: int = 10):
        """Ослабить связи с малым количеством co-occurrence."""
        mask = self.co_occurrence_count < min_usage
        self.affinity[mask] *= 0.9
        self.co_occurrence_count[mask] = self.co_occurrence_count[mask] * 0.5

    def get_continuation_potential(self, char_idx: int) -> torch.Tensor:
        """
        Потенциал продолжения для символа:
        какие символы логично следуют за ним.
        Возвращает вектор [vocab_size] — потенциал каждого символа как продолжения.
        """
        return self.affinity[char_idx]

    def get_affinity_matrix(self, normalize: bool = True) -> torch.Tensor:
        """Нормализованная матрица аффинности."""
        A = self.affinity.clone()
        if normalize:
            row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            A = A / row_sums
        return A

    def get_affinity_submatrix(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Суб-матрица аффинности для группы символов."""
        idx = torch.tensor(indices, dtype=torch.long)
        sub = self.affinity[idx][:, idx]
        usage = self.co_occurrence_count[idx][:, idx]
        return sub, usage

    def top_continuations(self, char_idx: int, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Топ-k символов-продолжений для данного символа."""
        potentials = self.get_continuation_potential(char_idx)
        values, indices = torch.topk(potentials, min(k, len(potentials)))
        return indices, values

    def semantic_distance(self, idx_a: int, idx_b: int) -> float:
        """Семантическое расстояние между двумя символами на основе потенциалов."""
        va = self.get_continuation_potential(idx_a)
        vb = self.get_continuation_potential(idx_b)
        cos_sim = torch.nn.functional.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0))
        return float(1.0 - cos_sim.item())

    def summary(self) -> str:
        mean_aff = self.affinity.mean().item()
        active_connections = (self.affinity > 0.5).sum().item()
        total = self.vocab_size * self.vocab_size
        return (
            f"PotentialField(V={self.vocab_size}, d={self.embed_dim}, "
            f"mean_aff={mean_aff:.3f}, active={active_connections}/{total})"
        )
