"""
PotentialDynamics — временная динамика потенциалов связей.

Реализует механизмы синаптической пластичности:
1. STDP (Spike-Timing-Dependent Plasticity):
   Если символ A активируется перед B → усиление A→B
   Если B перед A → ослабление A→B
2. LTP/DLTP (Long-Term Potentiation/Depression):
   Часто используемые связи усиливаются, неиспользуемые — ослабляются
3. Homeostatic scaling:
   Суммарный потенциал нейрона сохраняется в заданном диапазоне
4. Metaplasticity:
   Порог пластичности адаптируется под активность

Ключевая идея: вес связи — не статическое число, а функция времени и истории.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from collections import deque
from loguru import logger


class PotentialDynamics(nn.Module):
    """
    Управляет временной эволюцией потенциалов связей.

    Каждая связь i→j имеет:
    - affinity: текущий потенциал (сила связи)
    - trace: след недавней активности (для STDP)
    - last_update: время последнего обновления
    - plasticity: обучаемость связи (мета-пластичность)
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Основная матрица аффинности [V, V]
        self.register_buffer("affinity", torch.full((vocab_size, vocab_size), 0.5))

        # Следы активности для STDP
        self.register_buffer("pre_trace", torch.zeros(vocab_size))
        self.register_buffer("post_trace", torch.zeros(vocab_size))

        # Счётчики использования
        self.register_buffer("usage_count", torch.zeros(vocab_size, vocab_size))
        self.register_buffer("last_update", torch.zeros(vocab_size, vocab_size))

        # Мета-пластичность: насколько легко менять связь
        self.register_buffer("plasticity", torch.ones(vocab_size, vocab_size))

        # История для анализа трендов
        self.affinity_history: deque = deque(maxlen=100)

        # Параметры STDP
        self.tau_pre: float = 20.0    # постоянная времени pre-synaptic trace
        self.tau_post: float = 20.0   # постоянная времени post-synaptic trace
        self.A_plus: float = 0.005    # амплитуда усиления (pre→post)
        self.A_minus: float = 0.003   # амплитуда ослабления (post→pre)

        # Параметры LTP/LTD
        self.ltp_rate: float = 0.01    # скорость долговременного усиления
        self.ltd_rate: float = 0.001   # скорость долговременного ослабления
        self.decay_tau: float = 10000  # постоянная времени распада

        # Homeostatic parameters
        self.target_mean: float = 0.5   # целевое среднее аффинности
        self.homeostatic_rate: float = 0.001

        # Bounds
        self.min_affinity: float = 0.001
        self.max_affinity: float = 10.0

        self.global_step: int = 0

    def stdp_update(self, pre_idx: int, post_idx: int, dt: float = 1.0):
        """
        STDP обновление: pre перед post → усиление, post перед pre → ослабление.

        pre_trace: экспоненциально затухающий след pre-активации
        post_trace: экспоненциально затухающий след post-активации

        Δw = A_plus * pre_trace[post] (если pre был недавно)
           - A_minus * post_trace[pre] (если post был недавно)
        """
        pre_trace = self.pre_trace[pre_idx].item()
        post_trace_val = self.post_trace[post_idx].item()

        if pre_trace > 0.01:
            delta = self.A_plus * pre_trace * self.plasticity[pre_idx, post_idx]
            self.affinity[pre_idx, post_idx] += delta

        if post_trace_val > 0.01:
            delta = -self.A_minus * post_trace_val * self.plasticity[pre_idx, post_idx]
            self.affinity[pre_idx, post_idx] += delta

        self.affinity[pre_idx, post_idx] = self.affinity[pre_idx, post_idx].clamp(
            self.min_affinity, self.max_affinity
        )

    def update_traces(self, active_indices: List[int], dt: float = 1.0):
        """Обновить следы активности: decay + новые активации."""
        decay_pre = np.exp(-dt / self.tau_pre)
        decay_post = np.exp(-dt / self.tau_post)

        self.pre_trace *= decay_pre
        self.post_trace *= decay_post

        for idx in active_indices:
            if 0 <= idx < self.vocab_size:
                self.pre_trace[idx] = 1.0
                self.post_trace[idx] = 1.0

    def reinforce_sequence(
        self,
        sequence: List[int],
        attention_weights: Optional[List[float]] = None,
        confidence: float = 0.5,
    ):
        """
        Усилить связи в последовательности с учётом STDP и attention.

        Для каждой пары i,j в последовательности:
        1. STDP: если i перед j, усиливаем i→j
        2. Attention: вес внимания между i и j модулирует усиление
        3. Confidence: общая уверенность масштабирует обновление
        """
        n = len(sequence)
        if n < 2:
            return

        for i in range(n):
            for j in range(i + 1, min(i + 5, n)):
                idx_i = sequence[i]
                idx_j = sequence[j]

                if idx_i >= self.vocab_size or idx_j >= self.vocab_size:
                    continue

                # STDP: pre before post → усиление
                self.stdp_update(idx_i, idx_j, dt=1.0)

                # Attention-вес (если есть)
                attn_w = 1.0
                if attention_weights and i < len(attention_weights) and j < len(attention_weights):
                    attn_w = attention_weights[i] * attention_weights[j]

                # LTP: долговременное усиление
                dw = self.ltp_rate * attn_w * confidence
                self.affinity[idx_i, idx_j] += dw

                # Обновить счётчики
                self.usage_count[idx_i, idx_j] += 1
                self.usage_count[idx_j, idx_i] += 1
                self.last_update[idx_i, idx_j] = self.global_step
                self.last_update[idx_j, idx_i] = self.global_step

    def long_term_depression(self, min_usage: int = 5):
        """
        LTD: ослабить связи, которые давно не использовались.
        """
        mask = (self.usage_count < min_usage) & (self.last_update < self.global_step - 1000)
        self.affinity[mask] -= self.ltd_rate * self.plasticity[mask]
        self.affinity = self.affinity.clamp(self.min_affinity, self.max_affinity)

    def homeostatic_scaling(self):
        """
        Гомеостатический scaling: суммарный потенциал каждого символа
        должен оставаться вблизи target_mean.
        """
        row_means = self.affinity.mean(dim=1)
        scaling_factors = (self.target_mean / row_means.clamp(min=1e-8))
        scaling_factors = 1.0 + self.homeostatic_rate * (scaling_factors - 1.0)
        self.affinity = self.affinity * scaling_factors.unsqueeze(1)
        self.affinity = self.affinity.clamp(self.min_affinity, self.max_affinity)

    def update_plasticity(self):
        """
        Мета-пластичность: связи, которые часто меняются, становятся
        более пластичными (легче менять). Стабильные связи — менее пластичными.
        """
        # Измеряем вариацию аффинности
        if len(self.affinity_history) > 10:
            recent = torch.stack(list(self.affinity_history)[-10:])
            variance = recent.var(dim=0)
            # Высокая вариация → высокая пластичность
            self.plasticity = 0.5 * self.plasticity + 0.5 * (1.0 + variance).clamp(0.1, 2.0)

        self.affinity_history.append(self.affinity.clone())

    def step(
        self,
        active_sequence: Optional[List[int]] = None,
        attention_weights: Optional[List[float]] = None,
        confidence: float = 0.5,
    ):
        """
        Один шаг динамики: обновить следы, применить STDP/LTP/LTD, scaling.
        """
        self.global_step += 1

        # Обновить следы
        if active_sequence:
            self.update_traces(active_sequence)
            self.reinforce_sequence(active_sequence, attention_weights, confidence)

        # Периодическое обслуживание
        if self.global_step % 100 == 0:
            self.long_term_depression()
            self.homeostatic_scaling()

        if self.global_step % 500 == 0:
            self.update_plasticity()

    def get_continuation_distribution(self, char_idx: int) -> np.ndarray:
        """Нормализованное распределение потенциалов продолжения."""
        potentials = self.affinity[char_idx].cpu().numpy()
        potentials = potentials - potentials.max()
        probs = np.exp(potentials / 0.5)
        probs = probs / (probs.sum() + 1e-8)
        return probs

    def get_affinity_submatrix(
        self, indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Суб-матрица аффинности для группы символов."""
        idx = torch.tensor(indices, dtype=torch.long)
        sub = self.affinity[idx][:, idx]
        usg = self.usage_count[idx][:, idx]
        return sub, usg

    def summary(self) -> str:
        mean_aff = self.affinity.mean().item()
        active = (self.usage_count > 10).sum().item()
        total = self.vocab_size * self.vocab_size
        avg_plast = self.plasticity.mean().item()
        return (
            f"Dynamics(V={self.vocab_size}, step={self.global_step}, "
            f"μ_aff={mean_aff:.3f}, active={active}/{total}, μ_plast={avg_plast:.3f})"
        )
