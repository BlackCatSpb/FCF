"""
TemporalConditionalBinding — контекстно-зависимые потенциалы связей.

Не все связи абсолютны. P(i→j) зависит от:
1. Левого контекста (что было ДО i)
2. Правого контекста (что ожидается ПОСЛЕ j)
3. Домена (в каком кластере многообразия находимся)
4. Глубины (на каком уровне иерархии: символ/слово/предложение)

Условный потенциал:
  P(i→j | context) = base_affinity(i,j) × context_factor × domain_factor

context_factor = cos(avg_context_vector, transition_vector(i→j))
domain_factor  = 1.0 если i и j в одном домене, 0.5 если в разных

Аналог в мозге: contextual modulation of synaptic strength.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
from loguru import logger


class TemporalConditionalBinding(nn.Module):
    """
    Контекстно-зависимые потенциалы: P(i→j | context).

    Каждая связь i→j имеет не только абсолютную силу,
    но и контекстный модификатор, зависящий от:
    - Левого контекста (предыдущие символы)
    - Доменной принадлежности
    - Позиции в сборке (начало/середина/конец)
    """

    def __init__(
        self,
        potential_field,
        topological_field,
        clusterer,
        context_window: int = 10,
        embed_dim: int = 256,
    ):
        super().__init__()

        self.potential_field = potential_field
        self.topological_field = topological_field
        self.clusterer = clusterer
        self.context_window = context_window
        self.embed_dim = embed_dim
        self.vocab_size = potential_field.vocab_size

        # Контекстный энкодер: сжимает окно контекста в вектор
        self.context_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Буферы для контекстной статистики
        self.register_buffer("context_affinity_cache", torch.zeros(1000, self.vocab_size))

        # Позиционные модификаторы (0=начало сборки, 1=середина, 2=конец)
        self.position_modifiers = nn.Parameter(torch.ones(3, self.vocab_size, self.vocab_size))

        self._update_step: int = 0

    def encode_context(self, context_symbols: List[int]) -> torch.Tensor:
        """
        Закодировать левый контекст в вектор.

        Контекстный вектор = среднее потенциалов символов в окне,
        взвешенное по позиции (ближе к концу → больший вес).
        """
        if not context_symbols:
            return torch.zeros(self.embed_dim)

        vectors = []
        weights = []

        for i, sym in enumerate(context_symbols[-self.context_window:]):
            if 0 <= sym < self.vocab_size:
                vec = self.potential_field.char_potential[sym]
                # Экспоненциальный вес: чем ближе к текущей позиции, тем важнее
                weight = np.exp(i / self.context_window)
                vectors.append(vec)
                weights.append(weight)

        if not vectors:
            return torch.zeros(self.embed_dim)

        vectors_t = torch.stack(vectors)
        weights_t = torch.tensor(weights, device=vectors_t.device).unsqueeze(-1)

        weighted_avg = (vectors_t * weights_t).sum(dim=0) / weights_t.sum()

        return weighted_avg

    def get_conditional_potential(
        self,
        symbol_i: int,
        symbol_j: int,
        context_vector: Optional[torch.Tensor] = None,
        position: int = 1,  # 0=начало, 1=середина, 2=конец
    ) -> float:
        """
        Условный потенциал P(i→j | context, position).

        base × context_factor × position_factor × domain_factor
        """
        # Базовый потенциал
        if symbol_i >= self.vocab_size or symbol_j >= self.vocab_size:
            return 0.0

        base = float(self.potential_field.affinity[symbol_i, symbol_j])

        # Контекстный фактор
        context_factor = 1.0
        if context_vector is not None and context_vector.norm() > 1e-8:
            # Вектор перехода i→j
            vi = self.potential_field.char_potential[symbol_i]
            vj = self.potential_field.char_potential[symbol_j]
            transition = vj - vi

            # cos(контекст, переход)
            cos_sim = F.cosine_similarity(
                context_vector.unsqueeze(0),
                transition.unsqueeze(0),
            ).item()
            context_factor = max(0.1, (cos_sim + 1.0) / 2.0)

        # Позиционный фактор
        pos = min(max(position, 0), 2)
        position_factor = float(self.position_modifiers[pos, symbol_i, symbol_j])

        # Доменный фактор
        domain_factor = 1.0
        if self.clusterer and len(self.clusterer.symbol_to_domain) > 0:
            di = self.clusterer.symbol_to_domain.get(symbol_i)
            dj = self.clusterer.symbol_to_domain.get(symbol_j)
            if di is not None and dj is not None:
                if di == dj:
                    domain_factor = 1.0  # Один домен → полная сила
                else:
                    # Проверяем иерархию
                    path_i = self.clusterer.get_domain_path(symbol_i)
                    path_j = self.clusterer.get_domain_path(symbol_j)
                    common = len(set(path_i) & set(path_j))
                    domain_factor = 0.3 + 0.7 * (common / max(len(path_i), len(path_j), 1))

        return base * context_factor * position_factor * domain_factor

    def get_conditional_distribution(
        self,
        symbol_i: int,
        context_symbols: List[int],
        position: int = 1,
    ) -> np.ndarray:
        """
        Получить условное распределение продолжений P(*|i, context).
        Возвращает [vocab_size] — вероятности продолжения.
        """
        context_vec = self.encode_context(context_symbols)

        potentials = np.zeros(self.vocab_size)
        for j in range(self.vocab_size):
            potentials[j] = self.get_conditional_potential(
                symbol_i, j, context_vec, position
            )

        # Негативные потенциалы → 0
        potentials = np.maximum(potentials, 0.0)

        # Softmax
        potentials = potentials - potentials.max()
        probs = np.exp(potentials / 0.5)
        probs = probs / (probs.sum() + 1e-8)

        return probs

    def update_position_modifiers(
        self,
        position: int,
        symbol_i: int,
        symbol_j: int,
        was_successful: bool,
        learning_rate: float = 0.01,
    ):
        """
        Обновить позиционные модификаторы на основе успеха/неуспеха.
        """
        if was_successful:
            self.position_modifiers[position, symbol_i, symbol_j] += learning_rate
        else:
            self.position_modifiers[position, symbol_i, symbol_j] -= learning_rate * 0.5

        self.position_modifiers.data = self.position_modifiers.data.clamp(0.1, 5.0)

    def summary(self) -> str:
        return f"ConditionalBinding(V={self.vocab_size}, window={self.context_window})"
