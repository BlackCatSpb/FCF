"""MultiLevelEncoder — 4-level λ_d-hierarchy: Symbol → Word → Sentence → Knowledge Cluster.

Каждый уровень:
  d=1: символы → слово  (λ₁=2.0,  F^(1)_n ёмкость)
  d=2: слова → предложение (λ₂=φ,  F^(2)_n ёмкость)
  d=3: предложения → кластер (λ₃≈1.839,  F^(3)_n ёмкость)
  d=4: кластеры → знание (λ₄≈1.928,  F^(4)_n ёмкость)

Принцип: каждый символ (слово, предложение) отклоняет луч (траекторию),
а λ_d управляет скоростью затухания предыдущих отклонений.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from eva.symbolic.fibonacci_utils import FibonacciUtils as _FU


@dataclass
class LevelEmit:
    """Вектор, испущенный уровнем при достижении границы."""
    level: int          # 1=word, 2=sentence, 3=cluster, 4=knowledge
    vector: np.ndarray  # результирующий вектор единицы
    count: int          # сколько токенов вошло в единицу


# ── λ_d для каждого уровня (d=1 отдельно) ──
_LAM: dict[int, float] = {
    1: 2.0,
    2: _FU.get_lambda(2),   # φ
    3: _FU.get_lambda(3),   # ≈1.839
    4: _FU.get_lambda(4),   # ≈1.928
}

# ── Максимальная ёмкость: F^(d)_n для разумного n ──
_MAX_CAPACITY: dict[int, int] = {
    1: 2 ** 5,            # F^(1)_6 = 32 символа на слово
    2: _FU.get_generalized(8, 2),   # F_9 = 34 слова на предложение
    3: _FU.get_generalized(5, 3),   # F^(3)_6 = 13 предложений на кластер
    4: _FU.get_generalized(4, 4),   # F^(4)_5 = 8 кластеров
}


class MultiLevelEncoder:
    """4-level λ_d-VSA hierarchical encoder.

    Символы → слова → предложения → кластеры → знания.
    Каждый уровень — λ_d-взвешенная траектория (луч) с позиционной привязкой.
    """

    def __init__(self, dim: int = 768, lam: float | None = None):
        self.dim = dim
        self.lam = lam or _LAM[2]  # system λ (λ₂)

        # λ_d для каждого уровня
        self.level_lam = _LAM.copy()
        if lam is not None:
            self.level_lam[2] = lam

        # Ёмкости
        self.max_capacity = _MAX_CAPACITY.copy()

        # Позиционные сдвиги (детерминированные, один на уровень)
        rng = np.random.default_rng(seed=42)
        self._pos_shift: dict[int, np.ndarray] = {}
        for level in range(1, 5):
            v = rng.normal(0.0, 1.0 / np.sqrt(dim), (dim,)).astype(np.float32)
            self._pos_shift[level] = v / max(np.linalg.norm(v), 1e-10)

        self.reset()

    def reset(self):
        """Обнулить все уровни."""
        z = np.zeros(self.dim, dtype=np.float32)
        self._state: dict[int, np.ndarray] = {1: z.copy(), 2: z.copy(),
                                               3: z.copy(), 4: z.copy()}
        self._count: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

    # ── Публичный API ──

    def step(self, vec: np.ndarray, *, break_level: int = 0, start_level: int = 1
             ) -> list[LevelEmit]:
        """Продвинуть один вектор по иерархии.

        Args:
            vec: Вектор символа (для level=1) или слова/предложения/кластера.
            break_level: Принудительная граница (1=конец слова, 2=конец предложения,
                         3=конец кластера). Вектор-разделитель НЕ накапливается.
            start_level: С какого уровня начинать (1=символ, 2=слово, 3=предложение).
                         Для BPE-токенов используйте start_level=2.

        Returns:
            Список LevelEmit (испускания уровней), может быть пустым.
        """
        emits: list[LevelEmit] = []

        # Принудительная эмиссия (разделитель)
        if break_level >= 1:
            for level in range(1, min(break_level, 4) + 1):
                emit = self._emit_level(level)
                if emit is not None:
                    emits.append(emit)
            return emits

        # Нормальное накопление — начинаем с указанного уровня
        self._accumulate(start_level, vec)

        # Автоматическая эмиссия при переполнении ёмкости
        for level in range(start_level, 5):
            while self._count[level] >= self.max_capacity[level]:
                emit = self._emit_level(level)
                if emit is not None:
                    emits.append(emit)

        return emits

    def state_vector(self, level: int) -> np.ndarray:
        """Текущий вектор состояния уровня (без копирования)."""
        return self._state[level]

    def level_count(self, level: int) -> int:
        """Сколько токенов накоплено на уровне."""
        return self._count[level]

    # ── Внутренние методы ──

    def _accumulate(self, level: int, vec: np.ndarray):
        """Накопить вектор на уровне как λ_d-взвешенную траекторию.

        state[t+1] = normalize(λ_d · state[t] + bind(vec, pos_shift))
        bind = поэлементное умножение (VSA binding в MAP-C).
        """
        l = self.level_lam[level]
        # Позиционная привязка: новая информация связывается со сдвигом
        bound = vec * self._pos_shift[level]  # bind(vec, position)
        self._state[level] = l * self._state[level] + bound
        norm = float(np.linalg.norm(self._state[level]))
        if norm > 1e-10:
            self._state[level] /= norm
        self._count[level] += 1

    def _finalize(self, level: int) -> np.ndarray:
        """Извлечь вектор уровня (без сброса)."""
        return self._state[level].copy()

    def _emit_level(self, level: int) -> LevelEmit | None:
        """Испустить текущий вектор уровня, сбросить его, подать на уровень+1."""
        cnt = self._count[level]
        if cnt == 0:
            return None
        vec = self._finalize(level)
        self._state[level].fill(0.0)
        self._count[level] = 0
        if level < 4:
            self._accumulate(level + 1, vec)
        return LevelEmit(level=level, vector=vec, count=cnt)

    def __repr__(self) -> str:
        counts = ', '.join(f'L{l}:{self._count[l]}' for l in range(1, 5))
        return f'<MultiLevelEncoder [{counts}]>'
