"""TransitionManifold — самоорганизующаяся паутина переходов.

Копит векторы T = unbind(v_next, v_prev), кластеризует их в лучи,
использует как дополнительный сигнал пластичности и генерации.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional


class TransitionManifold:
    """Буфер переходов + лучи (центроиды VSA-связок).

    Параметры (все из FCFConfig.beam_*):
      dim              — размерность пространства (768)
      buffer_size      — макс. число хранимых переходов (10000)
      cos_threshold    — порог косинуса для объединения в луч (0.8)
      max_beams        — макс. число лучей (100)
      rebuild_interval — через сколько переходов перестраивать лучи (100)
    """

    def __init__(self, dim: int = 768, buffer_size: int = 10000,
                 cos_threshold: float = 0.8, max_beams: int = 100,
                 rebuild_interval: int = 100, eps: float = 1e-10,
                 min_count_base: int = 3, min_count_divisor: int = 4):
        self.dim = dim
        self.cos_threshold = cos_threshold
        self.max_beams = max_beams
        self.rebuild_interval = rebuild_interval
        self._eps = eps
        self._min_count_base = min_count_base
        self._min_count_divisor = min_count_divisor

        # Кольцевой буфер (фиксированный numpy, без фрагментации)
        self._buf = np.zeros((buffer_size, dim), dtype=np.float32)
        self._buf_size = buffer_size
        self._idx = 0        # следующий слот для записи
        self._total = 0      # сколько всего накоплено

        # Лучи: список (центроид, счётчик, дисперсия)
        self.beams: List[Tuple[np.ndarray, int, float]] = []

    # ── публичный API ────────────────────────────────────────────

    def push(self, T: np.ndarray) -> None:
        """Добавить один вектор перехода (нормированный)."""
        self._buf[self._idx % self._buf_size] = T
        self._idx += 1
        self._total += 1
        if self._total % self.rebuild_interval == 0:
            self._rebuild_beams()

    def push_batch(self, batch: np.ndarray) -> None:
        """Добавить массив переходов shape (N, dim)."""
        n = len(batch)
        space = self._buf_size - (self._idx % self._buf_size)
        if n <= space:
            self._buf[self._idx % self._buf_size:self._idx % self._buf_size + n] = batch
        else:
            first = batch[:space]
            second = batch[space:]
            self._buf[self._idx % self._buf_size:self._idx % self._buf_size + len(first)] = first
            self._buf[:len(second)] = second
        self._idx += n
        self._total += n
        if self._total // self.rebuild_interval != (self._total - n) // self.rebuild_interval:
            self._rebuild_beams()

    def nearest_beam(self, v: np.ndarray) -> Tuple[Optional[np.ndarray], float, int]:
        """Ближайший луч к вектору v (включительно сам v)."""
        if not self.beams:
            return None, 0.0, 0
        best_idx, best_sim = 0, -1.0
        for i, (cent, cnt, _var) in enumerate(self.beams):
            sim = float(np.dot(v, cent))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        cent, cnt, var = self.beams[best_idx]
        return cent, best_sim, cnt

    def beam_entropy(self, v: np.ndarray, eps: float = 1e-10) -> float:
        """Энтропия распределения косинусов по лучам — мера неопределённости."""
        if not self.beams:
            return 0.0
        sims = np.array([float(np.dot(v, cent)) for cent, _cnt, _var in self.beams])
        sims = np.clip(sims, -1, 1)
        weights = (sims - sims.min() + eps)
        weights /= weights.sum()
        return -float(np.sum(weights * np.log(weights + eps)))

    def n_beams(self) -> int:
        return len(self.beams)

    # ── внутреннее ───────────────────────────────────────────────

    def _rebuild_beams(self) -> None:
        """Жадная VSA-кластеризация: каждый переход → ближайший луч или новый."""
        n_valid = min(self._total, self._buf_size)
        samples = self._buf[:n_valid]
        if n_valid < 2:
            return

        np.random.shuffle(samples)  # случайный порядок для стабильности
        new_beams = []
        n_samples = len(samples)

        for T in samples:
            norm = np.linalg.norm(T)
            if norm < self._eps:
                continue
            T_norm = T / norm
            matched = False
            for i, (cent, cnt, var) in enumerate(new_beams):
                sim = float(np.dot(T_norm, cent))
                if sim > self.cos_threshold:
                    # VSA bundle: weighted average + renormalisation
                    new_cent = cent * cnt + T_norm
                    nc = np.linalg.norm(new_cent)
                    if nc > self._eps:
                        new_cent = new_cent / nc
                    # дисперсия: EMA квадрата расстояния
                    dist_sq = max(0.0, 1.0 - sim * sim)
                    new_var = (var * cnt + dist_sq) / (cnt + 1)
                    new_beams[i] = (new_cent, cnt + 1, new_var)
                    matched = True
                    break
            if not matched and len(new_beams) < self.max_beams:
                new_beams.append((T_norm.copy(), 1, 0.0))

        # Отсев лучей с малым числом попаданий
        min_count = max(self._min_count_base, n_samples // (self.max_beams * self._min_count_divisor))
        self.beams = [(c, cnt, v) for c, cnt, v in new_beams if cnt >= min_count]

    def _to_tangent(self, v_next: np.ndarray, v_prev: np.ndarray) -> np.ndarray:
        """Компонента v_next, ортогональная v_prev (направление перехода на сфере)."""
        cos_sim = float(np.dot(v_next, v_prev))
        T = v_next - cos_sim * v_prev
        n = np.linalg.norm(T)
        return T / n if n > self._eps else np.zeros(self.dim, dtype=np.float32)
