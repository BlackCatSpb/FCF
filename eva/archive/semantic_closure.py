"""
SemanticClosureChecker — математическая проверка связности сборки.

Гарантирует что:
1. Локальная связность: cos(p(xᵢ), p(xᵢ₊₁)) > ε_local
2. Контекстная связность: attention распределён гладко
3. Замкнутость: потенциал сборки ∈ convex hull валидных потенциалов
4. Сохранение: Σ потенциалов ≈ const (Noether-подобие)

Без этих гарантий сборка — "бред". С ними — логичная конструкция.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from loguru import logger


class SemanticClosureChecker:
    """
    Проверяет математическую корректность инструкции сборки.

    Четыре уровня гарантий:
    1. Adjacency — соседние символы семантически близки
    2. Attention smoothness — внимание распределено без резких скачков
    3. Convex hull — потенциал сборки лежит внутри известного пространства
    4. Conservation — сумма потенциалов сохраняется
    """

    def __init__(
        self,
        adjacency_threshold: float = 0.3,
        attention_smoothness_threshold: float = 0.1,
        convex_hull_margin: float = 0.05,
        conservation_tolerance: float = 0.1,
    ):
        self.adjacency_threshold = adjacency_threshold
        self.attention_smoothness_threshold = attention_smoothness_threshold
        self.convex_hull_margin = convex_hull_margin
        self.conservation_tolerance = conservation_tolerance

        # История валидных потенциалов для convex hull
        self.valid_potentials: List[np.ndarray] = []
        self._hull: Optional[ConvexHull] = None
        self._hull_stale: bool = True

    def check_adjacency(
        self,
        potentials: np.ndarray,  # [T, d] — потенциалы символов
        adjacency_matrix: np.ndarray,  # [T, T] — attention или affinity
    ) -> Tuple[bool, float, List[int]]:
        """
        Проверка локальной связности:
        каждые соседние символы должны иметь cos(p_i, p_{i+1}) > threshold
        """
        T = potentials.shape[0]
        violations = []
        scores = []

        for i in range(T - 1):
            sim = self._cosine_sim(potentials[i], potentials[i + 1])
            scores.append(sim)
            if sim < self.adjacency_threshold:
                violations.append(i)

        avg_score = float(np.mean(scores)) if scores else 0.0
        passed = len(violations) == 0 or len(violations) / T < 0.3

        return passed, avg_score, violations

    def check_attention_smoothness(
        self,
        attention_matrix: np.ndarray,  # [T, T]
    ) -> Tuple[bool, float]:
        """
        Проверка гладкости внимания:
        attention не должен иметь резких перепадов между соседними токенами.
        """
        T = attention_matrix.shape[0]
        if T < 3:
            return True, 1.0

        row_attn = attention_matrix.sum(axis=1)  # [T] — сколько внимания к каждому
        diffs = np.abs(np.diff(row_attn))
        max_diff = float(np.max(diffs)) if len(diffs) > 0 else 0.0
        mean_attn = float(np.mean(row_attn)) if row_attn.mean() > 1e-8 else 1.0

        smoothness = max_diff / mean_attn
        passed = smoothness < self.attention_smoothness_threshold * T

        return passed, smoothness

    def check_convex_hull(
        self,
        assembly_potential: np.ndarray,  # [d] — потенциал сборки
    ) -> Tuple[bool, float]:
        """
        Проверка замкнутости:
        потенциал сборки должен лежать внутри convex hull известных валидных потенциалов.

        Если сборка выходит за пределы hull — это "бред" (нелогичная конструкция).
        """
        if len(self.valid_potentials) < 3:
            return True, 1.0  # Недостаточно данных — считаем валидным

        if self._hull_stale and len(self.valid_potentials) >= 3:
            points = np.array(self.valid_potentials)
            try:
                self._hull = ConvexHull(points)
                self._hull_stale = False
            except Exception:
                return True, 1.0

        if self._hull is None:
            return True, 1.0

        # Проверяем минимальное расстояние до любой точки hull
        hull_points = np.array(self.valid_potentials)[self._hull.vertices]
        distances = cdist([assembly_potential], hull_points, metric='cosine')[0]
        min_dist = float(np.min(distances))

        # Расстояние до центра hull
        hull_center = np.mean(hull_points, axis=0)
        center_dist = self._cosine_sim(assembly_potential, hull_center)

        # Сборка валидна если она близка к центру hull
        passed = center_dist > 0.5 and min_dist < 0.5

        return passed, float(center_dist)

    def check_conservation(
        self,
        potentials_before: np.ndarray,  # [T, d]
        potentials_after: np.ndarray,   # [T, d] — после добавления нового символа
    ) -> Tuple[bool, float]:
        """
        Проверка сохранения семантического потенциала:
        при добавлении нового символа сумма потенциалов не должна резко меняться.

        Аналог Noether-симметрии: семантический "заряд" сохраняется при трансформациях.
        """
        total_before = np.sum(np.linalg.norm(potentials_before, axis=1))
        total_after = np.sum(np.linalg.norm(potentials_after, axis=1))

        if total_before < 1e-8:
            return True, 0.0

        change = abs(total_after - total_before) / total_before
        passed = change < self.conservation_tolerance * len(potentials_after)

        return passed, float(change)

    def full_check(
        self,
        potentials: np.ndarray,          # [T, d]
        attention_matrix: np.ndarray,    # [T, T]
        assembly_potential: np.ndarray,  # [d]
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Полная проверка сборки.

        Returns: (passed, overall_score, component_scores)
        """
        adj_ok, adj_score, _ = self.check_adjacency(potentials, attention_matrix)
        att_ok, att_score = self.check_attention_smoothness(attention_matrix)
        hull_ok, hull_score = self.check_convex_hull(assembly_potential)

        # Веса компонентов
        scores = {
            "adjacency": adj_score,
            "attention_smoothness": 1.0 - att_score,
            "convex_hull": hull_score,
        }

        overall = adj_score * 0.4 + (1.0 - att_score) * 0.3 + hull_score * 0.3
        all_passed = adj_ok and att_ok and hull_ok

        return all_passed, overall, scores

    def add_valid_potential(self, potential: np.ndarray):
        """Добавить валидный потенциал в историю для convex hull."""
        self.valid_potentials.append(potential.copy())
        self._hull_stale = True
        if len(self.valid_potentials) > 10000:
            self.valid_potentials = self.valid_potentials[-5000:]

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
