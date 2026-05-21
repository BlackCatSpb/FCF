"""
CurvatureAnalyzer — анализ кривизны семантического многообразия.

Кривизна показывает где метрика "рвётся":
- Низкая кривизна: связи гладкие, логика предсказуема
- Высокая кривизна: связи резкие, противоречия, неожиданные переходы
- Отрицательная кривизна: "седловые точки" — места выбора между альтернативами

Типы кривизны:
1. Ricci curvature (Ollivier): насколько расходятся геодезические
2. Sectional curvature: насколько соседи "согласованы"
3. Scalar curvature: общая "изогнутость" в точке

Применение:
- Места высокой кривизны → потенциальные противоречия
- Места нулевой кривизны → устойчивые паттерны (можно сжимать)
- Отрицательная кривизна → точки ветвления (разные смыслы одного символа)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from loguru import logger


class CurvatureAnalyzer:
    """
    Анализирует кривизну многообразия символьных связей.

    Кривизна = мера того, насколько "неплоским" является
    пространство аффинности в окрестности символа.
    """

    def __init__(self, potential_field, topological_field):
        self.potential_field = potential_field
        self.topological_field = topological_field

    def compute_ollivier_ricci(
        self,
        symbol_idx: int,
        neighbor_count: int = 5,
        epsilon: float = 1.0,
    ) -> float:
        """
        Ollivier-Ricci curvature для символа.

        Измеряет насколько расходятся/сходятся "геодезические"
        от символа к его соседям.

        κ > 0: соседи близки друг к другу (сфера-like)
        κ = 0: соседи равномерно распределены (плоскость)
        κ < 0: соседи разбегаются (гиперболичность)
        """
        if symbol_idx not in self.topological_field.points:
            return 0.0

        aff = self.potential_field.affinity.cpu().numpy()
        n = min(self.potential_field.vocab_size, aff.shape[0])

        # Берём top-k соседей по аффинности
        neighbors = np.argsort(aff[symbol_idx])[::-1][:neighbor_count + 1]
        neighbors = [int(nb) for nb in neighbors if nb != symbol_idx][:neighbor_count]

        if len(neighbors) < 2:
            return 0.0

        # Измеряем pairwise расстояние между соседями
        # (Wasserstein 1-distance между их распределениями продолжений)
        distances = []
        for i, ni in enumerate(neighbors):
            for j, nj in enumerate(neighbors):
                if i < j:
                    # Расстояние между распределениями продолжений
                    dist_ni = aff[ni] / (np.sum(aff[ni]) + 1e-8)
                    dist_nj = aff[nj] / (np.sum(aff[nj]) + 1e-8)
                    w_dist = np.sum(np.abs(dist_ni - dist_nj)) / 2
                    distances.append(w_dist)

        mean_pairwise_distance = np.mean(distances) if distances else 0.0

        # Нормализуем в [-1, 1]
        curvature = 1.0 - 2.0 * mean_pairwise_distance
        return float(curvature)

    def compute_sectional_curvature(
        self,
        symbol_a: int,
        symbol_b: int,
    ) -> float:
        """
        Секционная кривизна между двумя символами.

        Измеряет насколько "согласованы" их окрестности.
        Если a и b ведут к разным наборам соседей → высокая кривизна.
        """
        aff = self.potential_field.affinity.cpu().numpy()
        n = min(self.potential_field.vocab_size, aff.shape[0])

        neighbors_a = set(np.where(aff[symbol_a] > 0.4)[0][:10])
        neighbors_b = set(np.where(aff[symbol_b] > 0.4)[0][:10])

        if not neighbors_a or not neighbors_b:
            return 0.0

        # Jaccard сходство окрестностей
        intersection = len(neighbors_a & neighbors_b)
        union = len(neighbors_a | neighbors_b)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # Кривизна: 1 - overlap
        # Полный overlap → плоско (0), полное расхождение → изгиб (1)
        curvature = 1.0 - jaccard
        return float(curvature)

    def compute_scalar_curvature(
        self,
        symbol_indices: List[int],
    ) -> float:
        """
        Скалярная кривизна для последовательности символов.

        Усреднённая Ricci curvature вдоль сборки.
        Высокая → сборка в "изогнутой" области многообразия (сложная, неочевидная).
        """
        if len(symbol_indices) < 2:
            return 0.0

        curvatures = []
        for si in symbol_indices:
            rc = self.compute_ollivier_ricci(si)
            curvatures.append(rc)

        return float(np.mean(curvatures))

    def find_curvature_anomalies(
        self,
        threshold: float = 0.5,
    ) -> List[Tuple[int, float, str]]:
        """
        Найти символы с аномальной кривизной.

        Аномалия = кривизна > threshold (резкие переходы).
        Возвращает: [(symbol_idx, curvature, reasoning)]
        """
        anomalies = []
        aff = self.potential_field.affinity.cpu().numpy()
        n = min(self.potential_field.vocab_size, aff.shape[0])

        for i in range(n):
            rc = self.compute_ollivier_ricci(i)

            if abs(rc) > threshold:
                if rc > 0:
                    reason = "high positive curvature: restricted meaning"
                else:
                    reason = "negative curvature: divergent meanings (ambiguity)"

                anomalies.append((i, rc, reason))

        anomalies.sort(key=lambda x: abs(x[1]), reverse=True)
        return anomalies[:20]

    def summary(self) -> str:
        aff = self.potential_field.affinity.cpu().numpy()
        n = min(self.potential_field.vocab_size, aff.shape[0])

        sample_curvatures = [self.compute_ollivier_ricci(i) for i in range(0, n, max(n//10, 1))]
        avg_curv = np.mean(sample_curvatures) if sample_curvatures else 0

        return f"CurvatureAnalyzer: avg_Ricci={avg_curv:.3f}"
