"""
TopologicalField — многообразие символьных сборок.

Превращает матрицу аффинности [V×V] в дифференцируемое многообразие:

- Каждый символ → точка в координатном пространстве
- Аффинность → метрика (расстояние между точками)
- Attention → связность (какие точки образуют пути)
- Плотность путей → домены (естественные кластеры)
- Кривизна → где метрика "рвётся" (противоречия)

Многообразие = атлас локальных карт + метрика + связность.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


@dataclass
class ManifoldPoint:
    """Точка на многообразии = координаты символа в семантическом пространстве."""
    symbol_idx: int
    coordinates: np.ndarray           # позиция в многообразии [d]
    local_density: float = 0.0        # локальная плотность путей через точку
    local_curvature: float = 0.0      # локальная кривизна (0 = плоско, >0 = изгиб)
    connected_symbols: Set[int] = field(default_factory=set)  # соседи по многообразию
    domain_id: Optional[int] = None   # домен (естественный кластер)
    is_boundary: bool = False         # граница между доменами


@dataclass
class LocalChart:
    """Локальная карта — окрестность символа с координатами соседей."""
    center_idx: int
    center_coords: np.ndarray
    neighbor_indices: List[int]
    neighbor_coords: np.ndarray       # [K, d]
    neighbor_distances: np.ndarray    # [K]
    tangent_basis: Optional[np.ndarray] = None  # базис касательного пространства
    chart_metric: Optional[np.ndarray] = None   # локальная метрика


class TopologicalField(nn.Module):
    """
    Многообразие символьных сборок.

    Строит координатное пространство из матрицы аффинности,
    вычисляет метрику, плотности, кривизну и домены.
    """

    def __init__(
        self,
        potential_field,
        coord_dim: int = 64,           # размерность координатного пространства
        neighbor_radius: float = 0.5,  # радиус соседства для локальных карт
        min_domain_density: float = 0.2,  # минимальная плотность для домена
    ):
        super().__init__()

        self.potential_field = potential_field
        self.vocab_size = potential_field.vocab_size
        self.coord_dim = coord_dim
        self.neighbor_radius = neighbor_radius
        self.min_domain_density = min_domain_density

        # Координаты точек (обучаемые — сдвигаются при накоплении данных)
        self.register_buffer("coordinates", torch.randn(self.vocab_size, coord_dim) * 0.1)

        # Нормализуем координаты из аффинности
        self._compute_coordinates_from_affinity()

        # Точки многообразия
        self.points: Dict[int, ManifoldPoint] = {}

        # Локальные карты
        self.charts: Dict[int, LocalChart] = {}

        # Домены
        self.domains: Dict[int, List[int]] = {}
        self.domain_centroids: Dict[int, np.ndarray] = {}
        self.next_domain_id: int = 0

        self._rebuild_manifold()

    def _compute_coordinates_from_affinity(self):
        """Вычислить координаты из матрицы аффинности через MDS-like проекцию."""
        aff = self.potential_field.affinity.cpu().numpy()
        n = min(self.vocab_size, aff.shape[0])

        # Строим матрицу расстояний D_ij = 1 - affinity_ij
        D = 1.0 - aff[:n, :n]
        np.fill_diagonal(D, 0.0)

        # Простая MDS: центрируем и берём top-k собственных векторов
        D_sq = D * D
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_sq @ J

        try:
            eigvals, eigvecs = np.linalg.eigh(B)
            idx = np.argsort(eigvals)[::-1]
            k = min(self.coord_dim, n - 1, eigvecs.shape[1], len(eigvals))
            k = max(k, 2)  # Минимум 2 измерения
            eigvals_top = eigvals[idx[:k]]
            eigvecs_top = eigvecs[:, idx[:k]]

            eigvals_top = np.maximum(eigvals_top, 0)
            coords = eigvecs_top * np.sqrt(eigvals_top)

            # Паддинг до coord_dim если k < coord_dim
            if coords.shape[1] < self.coord_dim:
                padding = np.zeros((n, self.coord_dim - coords.shape[1]), dtype=np.float32)
                coords = np.concatenate([coords, padding], axis=1)

            self.coordinates[:n, :] = torch.tensor(coords, dtype=torch.float32)
        except Exception as e:
            logger.debug(f"[TopologicalField] MDS failed: {e}, using random init")

    def _rebuild_manifold(self):
        """Перестроить многообразие: точки, карты, домены."""
        aff = self.potential_field.affinity.cpu().numpy()
        n = min(self.vocab_size, aff.shape[0])
        coords = self.coordinates[:n].cpu().numpy()

        self.points.clear()
        self.charts.clear()

        # Строим точки и локальные карты
        for i in range(n):
            # Соседи: символы с аффинностью > neighbor_radius
            neighbors = np.where(aff[i] > self.neighbor_radius)[0]
            neighbors = [int(nb) for nb in neighbors if nb != i]

            if neighbors:
                neighbor_coords = coords[neighbors]
                center = coords[i]
                diffs = neighbor_coords - center
                distances = np.linalg.norm(diffs, axis=1)

                # Плотность: сумма аффинностей соседей
                density = float(np.sum(aff[i, neighbors]))

                # Кривизна: насколько соседи "расходятся" от центра
                if len(neighbors) >= 2:
                    radial_vectors = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8)
                    pairwise_dots = radial_vectors @ radial_vectors.T
                    mean_dot = float(np.mean(pairwise_dots))
                    curvature = 1.0 - abs(mean_dot)  # 0 = плоско, 1 = сильно искривлено
                else:
                    curvature = 0.0

                self.points[i] = ManifoldPoint(
                    symbol_idx=i,
                    coordinates=center.copy(),
                    local_density=density,
                    local_curvature=curvature,
                    connected_symbols=set(neighbors),
                )

                # Локальная карта
                chart = LocalChart(
                    center_idx=i,
                    center_coords=center.copy(),
                    neighbor_indices=neighbors,
                    neighbor_coords=neighbor_coords,
                    neighbor_distances=distances,
                )
                self.charts[i] = chart

    def get_metric(self, i: int, j: int) -> float:
        """Метрика многообразия: семантическое расстояние между символами."""
        if i in self.points and j in self.points:
            ci = self.points[i].coordinates
            cj = self.points[j].coordinates
            return float(np.linalg.norm(ci - cj))
        return 1.0

    def get_local_density(self, symbol_idx: int) -> float:
        if symbol_idx in self.points:
            return self.points[symbol_idx].local_density
        return 0.0

    def get_local_curvature(self, symbol_idx: int) -> float:
        if symbol_idx in self.points:
            return self.points[symbol_idx].local_curvature
        return 0.0

    def find_path_via_affinity(
        self,
        from_idx: int,
        to_idx: int,
        max_length: int = 20,
    ) -> Optional[List[int]]:
        """
        Найти путь по многообразию между двумя символами.
        Использует граф аффинности + A*-like поиск.
        """
        if from_idx not in self.points or to_idx not in self.points:
            return None

        target_coords = self.points[to_idx].coordinates

        visited = {from_idx}
        path = [from_idx]
        current = from_idx

        for _ in range(max_length):
            if current == to_idx:
                return path

            if current not in self.charts:
                break

            chart = self.charts[current]
            best_score = float('inf')
            best_next = None

            for nb in chart.neighbor_indices:
                if nb in visited:
                    continue
                nb_coords = self.points[nb].coordinates
                dist = np.linalg.norm(nb_coords - target_coords)
                if dist < best_score:
                    best_score = dist
                    best_next = nb

            if best_next is None:
                break

            visited.add(best_next)
            path.append(best_next)
            current = best_next

        return path

    def compute_assembly_coordinates(
        self,
        symbol_indices: List[int],
    ) -> np.ndarray:
        """
        Вычислить координаты сборки в многообразии.

        Координата сборки = взвешенный центр масс символов,
        где вес = локальная плотность символа.
        """
        if not symbol_indices:
            return np.zeros(self.coord_dim)

        coords = []
        weights = []
        for si in symbol_indices:
            if si in self.points:
                coords.append(self.points[si].coordinates)
                weights.append(max(self.points[si].local_density, 0.01))
            else:
                if si < self.coordinates.shape[0]:
                    coords.append(self.coordinates[si].cpu().numpy())
                    weights.append(0.01)

        if not coords:
            return np.zeros(self.coord_dim)

        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)
        result = np.average(coords, axis=0, weights=weights)
        if isinstance(result, np.ndarray):
            return result
        return np.array(result)

    def update_after_learning(self):
        """Обновить координаты после обучения (накопления потенциалов)."""
        self._compute_coordinates_from_affinity()
        self._rebuild_manifold()

    def summary(self) -> str:
        n_points = len(self.points)
        avg_density = np.mean([p.local_density for p in self.points.values()]) if self.points else 0
        avg_curvature = np.mean([p.local_curvature for p in self.points.values()]) if self.points else 0
        return (
            f"TopologicalField(V={self.vocab_size}, d={self.coord_dim}, "
            f"points={n_points}, avg_density={avg_density:.3f}, avg_curvature={avg_curvature:.3f})"
        )
