"""
NaturalClusterer — обнаружение доменов как естественных уплотнений.

Домены НЕ задаются вручную. Они ВОЗНИКАЮТ из данных:
1. Области высокой плотности в многообразии → домены
2. Области низкой плотности → границы между доменами
3. Иерархия: под-домены внутри доменов

Методы:
- MeanShift на координатном пространстве (пики плотности)
- DBSCAN на графе аффинности (связные компоненты)
- Watershed на поле плотности (водораздел = граница доменов)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Domain:
    """Естественный домен — область плотности в многообразии."""
    domain_id: int
    symbol_indices: List[int]
    centroid: np.ndarray
    density: float                   # средняя плотность
    coherence: float                 # внутренняя связность (mean intra-affinity)
    boundary_symbols: List[int]      # граничные символы (на стыке с другими доменами)
    parent_domain_id: Optional[int] = None  # для иерархии
    child_domain_ids: List[int] = field(default_factory=list)
    size: int = 0

    @property
    def inner_symbols(self) -> List[int]:
        """Внутренние символы (не граничные)."""
        boundary_set = set(self.boundary_symbols)
        return [s for s in self.symbol_indices if s not in boundary_set]


class NaturalClusterer:
    """
    Обнаруживает естественные домены без предопределённых категорий.

    Домен = область в многообразии где:
    - Локальная плотность выше порога
    - Символы семантически близки (высокая intra-аффинность)
    - Границы проходят по "долинам" плотности и низкой cross-аффинности
    """

    def __init__(
        self,
        potential_field,
        topological_field,
        min_domain_size: int = 3,
        density_threshold: float = 0.3,
        boundary_density_ratio: float = 0.5,  # граница если плотность падает в 2 раза
        max_domains: int = 50,
    ):
        self.potential_field = potential_field
        self.topological_field = topological_field
        self.min_domain_size = min_domain_size
        self.density_threshold = density_threshold
        self.boundary_density_ratio = boundary_density_ratio
        self.max_domains = max_domains

        self.domains: Dict[int, Domain] = {}
        self.symbol_to_domain: Dict[int, int] = {}  # символ → домен
        self.next_id: int = 0

    def cluster_by_density(self) -> List[Domain]:
        """
        MeanShift-like кластеризация по пикам плотности.

        Алгоритм:
        1. Для каждого символа: идти в направлении роста плотности (gradient ascent)
        2. Точки, сошедшиеся к одному пику → один домен
        3. Пики с высокой плотностью → центры доменов
        """
        new_domains = []
        aff = self.potential_field.affinity.cpu().numpy()
        n = min(self.potential_field.vocab_size, aff.shape[0])

        # Плотность точки i = сумма аффинностей к соседям > порога
        density = np.sum(aff * (aff > self.density_threshold), axis=1)

        # Сглаживаем плотность
        smoothed_density = density.copy()
        for _ in range(3):  # Несколько итераций сглаживания
            smoothed_density = aff @ smoothed_density
            smoothed_density = smoothed_density / (np.sum(smoothed_density) + 1e-8) * np.sum(density)

        # Ищем локальные максимумы (пики)
        peak_mask = np.ones(n, dtype=bool)
        for i in range(n):
            neighbors = np.where(aff[i] > self.density_threshold)[0]
            for nb in neighbors:
                if density[nb] > density[i]:
                    peak_mask[i] = False
                    break

        peaks = np.where(peak_mask)[0]
        peaks = sorted(peaks, key=lambda p: density[p], reverse=True)[:self.max_domains]

        # Назначаем точки ближайшим пикам (по аффинности)
        assignments = np.full(n, -1, dtype=int)
        for i in range(n):
            if density[i] < self.density_threshold:
                continue
            best_peak = -1
            best_aff = -1
            for pi, peak in enumerate(peaks):
                a = aff[i, peak]
                if a > best_aff:
                    best_aff = a
                    best_peak = pi
            if best_peak >= 0:
                assignments[i] = best_peak

        # Формируем домены
        for pi, peak in enumerate(peaks):
            members = list(np.where(assignments == pi)[0])
            if len(members) < self.min_domain_size:
                continue

            intra_aff = np.mean([aff[i, j] for i in members for j in members if i != j]) if len(members) > 1 else 1.0

            boundary = self._find_boundary_symbols(members, aff)

            domain = Domain(
                domain_id=self.next_id,
                symbol_indices=members,
                centroid=self.topological_field.coordinates[peak].cpu().numpy(),
                density=float(density[peak]),
                coherence=float(intra_aff),
                boundary_symbols=boundary,
                size=len(members),
            )
            self.domains[self.next_id] = domain
            for s in members:
                self.symbol_to_domain[s] = self.next_id
            new_domains.append(domain)
            self.next_id += 1

        return new_domains

    def _find_boundary_symbols(self, members: List[int], aff: np.ndarray) -> List[int]:
        """Найти граничные символы домена (на стыке с другими)."""
        boundary = []
        member_set = set(members)

        for i in members:
            neighbors = np.where(aff[i] > self.density_threshold)[0]
            external = [n for n in neighbors if n not in member_set]

            if len(external) > 0:
                avg_external_aff = np.mean([aff[i, e] for e in external])
                avg_internal_aff = np.mean([aff[i, m] for m in members if m != i]) if len(members) > 1 else 1.0

                # Граничный символ: внешние связи почти так же сильны, как внутренние
                if avg_external_aff > avg_internal_aff * self.boundary_density_ratio:
                    boundary.append(i)

        return boundary

    def discover_hierarchy(self) -> List[Domain]:
        """
        Обнаружить иерархию доменов:
        маленькие домены внутри больших → дочерние.
        """
        if len(self.domains) < 2:
            return []

        # Сортируем по размеру (большие → родители)
        sorted_domains = sorted(self.domains.values(), key=lambda d: d.size, reverse=True)

        for i, parent in enumerate(sorted_domains):
            for j, child in enumerate(sorted_domains):
                if i == j:
                    continue
                if child.size > parent.size:
                    continue

                # Проверяем overlap по символам
                overlap = len(set(parent.symbol_indices) & set(child.symbol_indices))
                containment = overlap / max(len(child.symbol_indices), 1)

                if containment > 0.7:
                    child.parent_domain_id = parent.domain_id
                    parent.child_domain_ids.append(child.domain_id)

        return sorted_domains

    def get_domain_path(self, symbol_idx: int) -> List[int]:
        """
        Получить путь по иерархии доменов для символа:
        от самого специфичного к самому общему.
        """
        if symbol_idx not in self.symbol_to_domain:
            return []

        domain_id = self.symbol_to_domain[symbol_idx]
        path = [domain_id]

        current = self.domains.get(domain_id)
        while current and current.parent_domain_id is not None:
            path.append(current.parent_domain_id)
            current = self.domains.get(current.parent_domain_id)

        return path

    def summary(self) -> str:
        hierarchy_info = ""
        for did, dom in self.domains.items():
            if dom.parent_domain_id is not None:
                hierarchy_info += f"  D{did} ⊂ D{dom.parent_domain_id}, "

        return (
            f"NaturalClusterer: domains={len(self.domains)}, "
            f"hierarchy={hierarchy_info}"
        )
