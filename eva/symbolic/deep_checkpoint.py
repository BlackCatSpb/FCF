"""
OrganicCheckpointSystem — естественная иерархия без фиксированных уровней.

НИКАКИХ предопределённых уровней L0-L8.
Домены рождаются как пики плотности в многообразии.
Иерархия = дерево вложенности кластеров.
Глубина = organic, выводится из данных.

Чекпоинт = узел в дереве плотности.
Навигация = спуск от корня к листьям через пики плотности.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


@dataclass
class OrganicCheckpoint:
    """Чекпоинт — узел в дереве плотности (не фиксированного уровня)."""
    checkpoint_id: int
    centroid: np.ndarray          # координаты в многообразии
    density: float                # локальная плотность (мера «важности»)
    radius: float                 # радиус окрестности
    member_symbols: List[int]     # символы в этой окрестности
    member_words: List[int] = field(default_factory=list)
    
    # Дерево
    parent_id: Optional[int] = None
    child_ids: List[int] = field(default_factory=list)
    depth: int = 0                # глубина в дереве (0 = корень)
    
    # Связность
    coherence: float = 0.0
    attention_mask: Optional[np.ndarray] = None  # [K, K] для навигации
    
    def __repr__(self):
        return f"CP#{self.checkpoint_id}(d={self.depth}, ρ={self.density:.3f}, n={len(self.member_symbols)})"


class OrganicCheckpointSystem:
    """
    Естественная иерархия чекпоинтов.

    Строится из пиков плотности в многообразии.
    Дерево растёт органически — нет фиксированных уровней.
    """

    def __init__(
        self,
        potential_field,
        topological_field,
        grammar,
        word_discovery,
        knowledge_base=None,
        min_density: float = 0.01,
        min_members: int = 3,
        max_depth: int = 10,
    ):
        self.pf = potential_field
        self.topo = topological_field
        self.grammar = grammar
        self.word_discovery = word_discovery
        self.kb = knowledge_base

        self.min_density = min_density
        self.min_members = min_members
        self.max_depth = max_depth

        # Дерево чекпоинтов
        self.checkpoints: Dict[int, OrganicCheckpoint] = {}
        self.root_id: Optional[int] = None
        self.next_id: int = 0

    def grow(self):
        """
        Вырастить дерево чекпоинтов из данных.

        Алгоритм:
        1. Найти ВСЕ пики плотности в многообразии
        2. Построить дерево вложенности (пик внутри пика = ребёнок)
        3. Глубина определяется вложенностью, не предзадана
        """
        aff = self.pf.affinity.cpu().numpy()
        coords = self.topo.coordinates.cpu().numpy()
        n = min(self.pf.vocab_size, coords.shape[0])

        if n < 3:
            return

        # 1. Вычислить плотность каждого символа
        density = np.sum(aff[:n, :n] * (aff[:n, :n] > 0.55), axis=1)
        density = density / (density.max() + 1e-8)

        # 2. Найти ВСЕ локальные пики (рекурсивно)
        peaks = self._find_all_peaks(aff[:n, :n], density, np.arange(n), depth=0)

        # 3. Построить дерево вложенности
        self._build_tree(peaks, coords)

    def _find_all_peaks(
        self,
        aff: np.ndarray,
        density: np.ndarray,
        member_indices: np.ndarray,
        depth: int,
    ) -> List[Dict]:
        """Рекурсивно найти все пики плотности."""
        if depth > self.max_depth or len(member_indices) < self.min_members:
            return []

        # Локальные пики в этой группе
        is_peak = np.ones(len(member_indices), dtype=bool)
        for i, mi in enumerate(member_indices):
            for j, mj in enumerate(member_indices):
                if i != j and density[mi] > density[mj]:
                    is_peak[j] = False

        peak_positions = np.where(is_peak)[0]
        peaks = []

        for pos in peak_positions:
            peak_idx = member_indices[pos]

            # Найти соседей этого пика (по аффинности > порога)
            neighbors = np.where(aff[peak_idx] > 0.55)[0]
            neighbors = np.intersect1d(neighbors, member_indices)

            if len(neighbors) < self.min_members:
                continue

            peak_info = {
                "index": int(peak_idx),
                "density": float(density[peak_idx]),
                "members": neighbors.tolist(),
                "depth": depth,
            }

            # Рекурсивно ищем под-пики среди соседей
            sub_peaks = self._find_all_peaks(aff, density, neighbors, depth + 1)
            if sub_peaks:
                peak_info["children"] = sub_peaks

            peaks.append(peak_info)

        return peaks

    def _build_tree(self, peaks: List[Dict], coords: np.ndarray, parent_id: Optional[int] = None):
        """Построить дерево чекпоинтов из структуры пиков."""
        for peak_info in peaks:
            cp = OrganicCheckpoint(
                checkpoint_id=self.next_id,
                centroid=coords[peak_info["index"]],
                density=peak_info["density"],
                radius=float(np.max(np.linalg.norm(
                    coords[peak_info["members"]] - coords[peak_info["index"]], axis=1
                ))) if peak_info["members"] else 1.0,
                member_symbols=peak_info["members"],
                parent_id=parent_id,
                depth=peak_info["depth"],
            )

            self.checkpoints[self.next_id] = cp
            if parent_id is not None:
                self.checkpoints[parent_id].child_ids.append(self.next_id)
            if self.root_id is None and parent_id is None:
                self.root_id = self.next_id

            current_id = self.next_id
            self.next_id += 1

            # Рекурсивно строим детей
            if "children" in peak_info:
                self._build_tree(peak_info["children"], coords, parent_id=current_id)

    def navigate(
        self,
        from_symbols: List[int],
        direction: str = "up",
        steps: int = 3,
    ) -> List[OrganicCheckpoint]:
        """
        Навигация по дереву от символов.

        direction="up": от листьев к корню (обобщение)
        direction="down": от корня к листьям (конкретизация)
        """
        # Найти листовые чекпоинты, содержащие эти символы
        leaf_matches = []
        for cid, cp in self.checkpoints.items():
            if not cp.child_ids:  # лист
                overlap = len(set(from_symbols) & set(cp.member_symbols))
                if overlap > 0:
                    leaf_matches.append((cid, overlap))

        if not leaf_matches:
            return []

        leaf_matches.sort(key=lambda x: x[1], reverse=True)
        start_id = leaf_matches[0][0]

        path = [self.checkpoints[start_id]]
        current = self.checkpoints[start_id]

        if direction == "up":
            for _ in range(steps):
                if current.parent_id is None:
                    break
                current = self.checkpoints[current.parent_id]
                path.append(current)
        else:
            for _ in range(steps):
                if not current.child_ids:
                    break
                # Выбрать ребёнка с максимальной плотностью
                best_child = max(current.child_ids,
                               key=lambda cid: self.checkpoints[cid].density)
                current = self.checkpoints[best_child]
                path.append(current)

        return path

    def find_context_for_symbol(
        self,
        symbol_idx: int,
    ) -> List[OrganicCheckpoint]:
        """
        Найти все контексты (пути от листа к корню) для символа.

        Символ может принадлежать нескольким листовым чекпоинтам
        (разные контексты → разная интерпретация).
        """
        paths = []

        # Найти все листья, содержащие символ
        leaves = []
        for cid, cp in self.checkpoints.items():
            if not cp.child_ids and symbol_idx in cp.member_symbols:
                leaves.append(cid)

        for leaf_id in leaves:
            path = [self.checkpoints[leaf_id]]
            current = self.checkpoints[leaf_id]
            while current.parent_id is not None:
                current = self.checkpoints[current.parent_id]
                path.insert(0, current)
            paths.append(path)

        return paths

    def compute_attention_path(
        self,
        from_symbols: List[int],
        to_symbols: List[int],
    ) -> Optional[Tuple[List[OrganicCheckpoint], float]]:
        """
        Вычислить путь внимания между двумя группами символов.

        Ищем ближайшего общего предка в дереве чекпоинтов.
        """
        from_paths = self.find_context_for_symbol(from_symbols[0]) if from_symbols else []
        to_paths = self.find_context_for_symbol(to_symbols[0]) if to_symbols else []

        if not from_paths or not to_paths:
            return None

        # Ищем ближайшего общего предка
        best_distance = float('inf')
        best_path = None

        for fp in from_paths:
            for tp in to_paths:
                # Найти глубину общего предка
                common_depth = 0
                for fa, ta in zip(fp, tp):
                    if fa.checkpoint_id == ta.checkpoint_id:
                        common_depth += 1
                    else:
                        break

                if common_depth > 0:
                    # Расстояние = 1 / общая_глубина
                    dist = 1.0 / common_depth
                    if dist < best_distance:
                        best_distance = dist
                        best_path = fp[:common_depth] + tp[common_depth:]

        return (best_path, 1.0 / (1.0 + best_distance)) if best_path else None

    def interpret_in_context(
        self,
        symbol_idx: int,
        context_checkpoints: List[OrganicCheckpoint],
    ) -> Tuple[np.ndarray, float]:
        """
        Интерпретировать символ в контексте пути чекпоинтов.

        Интерпретация = базовый вектор символа, смещённый контекстом.
        """
        base = np.zeros(3)
        if symbol_idx < self.topo.coordinates.shape[0]:
            base = self.topo.coordinates[symbol_idx].cpu().numpy()

        if not context_checkpoints:
            return base, 0.5

        # Контекстное смещение: усреднённый вектор пути
        context_vec = np.zeros(3)
        weight_sum = 0
        for k, cp in enumerate(context_checkpoints):
            w = 0.5 ** k  # decay по глубине
            context_vec += w * cp.centroid
            weight_sum += w

        if weight_sum > 0:
            context_vec /= weight_sum

        interpretation = 0.6 * base + 0.4 * context_vec
        confidence = 1.0 / (1.0 + np.linalg.norm(interpretation - base))

        return interpretation, float(confidence)

    def summary(self) -> str:
        if not self.checkpoints:
            return "OrganicCheckpoints: empty (need training data)"

        depths = [cp.depth for cp in self.checkpoints.values()]
        max_d = max(depths)
        leaves = sum(1 for cp in self.checkpoints.values() if not cp.child_ids)
        roots = sum(1 for cp in self.checkpoints.values() if cp.parent_id is None)

        return (
            f"OrganicCheckpoints: {len(self.checkpoints)} nodes, "
            f"depth={max_d}, leaves={leaves}, roots={roots}"
        )
