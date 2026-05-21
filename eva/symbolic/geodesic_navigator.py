"""
GeodesicNavigator + TangentSpace — навигация по многообразию сборок.

GeodesicNavigator:
  Находит кратчайшие логические пути между сборками.
  Геодезическая = минимальная цепочка трансформаций,
  где каждый шаг сохраняет семантическую связность.

TangentSpace:
  В каждой точке многообразия — допустимые направления движения.
  Касательный вектор = операция трансформации (insert/delete/substitute).
  Допустимые направления = те, что не нарушают связность.
"""

import numpy as np
import heapq
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from loguru import logger


class TangentDirection(Enum):
    INSERT = "insert"
    DELETE = "delete"
    SUBSTITUTE = "substitute"
    MERGE = "merge"
    SPLIT = "split"


@dataclass
class TangentVector:
    """Касательный вектор = допустимая трансформация в точке."""
    direction: TangentDirection
    target_symbol: int               # символ для операции
    position: int = 0                # позиция в сборке
    semantic_cost: float = 0.0       # семантическая цена трансформации
    coherence_after: float = 0.0     # связность после трансформации


@dataclass
class GeodesicPath:
    """Геодезический путь между двумя сборками."""
    source_sequence: List[int]
    target_sequence: List[int]
    steps: List[TangentVector]       # последовательность касательных векторов
    total_cost: float                # суммарная семантическая цена
    coherence_profile: List[float]   # связность на каждом шаге


class TangentSpace:
    """
    Касательное пространство в точке многообразия.

    Для данной сборки вычисляет все допустимые направления
    трансформации и их семантическую цену.
    """

    def __init__(self, potential_field, topological_field):
        self.potential_field = potential_field
        self.topological_field = topological_field

    def compute_tangent_vectors(
        self,
        symbol_indices: List[int],
        max_vectors: int = 20,
    ) -> List[TangentVector]:
        """
        Вычислить допустимые касательные векторы в точке сборки.

        Для каждой позиции i:
        - INSERT: какие символы можно вставить после i с низкой ценой?
        - DELETE: какова цена удаления символа i?
        - SUBSTITUTE: на какие символы можно заменить i с низкой ценой?
        """
        vectors = []
        n = len(symbol_indices)
        if n == 0:
            return vectors

        aff = self.potential_field.affinity.cpu().numpy()
        V = min(self.potential_field.vocab_size, aff.shape[0])

        for i in range(n):
            si = symbol_indices[i]

            # DELETE: цена = потеря аффинности с соседями
            del_cost = 0.0
            if i > 0:
                del_cost += 1.0 - aff[symbol_indices[i-1], symbol_indices[i+1]] if i+1 < n else 0.5
            if i < n - 1:
                del_cost += 0.5
            del_cost /= max(n - 1, 1)

            vectors.append(TangentVector(
                direction=TangentDirection.DELETE,
                target_symbol=si,
                position=i,
                semantic_cost=del_cost,
                coherence_after=1.0 - del_cost,
            ))

            # INSERT: цена вставки нового символа
            best_insert_symbol = -1
            best_insert_cost = 1.0
            for j in range(V):
                if j in symbol_indices:
                    continue
                cost = 1.0 - aff[si, j]
                if i < n - 1:
                    cost = (cost + 1.0 - aff[j, symbol_indices[i+1]]) / 2
                if cost < best_insert_cost:
                    best_insert_cost = cost
                    best_insert_symbol = j

            if best_insert_symbol >= 0:
                vectors.append(TangentVector(
                    direction=TangentDirection.INSERT,
                    target_symbol=best_insert_symbol,
                    position=i,
                    semantic_cost=best_insert_cost,
                    coherence_after=1.0 - best_insert_cost,
                ))

            # SUBSTITUTE: замена на другой символ
            best_sub_symbol = -1
            best_sub_cost = 1.0
            for j in range(V):
                if j == si or j in symbol_indices:
                    continue
                cost = 1.0 - aff[si, j]
                if i > 0:
                    cost = (cost + 1.0 - aff[symbol_indices[i-1], j]) / 2
                if i < n - 1:
                    cost = (cost + 1.0 - aff[j, symbol_indices[i+1]]) / 3
                if cost < best_sub_cost:
                    best_sub_cost = cost
                    best_sub_symbol = j

            if best_sub_symbol >= 0:
                vectors.append(TangentVector(
                    direction=TangentDirection.SUBSTITUTE,
                    target_symbol=best_sub_symbol,
                    position=i,
                    semantic_cost=best_sub_cost,
                    coherence_after=1.0 - best_sub_cost,
                ))

            if len(vectors) >= max_vectors:
                break

        # Сортируем по цене
        vectors.sort(key=lambda v: v.semantic_cost)
        return vectors[:max_vectors]

    def is_valid_transition(
        self,
        from_symbols: List[int],
        to_symbols: List[int],
    ) -> Tuple[bool, float]:
        """
        Проверить, является ли переход from→to допустимым
        (не нарушает ли связность).

        Возвращает (is_valid, coherence_after).
        """
        aff = self.potential_field.affinity.cpu().numpy()

        if len(from_symbols) == len(to_symbols):
            # Проверяем позаментную связность
            scores = []
            for i in range(len(from_symbols)):
                a = float(aff[from_symbols[i], to_symbols[i]])
                scores.append(a)
            avg = np.mean(scores)
            return avg > 0.4, avg

        # Разная длина — проверяем cross-аффинность
        scores = []
        for si in from_symbols:
            for tj in to_symbols:
                scores.append(float(aff[si, tj]))
        avg = np.mean(scores)
        return avg > 0.3, avg


class GeodesicNavigator:
    """
    Находит кратчайшие логические пути в многообразии.

    Геодезическая = путь минимальной семантической цены,
    где каждый шаг сохраняет связность > порога.
    """

    def __init__(
        self,
        potential_field,
        topological_field,
        tangent_space: TangentSpace,
        max_path_length: int = 20,
        coherence_threshold: float = 0.3,
    ):
        self.potential_field = potential_field
        self.topological_field = topological_field
        self.tangent_space = tangent_space
        self.max_path_length = max_path_length
        self.coherence_threshold = coherence_threshold

    def find_geodesic(
        self,
        from_symbols: List[int],
        to_symbols: List[int],
    ) -> Optional[GeodesicPath]:
        """
        Найти геодезический путь от from к to.

        Использует A*-like поиск в пространстве трансформаций.
        Эвристика = расстояние между координатами точек в многообразии.
        """
        from_key = tuple(from_symbols[:20])
        to_key = tuple(to_symbols[:20])

        if from_key == to_key:
            return GeodesicPath(from_symbols, to_symbols, [], 0.0, [])

        # A* search
        start_coords = self.topological_field.compute_assembly_coordinates(from_symbols)
        target_coords = self.topological_field.compute_assembly_coordinates(to_symbols)

        # Heuristic: расстояние между координатами сборок
        def heuristic(seq: List[int]) -> float:
            c = self.topological_field.compute_assembly_coordinates(seq)
            return float(np.linalg.norm(c - target_coords))

        # Priority queue: (f = g + h, g, seq, path, coherence_history)
        start_h = heuristic(from_symbols)
        pq = [(start_h, 0.0, from_symbols[:], [], [1.0])]
        visited = {from_key}

        while pq and len(pq) < 10000:
            f, g, current, path_steps, coh_history = heapq.heappop(pq)

            if g > self.max_path_length:
                continue

            current_key = tuple(current[:20])
            if current_key == to_key:
                return GeodesicPath(
                    from_symbols, to_symbols,
                    path_steps, g, coh_history,
                )

            # Генерируем соседей через касательное пространство
            vectors = self.tangent_space.compute_tangent_vectors(current, max_vectors=5)

            for vec in vectors:
                if vec.semantic_cost > 0.6:
                    continue

                # Применяем трансформацию
                if vec.direction == TangentDirection.INSERT:
                    new_seq = current[:vec.position+1] + [vec.target_symbol] + current[vec.position+1:]
                elif vec.direction == TangentDirection.DELETE:
                    new_seq = current[:vec.position] + current[vec.position+1:]
                elif vec.direction == TangentDirection.SUBSTITUTE:
                    new_seq = current[:]
                    new_seq[vec.position] = vec.target_symbol
                else:
                    continue

                new_key = tuple(new_seq[:20])
                if new_key in visited:
                    continue

                new_g = g + vec.semantic_cost
                new_h = heuristic(new_seq)
                new_f = new_g + new_h

                new_coh = coh_history + [vec.coherence_after]
                new_steps = path_steps + [vec]

                heapq.heappush(pq, (new_f, new_g, new_seq, new_steps, new_coh))
                visited.add(new_key)

        return None  # Путь не найден

    def compute_path_similarity(
        self,
        path_a: GeodesicPath,
        path_b: GeodesicPath,
    ) -> float:
        """
        Вычислить сходство двух геодезических путей.

        Сходство = overlap операций / длина пути.
        """
        if not path_a.steps or not path_b.steps:
            return 0.0

        ops_a = [(s.direction, s.target_symbol) for s in path_a.steps]
        ops_b = [(s.direction, s.target_symbol) for s in path_b.steps]

        matches = sum(1 for op in ops_a if op in ops_b)
        return matches / max(len(ops_a), len(ops_b), 1)

    def summary(self) -> str:
        return f"GeodesicNavigator(max_len={self.max_path_length}, coh_thr={self.coherence_threshold})"
