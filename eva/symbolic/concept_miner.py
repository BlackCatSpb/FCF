"""
SymbolicConceptMiner — поиск потенциальных концептов в свободном пространстве.

Пространство:
  Все связи - Противоречия - Известное = Свободное пространство

В свободном пространстве ищем КОНЦЕПТЫ — устойчивые группы связей,
которые:
1. Не противоречат известному (прошли фильтр противоречий)
2. Не дублируют уже известные паттерны
3. Образуют семантически замкнутые структуры (прошли closure check)
4. Имеют достаточную плотность в многообразии

Концепт = "новое знание" — то, что модель ОТКРЫЛА, а не запомнила.

Алгоритм:
1. Выбрать область свободного пространства (случайно или по плотности)
2. Построить геодезическую от известного паттерна в эту область
3. Проверить связность, замкнутость, непротиворечивость
4. Если прошло → создать концепт-кандидат
5. При повторном обнаружении → подтвердить концепт
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict, deque
from loguru import logger
import random as py_random


@dataclass
class Concept:
    """Концепт — потенциальное новое знание."""
    concept_id: str
    symbol_indices: List[int]            # символьная последовательность
    assembly_coordinates: np.ndarray      # координаты в многообразии
    density: float                        # плотность в многообразии
    coherence_score: float                # семантическая связность
    contradiction_score: float            # 0 = нет противоречий, 1 = много
    novelty_score: float                  # 0 = полный дубликат, 1 = полностью новое
    
    # Жизненный цикл
    status: str = "candidate"             # candidate → confirmed → stable
    confirmation_count: int = 0
    created_at: float = 0.0
    last_validated: float = 0.0
    
    source_known_pattern: Optional[str] = None  # из какого известного вышел
    
    @property
    def quality(self) -> float:
        """Качество концепта: связность + (1 - противоречивость) + новизна."""
        return (self.coherence_score + (1.0 - self.contradiction_score) + self.novelty_score) / 3.0


class SymbolicConceptMiner:
    """
    Ищет новые концепты в свободном пространстве.

    Свободное пространство = всё что не запрещено и не известно.
    Концепт = устойчивая структура в этом пространстве.
    """

    def __init__(
        self,
        potential_field,
        topological_field,
        contradiction_filter,
        grammar,
        logic_bridge,
        geodesic_navigator,
        min_concept_quality: float = 0.5,
        max_concepts: int = 10000,
    ):
        self.potential_field = potential_field
        self.topological_field = topological_field
        self.contradiction_filter = contradiction_filter
        self.grammar = grammar
        self.logic_bridge = logic_bridge
        self.geodesic_navigator = geodesic_navigator
        self.min_concept_quality = min_concept_quality
        self.max_concepts = max_concepts

        self.concepts: Dict[str, Concept] = {}
        self._next_concept_id: int = 0

        # Статистика
        self.total_searches: int = 0
        self.total_discoveries: int = 0

    def search_free_space(
        self,
        from_known: Optional[List[int]] = None,
        search_radius: float = 0.5,
        max_tries: int = 20,
    ) -> List[Concept]:
        """
        Поиск в свободном пространстве от известного паттерна.

        1. Выбрать известный паттерн (или случайную точку)
        2. Идти по касательным векторам в неизвестную область
        3. Проверить — не запрещено ли, не известно ли
        4. Если новое и связное → концепт
        """
        discovered = []

        for _ in range(max_tries):
            # Выбираем отправную точку
            if from_known is None:
                from_known = self._pick_random_known_point()

            if from_known is None or len(from_known) < 2:
                continue

            # Делаем шаг в "свободную" сторону
            candidate = self._step_into_free_space(from_known, search_radius)
            if candidate is None or len(candidate) < 2:
                continue

            # Проверяем на противоречия
            contra_score = self._check_contradictions(candidate)
            if contra_score > 0.5:
                continue

            # Проверяем на известность
            is_known, known_similarity = self._check_known(candidate)
            if is_known:
                continue

            # Проверяем связность
            coherence = self._evaluate_coherence(candidate)
            if coherence < self.min_concept_quality:
                continue

            # Создаём концепт
            coords = self.topological_field.compute_assembly_coordinates(candidate)
            concept = Concept(
                concept_id=f"cpt_{self._next_concept_id:06d}",
                symbol_indices=candidate,
                assembly_coordinates=coords,
                density=self._estimate_density(candidate),
                coherence_score=coherence,
                contradiction_score=contra_score,
                novelty_score=1.0 - known_similarity,
                source_known_pattern=self._hash_sequence(from_known) if from_known else None,
                created_at=np.float64(__import__('time').time()),
                last_validated=np.float64(__import__('time').time()),
            )

            self.concepts[concept.concept_id] = concept
            self._next_concept_id += 1
            self.total_discoveries += 1
            discovered.append(concept)

            if len(self.concepts) > self.max_concepts:
                self._prune_weak_concepts()

        self.total_searches += 1
        return discovered

    def _step_into_free_space(
        self,
        from_sequence: List[int],
        radius: float,
    ) -> Optional[List[int]]:
        """Сделать шаг в свободное пространство."""
        result = list(from_sequence)

        # Получаем касательные векторы
        if not hasattr(self, '_tangent_space'):
            from .geodesic_navigator import TangentSpace
            self._tangent_space = TangentSpace(self.potential_field, self.topological_field)

        vectors = self._tangent_space.compute_tangent_vectors(result, max_vectors=10)
        py_random.shuffle(vectors)

        for vec in vectors:
            # Применяем только если НЕ запрещено
            if vec.direction.name == 'INSERT':
                forbidden, _, _ = self.contradiction_filter.is_forbidden(
                    result[:vec.position+1], vec.target_symbol
                )
                if not forbidden:
                    result = result[:vec.position+1] + [vec.target_symbol] + result[vec.position+1:]
                    break
            elif vec.direction.name == 'SUBSTITUTE':
                forbidden, _, _ = self.contradiction_filter.is_forbidden(
                    result[:vec.position], vec.target_symbol
                )
                if not forbidden:
                    result = result[:]
                    result[vec.position] = vec.target_symbol
                    break

        return result if len(result) > len(from_sequence) or result != from_sequence else None

    def _check_contradictions(self, sequence: List[int]) -> float:
        """Оценить противоречивость последовательности."""
        contra_count = 0
        for i in range(len(sequence) - 1):
            forbidden, conf, _ = self.contradiction_filter.is_forbidden(
                sequence[:i+1], sequence[i+1]
            )
            if forbidden:
                contra_count += 1

        return contra_count / max(len(sequence) - 1, 1)

    def _check_known(self, sequence: List[int]) -> Tuple[bool, float]:
        """Проверить, известна ли уже эта последовательность."""
        best_similarity = 0.0

        for level, pats in self.grammar.patterns.items():
            for ph, pattern in pats.items():
                if pattern.length < 2:
                    continue
                overlap = len(set(sequence) & set(pattern.symbol_indices))
                sim = overlap / max(len(sequence), len(pattern.symbol_indices))
                if sim > best_similarity:
                    best_similarity = sim

        # Также проверяем концепты
        for cid, concept in self.concepts.items():
            overlap = len(set(sequence) & set(concept.symbol_indices))
            sim = overlap / max(len(sequence), len(concept.symbol_indices))
            if sim > best_similarity:
                best_similarity = sim

        return best_similarity > 0.8, best_similarity

    def _evaluate_coherence(self, sequence: List[int]) -> float:
        """Оценить семантическую связность."""
        if len(sequence) < 2:
            return 0.0
        aff = self.potential_field.affinity.cpu().numpy()
        scores = []
        for i in range(len(sequence) - 1):
            scores.append(float(aff[sequence[i], sequence[i + 1]]))
        return float(np.mean(scores)) if scores else 0.0

    def _estimate_density(self, sequence: List[int]) -> float:
        """Оценить плотность в многообразии."""
        densities = [self.topological_field.get_local_density(s) for s in sequence]
        return float(np.mean(densities)) if densities else 0.0

    def _pick_random_known_point(self) -> Optional[List[int]]:
        """Выбрать случайный известный паттерн."""
        all_patterns = []
        for level, pats in self.grammar.patterns.items():
            for ph, pattern in pats.items():
                if pattern.length >= 2:
                    all_patterns.append(pattern.symbol_indices)

        if all_patterns:
            return py_random.choice(all_patterns)
        return None

    def _hash_sequence(self, seq: List[int]) -> str:
        return "|".join(str(s) for s in seq[:20])

    def _prune_weak_concepts(self):
        """Удалить слабые концепты."""
        sorted_concepts = sorted(
            self.concepts.values(),
            key=lambda c: c.quality,
        )
        keep = sorted_concepts[-self.max_concepts:]
        self.concepts = {c.concept_id: c for c in keep}

    def confirm_concept(self, concept_id: str):
        """Подтвердить концепт (найден повторно)."""
        if concept_id in self.concepts:
            c = self.concepts[concept_id]
            c.confirmation_count += 1
            c.last_validated = np.float64(__import__('time').time())
            if c.confirmation_count >= 3:
                c.status = "confirmed"
            if c.confirmation_count >= 10:
                c.status = "stable"

    def get_best_concepts(self, n: int = 10) -> List[Concept]:
        """Топ-N концептов по качеству."""
        return sorted(
            self.concepts.values(),
            key=lambda c: c.quality * c.confirmation_count,
            reverse=True,
        )[:n]

    def summary(self) -> str:
        confirmed = sum(1 for c in self.concepts.values() if c.status == "confirmed")
        stable = sum(1 for c in self.concepts.values() if c.status == "stable")
        return (
            f"ConceptMiner: concepts={len(self.concepts)} "
            f"(confirmed={confirmed}, stable={stable}), "
            f"discoveries={self.total_discoveries}"
        )
