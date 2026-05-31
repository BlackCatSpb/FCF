"""
AssemblyExplorer — curiosity-driven исследование символьных комбинаций.

Модель активно ищет новые валидные сборки через:
1. Random walk по графу аффинности
2. Комбинаторное исследование (cross-product известных паттернов)
3. Gap-filling: поиск "пробелов" в пространстве связей
4. Temperature-annealing: от случайных к структурно-обоснованным
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from loguru import logger
import random as py_random


@dataclass
class ExplorationResult:
    sequence: List[int]
    coherence_score: float
    is_novel: bool
    distance_to_known: float
    potential_quality: float


class AssemblyExplorer:
    """
    Исследует пространство символьных сборок.

    Стратегии:
    - AffinityWalk: следовать по графу аффинности
    - PatternCross: скрещивать известные паттерны
    - GapFill: заполнять семантические пробелы
    - AnnealedExplore: температура снижается → меньше случайности
    """

    def __init__(
        self,
        potential_field,
        grammar,
        validator,
        embed_dim: int = 256,
    ):
        self.potential_field = potential_field
        self.grammar = grammar
        self.validator = validator
        self.embed_dim = embed_dim

        # История исследований
        self.explored_hashes: Set[int] = set()
        self.discoveries: List[ExplorationResult] = []
        self.total_explorations: int = 0
        self.useful_discoveries: int = 0

        # Температура (для annealed exploration)
        self.temperature: float = 1.0
        self.temperature_decay: float = 0.9999
        self.min_temperature: float = 0.01

        # Статистика для gap detection
        self.symbol_frequency: np.ndarray = np.ones(potential_field.vocab_size)

    def _hash_sequence(self, seq: List[int]) -> int:
        return hash(tuple(seq[:50]))

    def affinity_walk(
        self,
        start_symbol: int,
        max_length: int = 20,
        temperature: float = None,
    ) -> ExplorationResult:
        """
        Случайное блуждание по графу аффинности.

        На каждом шаге выбираем следующий символ пропорционально
        аффинности, с температурой для случайности.
        """
        temp = temperature if temperature is not None else self.temperature

        sequence = [start_symbol]
        current = start_symbol

        for _ in range(max_length - 1):
            affinities = self.potential_field.affinity[current].cpu().numpy()
            affinities = affinities / temp

            # Softmax с температурой
            affinities = affinities - affinities.max()
            probs = np.exp(np.clip(affinities, -10, 10))
            probs = probs / (probs.sum() + 1e-8)

            next_symbol = np.random.choice(len(probs), p=probs)
            sequence.append(int(next_symbol))
            current = next_symbol

        # Оцениваем
        coherence = self._evaluate_coherence(sequence)
        is_novel = self._hash_sequence(sequence) not in self.explored_hashes
        distance = self._distance_to_known(sequence)

        return ExplorationResult(
            sequence=sequence,
            coherence_score=coherence,
            is_novel=is_novel,
            distance_to_known=distance,
            potential_quality=coherence * (1.0 - distance),
        )

    def pattern_cross(
        self,
        pattern_a_indices: List[int],
        pattern_b_indices: List[int],
    ) -> ExplorationResult:
        """
        Скрещивание двух паттернов: берём первую половину A + вторую половину B.

        Если cross-аффинность в точке соединения высокая — паттерн валиден.
        """
        mid_a = len(pattern_a_indices) // 2
        mid_b = len(pattern_b_indices) // 2

        combined = pattern_a_indices[:mid_a] + pattern_b_indices[mid_b:]

        if len(combined) > 50:
            combined = combined[:50]

        coherence = self._evaluate_coherence(combined)
        is_novel = self._hash_sequence(combined) not in self.explored_hashes
        distance = self._distance_to_known(combined)

        return ExplorationResult(
            sequence=combined,
            coherence_score=coherence,
            is_novel=is_novel,
            distance_to_known=distance,
            potential_quality=coherence * (1.0 - distance) * (1.0 if is_novel else 0.3),
        )

    def gap_fill(
        self,
        prefix: List[int],
        target: List[int],
        max_fill: int = 5,
    ) -> ExplorationResult:
        """
        Заполнить пробел между prefix и target.

        Найти цепочку символов, которая соединяет prefix и target
        с максимальной аффинностью на каждом шаге.
        """
        if not prefix or not target:
            return ExplorationResult([], 0.0, False, 1.0, 0.0)

        current = prefix[-1]
        target_first = target[0]
        fill = []

        for _ in range(max_fill):
            if current == target_first:
                break

            aff = self.potential_field.affinity[current].cpu().numpy()
            best_next = np.argmax(aff)

            if best_next == current:
                break

            fill.append(int(best_next))
            current = int(best_next)

        sequence = prefix + fill + target[1:]
        sequence = sequence[:50]

        coherence = self._evaluate_coherence(sequence)
        is_novel = self._hash_sequence(sequence) not in self.explored_hashes
        distance = self._distance_to_known(sequence)

        return ExplorationResult(
            sequence=sequence,
            coherence_score=coherence,
            is_novel=is_novel,
            distance_to_known=distance,
            potential_quality=coherence * (1.0 - distance),
        )

    def explore(
        self,
        n_attempts: int = 10,
    ) -> List[ExplorationResult]:
        """
        Запустить исследование: перебрать стратегии, выбрать лучшие.
        """
        results = []

        # Стратегия 1: Affinity walk от частых символов
        top_symbols = np.argsort(self.symbol_frequency)[-5:][::-1]
        for sym in top_symbols[:3]:
            result = self.affinity_walk(int(sym), max_length=15)
            results.append(result)

        # Стратегия 2: Pattern cross
        if self.grammar:
            digrams = list(self.grammar.patterns[0].values())
            if len(digrams) >= 2:
                a = py_random.choice(digrams)
                b = py_random.choice(digrams)
                result = self.pattern_cross(a.symbol_indices, b.symbol_indices)
                results.append(result)

        # Стратегия 3: Gap fill
        if len(results) >= 2:
            result = self.gap_fill(
                results[0].sequence[:5],
                results[-1].sequence[:5],
            )
            results.append(result)

        # Оцениваем и сохраняем
        for result in results:
            self._record_exploration(result)

        self.temperature = max(self.temperature * self.temperature_decay, self.min_temperature)

        return results

    def _evaluate_coherence(self, sequence: List[int]) -> float:
        """Оценить связность последовательности через аффинность."""
        if len(sequence) < 2:
            return 0.0
        scores = []
        for i in range(len(sequence) - 1):
            a = float(self.potential_field.affinity[sequence[i], sequence[i + 1]])
            scores.append(a)
        return float(np.mean(scores)) if scores else 0.0

    def _distance_to_known(self, sequence: List[int]) -> float:
        """Расстояние до ближайшего известного паттерна."""
        if self.grammar is None:
            return 1.0

        min_dist = 1.0
        for level, pats in self.grammar.patterns.items():
            for ph, pattern in pats.items():
                if pattern.length < 2:
                    continue
                overlap = len(set(sequence) & set(pattern.symbol_indices))
                dist = 1.0 - overlap / max(len(sequence), len(pattern.symbol_indices))
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    def _record_exploration(self, result: ExplorationResult):
        self.total_explorations += 1
        h = self._hash_sequence(result.sequence)
        self.explored_hashes.add(h)

        if result.potential_quality > 0.5:
            self.discoveries.append(result)
            self.useful_discoveries += 1

        for sym in result.sequence:
            if sym < len(self.symbol_frequency):
                self.symbol_frequency[sym] += 1

        if len(self.discoveries) > 5000:
            self.discoveries = self.discoveries[-2500:]

    def summary(self) -> str:
        return (
            f"Explorer: explored={self.total_explorations}, "
            f"discoveries={self.useful_discoveries}, "
            f"temperature={self.temperature:.3f}"
        )
