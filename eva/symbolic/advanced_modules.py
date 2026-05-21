"""
UncertaintyQuantifier + ActiveInference + CausalDiscovery.

Три завершающих модуля символьной архитектуры.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
from loguru import logger


class UncertaintyQuantifier:
    """
    Квантификация неопределённости: не только valid/invalid, но и НАСКОЛЬКО.

    Типы неопределённости:
    1. Aleatoric (случайная): природная вариативность связей
    2. Epistemic (знаниевая): недостаток данных о связи
    3. Structural (структурная): противоречивые сигналы о связи

    total_uncertainty = aleatoric + epistemic + structural
    """

    def __init__(self, potential_field):
        self.potential_field = potential_field
        self.affinity_variance = np.zeros((potential_field.vocab_size, potential_field.vocab_size))
        self.observation_count = np.zeros((potential_field.vocab_size, potential_field.vocab_size))
        self.conflict_count = np.zeros((potential_field.vocab_size, potential_field.vocab_size))

    def record_observation(self, i: int, j: int, affinity: float, was_valid: bool):
        """Записать наблюдение о связи i→j."""
        if i >= self.potential_field.vocab_size or j >= self.potential_field.vocab_size:
            return

        n = self.observation_count[i, j]
        if n > 0:
            old_mean = self.affinity_variance[i, j]
            self.affinity_variance[i, j] = (n * old_mean + (affinity - old_mean)**2 / (n + 1)) / (n + 1)

        self.observation_count[i, j] += 1
        if not was_valid:
            self.conflict_count[i, j] += 1

    def aleatoric_uncertainty(self, i: int, j: int) -> float:
        """Природная вариативность."""
        return float(self.affinity_variance[i, j])

    def epistemic_uncertainty(self, i: int, j: int) -> float:
        """Недостаток данных: высокая когда мало наблюдений."""
        n = self.observation_count[i, j]
        if n < 1:
            return 1.0
        return 1.0 / np.sqrt(n + 1)

    def structural_uncertainty(self, i: int, j: int) -> float:
        """Противоречивые сигналы."""
        n = self.observation_count[i, j]
        if n < 1:
            return 0.0
        return self.conflict_count[i, j] / n

    def total_uncertainty(self, i: int, j: int) -> float:
        """Общая неопределённость."""
        return float(
            self.aleatoric_uncertainty(i, j) * 0.3 +
            self.epistemic_uncertainty(i, j) * 0.4 +
            self.structural_uncertainty(i, j) * 0.3
        )

    def most_uncertain_connections(self, top_k: int = 20) -> List[Tuple[int, int, float, str]]:
        """Топ-K самых неопределённых связей."""
        scores = []
        n = min(self.potential_field.vocab_size, self.observation_count.shape[0])
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                u = self.total_uncertainty(i, j)
                if u > 0.3:
                    # Определяем доминирующий тип
                    a = self.aleatoric_uncertainty(i, j)
                    e = self.epistemic_uncertainty(i, j)
                    s = self.structural_uncertainty(i, j)
                    dom = max((a, "aleatoric"), (e, "epistemic"), (s, "structural"), key=lambda x: x[0])
                    scores.append((i, j, u, dom[1]))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    def summary(self) -> str:
        avg_obs = self.observation_count.mean()
        avg_epistemic = np.mean([self.epistemic_uncertainty(i, j) for i in range(10) for j in range(10)])
        return f"Uncertainty(avg_obs={avg_obs:.1f}, avg_epistemic={avg_epistemic:.3f})"


class ActiveInference:
    """
    Активный вывод: модель ищет информацию для снижения неопределённости.

    Вместо пассивного ожидания данных, модель:
    1. Находит самые неопределённые связи
    2. Генерирует "внутренние запросы" для их проверки
    3. Исследует эти связи через AssemblyExplorer
    4. Обновляет уверенность на основе результатов
    """

    def __init__(
        self,
        uncertainty_quantifier: UncertaintyQuantifier,
        explorer,
        concept_miner,
        query_budget: int = 10,
    ):
        self.uncertainty = uncertainty_quantifier
        self.explorer = explorer
        self.concept_miner = concept_miner
        self.query_budget = query_budget

        self.internal_queries: deque = deque(maxlen=100)
        self.resolved_count = 0
        self.total_queries = 0

    def generate_internal_queries(self) -> List[Tuple[int, int, float]]:
        """
        Сгенерировать внутренние запросы: какие связи нужно проверить.
        """
        uncertain = self.uncertainty.most_uncertain_connections(top_k=self.query_budget)
        queries = [(i, j, u) for i, j, u, _ in uncertain]
        self.internal_queries.extend(queries)
        self.total_queries += len(queries)
        return queries

    def resolve_query(self, i: int, j: int) -> bool:
        """
        Разрешить запрос: проверить связь i→j через исследование.

        Использует AssemblyExplorer для проверки гипотетической связи.
        """
        result = self.explorer.affinity_walk(i, max_length=3)
        if j in result.sequence:
            self.uncertainty.record_observation(i, j, result.coherence_score, True)
            self.resolved_count += 1
            return True

        # Проверяем через концепт-майнер
        concepts = self.concept_miner.search_free_space(from_known=[i], max_tries=3)
        for concept in concepts:
            if j in concept.symbol_indices:
                self.uncertainty.record_observation(i, j, concept.coherence_score, True)
                self.resolved_count += 1
                return True

        self.uncertainty.record_observation(i, j, 0.1, False)
        self.resolved_count += 1
        return False

    def run_inference_cycle(self):
        """Один цикл активного вывода."""
        queries = self.generate_internal_queries()
        for i, j, u in queries[:self.query_budget]:
            self.resolve_query(i, j)

    def summary(self) -> str:
        return (
            f"ActiveInference(queries={self.total_queries}, resolved={self.resolved_count}, "
            f"pending={len(self.internal_queries)})"
        )


class CausalDiscovery:
    """
    Обнаружение причинно-следственных связей между паттернами.

    Не "A и B часто вместе", а "A ВЫЗЫВАЕТ B".

    Метод: проверка причинности по Грейнджеру на временных рядах сборок.
    Если появление паттерна A предшествует B с высокой вероятностью,
    и удаление A снижает вероятность B → A → B (причинность).
    """

    def __init__(self, potential_field, grammar, history_window: int = 500):
        self.potential_field = potential_field
        self.grammar = grammar
        self.history_window = history_window

        # История появлений паттернов: pattern_id → временной ряд появлений
        self.pattern_timeline: Dict[str, List[int]] = defaultdict(list)
        self.global_step = 0

        # Обнаруженные причинные связи
        self.causal_links: List[Tuple[str, str, float]] = []

    def record_pattern_occurrence(self, pattern_id: str):
        """Записать появление паттерна."""
        self.pattern_timeline[pattern_id].append(self.global_step)
        self.global_step += 1

        # Ограничиваем историю
        if len(self.pattern_timeline[pattern_id]) > self.history_window:
            self.pattern_timeline[pattern_id] = self.pattern_timeline[pattern_id][-self.history_window:]

    def discover_causal_links(self) -> List[Tuple[str, str, float]]:
        """
        Обнаружить причинные связи через тест Грейнджера.

        Для каждой пары паттернов A, B:
        P(B_t | B_{t-1}, A_{t-1}) > P(B_t | B_{t-1}) → A → B
        """
        new_links = []
        pat_ids = list(self.pattern_timeline.keys())

        for i, a_id in enumerate(pat_ids):
            a_timeline = self.pattern_timeline[a_id]
            if len(a_timeline) < 10:
                continue

            for j, b_id in enumerate(pat_ids):
                if i == j:
                    continue
                b_timeline = self.pattern_timeline[b_id]
                if len(b_timeline) < 10:
                    continue

                # Упрощённый тест: корреляция с лагом 1
                causality_score = self._granger_simple(a_timeline, b_timeline)
                if causality_score > 0.6:
                    new_links.append((a_id, b_id, causality_score))

        new_links.sort(key=lambda x: x[2], reverse=True)
        self.causal_links = new_links[:50]
        return new_links

    def _granger_simple(self, a_timeline: List[int], b_timeline: List[int]) -> float:
        """
        Упрощённый тест Грейнджера: насколько A предшествует B.
        """
        if not a_timeline or not b_timeline:
            return 0.0

        # Создаём бинарные временные ряды
        all_steps = sorted(set(a_timeline + b_timeline))
        if len(all_steps) < 10:
            return 0.0

        max_step = max(all_steps)
        a_series = np.zeros(max_step + 2)
        b_series = np.zeros(max_step + 2)
        a_series[a_timeline] = 1
        b_series[b_timeline] = 1

        # Проверяем: предсказывает ли a_{t-1} b_t лучше чем b_{t-1}?
        matches = 0
        trials = 0
        for t in range(2, max_step):
            if b_series[t] == 1:
                trials += 1
                if a_series[t - 1] == 1 or b_series[t - 1] == 1:
                    matches += 1

        if trials < 5:
            return 0.0
        return matches / trials

    def summary(self) -> str:
        return f"CausalDiscovery(links={len(self.causal_links)}, step={self.global_step})"
