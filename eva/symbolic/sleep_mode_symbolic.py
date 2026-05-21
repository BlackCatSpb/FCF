"""
SleepModeSymbolic — консолидация символьной памяти.

Аналог сна в мозге:
1. Replay: переигрывание недавних сборок → усиление важных связей
2. Consolidation: перенос из быстрой памяти (states) в медленную (patterns)
3. Dream: случайная рекомбинация → поиск новых валидных паттернов
4. Pruning: удаление слабых/неиспользуемых паттернов
5. Generalization: объединение похожих паттернов в абстрактные

Цикл:
  while in_sleep_mode:
    replay_recent()
    consolidate()
    dream()
    prune_weak()
    generalize()
"""

import torch
import numpy as np
import time
import random as py_random
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
from loguru import logger


class SleepModeSymbolic:
    """
    Режим сна для символьной системы.

    Во время "сна" (нет новых данных):
    - Проигрывает недавние сборки (replay) → STDP усиление
    - Консолидирует: состояния → паттерны → правила
    - Мечтает: рекомбинирует паттерны → новые комбинации
    - Чистит: удаляет слабые связи
    - Обобщает: находит общие структуры в похожих паттернах
    """

    def __init__(
        self,
        potential_dynamics,
        grammar,
        validator,
        explorer,
        logic_bridge,
        idle_timeout: float = 300.0,   # сек бездействия до сна
        sleep_duration: float = 60.0,   # длительность сна
        replay_count: int = 100,        # сколько сборок переиграть
        dream_count: int = 50,          # сколько рекомбинаций
    ):
        self.dynamics = potential_dynamics
        self.grammar = grammar
        self.validator = validator
        self.explorer = explorer
        self.logic_bridge = logic_bridge

        self.idle_timeout = idle_timeout
        self.sleep_duration = sleep_duration
        self.replay_count = replay_count
        self.dream_count = dream_count

        # Буфер недавних сборок для replay
        self.recent_assemblies: deque = deque(maxlen=500)

        # Состояние
        self._sleeping = False
        self._last_activity = time.time()
        self._sleep_cycles = 0
        self._total_pruned = 0
        self._total_merged = 0

    def record_assembly(self, symbol_indices: List[int], attention_weights: Optional[List[float]] = None):
        """Записать сборку в буфер для будущего replay."""
        self.recent_assemblies.append((symbol_indices, attention_weights))
        self._last_activity = time.time()

    def should_sleep(self) -> bool:
        """Пора ли спать? (idle > timeout)"""
        return (time.time() - self._last_activity) > self.idle_timeout and not self._sleeping

    def run_sleep_cycle(self):
        """Один цикл сна."""
        if self._sleeping:
            return

        self._sleeping = True
        self._sleep_cycles += 1
        start_time = time.time()
        logger.info(f"[SleepMode] Цикл #{self._sleep_cycles} начат")

        try:
            # 1. Replay: переигрывание недавних сборок
            self._replay()

            # 2. Consolidation: сборки → паттерны
            self._consolidate()

            # 3. Dream: рекомбинация паттернов
            self._dream()

            # 4. Pruning: удаление слабых связей
            self._prune()

            # 5. Generalization: объединение похожих паттернов
            self._generalize()

        except Exception as e:
            logger.warning(f"[SleepMode] Ошибка: {e}")

        elapsed = time.time() - start_time
        logger.info(f"[SleepMode] Цикл завершён за {elapsed:.1f}с")
        self._sleeping = False

    def _replay(self):
        """Переиграть недавние сборки для усиления связей."""
        assemblies = list(self.recent_assemblies)[-self.replay_count:]
        if not assemblies:
            return

        logger.info(f"[SleepMode] Replay: {len(assemblies)} сборок")

        for symbols, attn_weights in assemblies:
            self.dynamics.reinforce_sequence(symbols, attn_weights, confidence=0.8)

        self.dynamics.long_term_depression()
        self.dynamics.homeostatic_scaling()

    def _consolidate(self):
        """Консолидация: перенести сборки в долговременную память (паттерны)."""
        logger.info("[SleepMode] Consolidation: discovery диграмм")
        new_digrams = self.grammar.discover_digrams(min_affinity=0.5)

        if new_digrams:
            logger.info(f"[SleepMode] Обнаружено диграмм: {len(new_digrams)}")

        # Discovery N-грамм из накопленных диграмм
        if len(self.grammar.patterns[0]) > 10:
            new_ngrams = self.grammar.discover_ngrams(max_n=4, min_coherence=0.4)
            if new_ngrams:
                logger.info(f"[SleepMode] Обнаружено N-грамм: {len(new_ngrams)}")

        # Строим мостики между новыми паттернами
        if self.logic_bridge:
            new_rules = self.logic_bridge.discover_expansion_rules(self.grammar)
            if new_rules:
                logger.info(f"[SleepMode] Обнаружено правил: {len(new_rules)}")

    def _dream(self):
        """Dream mode: случайная рекомбинация известных паттернов."""
        logger.info(f"[SleepMode] Dream: {self.dream_count} рекомбинаций")

        discovered = 0
        for _ in range(self.dream_count):
            results = self.explorer.explore(n_attempts=3)

            for result in results:
                if result.coherence_score > 0.4 and result.is_novel:
                    # Строим мостик к ближайшему известному паттерну
                    for level, pats in self.grammar.patterns.items():
                        for ph, pattern in pats.items():
                            if pattern.length > 1:
                                bridge = self.logic_bridge.build_bridge(
                                    result.sequence[:pattern.length],
                                    pattern.symbol_indices[:pattern.length],
                                )
                                if bridge:
                                    discovered += 1
                                    break
                        if discovered > 0:
                            break

        if discovered > 0:
            logger.info(f"[SleepMode] Dream discoveries: {discovered}")

    def _prune(self):
        """Pruning: удалить слабые/неиспользуемые паттерны и связи."""
        removed = 0

        # Удаляем диграммы с низкой аффинностью
        for ph in list(self.grammar.patterns[0].keys()):
            pattern = self.grammar.patterns[0][ph]
            if pattern.coherence_score < 0.3 and pattern.usage_count < 2:
                del self.grammar.patterns[0][ph]
                for s in pattern.symbol_indices:
                    self.grammar.symbol_to_patterns[s].discard(ph)
                removed += 1

        self._total_pruned += removed

    def _generalize(self):
        """Generalization: объединить похожие паттерны в абстрактные."""
        merged = 0

        pats_1 = list(self.grammar.patterns[1].values())
        for i in range(len(pats_1)):
            for j in range(i + 1, len(pats_1)):
                a, b = pats_1[i], pats_1[j]

                # Оцениваем сходство через edit distance
                rules, cost = self.logic_bridge.compute_edit_distance(
                    a.symbol_indices[:10], b.symbol_indices[:10]
                )

                if cost < 0.3:  # Очень похожи
                    composed = self.grammar.compose(a, b)
                    if composed:
                        merged += 1

        self._total_merged += merged
        if merged > 0:
            logger.info(f"[SleepMode] Обобщено: {merged} паттернов")

    def summary(self) -> str:
        return (
            f"SleepMode: cycles={self._sleep_cycles}, "
            f"sleeping={self._sleeping}, "
            f"recent_assemblies={len(self.recent_assemblies)}, "
            f"pruned={self._total_pruned}, merged={self._total_merged}"
        )
