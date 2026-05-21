"""
SymbolicContradictionFilter — отсечение невозможных связей.

НЕ вычисляет все V×V варианты.
Вместо этого поддерживает МНОЖЕСТВО запретов:

Запрет = связь между символами/паттернами, которая
противоречит известной логике сборки.

Типы противоречий:
1. Структурное: A→B→A (цикл без прогрессии)
2. Порядковое: B после A логически несовместимо (январь→июль→март)
3. Семантическое: cos(potential_A, continuation_B) < 0 (вектора противоположны)
4. Контекстное: связь валидна в одном домене, невозможна в другом
5. Частотное: связь никогда не встречалась и противоречит частотным паттернам

Фильтр работает как "иммунная система" — запоминает что НЕЛЬЗЯ,
и при генерации отсекает эти пути ДО того как модель их попробует.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
from loguru import logger


class ContradictionType(Enum):
    STRUCTURAL = "structural"      # цикл, разрыв
    ORDINAL = "ordinal"            # нарушение порядка
    SEMANTIC = "semantic"          # cos < 0
    CONTEXTUAL = "contextual"      # не в том домене
    FREQUENCY = "frequency"        # никогда не встречалось
    LOGICAL = "logical"            # нарушение правил LogicBridge


@dataclass
class ForbiddenConnection:
    """Запрещённая связь: то, чего НЕ должно быть в сборке."""
    symbols_before: Tuple[int, ...]   # что было до
    symbols_after: Tuple[int, ...]    # что пытается следовать
    contradiction_type: ContradictionType
    confidence: float                  # насколько уверены что это противоречие
    first_detected: float = 0.0
    detection_count: int = 0
    counterexample: Optional[str] = None  # текст-контрпример

    @property
    def key(self) -> str:
        before = "|".join(map(str, self.symbols_before))
        after = "|".join(map(str, self.symbols_after))
        return f"{before}->{after}"


class SymbolicContradictionFilter:
    """
    Фильтр невозможного: что НЕЛЬЗЯ соединять.

    Поддерживает чёрный список связей, которые противоречат
    известной логике. При генерации сборки — отсекает их.
    """

    def __init__(
        self,
        potential_field,
        topological_field,
        logic_bridge=None,
        max_forbidden: int = 50000,
        semantic_contra_threshold: float = -0.3,
    ):
        self.potential_field = potential_field
        self.topological_field = topological_field
        self.logic_bridge = logic_bridge
        self.max_forbidden = max_forbidden
        self.semantic_contra_threshold = semantic_contra_threshold

        # Запрещённые связи: key → ForbiddenConnection
        self.forbidden: Dict[str, ForbiddenConnection] = {}

        # Быстрый lookup: символ → запрещённые продолжения
        self.forbidden_continuations: Dict[int, Set[int]] = defaultdict(set)

        # История для decay (противоречие может "ослабнуть" с новыми данными)
        self.confidence_decay_rate: float = 0.999

        # Статистика
        self.total_detected: int = 0
        self.total_activated: int = 0  # сколько раз фильтр сработал

    def forbid(
        self,
        symbols_before: List[int],
        symbols_after: List[int],
        contra_type: ContradictionType,
        confidence: float = 0.5,
    ):
        """Добавить запрет."""
        key_before = tuple(symbols_before[:10])
        key_after = tuple(symbols_after[:10])

        conn = ForbiddenConnection(
            symbols_before=key_before,
            symbols_after=key_after,
            contradiction_type=contra_type,
            confidence=confidence,
            first_detected=np.float64(time_module()),
            detection_count=1,
        )

        fkey = conn.key
        if fkey in self.forbidden:
            self.forbidden[fkey].detection_count += 1
            self.forbidden[fkey].confidence = min(
                1.0,
                self.forbidden[fkey].confidence + 0.1 * confidence,
            )
        else:
            self.forbidden[fkey] = conn
            self.total_detected += 1

        # Быстрый lookup
        if symbols_before:
            last = symbols_before[-1]
            if symbols_after:
                self.forbidden_continuations[last].add(symbols_after[0])

        # Ограничиваем размер
        if len(self.forbidden) > self.max_forbidden:
            sorted_items = sorted(
                self.forbidden.items(),
                key=lambda x: x[1].confidence * x[1].detection_count,
            )
            for old_key, _ in sorted_items[:len(self.forbidden)//4]:
                del self.forbidden[old_key]

    def is_forbidden(
        self,
        symbols_before: List[int],
        next_symbol: int,
    ) -> Tuple[bool, float, Optional[ContradictionType]]:
        """
        Проверить: запрещено ли добавлять next_symbol после before?

        Returns: (is_forbidden, confidence, contradiction_type)
        """
        # Быстрая проверка через lookup
        if symbols_before:
            last = symbols_before[-1]
            if next_symbol in self.forbidden_continuations.get(last, set()):
                return True, 0.8, ContradictionType.SEMANTIC

        # Полная проверка через запреты
        key_before = tuple(symbols_before[-5:])
        for length in range(1, min(len(symbols_before), 5) + 1):
            test_before = tuple(symbols_before[-length:])
            test_after = (next_symbol,)
            test_key = f"{'|'.join(map(str, test_before))}->{'|'.join(map(str, test_after))}"

            if test_key in self.forbidden:
                fc = self.forbidden[test_key]
                return True, fc.confidence, fc.contradiction_type

        return False, 0.0, None

    def detect_structural_contradictions(
        self,
        assembly: List[int],
        max_depth: int = 5,
    ) -> List[ForbiddenConnection]:
        """
        Обнаружить структурные противоречия в сборке:
        - Циклы A→B→A без семантической прогрессии
        - Разрывы: соседние символы с очень низкой аффинностью
        """
        detected = []
        n = len(assembly)
        if n < 3:
            return detected

        aff = self.potential_field.affinity.cpu().numpy()

        for i in range(n - 2):
            # Проверка на цикл: A→B→A
            if assembly[i] == assembly[i + 2] and assembly[i] != assembly[i + 1]:
                # Это цикл. Проверяем есть ли семантическая прогрессия
                a_ab = aff[assembly[i], assembly[i + 1]]
                a_ba = aff[assembly[i + 1], assembly[i + 2]]
                if a_ab > 0.5 and a_ba > 0.5:
                    # Оба направления сильные → это валидный цикл (например "не не")
                    pass
                else:
                    # Запрещаем
                    self.forbid(
                        [assembly[i], assembly[i + 1]],
                        [assembly[i + 2]],
                        ContradictionType.STRUCTURAL,
                        confidence=0.7,
                    )
                    detected.append(self.forbidden[
                        f"{assembly[i]}|{assembly[i+1]}->{assembly[i+2]}"
                    ])

            # Проверка разрыва
            a_curr_next = aff[assembly[i], assembly[i + 1]]
            if a_curr_next < 0.05:
                self.forbid(
                    [assembly[i]],
                    [assembly[i + 1]],
                    ContradictionType.STRUCTURAL,
                    confidence=0.6,
                )

        return detected

    def detect_ordinal_contradictions(
        self,
        known_sequence: List[int],
        candidate_sequence: List[int],
    ) -> List[ForbiddenConnection]:
        """
        Обнаружить порядковые противоречия:
        Известная последовательность [январь, февраль, март]
        Кандидат [январь, март, февраль] → противоречие порядка.
        """
        detected = []

        if len(known_sequence) < 3:
            return detected

        # Строим индекс: элемент → позиция в эталонной последовательности
        known_positions = {s: i for i, s in enumerate(known_sequence)}

        # Проверяем candidate
        for i in range(len(candidate_sequence) - 1):
            a = candidate_sequence[i]
            b = candidate_sequence[i + 1]

            if a in known_positions and b in known_positions:
                if known_positions[a] > known_positions[b]:
                    # В эталоне a перед b, а в кандидате b перед a → противоречие
                    self.forbid(
                        [a], [b],
                        ContradictionType.ORDINAL,
                        confidence=0.8,
                    )
                    detected.append(self.forbidden[f"{a}->{b}"])

        return detected

    def detect_semantic_contradictions(
        self,
        symbol_idx: int,
        min_samples: int = 5,
    ) -> List[ForbiddenConnection]:
        """
        Обнаружить семантические противоречия:
        Символы, чьи распределения продолжений противоположны.
        """
        detected = []
        aff = self.potential_field.affinity.cpu().numpy()
        n = min(self.potential_field.vocab_size, aff.shape[0])

        cont_i = aff[symbol_idx] / (aff[symbol_idx].sum() + 1e-8)

        for j in range(n):
            if j == symbol_idx:
                continue
            cont_j = aff[j] / (aff[j].sum() + 1e-8)
            cos_sim = np.dot(cont_i, cont_j) / (np.linalg.norm(cont_i) * np.linalg.norm(cont_j) + 1e-8)

            if cos_sim < self.semantic_contra_threshold:
                self.forbid(
                    [symbol_idx], [j],
                    ContradictionType.SEMANTIC,
                    confidence=float(abs(cos_sim)),
                )
                detected.append(self.forbidden[f"{symbol_idx}->{j}"])

        return detected

    def decay_forbidden(self):
        """Ослабить старые противоречия (новые данные могут их опровергнуть)."""
        to_remove = []
        for key, fc in self.forbidden.items():
            fc.confidence *= self.confidence_decay_rate
            if fc.confidence < 0.1 and fc.detection_count < 3:
                to_remove.append(key)

        for key in to_remove:
            del self.forbidden[key]

    def get_forbidden_mask(
        self,
        symbols_before: List[int],
    ) -> np.ndarray:
        """
        Получить маску запрещённых символов для генерации.
        Возвращает [vocab_size] bool: True = запрещено.
        """
        mask = np.zeros(self.potential_field.vocab_size, dtype=bool)

        if not symbols_before:
            return mask

        for sym in range(self.potential_field.vocab_size):
            forbidden, _, _ = self.is_forbidden(symbols_before, sym)
            mask[sym] = forbidden

        return mask

    def summary(self) -> str:
        return (
            f"ContradictionFilter: forbidden={len(self.forbidden)}, "
            f"detected={self.total_detected}, activated={self.total_activated}"
        )


def time_module():
    import time
    return time.time()
