"""
IntrinsicReward — внутренняя самооценка как обучающий сигнал.

Модель НЕ ждёт внешней похвалы (loss от учителя).
Она сама оценивает качество своих сборок по:
1. Coherence (насколько связно) 
2. Novelty (насколько ново — баланс exploitation/exploration)
3. Compression (насколько компактно — бритва Оккама)
4. Utility (насколько часто переиспользуется)

Reward = w_c * coherence + w_n * novelty - w_l * length_penalty

Этот reward используется для:
- Усиления успешных паттернов (больше reward → сильнее STDP)
- Ослабления неуспешных
- Выбора exploration strategy
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from collections import deque, defaultdict
from loguru import logger


class IntrinsicReward:
    """Внутренняя награда за качество сборки."""

    def __init__(
        self,
        w_coherence: float = 0.4,
        w_novelty: float = 0.3, 
        w_utility: float = 0.2,
        w_compression: float = 0.1,
        length_penalty: float = 0.001,
    ):
        self.w_coherence = w_coherence
        self.w_novelty = w_novelty
        self.w_utility = w_utility
        self.w_compression = w_compression
        self.length_penalty = length_penalty

        self._coherence_baseline = deque(maxlen=1000)
        self._utility_counter: Dict[str, int] = {}

    def compute(
        self,
        coherence: float,
        novelty: float,
        length: int,
        template_id: Optional[str] = None,
    ) -> float:
        """
        Вычислить внутреннюю награду.

        coherence: 0-1, насколько связна сборка
        novelty: 0-1, насколько нова (0=полный повтор, 1=полностью новое)
        length: количество символов (длиннее → больше штраф)
        template_id: если сборка из шаблона — учитываем utility
        """
        utility = 0.0
        if template_id:
            utility = min(1.0, self._utility_counter.get(template_id, 0) / 10.0)
            self._utility_counter[template_id] = self._utility_counter.get(template_id, 0) + 1

        compression = 1.0 / max(length, 1)

        reward = (
            self.w_coherence * coherence +
            self.w_novelty * novelty +
            self.w_utility * utility +
            self.w_compression * compression -
            self.length_penalty * length
        )

        self._coherence_baseline.append(coherence)
        return float(reward)

    def get_exploration_bonus(self, novelty: float, coherence: float) -> float:
        """Бонус за исследование: высокий при novelty И coherence."""
        return novelty * coherence * 0.5

    def summary(self) -> str:
        avg_coh = np.mean(self._coherence_baseline) if self._coherence_baseline else 0
        return f"IntrinsicReward(avg_coherence={avg_coh:.3f}, w_c={self.w_coherence})"


class MetaPatterns:
    """
    Закономерности о закономерностях — Meta-уровень.

    Обнаруживает паттерны НЕ между символами, а между ПАТТЕРНАМИ:
    - "Все глагольные паттерны заканчиваются на похожие последовательности"
    - "Вопросительные конструкции начинаются с {к, ч, г}"
    - "Паттерны типа A→B→C часто следуют за паттернами типа X→Y"

    MetaPattern = кластер паттернов со схожей структурой.
    """

    def __init__(self, grammar, potential_field, topological_field):
        self.grammar = grammar
        self.potential_field = potential_field
        self.topological_field = topological_field
        self.meta_patterns = []  # List of (description, pattern_ids, confidence)

    def discover_structural_metapatterns(self) -> list:
        """
        Найти мета-паттерны: группы паттернов с одинаковой структурой.

        Структура паттерна = последовательность {длин} составляющих.
        Например: [2,3] = диграмма + триграмма → слово из 5 символов.
        """
        structure_groups = defaultdict(list)

        for level, pats in self.grammar.patterns.items():
            for ph, pattern in pats.items():
                structure = tuple(pattern.symbol_indices[:5])
                structure_groups[structure].append(ph)

        metapatterns = []
        for structure, pat_ids in structure_groups.items():
            if len(pat_ids) >= 3:  # Минимум 3 паттерна с одинаковой структурой
                # Это мета-паттерн: структурный шаблон
                sample_symbols = list(structure)
                metapatterns.append({
                    "type": "structural",
                    "structure": sample_symbols,
                    "pattern_count": len(pat_ids),
                    "pattern_ids": pat_ids,
                })

        return metapatterns

    def discover_distributional_metapatterns(self) -> list:
        """
        Найти мета-паттерны: паттерны со схожими распределениями продолжений.
        """
        metapatterns = []
        pat_continuations = {}

        for level in [0, 1]:
            for ph, pattern in self.grammar.patterns[level].items():
                if pattern.length > 0:
                    last = pattern.symbol_indices[-1]
                    cont = self.potential_field.get_continuation_distribution(last)
                    pat_continuations[ph] = cont

        pat_ids = list(pat_continuations.keys())
        for i in range(len(pat_ids)):
            for j in range(i + 1, len(pat_ids)):
                ca = pat_continuations[pat_ids[i]]
                cb = pat_continuations[pat_ids[j]]
                cos_sim = np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-8)
                if cos_sim > 0.9:  # Почти идентичные продолжения
                    metapatterns.append({
                        "type": "distributional",
                        "pattern_a": pat_ids[i],
                        "pattern_b": pat_ids[j],
                        "similarity": float(cos_sim),
                    })

        return metapatterns

    def summary(self) -> str:
        structural = self.discover_structural_metapatterns()
        distributional = self.discover_distributional_metapatterns()
        return (
            f"MetaPatterns: structural={len(structural)}, distributional={len(distributional)}"
        )


class HierarchicalCompressor:
    """
    Компиляция стабильных паттернов в токены верхнего уровня.

    Когда паттерн подтверждён многократно (usage > threshold),
    он "компилируется" в единый виртуальный токен.

    Это позволяет:
    - Сократить длину последовательностей (эффективность)
    - Работать с "понятиями" вместо "букв" на верхних уровнях
    - Строить иерархию: символ → слово → фраза → предложение
    """

    def __init__(self, grammar, confirmation_threshold: int = 10):
        self.grammar = grammar
        self.confirmation_threshold = confirmation_threshold
        self.compiled_tokens: Dict[str, int] = {}  # pattern_id → virtual_token_id
        self.token_to_pattern: Dict[int, str] = {}  # virtual_token_id → pattern_id
        self.next_token_id: int = 1000  # начинаем после символьного vocab

    def compile_stable_patterns(self) -> int:
        """Скомпилировать все стабильные паттерны в токены."""
        compiled_count = 0

        for level, pats in self.grammar.patterns.items():
            for ph, pattern in pats.items():
                if pattern.usage_count >= self.confirmation_threshold:
                    if ph not in self.compiled_tokens:
                        token_id = self.next_token_id
                        self.compiled_tokens[ph] = token_id
                        self.token_to_pattern[token_id] = ph
                        self.next_token_id += 1
                        compiled_count += 1

        return compiled_count

    def compress_sequence(self, symbol_indices: List[int]) -> List[int]:
        """Сжать последовательность, заменяя известные паттерны токенами."""
        if not self.compiled_tokens:
            return symbol_indices

        result = []
        i = 0
        while i < len(symbol_indices):
            replaced = False
            for ph, token_id in self.compiled_tokens.items():
                pattern = self._get_pattern(ph)
                if pattern is None:
                    continue
                pat_len = pattern.length if hasattr(pattern, 'length') else len(pattern.symbol_indices)
                if i + pat_len <= len(symbol_indices):
                    if list(symbol_indices[i:i+pat_len]) == list(pattern.symbol_indices[:pat_len]):
                        result.append(token_id)
                        i += pat_len
                        replaced = True
                        break
            if not replaced:
                result.append(symbol_indices[i])
                i += 1

        return result

    def decompress_sequence(self, tokens: List[int]) -> List[int]:
        """Разжать токены обратно в символы."""
        result = []
        for tok in tokens:
            if tok in self.token_to_pattern:
                pattern = self._get_pattern(self.token_to_pattern[tok])
                if pattern:
                    result.extend(pattern.symbol_indices)
            else:
                result.append(tok)
        return result

    def _get_pattern(self, ph: str):
        for level_pats in self.grammar.patterns.values():
            if ph in level_pats:
                return level_pats[ph]
        return None

    def summary(self) -> str:
        return f"HCompressor: compiled={len(self.compiled_tokens)}, next_id={self.next_token_id}"
