"""
AssemblyTemplateBank — банк эталонных сборок для быстрого извлечения.

Когда модель успешно собрала связную конструкцию — сохраняет её
как "шаблон". При последующем похожем контексте — извлекает шаблон
и использует как основу вместо построения с нуля.

Ключевая идея: не изобретать заново то, что уже работает.
Аналог: procedural memory (как езда на велосипеде — не думаешь, а делаешь).

Шаблоны организованы в TRIE-подобную структуру для быстрого поиска:
  контекст → префикс-дерево → топ-N продолжений с весами.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


@dataclass
class AssemblyTemplate:
    """Шаблон успешной сборки."""
    template_id: str
    symbol_sequence: List[int]       # полная последовательность
    context_prefix: Tuple[int, ...]  # ключ для поиска (первые N символов)
    continuation: List[int]          # продолжение после префикса
    usage_count: int = 0
    success_rate: float = 1.0
    coherence_baseline: float = 0.7
    last_used: float = 0.0
    created_at: float = 0.0

    @property
    def prefix_len(self) -> int:
        return len(self.context_prefix)

    @property
    def total_len(self) -> int:
        return len(self.symbol_sequence)


class AssemblyTemplateBank:
    """
    Банк шаблонов: префикс-дерево для быстрого извлечения.

    TRIE: context_prefix → список шаблонов, отсортированных по success_rate.
    """

    def __init__(self, max_templates: int = 50000, min_prefix_len: int = 2, max_prefix_len: int = 10):
        self.max_templates = max_templates
        self.min_prefix_len = min_prefix_len
        self.max_prefix_len = max_prefix_len

        # TRIE: (prefix_tuple) → list of templates
        self._trie: Dict[Tuple[int, ...], List[AssemblyTemplate]] = defaultdict(list)

        # Хеш-таблица: template_id → template
        self._templates: Dict[str, AssemblyTemplate] = {}
        self._next_id: int = 0

        # Статистика
        self.total_saved = 0
        self.total_retrieved = 0
        self.cache_hits = 0

    def save_template(
        self,
        symbol_sequence: List[int],
        coherence: float = 0.7,
    ) -> Optional[str]:
        """
        Сохранить успешную сборку как шаблон.

        Сохраняем если:
        1. Последовательность достаточно длинная
        2. Связность выше порога
        3. Нет дубликата
        """
        if len(symbol_sequence) < self.min_prefix_len + 2:
            return None
        if coherence < 0.4:
            return None

        # Строим префиксы разной длины для быстрого поиска
        seq_tuple = tuple(symbol_sequence)

        for prefix_len in range(self.min_prefix_len, min(len(symbol_sequence) - 1, self.max_prefix_len + 1)):
            prefix = tuple(symbol_sequence[:prefix_len])
            continuation = symbol_sequence[prefix_len:]

            # Проверяем на дубликат
            for template in self._trie.get(prefix, []):
                if template.symbol_sequence == symbol_sequence:
                    template.usage_count += 1
                    template.last_used = __import__('time').time()
                    template.success_rate = min(1.0, template.success_rate + 0.05)
                    return template.template_id

            tid = f"tpl_{self._next_id:06d}"
            self._next_id += 1

            template = AssemblyTemplate(
                template_id=tid,
                symbol_sequence=symbol_sequence,
                context_prefix=prefix,
                continuation=continuation,
                coherence_baseline=coherence,
                created_at=__import__('time').time(),
                last_used=__import__('time').time(),
            )

            self._trie[prefix].append(template)
            self._templates[tid] = template
            self.total_saved += 1

        # Ограничение размера
        if len(self._templates) > self.max_templates:
            self._prune()

        return tid

    def find_templates(
        self,
        context: List[int],
        max_results: int = 10,
    ) -> List[Tuple[AssemblyTemplate, float]]:
        """
        Найти шаблоны, соответствующие контексту.

        Ищет по префиксу контекста. Чем длиннее совпадение — тем выше score.
        """
        if len(context) < self.min_prefix_len:
            return []

        candidates = []

        for prefix_len in range(min(len(context), self.max_prefix_len), self.min_prefix_len - 1, -1):
            prefix = tuple(context[-prefix_len:])
            if prefix in self._trie:
                for template in self._trie[prefix]:
                    score = template.success_rate * (template.usage_count + 1)
                    score *= (prefix_len / self.max_prefix_len)  # Длиннее префикс → выше score
                    candidates.append((template, score))

        if candidates:
            self.cache_hits += 1

        candidates.sort(key=lambda x: x[1], reverse=True)
        self.total_retrieved += 1
        return candidates[:max_results]

    def get_best_continuation(
        self,
        context: List[int],
        max_symbols: int = 20,
    ) -> Optional[List[int]]:
        """
        Получить лучшее продолжение из шаблонов.
        """
        templates = self.find_templates(context, max_results=3)
        if not templates:
            return None

        best_template, score = templates[0]
        if score < 0.5:
            return None

        continuation = best_template.continuation[:max_symbols]
        best_template.usage_count += 1
        best_template.last_used = __import__('time').time()

        return continuation

    def _prune(self):
        """Удалить слабые шаблоны."""
        sorted_templates = sorted(
            self._templates.values(),
            key=lambda t: t.success_rate * (t.usage_count + 1),
        )
        keep_count = self.max_templates // 2
        to_remove = sorted_templates[:-keep_count]

        for t in to_remove:
            del self._templates[t.template_id]
            for prefix in list(self._trie.keys()):
                self._trie[prefix] = [x for x in self._trie[prefix] if x.template_id not in self._templates]
                if not self._trie[prefix]:
                    del self._trie[prefix]

    def summary(self) -> str:
        hit_rate = self.cache_hits / max(self.total_retrieved, 1)
        return (
            f"TemplateBank: templates={len(self._templates)}, "
            f"cache_hit_rate={hit_rate:.1%}"
        )
