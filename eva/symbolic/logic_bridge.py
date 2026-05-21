"""
LogicBridge — мостик логики между паттернами сборки.

Ключевая идея: если два паттерна семантически близки, существует
"мост" — минимальная трансформация, переводящая один в другой.

Мостик логики = набор правил трансформации:
- insert(symbol, position): вставить символ
- delete(position): удалить символ
- substitute(old, new, position): заменить символ
- reorder(positions): переставить символы
- merge(pat_a, pat_b): объединить два паттерна в один
- split(pat, at): разделить паттерн

Правила с высокой частотой становятся "логическими законами" —
устойчивыми способами преобразования смысла.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
from loguru import logger


class TransformOp(Enum):
    INSERT = "insert"
    DELETE = "delete"
    SUBSTITUTE = "substitute"
    REORDER = "reorder"
    MERGE = "merge"
    SPLIT = "split"
    EXPAND = "expand"       # вставить группу символов
    CONTRACT = "contract"   # сжать группу в один символ


@dataclass
class TransformRule:
    """Правило трансформации между паттернами."""
    op: TransformOp
    symbols_before: List[int]   # символы до трансформации
    symbols_after: List[int]    # символы после трансформации
    position: int = 0           # позиция операции
    frequency: int = 0           # сколько раз правило применялось
    confidence: float = 0.0      # уверенность в правиле
    semantic_cost: float = 0.0   # семантическая "цена" трансформации

    @property
    def rule_id(self) -> str:
        return f"{self.op.value}|{'|'.join(map(str, self.symbols_before))}|{'|'.join(map(str, self.symbols_after))}"


class LogicBridge:
    """
    Обнаруживает и хранит логические мостики между паттернами.

    Мостик = минимальная трансформация с семантической ценой.
    """

    def __init__(
        self,
        potential_field,
        vocab_size: int,
        min_rule_frequency: int = 3,
    ):
        self.potential_field = potential_field
        self.vocab_size = vocab_size
        self.min_rule_frequency = min_rule_frequency

        # Правила: rule_id → TransformRule
        self.rules: Dict[str, TransformRule] = {}

        # Инвертированный индекс: op → список правил
        self.rules_by_op: Dict[TransformOp, List[str]] = {
            op: [] for op in TransformOp
        }

        # Граф трансформаций: symbol → symbol (через какие правила можно перейти)
        self.transform_graph: Dict[int, Set[Tuple[int, str]]] = defaultdict(set)

        # Статистика
        self.total_bridges_built: int = 0
        self.total_rules_applied: int = 0

    def compute_edit_distance(
        self,
        seq_a: List[int],
        seq_b: List[int],
    ) -> Tuple[List[TransformRule], float]:
        """
        Вычислить минимальную трансформацию между двумя последовательностями.

        Использует взвешенный Левенштейн с семантической ценой:
        - insert/delete: цена = 1.0 - affinity(prev_symbol, inserted_symbol)
        - substitute: цена = 1.0 - affinity(old_symbol, new_symbol)
        - reorder: цена = 0.5 (перестановка — дешевле замены)

        Returns: (список правил, суммарная семантическая цена)
        """
        m, n = len(seq_a), len(seq_b)
        if m == 0:
            return [TransformRule(TransformOp.INSERT, [], seq_b, 0)], float(n)
        if n == 0:
            return [TransformRule(TransformOp.DELETE, seq_a, [], 0)], float(m)

        # DP таблица
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        ops = [[[] for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i * 0.5
            if i > 0:
                ops[i][0] = ops[i - 1][0] + [
                    TransformRule(TransformOp.DELETE, [seq_a[i - 1]], [], i - 1)
                ]

        for j in range(n + 1):
            dp[0][j] = j * 0.5
            if j > 0:
                ops[0][j] = ops[0][j - 1] + [
                    TransformRule(TransformOp.INSERT, [], [seq_b[j - 1]], 0)
                ]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    cost = 0.0
                else:
                    aff = float(self.potential_field.affinity[seq_a[i - 1], seq_b[j - 1]])
                    cost = 1.0 - aff

                sub_cost = dp[i - 1][j - 1] + cost
                del_cost = dp[i - 1][j] + 0.5
                ins_cost = dp[i][j - 1] + 0.5

                if sub_cost <= del_cost and sub_cost <= ins_cost:
                    dp[i][j] = sub_cost
                    ops[i][j] = ops[i - 1][j - 1]
                    if cost > 0:
                        ops[i][j] = ops[i][j] + [
                            TransformRule(TransformOp.SUBSTITUTE, [seq_a[i - 1]], [seq_b[j - 1]], i - 1)
                        ]
                elif del_cost <= ins_cost:
                    dp[i][j] = del_cost
                    ops[i][j] = ops[i - 1][j] + [
                        TransformRule(TransformOp.DELETE, [seq_a[i - 1]], [], i - 1)
                    ]
                else:
                    dp[i][j] = ins_cost
                    ops[i][j] = ops[i][j - 1] + [
                        TransformRule(TransformOp.INSERT, [], [seq_b[j - 1]], j - 1)
                    ]

        return ops[m][n], dp[m][n] / max(m, n, 1)

    def build_bridge(
        self,
        pattern_a_symbols: List[int],
        pattern_b_symbols: List[int],
    ) -> Optional[List[TransformRule]]:
        """
        Построить мостик логики между двумя паттернами.

        Если трансформация "дешёвая" (низкая семантическая цена),
        правило становится устойчивым логическим законом.
        """
        rules, cost = self.compute_edit_distance(pattern_a_symbols, pattern_b_symbols)

        if cost > 0.7:  # Слишком дорогая трансформация — не логично
            return None

        for rule in rules:
            rid = rule.rule_id
            if rid in self.rules:
                self.rules[rid].frequency += 1
                self.rules[rid].confidence = min(
                    1.0,
                    self.rules[rid].frequency / self.min_rule_frequency,
                )
                self.rules[rid].semantic_cost = (
                    0.7 * self.rules[rid].semantic_cost + 0.3 * cost
                )
            else:
                rule.frequency = 1
                rule.semantic_cost = cost
                self.rules[rid] = rule
                self.rules_by_op[rule.op].append(rid)

            # Обновить граф трансформаций
            for sb in rule.symbols_before:
                for sa in rule.symbols_after:
                    self.transform_graph[sb].add((sa, rid))

        self.total_bridges_built += 1
        return rules if cost < 0.5 else None

    def apply_rule(
        self,
        symbols: List[int],
        rule: TransformRule,
    ) -> Optional[List[int]]:
        """Применить правило трансформации к последовательности."""
        result = symbols.copy()

        try:
            if rule.op == TransformOp.INSERT:
                result.insert(rule.position, rule.symbols_after[0])
            elif rule.op == TransformOp.DELETE:
                if rule.position < len(result):
                    result.pop(rule.position)
            elif rule.op == TransformOp.SUBSTITUTE:
                if rule.position < len(result):
                    result[rule.position] = rule.symbols_after[0]
            elif rule.op == TransformOp.EXPAND:
                if rule.position < len(result):
                    result = result[:rule.position] + rule.symbols_after + result[rule.position + 1:]
            elif rule.op == TransformOp.CONTRACT:
                if rule.position + 1 < len(result):
                    result = result[:rule.position] + [rule.symbols_after[0]] + result[rule.position + 2:]
            elif rule.op == TransformOp.MERGE:
                result.extend(rule.symbols_after)
            elif rule.op == TransformOp.SPLIT:
                mid = len(result) // 2
                result = result[:mid]

            self.total_rules_applied += 1
            return result
        except Exception as e:
            logger.debug(f"[LogicBridge] apply_rule error: {e}")
            return None

    def find_path(
        self,
        from_symbols: List[int],
        to_symbols: List[int],
        max_depth: int = 5,
    ) -> Optional[List[TransformRule]]:
        """
        Найти путь трансформаций от from к to через известные правила.

        BFS по графу правил.
        """
        if from_symbols == to_symbols:
            return []

        from_key = tuple(from_symbols[:20])
        to_key = tuple(to_symbols[:20])

        visited = {from_key}
        queue = [(from_symbols, [], 0)]

        while queue:
            current, path, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            # Пробуем все правила для текущего состояния
            for rid, rule in list(self.rules.items())[:100]:  # Ограничиваем для скорости
                if rule.confidence < 0.5:
                    continue
                result = self.apply_rule(current, rule)
                if result is None:
                    continue

                result_key = tuple(result[:20])
                if result_key in visited:
                    continue

                new_path = path + [rule]
                if result == to_symbols[:len(result)]:
                    return new_path

                visited.add(result_key)
                queue.append((result, new_path, depth + 1))

        return None

    def discover_expansion_rules(self, grammar) -> List[TransformRule]:
        """
        Discovery EXPAND/CONTRACT правил:
        найти диграммы которые часто заменяют одиночные символы.
        """
        new_rules = []

        for ph, pattern in grammar.patterns[0].items():
            if pattern.length == 2:
                i, j = pattern.symbol_indices[:2]
                aff = float(self.potential_field.affinity[i, j])

                if aff > 0.7:
                    # CONTRACT: диграмма → символ (сжатие)
                    contract_rule = TransformRule(
                        TransformOp.CONTRACT, [i, j], [i], 0,
                        frequency=1, semantic_cost=1.0 - aff,
                    )
                    rid = contract_rule.rule_id
                    if rid not in self.rules:
                        self.rules[rid] = contract_rule
                        new_rules.append(contract_rule)

                    # EXPAND: символ → диграмма (расширение)
                    expand_rule = TransformRule(
                        TransformOp.EXPAND, [i], [i, j], 0,
                        frequency=1, semantic_cost=1.0 - aff,
                    )
                    rid = expand_rule.rule_id
                    if rid not in self.rules:
                        self.rules[rid] = expand_rule
                        new_rules.append(expand_rule)

        return new_rules

    def summary(self) -> str:
        total_rules = len(self.rules)
        confident_rules = sum(1 for r in self.rules.values() if r.confidence > 0.5)
        return (
            f"LogicBridge: rules={total_rules} (confident={confident_rules}), "
            f"bridges={self.total_bridges_built}, applied={self.total_rules_applied}"
        )
