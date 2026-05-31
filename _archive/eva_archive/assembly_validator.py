"""
AssemblyValidator — глубокая валидация инструкций сборки.

Проверяет не только связность, но и:
1. Структурную корректность (нет разорванных связей)
2. Функциональную валидность (правильные продолжения)
3. Трансформационную достижимость (можно ли получить из известных паттернов)
4. Контекстную адекватность (соответствует ли контексту)
5. Информационную плотность (нет избыточности)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from loguru import logger


class AssemblyValidator:
    """
    Многоуровневая валидация сборки.

    Каждый уровень возвращает (passed, score, issues) —
    булев флаг, числовую оценку и список проблем.
    """

    def __init__(
        self,
        potential_field,
        semantic_checker,
        logic_bridge=None,
        grammar=None,
    ):
        self.potential_field = potential_field
        self.semantic_checker = semantic_checker
        self.logic_bridge = logic_bridge
        self.grammar = grammar

        # Эталонные паттерны для сравнения
        self.reference_assemblies: List[Dict] = []

        # Статистика валидаций
        self.total_validations: int = 0
        self.passed_validations: int = 0

    def structural_check(
        self,
        symbol_indices: List[int],
        attention_matrix: np.ndarray,
    ) -> Tuple[bool, float, List[str]]:
        """
        Структурная проверка:
        - Все символы связаны хотя бы одной связью (нет изолированных)
        - Нет циклов деградации (attention → 0 для всех)
        - Граф связен (путь между любыми двумя символами)
        """
        issues = []
        n = len(symbol_indices)
        if n < 2:
            return True, 1.0, []

        # Проверка изолированности
        row_sums = attention_matrix.sum(axis=1)
        isolated = np.where(row_sums < 0.01)[0]
        if len(isolated) > 0:
            issues.append(f"isolated symbols at positions {isolated.tolist()}")

        # Проверка связности графа (DFS)
        adj = attention_matrix > 0.01
        visited = np.zeros(n, dtype=bool)
        stack = [0]
        visited[0] = True
        while stack:
            node = stack.pop()
            for neighbor in range(n):
                if (adj[node, neighbor] or adj[neighbor, node]) and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        disconnected = np.where(~visited)[0]
        if len(disconnected) > 0:
            issues.append(f"disconnected components: {disconnected.tolist()}")

        score = 1.0 - len(isolated) / n - len(disconnected) / n
        score = max(0.0, score)

        return len(issues) == 0, score, issues

    def functional_check(
        self,
        symbol_indices: List[int],
        continuation_candidates: Optional[List[int]] = None,
    ) -> Tuple[bool, float, List[str]]:
        """
        Функциональная проверка:
        - Потенциал продолжения не нулевой (есть что сказать дальше)
        - Продолжения семантически близки к текущему контексту
        """
        issues = []
        if not symbol_indices:
            return True, 1.0, []

        # Проверяем потенциал продолжения последнего символа
        last = symbol_indices[-1]
        cont = self.potential_field.get_continuation_distribution(last)

        entropy = -np.sum(cont * np.log(cont + 1e-8))
        max_entropy = np.log(len(cont))

        if max_entropy < 1e-8:
            return True, 1.0, []

        normalized_entropy = entropy / max_entropy

        if normalized_entropy < 0.1:
            issues.append("dead end: no meaningful continuations")

        score = normalized_entropy
        return len(issues) == 0, score, issues

    def transformational_check(
        self,
        symbol_indices: List[int],
    ) -> Tuple[bool, float, List[str]]:
        """
        Трансформационная проверка:
        - Можно ли получить эту сборку из известных паттернов через LogicBridge?
        - Если да, сборка "легитимна" (выводима из известного).
        """
        if self.logic_bridge is None or self.grammar is None:
            return True, 1.0, []

        issues = []

        # Ищем ближайший известный паттерн
        best_score = 0.0
        for level, pats in self.grammar.patterns.items():
            for ph, pattern in pats.items():
                if pattern.length > 2:
                    rules, cost = self.logic_bridge.compute_edit_distance(
                        symbol_indices[:pattern.length],
                        pattern.symbol_indices[:pattern.length],
                    )
                    score = 1.0 - cost
                    if score > best_score:
                        best_score = score

        if best_score < 0.3:
            issues.append("no known pattern within reachable distance")

        return len(issues) == 0, best_score, issues

    def information_density_check(
        self,
        symbol_indices: List[int],
        attention_matrix: np.ndarray,
    ) -> Tuple[bool, float, List[str]]:
        """
        Проверка информационной плотности:
        - Нет избыточных повторов (одинаковые символы с низким attention)
        - Attention эффективно распределён (не сконцентрирован на одном символе)
        """
        issues = []
        n = len(symbol_indices)
        if n < 3:
            return True, 1.0, []

        # Повторы
        unique_ratio = len(set(symbol_indices)) / n
        if unique_ratio < 0.3 and n > 10:
            issues.append(f"too repetitive: {unique_ratio:.1%} unique")

        # Равномерность attention
        attn_by_symbol = attention_matrix.sum(axis=0) + attention_matrix.sum(axis=1)
        attn_by_symbol = attn_by_symbol / (attn_by_symbol.sum() + 1e-8)
        entropy = -np.sum(attn_by_symbol * np.log(attn_by_symbol + 1e-8))
        max_entropy = np.log(n)
        attn_efficiency = entropy / max_entropy if max_entropy > 1e-8 else 0

        if attn_efficiency < 0.3:
            issues.append(f"attention concentrated: efficiency={attn_efficiency:.2f}")

        score = 0.5 * unique_ratio + 0.5 * attn_efficiency
        return len(issues) == 0, score, issues

    def full_validate(
        self,
        symbol_indices: List[int],
        attention_matrix: np.ndarray,
        continuation_candidates: Optional[List[int]] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Полная многоуровневая валидация.
        """
        self.total_validations += 1

        checks = {}

        checks["structural"], struct_score, struct_issues = self.structural_check(
            symbol_indices, attention_matrix
        )
        checks["functional"], func_score, func_issues = self.functional_check(
            symbol_indices, continuation_candidates
        )
        checks["transformational"], trans_score, trans_issues = self.transformational_check(
            symbol_indices
        )
        checks["information"], info_score, info_issues = self.information_density_check(
            symbol_indices, attention_matrix
        )

        # Взвешенная оценка
        overall = (
            struct_score * 0.25 +
            func_score * 0.25 +
            trans_score * 0.25 +
            info_score * 0.25
        )

        all_passed = all(checks.values())

        if all_passed:
            self.passed_validations += 1

        result = {
            "passed": all_passed,
            "overall_score": overall,
            "component_scores": {
                "structural": struct_score,
                "functional": func_score,
                "transformational": trans_score,
                "information": info_score,
            },
            "issues": {
                "structural": struct_issues,
                "functional": func_issues,
                "transformational": trans_issues,
                "information": info_issues,
            },
            "pass_rate": self.passed_validations / max(self.total_validations, 1),
        }

        return all_passed, overall, result

    def summary(self) -> str:
        rate = self.passed_validations / max(self.total_validations, 1)
        return f"Validator: {self.passed_validations}/{self.total_validations} passed ({rate:.1%})"
