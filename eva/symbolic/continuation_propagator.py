"""
ContinuationPropagator — иерархическое распространение потенциалов.

Символ → Группа → Слово → Фраза → Предложение → Контекст

На каждом уровне:
1. Вычисляется общий потенциал конструкции
2. Вычисляется потенциал продолжения (что может идти дальше)
3. Проверяется семантическая связность через SemanticClosureChecker

Ключевая идея: "вес" слова/предложения — это не число, а
распределение потенциалов продолжения в семантическом пространстве.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from loguru import logger


class ContinuationPropagator:
    """
    Распространяет потенциалы иерархически.

    Уровень 0: символы → потенциалы из PotentialField
    Уровень 1: группы символов → композиция потенциалов
    Уровень 2: слова → агрегация групповых потенциалов
    Уровень 3: предложения → вектор контекстного продолжения
    """

    def __init__(self, potential_field, semantic_checker, embed_dim: int = 256):
        self.potential_field = potential_field
        self.semantic_checker = semantic_checker
        self.embed_dim = embed_dim

    def compute_symbol_potential(self, char_idx: int) -> np.ndarray:
        """
        Потенциал символа = его embedding + вектор продолжения.
        Символ "знает" что может идти после него.
        """
        with torch.no_grad():
            emb = self.potential_field.char_potential[char_idx].cpu().numpy()
            continuation = self.potential_field.get_continuation_potential(char_idx).cpu().numpy()
            # Комбинируем: сам символ + его "ожидания"
            combined = np.concatenate([
                emb,
                continuation[:self.embed_dim // 2] if len(continuation) > self.embed_dim // 2
                else np.pad(continuation, (0, self.embed_dim // 2 - len(continuation)))
            ])
            # Нормализуем до целевой размерности
            if len(combined) > self.embed_dim:
                combined = combined[:self.embed_dim]
            elif len(combined) < self.embed_dim:
                combined = np.pad(combined, (0, self.embed_dim - len(combined)))
        return combined

    def compute_group_potential(
        self,
        char_indices: List[int],
        attention_weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Потенциал группы символов = взвешенная сумма потенциалов символов.

        Веса — из attention: символы с высоким вниманием вносят больший вклад.
        """
        if not char_indices:
            return np.zeros(self.embed_dim)

        potentials = [self.compute_symbol_potential(ci) for ci in char_indices]

        if attention_weights and len(attention_weights) == len(potentials):
            total_w = sum(attention_weights)
            if total_w > 1e-8:
                weights = [w / total_w for w in attention_weights]
            else:
                weights = [1.0 / len(potentials)] * len(potentials)
        else:
            weights = [1.0 / len(potentials)] * len(potentials)

        return np.average(potentials, axis=0, weights=weights).astype(np.float32)

    def compute_continuation_potential(
        self,
        current_potential: np.ndarray,
        context_potentials: List[np.ndarray],
    ) -> np.ndarray:
        """
        Потенциал продолжения: какой следующий элемент логичен.

        Вычисляется как "разность" между текущим потенциалом и потенциалами
        контекста — модель "ожидает" что следующий элемент заполнит семантический пробел.
        """
        if not context_potentials:
            return np.zeros_like(current_potential)

        context_avg = np.mean(context_potentials, axis=0)

        # Ожидаемое продолжение = направление от текущего к контексту
        continuation = context_avg - current_potential

        # Нормализуем
        norm = np.linalg.norm(continuation)
        if norm > 1e-8:
            continuation = continuation / norm * np.linalg.norm(current_potential)

        return continuation

    def propagate_sequence(
        self,
        char_indices: List[int],
        attention_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Полное иерархическое распространение для последовательности символов.

        Returns:
            potentials: потенциалы на каждом уровне для каждого символа
            sequence_potential: общий потенциал всей последовательности
        """
        T = len(char_indices)

        # Уровень 0: символьные потенциалы
        symbol_potentials = [self.compute_symbol_potential(ci) for ci in char_indices]
        symbol_potentials = np.array(symbol_potentials)

        # Уровень 1-2: групповые потенциалы (скользящее окно)
        window = max(2, T // 8)
        group_potentials = []
        for i in range(0, T, max(1, window // 2)):
            end = min(i + window, T)
            group_chars = char_indices[i:end]
            if attention_matrix is not None:
                attn_w = attention_matrix[i:end, i:end].sum(axis=1).tolist()
            else:
                attn_w = None
            gp = self.compute_group_potential(group_chars, attn_w)
            group_potentials.append(gp)

        # Уровень 3: потенциал всей последовательности
        sequence_potential = np.mean(group_potentials, axis=0) if group_potentials else np.zeros(self.embed_dim)

        return symbol_potentials, sequence_potential

    def check_assembly_coherence(
        self,
        char_indices: List[int],
        attention_matrix: np.ndarray,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Проверяет семантическую связность сборки на всех уровнях.
        """
        symbol_potentials, sequence_potential = self.propagate_sequence(
            char_indices, attention_matrix
        )

        passed, score, components = self.semantic_checker.full_check(
            symbol_potentials,
            attention_matrix,
            sequence_potential,
        )

        if passed:
            self.semantic_checker.add_valid_potential(sequence_potential)

        return passed, score, components

    def compute_continuation_distribution(
        self,
        current_potential: np.ndarray,
        candidate_potentials: np.ndarray,  # [K, d]
    ) -> np.ndarray:
        """
        Распределение вероятностей продолжения:
        какие кандидаты логично следуют за текущим потенциалом.

        Использует семантическую близость + потенциал аффинности.
        """
        K = candidate_potentials.shape[0]

        # Косинусное сходство
        similarities = np.zeros(K)
        for i in range(K):
            sim = SemanticClosureChecker._cosine_sim(current_potential, candidate_potentials[i])
            similarities[i] = sim

        # Softmax с температурой
        temperature = 0.5
        similarities = (similarities - similarities.max()) / temperature
        probs = np.exp(similarities)
        probs = probs / probs.sum()

        return probs

    @staticmethod
    def _cosine_sim_static(a, b):
        from .semantic_closure import SemanticClosureChecker
        return SemanticClosureChecker._cosine_sim(a, b)
