"""
Hierarchical Layer — многослойное многообразие EVA.

Ключевой принцип: свойства верхнего уровня МАТЕМАТИЧЕСКИ ВЫВОДЯТСЯ
из нижнего, а не заучиваются отдельно.

Уровень 0: символы (160). Связи через co-occurrence. DONE.
Уровень 1: слова. Координаты = спектральная интерполяция символов.
           Связи = геодезические в многообразии. НЕ обучаются отдельно.
Уровень 2: фразы. Координаты = композиция слов.
           Связи = продолжение через касательное пространство.
Уровень 3: предложения. Замыкание семантического пространства.

Математика:
- Слово W = [c₁, ..., cₙ]. Координата W в многообразии:
  coord(W) = Σᵢ αᵢ · coord(cᵢ) / Σᵢ αᵢ
  где αᵢ = affinity(cᵢ, cᵢ₊₁) — вес по связности

- Семантическое расстояние между словами:
  d(W₁, W₂) = geodesic_distance(coord(W₁), coord(W₂))

- Потенциал продолжения слова:
  P(Wⱼ | Wᵢ) ∝ exp(-d(Wᵢ, Wⱼ)² / 2σ²)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


# ============================================================
# 1. WordToken — слово как точка в многообразии
# ============================================================

@dataclass
class WordToken:
    """Слово — стабильная сборка символов с позицией в многообразии."""
    token_id: int
    symbols: List[int]
    text: str
    # Координаты в многообразии (выводятся из символов)
    coordinates: Optional[np.ndarray] = None
    # Статистика
    occurrence_count: int = 0
    confidence: float = 0.5
    # Связи с другими словами (выводятся, не заучиваются)
    continuation_potentials: Optional[np.ndarray] = None
    # Для иерархии
    parent_phrase_id: Optional[int] = None


class WordDiscovery:
    """
    Обнаруживает слова как стабильные символьные сборки.

    Слово = последовательность символов, которая:
    1. Часто встречается как единое целое
    2. Имеет высокую внутреннюю связность (affinity между соседними символами)
    3. Имеет стабильные продолжения (word boundary)
    """

    def __init__(self, grammar, potential_field, char_vocab, min_confidence: float = 0.6):
        self.grammar = grammar
        self.pf = potential_field
        self.char_vocab = char_vocab
        self.min_confidence = min_confidence

        self.words: Dict[int, WordToken] = {}
        self.symbol_to_words: Dict[int, Set[int]] = defaultdict(set)
        self.next_word_id: int = 1000  # начинаем после символьного диапазона

    def discover_from_grammar(self) -> List[WordToken]:
        """
        Извлечь слова из грамматики:
        N-граммы уровня 1 с высокой связностью → слова.
        """
        new_words = []
        aff = self.pf.affinity.cpu().numpy()

        for level, pats in self.grammar.patterns.items():
            for ph, pattern in pats.items():
                if pattern.length < 2:
                    continue
                if pattern.coherence_score < self.min_confidence:
                    continue

                # Проверяем: это слово целиком или только часть?
                symbols = pattern.symbol_indices[:pattern.length]

                # Вычисляем внутреннюю связность
                if len(symbols) >= 2:
                    internal_aff = []
                    for k in range(len(symbols) - 1):
                        internal_aff.append(aff[symbols[k], symbols[k + 1]])
                    avg_internal = np.mean(internal_aff) if internal_aff else 0
                else:
                    avg_internal = 1.0

                if avg_internal < 0.55:
                    continue

                # Проверяем, не является ли подстрокой существующего слова
                is_substring = False
                for wid, word in self.words.items():
                    if len(word.symbols) > len(symbols):
                        if self._is_subsequence(symbols, word.symbols):
                            is_substring = True
                            break

                if is_substring and len(symbols) < 4:
                    continue

                word_id = self.next_word_id
                self.next_word_id += 1

                text = self.char_vocab.decode(symbols)

                word = WordToken(
                    token_id=word_id,
                    symbols=symbols,
                    text=text,
                    confidence=avg_internal,
                    occurrence_count=pattern.usage_count,
                )
                self.words[word_id] = word
                for s in symbols:
                    self.symbol_to_words[s].add(word_id)
                new_words.append(word)

        return new_words

    def _is_subsequence(self, short: List[int], long: List[int]) -> bool:
        """Проверяет, является ли short подпоследовательностью long."""
        si = 0
        for li in long:
            if si < len(short) and short[si] == li:
                si += 1
        return si == len(short)

    def get_word_by_symbols(self, symbols: List[int]) -> Optional[WordToken]:
        """Найти слово по точному совпадению символов."""
        for wid, word in self.words.items():
            if word.symbols == symbols:
                return word
        return None


# ============================================================
# 2. MultiLayerManifold — координаты на всех уровнях
# ============================================================

class MultiLayerManifold:
    """
    Многообразие с координатами на всех уровнях.

    Координаты уровня N+1 выводятся из уровня N математически,
    без отдельного обучения.
    """

    def __init__(self, topological_field, potential_field, word_discovery: WordDiscovery):
        self.topo = topological_field
        self.pf = potential_field
        self.words = word_discovery

        # Координаты слов (вычисляются лениво)
        self.word_coordinates: Dict[int, np.ndarray] = {}

    def compute_word_coordinates(self, word: WordToken) -> np.ndarray:
        """
        Координата слова в многообразии = взвешенная сумма координат символов.

        Вес символа = 1 + affinity(предыдущий, этот).
        Символы в сильных связях вносят больший вклад в позицию слова.
        """
        if word.token_id in self.word_coordinates:
            return self.word_coordinates[word.token_id]

        aff = self.pf.affinity.cpu().numpy()
        symbols = word.symbols

        if not symbols:
            return np.zeros(self.topo.coord_dim)

        coords = []
        weights = []

        for i, si in enumerate(symbols):
            if si < self.topo.coordinates.shape[0]:
                c = self.topo.coordinates[si].cpu().numpy()
                coords.append(c)

                # Вес: 1 + affinity с предыдущим
                w = 1.0
                if i > 0 and symbols[i - 1] < aff.shape[0]:
                    w += float(aff[symbols[i - 1], si])
                weights.append(w)

        if not coords:
            return np.zeros(self.topo.coord_dim)

        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)
        result = np.average(coords, axis=0, weights=weights)

        if isinstance(result, np.ndarray):
            self.word_coordinates[word.token_id] = result
        else:
            self.word_coordinates[word.token_id] = np.array(result)

        return self.word_coordinates[word.token_id]

    def word_distance(self, word_a: WordToken, word_b: WordToken) -> float:
        """Семантическое расстояние между словами в многообразии."""
        ca = self.compute_word_coordinates(word_a)
        cb = self.compute_word_coordinates(word_b)
        return float(np.linalg.norm(ca - cb))

    def nearest_words(
        self, target: WordToken, k: int = 10
    ) -> List[Tuple[WordToken, float]]:
        """Ближайшие слова к целевому в многообразии."""
        tc = self.compute_word_coordinates(target)
        distances = []
        for wid, word in self.words.words.items():
            if wid == target.token_id:
                continue
            wc = self.compute_word_coordinates(word)
            d = float(np.linalg.norm(tc - wc))
            distances.append((word, d))

        distances.sort(key=lambda x: x[1])
        return distances[:k]


# ============================================================
# 3. HierarchicalPredictor — предсказание на всех уровнях
# ============================================================

class HierarchicalPredictor:
    """
    Предсказание следующего токена с учётом всех уровней.

    Уровень 0 (символы): P(s_next | last_symbol) — из affinity
    Уровень 1 (слова): P(w_next | last_word) — из геодезических в многообразии
    Уровень 2 (фразы): P(p_next | recent_words) — из касательного пространства

    Финальное предсказание — взвешенная сумма уровней.
    """

    def __init__(
        self,
        potential_field,
        word_discovery: WordDiscovery,
        manifold: MultiLayerManifold,
        char_vocab,
    ):
        self.pf = potential_field
        self.words = word_discovery
        self.manifold = manifold
        self.char_vocab = char_vocab

        # Веса уровней
        self.w_symbol: float = 0.5
        self.w_word: float = 0.3
        self.w_phrase: float = 0.2

    def predict_next_symbols(
        self,
        context_symbols: List[int],
        top_k: int = 50,
    ) -> np.ndarray:
        """
        Предсказать следующий символ с учётом всех уровней.

        Returns: распределение вероятностей [vocab_size]
        """
        V = self.pf.vocab_size
        result = np.zeros(V)

        # Уровень 0: символьное предсказание (affinity)
        if context_symbols:
            last = context_symbols[-1]
            symbol_pred = self.pf.get_continuation_potential(last).cpu().numpy()
            result += self.w_symbol * symbol_pred

        # Уровень 1: словесное предсказание
        word_pred = self._word_level_prediction(context_symbols, V)
        if word_pred is not None:
            result += self.w_word * word_pred

        # Уровень 2: фразовое предсказание
        phrase_pred = self._phrase_level_prediction(context_symbols, V)
        if phrase_pred is not None:
            result += self.w_phrase * phrase_pred

        return result

    def _word_level_prediction(
        self, context_symbols: List[int], V: int
    ) -> Optional[np.ndarray]:
        """
        Предсказание на уровне слов: найти ближайшие слова
        в многообразии, взять их первые символы как кандидаты.
        """
        # Ищем: является ли хвост контекста началом известного слова?
        result = np.zeros(V)
        found = False

        for wid, word in self.words.words.items():
            ws = word.symbols
            # Проверяем overlap: контекст — префикс слова?
            ctx_tail = context_symbols[-min(len(context_symbols), len(ws)):]
            match_len = 0
            for k in range(min(len(ctx_tail), len(ws))):
                if ctx_tail[-(k + 1)] == ws[-(k + 1)]:
                    match_len += 1
                else:
                    break

            if match_len >= 2 and match_len < len(ws):
                next_sym = ws[match_len]
                if 0 <= next_sym < V:
                    result[next_sym] += word.confidence
                    found = True

        return result if found else None

    def _phrase_level_prediction(
        self, context_symbols: List[int], V: int
    ) -> Optional[np.ndarray]:
        """
        Предсказание на уровне фраз: используем геодезическую навигацию.

        Найти ближайшие слова к последним символам контекста,
        использовать их первые символы.
        """
        if len(context_symbols) < 3:
            return None

        result = np.zeros(V)
        found = False

        # Ищем слова, начинающиеся с последних 2-3 символов контекста
        tail_2 = tuple(context_symbols[-2:])
        tail_3 = tuple(context_symbols[-3:]) if len(context_symbols) >= 3 else None

        for wid, word in self.words.words.items():
            ws = word.symbols
            if len(ws) <= len(context_symbols):
                continue

            # Слово начинается с tail?
            starts_with = (
                (len(ws) >= 3 and tail_3 and tuple(ws[:3]) == tail_3) or
                (len(ws) >= 2 and tuple(ws[:2]) == tail_2)
            )

            if starts_with and len(ws) > len(context_symbols):
                next_sym = ws[len(context_symbols)]
                if 0 <= next_sym < V:
                    result[next_sym] += word.confidence * word.occurrence_count
                    found = True

        return result if found else None

    def predict_next_word(
        self,
        context_words: List[WordToken],
        top_k: int = 10,
    ) -> List[Tuple[WordToken, float]]:
        """
        Предсказать следующее СЛОВО (не символ).
        Использует геодезическую близость в многообразии.
        """
        if not context_words:
            return []

        last_word = context_words[-1]
        nearby = self.manifold.nearest_words(last_word, k=50)

        # Упорядочиваем по расстоянию
        nearby.sort(key=lambda x: x[1])
        return nearby[:top_k]

    def summary(self) -> str:
        return (
            f"HierarchicalPredictor(words={len(self.words.words)}, "
            f"w_sym={self.w_symbol}, w_word={self.w_word})"
        )


# ============================================================
# 4. LogicCompiler — математический вывод связей
# ============================================================

class LogicCompiler:
    """
    Выводит связи верхнего уровня из нижнего МАТЕМАТИЧЕСКИ.

    Не требует обучения. Использует:
    - Геодезические в многообразии для семантической близости
    - Касательное пространство для направлений трансформации
    - Convex hull для проверки замкнутости

    «Мостик логики» = минимальная геодезическая между точками.
    """

    def __init__(self, manifold: MultiLayerManifold, potential_field):
        self.manifold = manifold
        self.pf = potential_field

        # Кеш вычисленных связей
        self._word_links: Dict[Tuple[int, int], float] = {}
        self._phrase_links: Dict[Tuple[int, int], float] = {}

    def word_affinity(self, word_a: WordToken, word_b: WordToken) -> float:
        """
        Аффинность между словами (вычисляется, не заучивается).

        = 1 / (1 + geodesic_distance(word_a, word_b))
        """
        key = (word_a.token_id, word_b.token_id)
        if key in self._word_links:
            return self._word_links[key]

        d = self.manifold.word_distance(word_a, word_b)
        affinity = 1.0 / (1.0 + d)

        self._word_links[key] = affinity
        return affinity

    def continuation_score(
        self, word_a: WordToken, word_b: WordToken
    ) -> float:
        """
        Насколько логично word_b после word_a.

        Вычисляется из:
        1. Геодезического расстояния (близость в пространстве смыслов)
        2. Cross-affinity между последним символом A и первым символом B
        3. Связности через касательное пространство
        """
        # Геодезическая близость
        d = self.manifold.word_distance(word_a, word_b)
        geo_score = 1.0 / (1.0 + d)

        # Cross-affinity символов
        last_a = word_a.symbols[-1]
        first_b = word_b.symbols[0]
        cross_aff = float(self.pf.affinity[last_a, first_b])

        # Комбинируем
        return 0.6 * geo_score + 0.4 * cross_aff

    def top_continuations(
        self, word: WordToken, top_k: int = 10
    ) -> List[Tuple[WordToken, float]]:
        """Топ-K логичных продолжений для слова."""
        scores = []
        for wid, candidate in self.manifold.words.words.items():
            if wid == word.token_id:
                continue
            score = self.continuation_score(word, candidate)
            scores.append((candidate, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def phrase_continuation(
        self,
        phrase_words: List[WordToken],
        top_k: int = 10,
    ) -> List[Tuple[WordToken, float]]:
        """
        Продолжение фразы: усреднённая оценка по всем словам фразы.
        """
        if not phrase_words:
            return []

        scores = defaultdict(float)
        decay = 1.0

        for word in reversed(phrase_words[-3:]):  # Последние 3 слова
            continuations = self.top_continuations(word, top_k=30)
            for candidate, score in continuations:
                scores[candidate.token_id] += score * decay
            decay *= 0.5

        result = []
        for wid, score in scores.items():
            if wid in self.manifold.words.words:
                result.append((self.manifold.words.words[wid], score))

        result.sort(key=lambda x: x[1], reverse=True)
        return result[:top_k]

    def summary(self) -> str:
        return f"LogicCompiler(links={len(self._word_links)})"
