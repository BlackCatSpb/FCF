"""
EVA Symbolic — Advanced Methods

Четыре доработки для снятия фундаментальных ограничений:
1. NGramContext — предсказание по последним N символам (не только последнему)
2. DynamicVocab — расширение алфавита на лету
3. PatternToConcept — связь символьных паттернов с "понятиями"
4. MultiLevelGrammar — иерархия: диграммы → N-граммы → слова → фразы
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
from loguru import logger


# ============================================================
# 1. NGramContext — предсказание по контексту из N последних символов
# ============================================================

class NGramContext:
    """
    Контекст из N последних символов для продолжения.

    Вместо P(next | last_symbol) использует:
    P(next | last_N_symbols) = weighted_average(
        affinity[last_1][next],
        affinity[last_2][next] * decay,
        affinity[last_3][next] * decay^2,
        ...
    )

    Это решает проблему «малого контекста».
    """

    def __init__(self, potential_field, max_context: int = 4, decay: float = 0.6):
        self.pf = potential_field
        self.max_context = max_context
        self.decay = decay

    def get_continuation(self, context_symbols: List[int]) -> np.ndarray:
        if not context_symbols:
            return np.full(self.pf.vocab_size, 0.5, dtype=np.float32)

        aff = self.pf.affinity.cpu().numpy()
        V = min(self.pf.vocab_size, aff.shape[0])
        result = np.zeros(V, dtype=np.float32)

        weight_sum = 0.0
        for i, sym in enumerate(reversed(context_symbols[-self.max_context:])):
            if 0 <= sym < V:
                w = self.decay ** i
                result += w * aff[sym]
                weight_sum += w

        if weight_sum > 0:
            result /= weight_sum

        return result

    def top_continuations(self, context_symbols: List[int], k: int = 10) -> List[Tuple[int, float]]:
        """Top-k продолжений с учётом контекста."""
        cont = self.get_continuation(context_symbols)
        top_idx = np.argsort(cont)[-k:][::-1]
        return [(int(i), float(cont[i])) for i in top_idx]


# ============================================================
# 2. DynamicVocab — расширение алфавита на лету
# ============================================================

class DynamicVocab:
    """
    Динамический словарь: добавляет новые символы при встрече.

    В отличие от CharacterVocab (фиксированный набор),
    этот словарь растёт по мере необходимости.

    При добавлении символа:
    - Расширяет матрицу аффинности (новая строка + столбец с 0.5)
    - Назначает новый индекс
    """

    def __init__(self, potential_field, base_vocab):
        self.pf = potential_field
        self.base_vocab = base_vocab
        self.char_to_idx: Dict[str, int] = dict(base_vocab._char_to_idx)
        self.idx_to_char: Dict[int, str] = dict(base_vocab._idx_to_char)
        self.next_idx = len(self.char_to_idx)

    def add_symbol(self, char: str) -> int:
        """Добавить новый символ в словарь и расширить матрицу."""
        if char in self.char_to_idx:
            return self.char_to_idx[char]

        idx = self.next_idx
        self.char_to_idx[char] = idx
        self.idx_to_char[idx] = char
        self.next_idx += 1

        # Расширяем матрицу аффинности
        old_size = self.pf.affinity.shape[0]
        if idx >= old_size:
            new_size = max(idx + 1, old_size * 2)
            old_aff = self.pf.affinity.cpu().numpy()
            old_count = self.pf.co_occurrence_count.cpu().numpy()

            new_aff = np.full((new_size, new_size), 0.5, dtype=np.float32)
            new_count = np.zeros((new_size, new_size), dtype=np.float32)

            new_aff[:old_size, :old_size] = old_aff
            new_count[:old_size, :old_size] = old_count

            self.pf.affinity = torch.tensor(new_aff)
            self.pf.co_occurrence_count = torch.tensor(new_count)
            self.pf.vocab_size = new_size

            logger.info(f"[DynamicVocab] Расширен до {new_size} символов (добавлен '{char}')")

        return idx

    def encode(self, text: str) -> List[int]:
        """Кодировать текст, добавляя новые символы при необходимости."""
        result = [self.base_vocab.BOS_IDX]
        for ch in text:
            if ch not in self.char_to_idx:
                self.add_symbol(ch)
            result.append(self.char_to_idx[ch])
        result.append(self.base_vocab.EOS_IDX)
        return result

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.idx_to_char.get(i, '?') for i in ids)

    def __len__(self) -> int:
        return self.next_idx


# ============================================================
# 3. PatternToConcept — связь паттернов с понятиями
# ============================================================

class PatternToConcept:
    """
    Связывает символьные паттерны (слова) с «понятиями».

    Понятие = кластер паттернов, которые:
    - Часто встречаются в похожих контекстах
    - Имеют схожие продолжения
    - Образуют семантическую группу

    Пример:
    Паттерны: «кошка», «собака», «тигр», «лев»
    Контексты: «животное», «домашнее», «дикое»
    → Понятие: «животные»
    """

    def __init__(self, grammar, potential_field, min_pattern_length: int = 3):
        self.grammar = grammar
        self.pf = potential_field
        self.min_pattern_length = min_pattern_length

        # Понятия: concept_id → {name, pattern_ids, centroid}
        self.concepts: Dict[str, Dict] = {}

        # Паттерн → понятие
        self.pattern_to_concept: Dict[str, str] = {}

        # Контекстные векторы для паттернов
        self.pattern_contexts: Dict[str, np.ndarray] = {}

    def extract_patterns(self) -> List[Dict]:
        """Извлечь все стабильные паттерны из грамматики."""
        patterns = []
        for level, pats in self.grammar.patterns.items():
            for ph, pattern in pats.items():
                if pattern.length >= self.min_pattern_length:
                    patterns.append({
                        'id': ph,
                        'symbols': pattern.symbol_indices,
                        'length': pattern.length,
                        'coherence': pattern.coherence_score,
                        'usage': pattern.usage_count,
                    })
        return patterns

    def compute_pattern_context(self, pattern_symbols: List[int]) -> np.ndarray:
        """
        Контекст паттерна = распределение того, что идёт ПОСЛЕ него.
        Два паттерна с похожими продолжениями — семантически близки.
        """
        if not pattern_symbols:
            return np.zeros(self.pf.vocab_size)

        last = pattern_symbols[-1]
        return self.pf.get_continuation_potential(last).cpu().numpy()

    def cluster_by_context(self, similarity_threshold: float = 0.7) -> int:
        """Кластеризовать паттерны по похожести контекстов."""
        patterns = self.extract_patterns()
        if len(patterns) < 2:
            return 0

        # Вычисляем контекстные векторы
        for p in patterns:
            ctx = self.compute_pattern_context(p['symbols'])
            self.pattern_contexts[p['id']] = ctx

        concept_id = 0
        pattern_ids = [p['id'] for p in patterns]

        for i, pid_a in enumerate(pattern_ids):
            for j, pid_b in enumerate(pattern_ids):
                if i >= j:
                    continue

                ca = self.pattern_contexts.get(pid_a)
                cb = self.pattern_contexts.get(pid_b)
                if ca is None or cb is None:
                    continue

                sim = float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-8))

                if sim > similarity_threshold:
                    # Эти паттерны принадлежат одному понятию
                    cid_a = self.pattern_to_concept.get(pid_a)
                    cid_b = self.pattern_to_concept.get(pid_b)

                    if cid_a and cid_b:
                        if cid_a != cid_b:
                            # Merge concepts
                            self.concepts[cid_a]['pattern_ids'].extend(
                                self.concepts.pop(cid_b, {}).get('pattern_ids', [])
                            )
                            for pid in self.concepts[cid_a]['pattern_ids']:
                                self.pattern_to_concept[pid] = cid_a
                    elif cid_a:
                        self.pattern_to_concept[pid_b] = cid_a
                        self.concepts[cid_a]['pattern_ids'].append(pid_b)
                    elif cid_b:
                        self.pattern_to_concept[pid_a] = cid_b
                        self.concepts[cid_b]['pattern_ids'].append(pid_a)
                    else:
                        cid = f"concept_{concept_id:04d}"
                        concept_id += 1
                        self.concepts[cid] = {
                            'name': f"concept_{concept_id}",
                            'pattern_ids': [pid_a, pid_b],
                            'similarity': float(sim),
                        }
                        self.pattern_to_concept[pid_a] = cid
                        self.pattern_to_concept[pid_b] = cid

        return concept_id

    def summary(self) -> str:
        return f"PatternToConcept: {len(self.concepts)} concepts from patterns"


# ============================================================
# 4. MultiLevelGrammar — иерархический вывод на всех уровнях
# ============================================================

class MultiLevelPredictor:
    """
    Предсказание на всех уровнях иерархии.

    Уровень 0: диграммы (пары символов) — самый надёжный
    Уровень 1: N-граммы (цепочки из 3-5 символов)
    Уровень 2: слова (стабильные паттерны из AssemblyGrammar)
    Уровень 3: фразы (последовательности слов)

    Предсказание = взвешенная сумма предсказаний всех уровней.
    """

    def __init__(self, potential_field, grammar, ngram_context: NGramContext):
        self.pf = potential_field
        self.grammar = grammar
        self.ngram = ngram_context

        # Веса уровней
        self.level_weights = {
            0: 0.5,  # диграммы
            1: 0.3,  # N-граммы
            2: 0.15, # слова
            3: 0.05, # фразы (слабый — мало данных)
        }

    def predict(self, context_symbols: List[int]) -> np.ndarray:
        """
        Многоуровневое предсказание следующего символа.
        """
        V = self.pf.vocab_size
        result = np.zeros(V)
        weight_sum = 0.0

        # Уровень 0: диграммы
        if context_symbols:
            last = context_symbols[-1]
            digram_pred = self.pf.get_continuation_potential(last).cpu().numpy()
            result += self.level_weights[0] * digram_pred
            weight_sum += self.level_weights[0]

        # Уровень 1: N-граммы (контекст из 2-3 символов)
        if len(context_symbols) >= 2:
            ngram_pred = self.ngram.get_continuation(context_symbols)
            result += self.level_weights[1] * ngram_pred
            weight_sum += self.level_weights[1]

        # Уровень 2: слова (паттерны из грамматики)
        if len(context_symbols) >= 2:
            word_pred = self._word_level_prediction(context_symbols, V)
            if word_pred is not None:
                result += self.level_weights[2] * word_pred
                weight_sum += self.level_weights[2]

        if weight_sum > 0:
            result /= weight_sum

        return result

    def _word_level_prediction(self, context: List[int], V: int) -> Optional[np.ndarray]:
        """Предсказание на уровне слов: завершить известный паттерн."""
        result = np.zeros(V)
        found = False

        for level, pats in self.grammar.patterns.items():
            for ph, pattern in pats.items():
                if pattern.length < 3:
                    continue
                pat_symbols = pattern.symbol_indices[:pattern.length]
                # Проверяем, является ли контекст префиксом паттерна
                context_tail = context[-min(len(context), len(pat_symbols)):]
                if len(context_tail) >= 2:
                    match_len = 0
                    for k in range(min(len(context_tail), len(pat_symbols))):
                        if context_tail[-(k+1)] == pat_symbols[-(k+1)]:
                            match_len += 1
                        else:
                            break
                    if match_len >= 2 and match_len < len(pat_symbols):
                        next_sym = pat_symbols[match_len]
                        if 0 <= next_sym < V:
                            result[next_sym] += pattern.coherence_score
                            found = True

        return result if found else None

    def summary(self) -> str:
        return f"MultiLevelPredictor(levels={len(self.level_weights)})"
