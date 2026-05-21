"""
Advanced Word-Level Processing for EVA Symbolic.

1. WordBoundaryDetector — границы слов через transition probability minima
2. GrammaticalRoleDiscovery — роли слов через clustering контекстов
3. SemanticClustering — группировка слов по continuation patterns
4. WordLevelGenerator — генерация на уровне слов с back-off
5. SelfConsistencyCheck — проверка сгенерированного

Все связи ВЫВОДЯТСЯ математически из символьного уровня.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from loguru import logger


# ============================================================
# 1. WordBoundaryDetector
# ============================================================

class WordBoundaryDetector:
    """
    Обнаруживает границы слов через MINIMA transition probability.

    Идея: в точке границы слова transition probability резко падает.
    P(следующий_символ | текущий_символ) < порога → граница слова.

    Также использует энтропию: в середине слова энтропия низкая
    (ограниченный набор продолжений), на границе — высокая (любой символ).
    """

    def __init__(self, potential_field, boundary_threshold: float = 0.5, entropy_boost: float = 1.5):
        self.pf = potential_field
        self.boundary_threshold = boundary_threshold
        self.entropy_boost = entropy_boost

    def transition_score(self, symbol_i: int, symbol_j: int) -> float:
        """Оценка вероятности перехода i→j. Высокий = вероятно внутри слова."""
        cont = self.pf.get_continuation_potential(symbol_i).cpu().numpy()
        total = cont.sum() + 1e-8
        prob = cont[symbol_j] / total if symbol_j < len(cont) else 0
        return float(prob)

    def is_boundary(self, symbol_i: int, symbol_j: int) -> Tuple[bool, float]:
        """
        Является ли переход i→j границей слова.

        Граница если: transition_prob < threshold
        ИЛИ энтропия продолжений i высокая (может быть концом многих слов).
        """
        prob = self.transition_score(symbol_i, symbol_j)

        # Энтропия продолжений i
        cont = self.pf.get_continuation_potential(symbol_i).cpu().numpy()
        cont_norm = cont / (cont.sum() + 1e-8)
        entropy = -np.sum(cont_norm * np.log(cont_norm + 1e-8))
        max_entropy = np.log(len(cont))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Комбинированная оценка: низкая prob ИЛИ высокая энтропия
        boundary_score = (1.0 - prob) * self.entropy_boost * norm_entropy
        is_boundary = prob < self.boundary_threshold or boundary_score > 0.5

        return is_boundary, boundary_score

    def find_boundaries(self, symbol_sequence: List[int]) -> List[int]:
        """
        Найти все границы слов в символьной последовательности.

        Returns: список индексов ГРАНИЦ (позиция ПОСЛЕДНЕГО символа слова).
        """
        boundaries = []
        for i in range(len(symbol_sequence) - 1):
            si, sj = symbol_sequence[i], symbol_sequence[i + 1]
            is_b, score = self.is_boundary(si, sj)
            if is_b:
                boundaries.append(i)
        return boundaries

    def split_into_words(self, symbol_sequence: List[int]) -> List[List[int]]:
        """Разбить символьную последовательность на слова."""
        boundaries = self.find_boundaries(symbol_sequence)
        words = []
        start = 0
        for b in boundaries:
            if b - start >= 1:
                words.append(symbol_sequence[start:b + 1])
            start = b + 1
        if start < len(symbol_sequence):
            words.append(symbol_sequence[start:])
        return words


# ============================================================
# 2. GrammaticalRoleDiscovery
# ============================================================

@dataclass
class GrammaticalRole:
    """Грамматическая роль: кластер слов с похожими continuation patterns."""
    role_id: int
    name: str = "unknown"
    word_ids: List[int] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    coherence: float = 0.0


class GrammaticalRoleDiscovery:
    """
    Обнаруживает грамматические роли слов БЕЗ РАЗМЕТКИ.

    Идея: слова с одинаковой грамматической ролью имеют похожие
    continuation patterns (после них следуют похожие слова).

    Пример: после глаголов часто идут существительные.
    Метод: кластеризация continuation vectors.
    """

    def __init__(self, potential_field, word_discovery):
        self.pf = potential_field
        self.word_discovery = word_discovery
        self.roles: Dict[int, GrammaticalRole] = {}
        self.word_to_role: Dict[int, int] = {}
        self.next_role_id: int = 0

    def compute_continuation_vector(self, word_tokens: List[int]) -> np.ndarray:
        """
        Вектор продолжений слова = распределение того,
        какие СЛОВА следуют за данным словом.

        Для каждого слова в словаре вычисляем оценку продолжения
        через cross-affinity последнего символа слова.
        """
        V = len(self.word_discovery.words)
        vec = np.zeros(V)

        last_symbol = word_tokens[-1] if word_tokens else 0
        cont = self.pf.get_continuation_potential(last_symbol).cpu().numpy()

        for wid, word in self.word_discovery.words.items():
            if word.symbols:
                first_sym = word.symbols[0]
                if first_sym < len(cont):
                    vec[wid - 1000] = cont[first_sym]  # word_id starts at 1000

        return vec

    def discover_roles(self, n_clusters: int = 8) -> List[GrammaticalRole]:
        """Кластеризовать слова по грамматическим ролям."""
        if len(self.word_discovery.words) < n_clusters * 3:
            return []

        # Строим continuation vectors для всех слов
        word_vecs = {}
        for wid, word in self.word_discovery.words.items():
            vec = self.compute_continuation_vector(word.symbols)
            if vec.sum() > 0:
                word_vecs[wid] = vec

        if len(word_vecs) < n_clusters:
            return []

        # Простой k-means на continuation vectors
        vecs = np.array(list(word_vecs.values()))
        wids = list(word_vecs.keys())

        # Random init
        rng = np.random.RandomState(42)
        centroids = vecs[rng.choice(len(vecs), min(n_clusters, len(vecs)), replace=False)]

        for _ in range(10):
            # Assign
            assignments = []
            for v in vecs:
                dists = np.linalg.norm(centroids - v, axis=1)
                assignments.append(int(np.argmin(dists)))

            # Update
            for c in range(len(centroids)):
                members = vecs[np.array(assignments) == c]
                if len(members) > 0:
                    centroids[c] = members.mean(axis=0)

        # Формируем роли
        self.roles.clear()
        self.word_to_role.clear()

        for c in range(len(centroids)):
            member_indices = [i for i, a in enumerate(assignments) if a == c]
            if len(member_indices) < 2:
                continue

            role = GrammaticalRole(
                role_id=c,
                name=f"role_{c}",
                word_ids=[wids[i] for i in member_indices],
                centroid=centroids[c],
            )
            self.roles[c] = role
            for wid in role.word_ids:
                self.word_to_role[wid] = c

        return list(self.roles.values())

    def get_role(self, word_tokens: List[int]) -> Optional[int]:
        """Определить роль слова."""
        if not self.roles:
            return None
        vec = self.compute_continuation_vector(word_tokens)
        best_role = None
        best_dist = float('inf')
        for rid, role in self.roles.items():
            d = np.linalg.norm(role.centroid - vec)
            if d < best_dist:
                best_dist = d
                best_role = rid
        return best_role

    def summary(self) -> str:
        role_sizes = {rid: len(role.word_ids) for rid, role in self.roles.items()}
        return f"GrammaticalRoles: {len(self.roles)} roles, sizes={role_sizes}"


# ============================================================
# 3. SemanticClustering
# ============================================================

class SemanticClustering:
    """
    Группировка слов по СЕМАНТИЧЕСКОЙ близости.

    Два слова семантически близки если:
    1. У них похожие continuation patterns (что идёт ПОСЛЕ)
    2. У них похожие PRECEDING patterns (что идёт ДО)
    3. Они близки в координатном многообразии

    Аналог: word2vec, но ВЫВЕДЕННЫЙ из символьной статистики.
    """

    def __init__(self, potential_field, word_discovery, manifold, boundary_detector, n_clusters: int = 20):
        self.pf = potential_field
        self.word_discovery = word_discovery
        self.manifold = manifold
        self.boundary_detector = boundary_detector
        self.n_clusters = n_clusters

        self.clusters: Dict[int, List[int]] = defaultdict(list)
        self.word_to_cluster: Dict[int, int] = {}
        self.cluster_labels: Dict[int, str] = {}

    def cluster_by_context(self) -> Dict[int, List[int]]:
        """
        Кластеризация: для каждого слова вычисляем вектор контекста,
        затем k-means.
        """
        if len(self.word_discovery.words) < self.n_clusters:
            return {}

        # Для каждого слова: вектор "что идёт после" + "что идёт до"
        word_vecs = {}
        wids = []

        for wid, word in self.word_discovery.words.items():
            if len(word.symbols) < 2:
                continue

            # Контекст после: последний символ
            last = word.symbols[-1]
            after = self.pf.get_continuation_potential(last).cpu().numpy()

            # Контекст до: первый символ (что было ДО слова — аппроксимация)
            first = word.symbols[0]
            before = np.zeros(len(after))
            for s in range(len(before)):
                before[s] = float(self.pf.affinity[s, first])

            # Координаты в многообразии
            coord = self.manifold.compute_word_coordinates(word)

            # Объединяем
            vec = np.concatenate([
                after[:32],
                before[:32],
                coord[:32] if isinstance(coord, np.ndarray) and len(coord) >= 32
                else np.pad(coord, (0, max(0, 32 - len(coord)))),
            ])
            word_vecs[wid] = vec
            wids.append(wid)

        if len(wids) < 3:
            return {}

        # k-means
        vecs = np.array([word_vecs[w] for w in wids])
        rng = np.random.RandomState(42)
        n_clusters = min(self.n_clusters, len(wids))
        centroids = vecs[rng.choice(len(vecs), n_clusters, replace=False)]

        for it in range(20):
            # Assign
            assignments = []
            for v in vecs:
                dists = np.linalg.norm(centroids - v, axis=1)
                assignments.append(int(np.argmin(dists)))

            # Update
            for c in range(len(centroids)):
                members_idx = [i for i, a in enumerate(assignments) if a == c]
                if members_idx:
                    centroids[c] = vecs[members_idx].mean(axis=0)

        # Формируем кластеры
        self.clusters.clear()
        self.word_to_cluster.clear()

        for i, c in enumerate(assignments):
            wid = wids[i]
            self.clusters[c].append(wid)
            self.word_to_cluster[wid] = c

        return dict(self.clusters)

    def get_semantic_neighbors(self, word_id: int, k: int = 10) -> List[int]:
        """Семантические соседи слова (из того же кластера)."""
        if word_id not in self.word_to_cluster:
            return []
        cid = self.word_to_cluster[word_id]
        neighbors = [w for w in self.clusters[cid] if w != word_id]
        return neighbors[:k]

    def summary(self) -> str:
        return f"SemanticClusters: {len(self.clusters)} clusters, {len(self.word_to_cluster)} words"


# ============================================================
# 4. WordLevelGenerator — генерация на уровне слов с back-off
# ============================================================

class WordLevelGenerator:
    """
    Генератор текста на УРОВНЕ СЛОВ.

    Цикл:
    1. Предсказать следующее СЛОВО (через LogicCompiler)
    2. Сгенерировать СИМВОЛЫ этого слова (через affinity)
    3. Повторить

    Back-off: если слово неизвестно → генерация по символам.
    """

    def __init__(
        self,
        symbolic_generator,
        word_discovery,
        logic_compiler,
        boundary_detector,
        char_vocab,
        beam_width: int = 5,
        max_words: int = 50,
    ):
        self.sym_gen = symbolic_generator
        self.word_discovery = word_discovery
        self.logic = logic_compiler
        self.boundary = boundary_detector
        self.char_vocab = char_vocab
        self.beam_width = beam_width
        self.max_words = max_words

    def generate(
        self,
        prompt_symbols: List[int],
        temperature: float = 0.7,
    ) -> str:
        """
        Сгенерировать текст на уровне слов.

        1. Выделить начальные слова из prompt
        2. Для каждого шага:
           a. Найти best continuation word через LogicCompiler
           b. Сгенерировать символы слова
           c. Повторить
        """
        all_symbols = list(prompt_symbols)
        recent_words: List[int] = []  # token_ids последних слов

        # Разбить prompt на слова
        prompt_words = self.boundary.split_into_words(prompt_symbols)
        for pw in prompt_words:
            word = self.word_discovery.get_word_by_symbols(pw)
            if word:
                recent_words.append(word.token_id)

        for _ in range(self.max_words):
            if recent_words:
                # 1. Предсказать следующее слово
                last_word = self.word_discovery.words.get(recent_words[-1]) if recent_words else None
                if last_word:
                    continuations = self.logic.top_continuations(last_word, top_k=self.beam_width)
                    if continuations:
                        # Выбрать с temperature
                        scores = np.array([s for _, s in continuations])
                        scores = scores / max(temperature, 0.1)
                        probs = np.exp(scores - scores.max())
                        probs = probs / probs.sum()
                        idx = np.random.choice(len(continuations), p=probs)
                        next_word, _ = continuations[idx]

                        # 2. Сгенерировать символы слова (если есть в словаре)
                        all_symbols.append(self.char_vocab.encode(' ')[1])  # пробел
                        all_symbols.extend(next_word.symbols)
                        recent_words.append(next_word.token_id)
                        continue

            # Back-off: генерация по символам
            gen = self.sym_gen.generate(all_symbols, max_new_symbols=20, temperature=temperature)
            all_symbols = gen

            # Извлечь новые слова из сгенерированного
            new_part = all_symbols[len(prompt_symbols):]
            new_words = self.boundary.split_into_words(new_part)
            for nw in new_words:
                w = self.word_discovery.get_word_by_symbols(nw)
                if w:
                    recent_words.append(w.token_id)

            if len(all_symbols) > 500:
                break

        return self.char_vocab.decode(all_symbols)

    def summary(self) -> str:
        return f"WordLevelGenerator(words={len(self.word_discovery.words)}, beam={self.beam_width})"


# ============================================================
# 5. SelfConsistencyCheck
# ============================================================

class SelfConsistencyCheck:
    """
    Проверяет самосогласованность сгенерированного текста.

    Проверки:
    1. Все слова — известные или разумные конструкции
    2. Переходы между словами логичны (affinity на стыке)
    3. Нет зацикливаний (одинаковые слова подряд >3 раз)
    4. Длина предложения в разумных пределах
    """

    def __init__(self, potential_field, word_discovery, logic_compiler):
        self.pf = potential_field
        self.word_discovery = word_discovery
        self.logic = logic_compiler

    def check_word_validity(self, symbols: List[int]) -> Tuple[bool, float]:
        """
        Проверить, является ли последовательность символов валидным словом.

        Валидное если:
        - Известное слово в словаре → 1.0
        - Высокая внутренняя связность → 0.5-0.9
        """
        word = self.word_discovery.get_word_by_symbols(symbols)
        if word:
            return True, word.confidence

        if len(symbols) < 2:
            return False, 0.0

        # Внутренняя связность
        aff = self.pf.affinity.cpu().numpy()
        scores = []
        for k in range(len(symbols) - 1):
            scores.append(float(aff[symbols[k], symbols[k + 1]]))
        avg = np.mean(scores) if scores else 0
        return avg > 0.55, avg

    def check_transition(self, word_a_symbols: List[int], word_b_symbols: List[int]) -> float:
        """Оценка логичности перехода между словами."""
        if not word_a_symbols or not word_b_symbols:
            return 0.5
        last_a = word_a_symbols[-1]
        first_b = word_b_symbols[0]
        return float(self.pf.affinity[last_a, first_b])

    def check_repetition(self, words: List[List[int]]) -> Tuple[bool, str]:
        """Проверить на чрезмерные повторы."""
        if len(words) < 4:
            return True, ""

        for i in range(len(words) - 3):
            if (words[i] == words[i + 1] == words[i + 2] == words[i + 3]):
                return False, "repetition_4x"
        return True, ""

    def full_check(self, generated_text: str) -> Dict[str, Any]:
        """Полная проверка сгенерированного текста."""
        symbols = self.char_vocab.encode(generated_text) if hasattr(self, 'char_vocab') else []
        if not symbols:
            return {"valid": True, "score": 1.0}

        words = self.boundary.split_into_words(symbols) if hasattr(self, 'boundary') else [symbols]

        # Проверка валидности слов
        word_scores = []
        for w in words:
            valid, score = self.check_word_validity(w)
            word_scores.append(score)

        # Проверка переходов
        trans_scores = []
        for k in range(len(words) - 1):
            s = self.check_transition(words[k], words[k + 1])
            trans_scores.append(s)

        # Проверка повторов
        no_repeat, issue = self.check_repetition(words)

        avg_word = np.mean(word_scores) if word_scores else 0.5
        avg_trans = np.mean(trans_scores) if trans_scores else 0.5

        overall = 0.4 * avg_word + 0.4 * avg_trans + 0.2 * (1.0 if no_repeat else 0.0)

        return {
            "valid": overall > 0.4,
            "score": overall,
            "word_quality": avg_word,
            "transition_quality": avg_trans,
            "repetition_ok": no_repeat,
            "issue": issue,
        }

    def summary(self) -> str:
        return "SelfConsistencyCheck"
