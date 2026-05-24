"""
AssemblyGrammar — грамматика сборки: иерархический discovery паттернов.

Уровни:
  L0: символ → символ (диграммы)
  L1: диграмма → диграмма (N-граммы / слова)
  L2: слово → слово (фразы)
  L3: фраза → фраза (предложения)

Для каждого уровня:
- Discover: найти устойчивые паттерны
- Compose: объединить два паттерна в один более высокого уровня
- Decompose: разбить паттерн на составные части
- Validate: проверить что паттерн логичен

Ключевая идея: паттерн не "хранится", а "выводится" из потенциалов связей.
Паттерн = граф связей с гарантированной семантической замкнутостью.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class AssemblyPattern:
    """Устойчивый паттерн сборки на определённом уровне."""
    pattern_id: str          # хеш-идентификатор
    level: int               # 0=диграмма, 1=N-грамма, 2=фраза, 3=предложение
    symbol_indices: List[int]  # индексы символов в паттерне
    affinity_submatrix: Optional[np.ndarray] = None  # матрица связей
    potential_vector: Optional[np.ndarray] = None   # потенциал паттерна
    continuation_distribution: Optional[np.ndarray] = None  # что может следовать
    coherence_score: float = 0.0
    usage_count: int = 0
    created_at: float = 0.0
    last_used: float = 0.0
    children: List[str] = field(default_factory=list)  # ID подпаттернов

    @property
    def length(self) -> int:
        return len(self.symbol_indices)

    def to_vec(self) -> np.ndarray:
        if self.potential_vector is not None:
            return self.potential_vector
        return np.array(self.symbol_indices, dtype=np.float32)


class AssemblyGrammar:
    """
    Грамматика сборки: обнаруживает, хранит и валидирует паттерны.

    Паттерны организованы иерархически по уровням.
    Discovery происходит через кластеризацию attention-графов.
    """

    def __init__(
        self,
        potential_field,
        vocab_size: int,
        embed_dim: int = 256,
        max_patterns_per_level: int = 10000,
    ):
        self.potential_field = potential_field
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Паттерны по уровням
        self.patterns: Dict[int, Dict[str, AssemblyPattern]] = {
            0: {},  # диграммы
            1: {},  # N-граммы / слова
            2: {},  # фразы
            3: {},  # предложения
        }
        self.max_patterns_per_level = max_patterns_per_level

        # Инвертированный индекс: символ → паттерны
        self.symbol_to_patterns: Dict[int, Set[str]] = defaultdict(set)

        # Статистика
        self.total_discoveries: int = 0
        self.total_compositions: int = 0

    def _pattern_hash(self, symbol_indices: List[int]) -> str:
        """Хеш для идентификации паттерна."""
        return "|".join(str(i) for i in symbol_indices[:50])

    def discover_digrams(self, min_affinity: float = 0.6) -> List[AssemblyPattern]:
        """
        Discovery уровня 0: найти пары символов с высокой аффинностью.

        Диграмма = два символа i,j где affinity[i,j] > threshold.
        Это базовая единица сборки.
        """
        import time
        new_patterns = []
        aff = self.potential_field.affinity.cpu().numpy()
        n = min(self.vocab_size, aff.shape[0])

        for i in range(n):
            for j in range(i + 1, n):
                a_ij = float(aff[i, j])
                a_ji = float(aff[j, i])

                if a_ij > min_affinity or a_ji > min_affinity:
                    indices = [i, j]
                    ph = self._pattern_hash(indices)

                    if ph not in self.patterns[0]:
                        sub, _ = self.potential_field.get_affinity_submatrix(indices)
                        sub = sub.cpu().numpy() if isinstance(sub, torch.Tensor) else sub

                        pattern = AssemblyPattern(
                            pattern_id=ph,
                            level=0,
                            symbol_indices=indices,
                            affinity_submatrix=sub,
                            coherence_score=float(max(a_ij, a_ji)),
                            created_at=time.time(),
                            last_used=time.time(),
                        )
                        self.patterns[0][ph] = pattern
                        self.symbol_to_patterns[i].add(ph)
                        self.symbol_to_patterns[j].add(ph)
                        new_patterns.append(pattern)
                        self.total_discoveries += 1

        return new_patterns

    def discover_ngrams(
        self,
        max_n: int = 5,
        min_coherence: float = 0.5,
    ) -> List[AssemblyPattern]:
        """
        Discovery уровня 1: найти N-граммы через композицию диграмм.

        N-грамма = цепочка диграмм с общими символами и высокой связностью.
        """
        import time
        new_patterns = []

        # Строим граф диграмм
        digram_graph: Dict[int, List[int]] = defaultdict(list)
        for ph, pattern in self.patterns[0].items():
            i, j = pattern.symbol_indices[:2]
            digram_graph[i].append(j)
            digram_graph[j].append(i)

        # Ищем пути длины N через BFS
        for start in digram_graph:
            visited = {start}
            queue = deque([(start, [start], 1)])
            while queue:
                node, path, depth = queue.popleft() if hasattr(queue, 'popleft') else queue.popleft()
                if depth >= max_n:
                    continue
                for next_node in digram_graph[node]:
                    if next_node not in visited and depth + 1 <= max_n:
                        new_path = path + [next_node]
                        new_visited = visited | {next_node}

                        # Проверяем связность пути
                        coherence = self._check_path_coherence(new_path)
                        if coherence > min_coherence and len(new_path) >= 3:
                            ph = self._pattern_hash(new_path)
                            if ph not in self.patterns[1]:
                                sub, _ = self.potential_field.get_affinity_submatrix(new_path)
                                pattern = AssemblyPattern(
                                    pattern_id=ph,
                                    level=1,
                                    symbol_indices=new_path,
                                    affinity_submatrix=sub.cpu().numpy() if isinstance(sub, torch.Tensor) else sub,
                                    coherence_score=coherence,
                                    created_at=time.time(),
                                    last_used=time.time(),
                                )
                                self.patterns[1][ph] = pattern
                                for s in new_path:
                                    self.symbol_to_patterns[s].add(ph)
                                new_patterns.append(pattern)
                                self.total_discoveries += 1

                        queue.append((next_node, new_path, depth + 1))
                        visited = new_visited

        if len(self.patterns[1]) > self.max_patterns_per_level:
            sorted_pats = sorted(self.patterns[1].values(),
                                 key=lambda p: p.coherence_score, reverse=True)
            self.patterns[1] = {p.pattern_id: p for p in sorted_pats[:self.max_patterns_per_level]}

        return new_patterns

    def _check_path_coherence(self, path: List[int]) -> float:
        """Проверить связность пути в матрице аффинности."""
        if len(path) < 2:
            return 0.0
        scores = []
        for k in range(len(path) - 1):
            a = float(self.potential_field.affinity[path[k], path[k + 1]])
            scores.append(a)
        return float(np.mean(scores)) if scores else 0.0

    def compose(
        self,
        pattern_a: AssemblyPattern,
        pattern_b: AssemblyPattern,
    ) -> Optional[AssemblyPattern]:
        """
        Композиция: объединить два паттерна в один более высокого уровня.

        Если a и b имеют общие символы или высокую cross-аффинность,
        создаём составной паттерн уровня max(level_a, level_b) + 1.
        """
        import time

        symbols_a = set(pattern_a.symbol_indices)
        symbols_b = set(pattern_b.symbol_indices)

        overlap = symbols_a & symbols_b
        if not overlap:
            # Проверяем cross-аффинность
            cross_scores = []
            for ia in pattern_a.symbol_indices:
                for ib in pattern_b.symbol_indices:
                    cross_scores.append(float(self.potential_field.affinity[ia, ib]))
            avg_cross = np.mean(cross_scores) if cross_scores else 0.0
            if avg_cross < 0.4:
                return None

        combined = list(pattern_a.symbol_indices) + [
            s for s in pattern_b.symbol_indices if s not in symbols_a
        ]

        if len(combined) > 50:
            combined = combined[:50]

        new_level = max(pattern_a.level, pattern_b.level) + 1
        new_level = min(new_level, 3)

        ph = self._pattern_hash(combined)
        if ph in self.patterns[new_level]:
            return self.patterns[new_level][ph]

        sub, _ = self.potential_field.get_affinity_submatrix(combined)
        coherence = self._check_path_coherence(combined)

        if coherence < 0.3:
            return None

        pattern = AssemblyPattern(
            pattern_id=ph,
            level=new_level,
            symbol_indices=combined,
            affinity_submatrix=sub.cpu().numpy() if isinstance(sub, torch.Tensor) else sub,
            coherence_score=coherence,
            children=[pattern_a.pattern_id, pattern_b.pattern_id],
            created_at=time.time(),
            last_used=time.time(),
        )
        self.patterns[new_level][ph] = pattern
        for s in combined:
            self.symbol_to_patterns[s].add(ph)
        self.total_compositions += 1

        return pattern

    def decompose(self, pattern: AssemblyPattern) -> List[AssemblyPattern]:
        """
        Декомпозиция: разбить паттерн на составные части.

        Использует children (если есть) или разрезает по минимуму аффинности.
        """
        if pattern.children:
            parts = []
            for child_id in pattern.children:
                for level in self.patterns:
                    if child_id in self.patterns[level]:
                        parts.append(self.patterns[level][child_id])
                        break
            if parts:
                return parts

        # Разрезаем: ищем точку с минимальной аффинностью
        indices = pattern.symbol_indices
        if len(indices) < 3:
            return [pattern]

        min_aff = float('inf')
        split_at = 1
        for k in range(1, len(indices)):
            aff = float(self.potential_field.affinity[indices[k - 1], indices[k]])
            if aff < min_aff:
                min_aff = aff
                split_at = k

        left_indices = indices[:split_at]
        right_indices = indices[split_at:]

        left_ph = self._pattern_hash(left_indices)
        right_ph = self._pattern_hash(right_indices)

        left_pat = self.patterns[pattern.level].get(left_ph)
        right_pat = self.patterns[pattern.level].get(right_ph)

        result = []
        if left_pat:
            result.append(left_pat)
        if right_pat:
            result.append(right_pat)

        return result if result else [pattern]

    def find_best_continuation(
        self,
        pattern: AssemblyPattern,
        top_k: int = 5,
    ) -> List[Tuple[AssemblyPattern, float]]:
        """
        Найти лучшие продолжения для паттерна:
        какие паттерны логично следуют за данным.
        """
        last_symbol = pattern.symbol_indices[-1]
        candidates = []

        for ph in self.symbol_to_patterns.get(last_symbol, set()):
            for level, pats in self.patterns.items():
                if ph in pats:
                    cand = pats[ph]
                    if cand.length <= 3 and cand.pattern_id != pattern.pattern_id:
                        # Оцениваем совместимость
                        cross_score = 0.0
                        for s in pattern.symbol_indices[-3:]:
                            for t in cand.symbol_indices[:3]:
                                cross_score += float(self.potential_field.affinity[s, t])
                        cross_score /= max(len(pattern.symbol_indices[-3:]) * len(cand.symbol_indices[:3]), 1)
                        candidates.append((cand, cross_score))
                    break

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def summary(self) -> str:
        counts = {level: len(pats) for level, pats in self.patterns.items()}
        return (
            f"AssemblyGrammar: "
            f"digrams={counts[0]}, ngrams={counts[1]}, "
            f"phrases={counts[2]}, sentences={counts[3]}, "
            f"discoveries={self.total_discoveries}, compositions={self.total_compositions}"
        )
