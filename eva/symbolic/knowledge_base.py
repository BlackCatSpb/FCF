"""
KnowledgeBase — самоорганизующаяся база знаний.

Многодоменность + иерархия возникают ЕСТЕСТВЕННО из топологии многообразия.
Никакие домены не заданы вручную.

Принцип:
1. Домен = область высокой плотности в координатном многообразии
2. Под-домен = локальный пик плотности внутри домена
3. Иерархия = вложенность областей плотности
4. Навигация между доменами = геодезические в многообразии

Структура:
  KnowledgeBase
    └── Domain "наука"
         ├── SubDomain "физика"
         │    ├── Patterns (слова, фразы)
         │    └── ContinuationRules
         ├── SubDomain "математика"
         └── ...
    └── Domain "спорт"
         └── ...
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from loguru import logger


# ============================================================
# 1. Domain — естественная область знаний
# ============================================================

@dataclass
class KnowledgeDomain:
    """Естественный домен знаний — сгусток в многообразии."""
    domain_id: int
    name: str = "unknown"
    
    # Границы в координатном пространстве
    centroid: Optional[np.ndarray] = None
    radius: float = 1.0                 # радиус домена
    
    # Содержимое
    symbol_indices: List[int] = field(default_factory=list)
    word_ids: List[int] = field(default_factory=list)
    pattern_ids: List[str] = field(default_factory=list)
    
    # Статистика
    density: float = 0.0                 # средняя плотность
    coherence: float = 0.0               # внутренняя связность
    usage_count: int = 0
    
    # Иерархия
    parent_domain_id: Optional[int] = None
    child_domain_ids: List[int] = field(default_factory=list)
    level: int = 0                       # 0=root, 1=domain, 2=subdomain
    
    # Правила продолжения (специфичные для домена)
    continuation_rules: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_leaf(self) -> bool:
        return len(self.child_domain_ids) == 0


# ============================================================
# 2. KnowledgeBase — самоорганизующееся хранилище
# ============================================================

class KnowledgeBase:
    """
    Иерархическая база знаний.

    Домены возникают как естественные кластеры плотности.
    Иерархия строится через вложенность кластеров.
    """

    def __init__(
        self,
        potential_field,
        topological_field,
        grammar,
        word_discovery,
        max_domains: int = 50,
        min_domain_size: int = 10,
        density_threshold: float = 0.02,
        split_coherence_threshold: float = 0.3,
        merge_similarity_threshold: float = 0.7,
    ):
        self.pf = potential_field
        self.topo = topological_field
        self.grammar = grammar
        self.word_discovery = word_discovery

        self.max_domains = max_domains
        self.min_domain_size = min_domain_size
        self.density_threshold = density_threshold
        self.split_threshold = split_coherence_threshold
        self.merge_threshold = merge_similarity_threshold

        self.domains: Dict[int, KnowledgeDomain] = {}
        self.symbol_to_domain: Dict[int, int] = {}
        self.word_to_domain: Dict[int, int] = {}
        self.next_id: int = 0

        self.total_reorganizations: int = 0

    def discover_domains(self):
        """
        Обнаружить домены как пики плотности в многообразии.

        Алгоритм (MeanShift на координатах + аффинностях):
        1. Для каждого символа: вычислить локальную плотность
        2. Найти локальные максимумы (пики)
        3. Назначить символы ближайшим пикам
        4. Сформировать домены
        """
        aff = self.pf.affinity.cpu().numpy()
        coords = self.topo.coordinates.cpu().numpy()
        n = min(self.pf.vocab_size, coords.shape[0])

        if n < 3:
            return []

        # Плотность: сумма аффинностей в окрестности
        density = np.sum(aff[:n, :n] * (aff[:n, :n] > 0.55), axis=1)
        density = density / max(density.max(), 1)

        # Найти локальные максимумы плотности
        is_peak = np.ones(n, dtype=bool)
        for i in range(n):
            neighbors = np.where(aff[i, :n] > 0.55)[0]
            for nb in neighbors:
                if density[nb] > density[i]:
                    is_peak[i] = False
                    break

        peak_indices = np.where(is_peak)[0]
        peak_indices = sorted(peak_indices, key=lambda p: density[p], reverse=True)

        # Назначаем символы ближайшим пикам (по аффинности)
        assignments = np.full(n, -1, dtype=int)
        for i in range(n):
            if density[i] < self.density_threshold:
                continue
            best_peak = -1
            best_aff = -1
            for pi, peak in enumerate(peak_indices):
                a = aff[i, peak]
                if a > best_aff:
                    best_aff = a
                    best_peak = pi
            if best_peak >= 0:
                assignments[i] = best_peak

        # Формируем домены
        new_domains = []
        self.domains.clear()
        self.symbol_to_domain.clear()
        self.next_id = 0

        for pi, peak in enumerate(peak_indices[:self.max_domains]):
            members = [int(i) for i in np.where(assignments == pi)[0]]
            if len(members) < self.min_domain_size:
                continue

            # Вычисляем centroid
            member_coords = coords[members]
            centroid = member_coords.mean(axis=0)

            # Вычисляем coherence
            if len(members) > 1:
                intra = [float(aff[i, j]) for i in members for j in members if i != j]
                coherence = np.mean(intra) if intra else 0.5
            else:
                coherence = 1.0

            domain = KnowledgeDomain(
                domain_id=self.next_id,
                name=f"domain_{self.next_id}",
                centroid=centroid,
                radius=float(np.max(np.linalg.norm(member_coords - centroid, axis=1))) if len(members) > 1 else 0.5,
                symbol_indices=members,
                density=float(density[peak]),
                coherence=float(coherence),
                level=0,
            )

            self.domains[self.next_id] = domain
            for s in members:
                self.symbol_to_domain[s] = self.next_id
            new_domains.append(domain)
            self.next_id += 1

        return new_domains

    def build_hierarchy(self):
        """
        Построить иерархию: под-домены внутри доменов.

        Домен B ⊂ домен A если:
        - Все (или >70%) символы B ∈ символы A
        - Центроид B близок к центроиду A
        """
        if len(self.domains) < 2:
            return

        # Сортируем по размеру (большие → возможные родители)
        sorted_doms = sorted(self.domains.values(), key=lambda d: len(d.symbol_indices), reverse=True)

        for parent in sorted_doms:
            for child in sorted_doms:
                if child.domain_id == parent.domain_id:
                    continue
                if len(child.symbol_indices) >= len(parent.symbol_indices):
                    continue

                # Проверяем containment
                overlap = len(set(child.symbol_indices) & set(parent.symbol_indices))
                containment = overlap / max(len(child.symbol_indices), 1)

                if containment > 0.7:
                    child.parent_domain_id = parent.domain_id
                    child.level = parent.level + 1
                    parent.child_domain_ids.append(child.domain_id)

    def split_domain(self, domain_id: int) -> Optional[int]:
        """
        Разделить домен если внутренняя связность упала.

        Ищем под-кластеры с высокой внутренней связностью.
        """
        if domain_id not in self.domains:
            return None

        domain = self.domains[domain_id]
        if len(domain.symbol_indices) < self.min_domain_size * 2:
            return None

        aff = self.pf.affinity.cpu().numpy()
        members = domain.symbol_indices

        # Вычисляем pairwise affinity matrix для членов домена
        n = len(members)
        if n < 4:
            return None
        sub_aff = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sub_aff[i, j] = aff[members[i], members[j]]

        # Ищем точку минимальной связности (разрез)
        min_total = float('inf')
        split_at = 1
        for k in range(1, n - 1):
            # Внутренняя связность левой и правой частей
            left_internal = sub_aff[:k, :k].mean() if k > 1 else 1.0
            right_internal = sub_aff[k:, k:].mean() if n - k > 1 else 1.0
            cross = sub_aff[:k, k:].mean()

            if cross < left_internal * 0.5 and cross < right_internal * 0.5:
                total = cross
                if total < min_total:
                    min_total = total
                    split_at = k

        if min_total >= domain.coherence * self.split_threshold:
            return None  # Недостаточно слабая связь для разделения

        # Разделяем
        left_members = members[:split_at]
        right_members = members[split_at:]

        if len(left_members) < self.min_domain_size or len(right_members) < self.min_domain_size:
            return None

        # Создаём новый домен для правой части
        right_domain = KnowledgeDomain(
            domain_id=self.next_id,
            name=f"{domain.name}_split_{self.next_id}",
            centroid=self.topo.coordinates[right_members].cpu().numpy().mean(axis=0),
            symbol_indices=right_members,
            level=domain.level,
            parent_domain_id=domain.parent_domain_id,
        )

        self.domains[self.next_id] = right_domain
        for s in right_members:
            self.symbol_to_domain[s] = self.next_id

        # Обновляем родителя
        if domain.parent_domain_id is not None:
            parent = self.domains.get(domain.parent_domain_id)
            if parent:
                parent.child_domain_ids.append(self.next_id)

        # Обновляем старый домен
        domain.symbol_indices = left_members

        self.next_id += 1
        self.total_reorganizations += 1
        logger.info(f"[KnowledgeBase] Domain {domain_id} split → {self.next_id - 1}")

        return self.next_id - 1

    def merge_domains(self, domain_a_id: int, domain_b_id: int) -> Optional[int]:
        """
        Объединить два домена если они слишком похожи.
        """
        if domain_a_id not in self.domains or domain_b_id not in self.domains:
            return None

        a, b = self.domains[domain_a_id], self.domains[domain_b_id]
        if a.centroid is None or b.centroid is None:
            return None

        similarity = float(1.0 / (1.0 + np.linalg.norm(a.centroid - b.centroid)))
        if similarity < self.merge_threshold:
            return None

        # Объединяем
        merged = KnowledgeDomain(
            domain_id=self.next_id,
            name=f"{a.name}+{b.name}",
            centroid=(a.centroid + b.centroid) / 2,
            symbol_indices=list(set(a.symbol_indices + b.symbol_indices)),
            level=min(a.level, b.level),
            child_domain_ids=list(set(a.child_domain_ids + b.child_domain_ids)),
        )

        self.domains[self.next_id] = merged
        for s in merged.symbol_indices:
            self.symbol_to_domain[s] = self.next_id

        # Удаляем старые
        del self.domains[domain_a_id]
        del self.domains[domain_b_id]

        self.next_id += 1
        self.total_reorganizations += 1
        logger.info(f"[KnowledgeBase] Merged {domain_a_id}+{domain_b_id} → {self.next_id - 1}")

        return self.next_id - 1

    def auto_maintain(self):
        """Автоматическое обслуживание: переоткрытие, разделение, слияние."""
        self.discover_domains()
        self.build_hierarchy()

        # Проверить на разделение
        for did in list(self.domains.keys()):
            domain = self.domains.get(did)
            if domain and domain.coherence < self.split_threshold:
                self.split_domain(did)

        # Проверить на слияние
        dom_ids = list(self.domains.keys())
        for i, da in enumerate(dom_ids):
            for db in dom_ids[i + 1:]:
                if da in self.domains and db in self.domains:
                    a, b = self.domains[da], self.domains[db]
                    if a.centroid is not None and b.centroid is not None:
                        d = np.linalg.norm(a.centroid - b.centroid)
                        if d < min(a.radius, b.radius):
                            self.merge_domains(da, db)

    def get_domain_for_symbols(self, symbols: List[int]) -> Optional[KnowledgeDomain]:
        """Определить домен для последовательности символов."""
        if not symbols:
            return None

        # Голосование: каждый символ голосует за свой домен
        votes = defaultdict(int)
        for s in symbols:
            if s in self.symbol_to_domain:
                votes[self.symbol_to_domain[s]] += 1

        if not votes:
            return None

        best_domain = max(votes, key=votes.get)
        return self.domains.get(best_domain)

    def get_domain_path(self, domain_id: int) -> List[int]:
        """Иерархический путь от домена к корню."""
        path = [domain_id]
        current = self.domains.get(domain_id)
        while current and current.parent_domain_id is not None:
            path.append(current.parent_domain_id)
            current = self.domains.get(current.parent_domain_id)
        return path

    def summary(self) -> str:
        hierarchy = ""
        for did, dom in self.domains.items():
            if dom.parent_domain_id is not None:
                hierarchy += f"D{did}⊂D{dom.parent_domain_id} "

        return (
            f"KnowledgeBase: {len(self.domains)} domains, "
            f"reorgs={self.total_reorganizations}, "
            f"hierarchy={hierarchy}"
        )


# ============================================================
# 3. CrossDomainNavigator — навигация между доменами
# ============================================================

class CrossDomainNavigator:
    """
    Навигация между доменами.

    Позволяет:
    1. Найти путь из домена A в домен B
    2. Оценить "стоимость" кросс-доменного перехода
    3. Определить, нужно ли переключать домен при генерации
    """

    def __init__(self, knowledge_base: KnowledgeBase, logic_bridge):
        self.kb = knowledge_base
        self.logic = logic_bridge

    def cross_domain_affinity(self, domain_a_id: int, domain_b_id: int) -> float:
        """
        Аффинность между доменами = средняя cross-affinity
        между символами из разных доменов.
        """
        a = self.kb.domains.get(domain_a_id)
        b = self.kb.domains.get(domain_b_id)
        if not a or not b:
            return 0.5

        aff = self.kb.pf.affinity.cpu().numpy()
        scores = []
        for sa in a.symbol_indices[:20]:
            for sb in b.symbol_indices[:20]:
                scores.append(float(aff[sa, sb]))

        return float(np.mean(scores)) if scores else 0.5

    def should_switch_domain(
        self, current_domain_id: int, next_symbols: List[int]
    ) -> Tuple[bool, Optional[int]]:
        """
        Определить, нужно ли переключить домен.

        Переключаем если:
        - Следующие символы принадлежат другому домену (>70%)
        - Cross-domain affinity достаточно высока (>0.5)
        """
        next_domain = self.kb.get_domain_for_symbols(next_symbols)
        if next_domain is None or next_domain.domain_id == current_domain_id:
            return False, None

        cross_aff = self.cross_domain_affinity(current_domain_id, next_domain.domain_id)
        if cross_aff < 0.5:
            return False, None

        return True, next_domain.domain_id

    def find_path_between_domains(
        self, from_domain_id: int, to_domain_id: int
    ) -> List[int]:
        """
        Найти путь между доменами через промежуточные.

        BFS по графу доменов (рёбра = cross-affinity > 0.5).
        """
        if from_domain_id == to_domain_id:
            return [from_domain_id]

        visited = {from_domain_id}
        queue = [([from_domain_id], from_domain_id)]

        while queue:
            path, current = queue.pop(0)
            if len(path) > 5:
                continue

            for did in self.kb.domains:
                if did in visited:
                    continue
                cross = self.cross_domain_affinity(current, did)
                if cross > 0.55:
                    new_path = path + [did]
                    if did == to_domain_id:
                        return new_path
                    visited.add(did)
                    queue.append((new_path, did))

        return []

    def summary(self) -> str:
        return f"CrossDomainNavigator(domains={len(self.kb.domains)})"


# ============================================================
# 4. IntelligentContextRouter — умная маршрутизация контекста
# ============================================================

class IntelligentContextRouter:
    """
    Определяет контекст генерации и выбирает релевантные домены.

    При генерации:
    1. Определить домен текущего контекста
    2. Использовать домен-специфичные continuation patterns
    3. Предсказывать когда нужно переключить домен
    4. Кросс-доменная навигация через геодезические
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        cross_navigator: CrossDomainNavigator,
        ngram_context,
        potential_field,
    ):
        self.kb = knowledge_base
        self.navigator = cross_navigator
        self.ngram = ngram_context
        self.pf = potential_field

        self.current_domain_id: Optional[int] = None
        self.domain_history: deque = deque(maxlen=20)

    def route(self, context_symbols: List[int]) -> Tuple[np.ndarray, Optional[int]]:
        """
        Определить контекст и вернуть релевантные continuation potentials.

        Returns: (continuation_distribution, domain_id)
        """
        # Определить текущий домен
        current_domain = self.kb.get_domain_for_symbols(context_symbols)
        domain_id = current_domain.domain_id if current_domain else None

        # Базовая continuation (NGram)
        base = self.ngram.get_continuation(context_symbols)

        # Если в домене — усиливаем домен-специфичные символы
        if current_domain and domain_id is not None:
            for sym in current_domain.symbol_indices:
                if sym < len(base):
                    base[sym] *= 1.1  # небольшой boost для символов домена

            # Добавляем boost для дочерних доменов
            for child_id in current_domain.child_domain_ids:
                child = self.kb.domains.get(child_id)
                if child:
                    for sym in child.symbol_indices:
                        if sym < len(base):
                            base[sym] *= 1.05

        self.current_domain_id = domain_id
        if domain_id is not None:
            self.domain_history.append(domain_id)

        return base, domain_id

    def should_switch(self, next_symbols: List[int]) -> Tuple[bool, Optional[int]]:
        """Предсказать переключение домена."""
        if self.current_domain_id is None:
            return False, None
        return self.navigator.should_switch_domain(
            self.current_domain_id, next_symbols
        )

    def summary(self) -> str:
        return (
            f"ContextRouter(current_domain={self.current_domain_id}, "
            f"history={len(self.domain_history)})"
        )
