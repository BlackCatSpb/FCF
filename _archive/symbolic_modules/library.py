"""
LibraryLibrarian — «смотритель библиотеки».

Метафора:
- Библиотека = KnowledgeBase (пустая → наполняется)
- Книги = Домены (возникают как плотностные кластеры)
- Полки = Иерархия (под-домены в доменах)
- Строки = Паттерны символов (слова, фразы)
- Каталог = DomainIndex (навигация: где что лежит)
- Смотритель = IntelligentContextRouter (понимает связи)

Что реализовано:
1. DomainAutoNamer — авто-именование доменов по характерным символам
2. DomainIndex — каталог: domain → key_patterns, cross-references
3. SimilarityRetrieval — «найти книгу, похожую на эту строку»
4. LibrarianMap — карта библиотеки (матрица domain×domain связей)
5. LibraryStats — статистика наполненности и порядка
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


# ============================================================
# 1. DomainAutoNamer
# ============================================================

class DomainAutoNamer:
    """
    Автоматически именует домены на основе их содержимого.

    Имя домена = самые характерные символы/паттерны,
    которые встречаются в нём чаще, чем в других доменах.

    Использует TF-IDF-like подход:
    tf(symbol, domain) = частота символа в домене
    idf(symbol) = log(всего_доменов / доменов_с_символом)
    """

    def __init__(self, knowledge_base, char_vocab, word_discovery):
        self.kb = knowledge_base
        self.char_vocab = char_vocab
        self.word_discovery = word_discovery

    def name_domain(self, domain, top_k: int = 5) -> str:
        """Имя домена из характерных слов, а НЕ случайных символов."""
        if not domain.symbol_indices:
            return f"domain_{domain.domain_id}"

        # Приоритет 1: реальные слова из word_discovery
        domain_words = []
        for wid, word in self.word_discovery.words.items():
            if len(word.symbols) >= 3:  # Только слова длиной 3+ символов
                overlap = sum(1 for s in word.symbols if s in domain.symbol_indices)
                if overlap / max(len(word.symbols), 1) > 0.7:
                    domain_words.append(word)

        if domain_words:
            sorted_words = sorted(domain_words, key=lambda w: w.occurrence_count, reverse=True)
            name_parts = [w.text for w in sorted_words[:top_k] if len(w.text) >= 2]
            if name_parts:
                return "/".join(name_parts[:3])

        # Приоритет 2: частые диграммы из grammar
        if hasattr(self.kb, 'grammar'):
            digrams = []
            for ph, p in self.kb.grammar.patterns.get(0, {}).items():
                if p.length >= 2 and p.symbol_indices[0] in domain.symbol_indices:
                    digrams.append((p.coherence_score, p))
            if digrams:
                digrams.sort(reverse=True)
                name_parts = [self.char_vocab.decode(d[1].symbol_indices[:2]) for d in digrams[:top_k]]
                return "/".join(name_parts[:3])

        # Приоритет 3: номер домена (честно)
        return f"D{domain.domain_id}"

    def name_all_domains(self):
        """Именовать все домены."""
        for did, domain in self.kb.domains.items():
            domain.name = self.name_domain(domain)
            logger.debug(f"  Domain {did} → '{domain.name}'")


# ============================================================
# 2. DomainIndex — каталог библиотеки
# ============================================================

@dataclass
class CatalogEntry:
    """Запись в каталоге: что находится в домене."""
    domain_id: int
    domain_name: str
    # Ключевые паттерны (слова/фразы)
    key_patterns: List[str] = field(default_factory=list)
    # Перекрёстные ссылки на связанные домены
    cross_references: List[Tuple[int, str, float]] = field(default_factory=list)
    # Статистика
    symbol_count: int = 0
    pattern_count: int = 0
    coherence: float = 0.0
    # Положение в иерархии
    level: int = 0
    parent_id: Optional[int] = None


class DomainIndex:
    """
    Каталог библиотеки: быстрый поиск «где что лежит».

    Позволяет:
    - По символу → найти домен
    - По паттерну → найти домен
    - По домену → получить все его паттерны и cross-refs
    """

    def __init__(self, knowledge_base, word_discovery, auto_namer):
        self.kb = knowledge_base
        self.word_discovery = word_discovery
        self.namer = auto_namer

        self.catalog: Dict[int, CatalogEntry] = {}

    def build_catalog(self):
        """Построить каталог по всем доменам."""
        self.catalog.clear()

        for did, domain in self.kb.domains.items():
            # Ключевые паттерны
            key_patterns = []
            for wid, word in self.word_discovery.words.items():
                if len(word.symbols) >= 2:
                    overlap = sum(1 for s in word.symbols if s in domain.symbol_indices)
                    if overlap / max(len(word.symbols), 1) > 0.7:
                        key_patterns.append(word.text)

            # Cross-references: домены с высокой cross-affinity
            cross_refs = []
            for other_did in self.kb.domains:
                if other_did == did:
                    continue
                cross_aff = self._cross_affinity(domain, self.kb.domains[other_did])
                if cross_aff > 0.55:
                    cross_refs.append((
                        other_did,
                        self.kb.domains[other_did].name,
                        float(cross_aff),
                    ))

            entry = CatalogEntry(
                domain_id=did,
                domain_name=domain.name,
                key_patterns=key_patterns[:20],
                cross_references=cross_refs[:10],
                symbol_count=len(domain.symbol_indices),
                pattern_count=len(key_patterns),
                coherence=domain.coherence,
                level=domain.level,
                parent_id=domain.parent_domain_id,
            )
            self.catalog[did] = entry

    def _cross_affinity(self, a, b) -> float:
        aff = self.kb.pf.affinity.cpu().numpy()
        scores = []
        for sa in a.symbol_indices[:20]:
            for sb in b.symbol_indices[:20]:
                scores.append(float(aff[sa, sb]))
        return float(np.mean(scores)) if scores else 0.5

    def find_domain_for_pattern(self, pattern_symbols: List[int]) -> Optional[int]:
        """Найти домен, к которому относится паттерн."""
        best_domain = None
        best_score = 0
        for did, domain in self.kb.domains.items():
            overlap = sum(1 for s in pattern_symbols if s in domain.symbol_indices)
            score = overlap / max(len(pattern_symbols), 1)
            if score > best_score:
                best_score = score
                best_domain = did
        return best_domain if best_score > 0.5 else None

    def find_similar_domains(self, domain_id: int, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """Найти домены, похожие на данный."""
        if domain_id not in self.catalog:
            return []
        entry = self.catalog[domain_id]
        refs = sorted(entry.cross_references, key=lambda x: x[2], reverse=True)
        return refs[:top_k]

    def summary(self) -> str:
        return f"DomainIndex: {len(self.catalog)} entries in catalog"


# ============================================================
# 3. LibrarianMap — карта библиотеки
# ============================================================

class LibrarianMap:
    """
    Карта библиотеки: визуализация связей между доменами.

    Матрица domain×domain:
    map[i][j] = cross_affinity(domain_i, domain_j)

    Это «карта» для смотрителя: насколько книги рядом.
    """

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.map_matrix: Optional[np.ndarray] = None
        self.domain_names: Dict[int, str] = {}

    def build_map(self):
        """Построить карту: матрица cross-affinity между всеми доменами."""
        n = len(self.kb.domains)
        if n < 2:
            self.map_matrix = np.eye(1)
            return

        aff = self.kb.pf.affinity.cpu().numpy()
        dids = sorted(self.kb.domains.keys())
        self.map_matrix = np.zeros((n, n))

        for i, da_id in enumerate(dids):
            da = self.kb.domains[da_id]
            self.domain_names[da_id] = da.name
            for j, db_id in enumerate(dids):
                if i == j:
                    self.map_matrix[i, j] = 1.0
                else:
                    db = self.kb.domains[db_id]
                    scores = []
                    for sa in da.symbol_indices[:20]:
                        for sb in db.symbol_indices[:20]:
                            scores.append(float(aff[sa, sb]))
                    self.map_matrix[i, j] = float(np.mean(scores)) if scores else 0.5

        # Симметризуем
        self.map_matrix = (self.map_matrix + self.map_matrix.T) / 2

    def get_neighbors(self, domain_id: int, threshold: float = 0.55) -> List[Tuple[int, str, float]]:
        """Ближайшие соседи домена на карте."""
        dids = sorted(self.kb.domains.keys())
        if domain_id not in dids:
            return []

        idx = dids.index(domain_id)
        row = self.map_matrix[idx]
        neighbors = []
        for j, did in enumerate(dids):
            if j != idx and row[j] > threshold:
                neighbors.append((did, self.domain_names.get(did, f"d{did}"), float(row[j])))

        neighbors.sort(key=lambda x: x[2], reverse=True)
        return neighbors

    def summary(self) -> str:
        if self.map_matrix is None:
            return "LibrarianMap: not built"
        n = self.map_matrix.shape[0]
        return f"LibrarianMap: {n}×{n}, avg_cross_aff={float(self.map_matrix.mean()):.3f}"


# ============================================================
# 4. LibraryStats — статистика наполненности
# ============================================================

@dataclass
class LibraryStats:
    """Статистика библиотеки."""
    total_domains: int = 0
    total_symbols: int = 0
    total_patterns: int = 0
    avg_domain_size: float = 0.0
    avg_domain_coherence: float = 0.0
    hierarchy_depth: int = 0
    orphans: int = 0  # домены без родителей
    coverage: float = 0.0  # % символов, покрытых доменами

    def to_dict(self) -> dict:
        return {
            "total_domains": self.total_domains,
            "total_symbols": self.total_symbols,
            "total_patterns": self.total_patterns,
            "avg_domain_size": self.avg_domain_size,
            "avg_domain_coherence": self.avg_domain_coherence,
            "hierarchy_depth": self.hierarchy_depth,
            "orphans": self.orphans,
            "coverage": self.coverage,
        }


class LibraryManager:
    """
    Главный управляющий библиотекой.

    Оркестрирует:
    - KnowledgeBase (домены)
    - DomainAutoNamer (имена)
    - DomainIndex (каталог)
    - LibrarianMap (карта)
    - LibraryStats (статистика)

    Это «смотритель библиотеки» верхнего уровня.
    """

    def __init__(
        self,
        knowledge_base,
        word_discovery,
        char_vocab,
    ):
        self.kb = knowledge_base
        self.word_discovery = word_discovery
        self.char_vocab = char_vocab

        self.namer = DomainAutoNamer(knowledge_base, char_vocab, word_discovery)
        self.index = DomainIndex(knowledge_base, word_discovery, self.namer)
        self.librarian_map = LibrarianMap(knowledge_base)

    def organize(self):
        """
        Организовать библиотеку: домены → имена → каталог → карта.

        Полный цикл того, что делает смотритель:
        1. Обнаружить домены (книги)
        2. Дать им имена
        3. Построить иерархию (расставить по полкам)
        4. Составить каталог
        5. Нарисовать карту
        """
        self.kb.auto_maintain()
        self.namer.name_all_domains()
        self.index.build_catalog()
        self.librarian_map.build_map()

    def get_stats(self) -> LibraryStats:
        """Собрать статистику библиотеки."""
        stats = LibraryStats()
        stats.total_domains = len(self.kb.domains)
        stats.total_patterns = len(self.word_discovery.words)

        if self.kb.domains:
            stats.avg_domain_size = np.mean([len(d.symbol_indices) for d in self.kb.domains.values()])
            stats.avg_domain_coherence = np.mean([d.coherence for d in self.kb.domains.values()])

            # Глубина иерархии
            for d in self.kb.domains.values():
                stats.hierarchy_depth = max(stats.hierarchy_depth, d.level)
                if d.parent_domain_id is None:
                    stats.orphans += 1

            # Покрытие символов доменами
            covered = set()
            for d in self.kb.domains.values():
                covered.update(d.symbol_indices)
            stats.coverage = len(covered) / max(self.kb.pf.vocab_size, 1)

        return stats

    def find(self, query: str) -> Dict:
        """
        Найти информацию по запросу.

        «Смотритель, где книга про X?»
        """
        symbols = self.char_vocab.encode(query)[1:-1]
        domain_id = self.index.find_domain_for_pattern(symbols)

        result = {"query": query, "domain_id": domain_id}

        if domain_id is not None and domain_id in self.index.catalog:
            entry = self.index.catalog[domain_id]
            result["domain_name"] = entry.domain_name
            result["key_patterns"] = entry.key_patterns[:10]
            result["related_domains"] = [
                {"id": rid, "name": rname, "affinity": raff}
                for rid, rname, raff in entry.cross_references[:5]
            ]
            result["path"] = self.kb.get_domain_path(domain_id)

        return result

    def summary(self) -> str:
        stats = self.get_stats()
        return (
            f"Library: {stats.total_domains} domains, "
            f"{stats.total_patterns} patterns, "
            f"coverage={stats.coverage:.1%}, "
            f"depth={stats.hierarchy_depth}, "
            f"catalog={len(self.index.catalog)} entries"
        )
