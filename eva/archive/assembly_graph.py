"""
AssemblyGraph — инструкция сборки смысла из символов.

Состояние EVA — это НЕ dense вектор, а граф сборки:
- Какие символы связаны (attention edges)
- В каком порядке собираются (assembly tree)
- Какие потенциалы продолжения у каждой собранной конструкции

Уровни иерархии:
  Символы → Группы символов → Слова → Фразы → Предложения
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class AssemblyEdge:
    """Связь между двумя символами в графе сборки."""
    src_idx: int       # позиция символа-источника
    dst_idx: int       # позиция символа-цели
    attention_weight: float  # сила внимания (из attention_matrix)
    affinity: float         # аффинность из PotentialField
    semantic_score: float   # семантическая близость

    def total_strength(self) -> float:
        return (self.attention_weight * 0.4 +
                self.affinity * 0.3 +
                self.semantic_score * 0.3)


@dataclass
class AssemblyNode:
    """Узел в дереве сборки."""
    start: int          # начальная позиция в последовательности
    end: int            # конечная позиция (эксклюзивно)
    level: int          # уровень: 0=символ, 1=группа, 2=слово, 3=фраза
    children: List['AssemblyNode'] = field(default_factory=list)
    potential_vector: Optional[np.ndarray] = None  # потенциал конструкции
    continuation_potential: Optional[np.ndarray] = None  # что может идти дальше
    coherence_score: float = 1.0


@dataclass
class AssemblyState:
    """
    Состояние = полная инструкция сборки смысла.

    Хранит:
    - последовательность символов
    - граф внимания (кто с кем связан)
    - дерево сборки (иерархическая группировка)
    - потенциалы на каждом уровне
    - семантическую связность
    """
    symbols: List[int]                       # индексы символов
    attention_matrix: Optional[np.ndarray] = None  # [T × T] матрица внимания
    edges: List[AssemblyEdge] = field(default_factory=list)
    root: Optional[AssemblyNode] = None      # корень дерева сборки
    coherence_score: float = 0.0              # математическая гарантия связности
    confidence: float = 0.0
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def num_symbols(self) -> int:
        return len(self.symbols)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_symbol_span(self, start: int, end: int) -> List[int]:
        """Получить символы в диапазоне."""
        return self.symbols[start:end]

    def get_node_potential(self, node: AssemblyNode) -> np.ndarray:
        """Вычислить потенциал узла как сумму потенциалов его детей."""
        if node.potential_vector is not None:
            return node.potential_vector
        if not node.children:
            return np.zeros(1)
        children_potentials = [self.get_node_potential(c) for c in node.children]
        return np.mean(children_potentials, axis=0)

    def max_edge_strength(self) -> float:
        if not self.edges:
            return 0.0
        return max(e.total_strength() for e in self.edges)

    def avg_edge_strength(self) -> float:
        if not self.edges:
            return 0.0
        return sum(e.total_strength() for e in self.edges) / len(self.edges)

    def to_dict(self) -> Dict:
        return {
            "num_symbols": self.num_symbols,
            "num_edges": self.num_edges,
            "coherence_score": self.coherence_score,
            "confidence": self.confidence,
            "avg_edge_strength": self.avg_edge_strength(),
            "timestamp": self.timestamp,
        }
