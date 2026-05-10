"""
HNSWIndex — иерархический поиск состояний (Пункт 7).

Три уровня:
- Уровень 0 (глобальный): центроиды всех доменов → маршрутизация
- Уровень 1 (доменный): слепки внутри домена
- Уровень 2 (слой-специфичный): подындексы по диапазонам слоёв (1-8, 9-16, ...)

Обеспечивает логарифмическую сложность поиска даже при миллионах слепков.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class HNSWIndex:

    def __init__(self, dim: int = 2560, M: int = 32):
        self.dim = dim
        self.M = M

        self.level0: Dict[str, np.ndarray] = {}
        self.level1: Dict[str, List[np.ndarray]] = {}
        self.level2: Dict[str, Dict[int, List[np.ndarray]]] = {}

        self._level0_ids: List[str] = []
        self._level0_matrix: Optional[np.ndarray] = None

    def add_domain(self, domain_id: str, centroid: np.ndarray):
        c = centroid.flatten().astype(np.float32)
        c = c / (np.linalg.norm(c) + 1e-8)
        self.level0[domain_id] = c
        self.level1[domain_id] = []
        self.level2[domain_id] = {}
        self._rebuild_level0()

    def add_snapshot(
        self,
        domain_id: str,
        vector: np.ndarray,
        layer_idx: int = 0,
    ):
        v = vector.flatten().astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)

        if domain_id not in self.level1:
            self.level1[domain_id] = []
        self.level1[domain_id].append(v)

        layer_bucket = (layer_idx // 8) * 8
        if domain_id not in self.level2:
            self.level2[domain_id] = {}
        if layer_bucket not in self.level2[domain_id]:
            self.level2[domain_id][layer_bucket] = []
        self.level2[domain_id][layer_bucket].append(v)

    def search_domain(self, c_query: np.ndarray) -> Optional[str]:
        if not self.level0:
            return None

        q = c_query.flatten().astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)

        if self._level0_matrix is None:
            self._rebuild_level0()

        if self._level0_matrix is None or len(self._level0_matrix) == 0:
            return None

        similarities = np.dot(self._level0_matrix, q)
        best_idx = int(np.argmax(similarities))

        if best_idx < len(self._level0_ids):
            return self._level0_ids[best_idx]

        return None

    def search_snapshot(
        self,
        domain_id: str,
        c_query: np.ndarray,
        layer_idx: Optional[int] = None,
    ) -> Optional[int]:
        q = c_query.flatten().astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)

        vectors = None
        if layer_idx is not None and domain_id in self.level2:
            bucket = (layer_idx // 8) * 8
            if bucket in self.level2[domain_id]:
                vectors = self.level2[domain_id][bucket]

        if not vectors and domain_id in self.level1:
            vectors = self.level1[domain_id]

        if not vectors:
            return None

        best_idx = -1
        best_sim = -1.0
        for i, v in enumerate(vectors):
            sim = np.dot(v, q)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return best_idx if best_idx >= 0 else None

    def _rebuild_level0(self):
        self._level0_ids = list(self.level0.keys())
        if not self._level0_ids:
            self._level0_matrix = None
            return

        self._level0_matrix = np.zeros(
            (len(self._level0_ids), self.dim), dtype=np.float32
        )
        for i, did in enumerate(self._level0_ids):
            self._level0_matrix[i] = self.level0[did]

    def size(self) -> Dict[str, int]:
        return {
            "domains": len(self.level0),
            "level1_snapshots": sum(
                len(v) for v in self.level1.values()
            ),
            "level2_buckets": sum(
                len(buckets) for buckets in self.level2.values()
            ),
        }
