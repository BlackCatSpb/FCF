"""
StateStorage — векторное хранилище состояний на базе FAISS.

Каждый «слепок» содержит:
- c: контекстный вектор (L2-нормированный, в FAISS-индексе)
- K, V: Key и Value тензоры последнего токена
- confidence: оценка SRG на момент сохранения
- domain: идентификатор домена (изначально "general")
- timestamp: время создания
- usage_count: счётчик использований
"""

import time
import numpy as np
from typing import Optional, Tuple
from loguru import logger

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS не установлен. Используется fallback-хранилище.")


class StateStorage:
    def __init__(self, dim: int, max_snapshots: int = 10000):
        self.dim = dim
        self.max_snapshots = max_snapshots

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None

        self.snapshots_meta: list = []
        self._vectors: list = []
        self._next_id: int = 0
        self._removed_ids: set = set()
        self._rebuild_pending: int = 0
        self.REBUILD_THRESHOLD = 50

    def add(
        self,
        c: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        confidence: float,
        domain: str = "general",
    ) -> int:
        if len(self.snapshots_meta) >= self.max_snapshots:
            oldest_idx = min(
                range(len(self.snapshots_meta)),
                key=lambda i: self.snapshots_meta[i]["timestamp"],
            )
            self._remove(oldest_idx)

        c_norm = c / (np.linalg.norm(c) + 1e-8)
        c_norm = c_norm.astype(np.float32)

        vector_id = self._next_id
        self._next_id += 1

        if self.index is not None:
            self.index.add(c_norm.reshape(1, -1))
        else:
            self._vectors.append(c_norm)

        meta = {
            "id": vector_id,
            "c": c_norm,
            "K": K.copy(),
            "V": V.copy(),
            "confidence": confidence,
            "domain": domain,
            "timestamp": time.time(),
            "usage_count": 1,
        }
        self.snapshots_meta.append(meta)
        return len(self.snapshots_meta) - 1

    def search(self, c_query: np.ndarray, threshold: float = 0.7) -> int:
        if len(self.snapshots_meta) == 0:
            return -1

        c_norm = c_query / (np.linalg.norm(c_query) + 1e-8)
        c_norm = c_norm.astype(np.float32)

        if self.index is not None and self.index.ntotal > 0:
            distances, indices = self.index.search(c_norm.reshape(1, -1), 1)
            if distances[0][0] >= threshold:
                faiss_idx = int(indices[0][0])
                for i, meta in enumerate(self.snapshots_meta):
                    if meta.get("id", i) == faiss_idx or i == faiss_idx:
                        self.snapshots_meta[i]["usage_count"] += 1
                        return i

        best_idx = -1
        best_sim = -1.0
        for i, meta in enumerate(self.snapshots_meta):
            sim = np.dot(c_norm.flatten(), meta["c"].flatten())
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= threshold and best_idx >= 0:
            self.snapshots_meta[best_idx]["usage_count"] += 1
            return best_idx

        return -1

    def get(self, idx: int) -> Optional[dict]:
        if 0 <= idx < len(self.snapshots_meta):
            return self.snapshots_meta[idx]
        return None

    def _remove(self, idx: int):
        if idx < 0 or idx >= len(self.snapshots_meta):
            return

        meta = self.snapshots_meta[idx]
        del self.snapshots_meta[idx]

        if self.index is not None:
            self._rebuild_pending += 1
            if self._rebuild_pending >= self.REBUILD_THRESHOLD:
                self._rebuild_index()
                self._rebuild_pending = 0
            else:
                self._removed_ids.add(meta.get("id", idx))

    def _rebuild_index(self):
        if self.index is None:
            return
        self.index.reset()
        for i, meta in enumerate(self.snapshots_meta):
            c = meta["c"].astype(np.float32).reshape(1, -1)
            self.index.add(c)
            meta["id"] = i
        self._next_id = len(self.snapshots_meta)
        self._removed_ids.clear()

    def get_all_vectors(self) -> np.ndarray:
        if self.index is not None and self.index.ntotal > 0:
            vectors = np.zeros((self.index.ntotal, self.dim), dtype=np.float32)
            for i in range(self.index.ntotal):
                self.index.reconstruct(i, vectors[i])
            return vectors
        if self._vectors:
            return np.array(self._vectors, dtype=np.float32)
        return np.array([], dtype=np.float32).reshape(0, self.dim)

    def rebuild_from_meta(self):
        if self.index is None:
            self._vectors = [m["c"].copy() for m in self.snapshots_meta]
        else:
            self._rebuild_index()
        self._rebuild_pending = 0

    def sync_to_hnsw(self, hnsw_index):
        if hnsw_index is None:
            return
        for meta in self.snapshots_meta:
            domain = meta.get("domain", "general")
            hnsw_index.add_snapshot(
                domain_id=domain,
                vector=meta["c"],
                code_id=str(meta.get("id", 0)),
            )

    def sync_from_hnsw(self, hnsw_index):
        pass

    def __len__(self) -> int:
        return len(self.snapshots_meta)

    def __bool__(self) -> bool:
        return len(self.snapshots_meta) > 0
