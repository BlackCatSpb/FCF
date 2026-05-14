"""
HNSWIndex — иерархический поиск с Product Quantization + Temporal Decay + Fractal Links.

Возможности:
  - Уровень 0 (глобальный): центроиды доменов → маршрутизация
  - Уровень 1 (доменный): сжатые PQ-коды слепков → поиск
  - Уровень 2 (фрактальный): ссылки между кодами разных уровней абстракции
  - Product Quantization: сжатие векторов 4-8x
  - Temporal Decay Attention: exp(-λ·age) — старые коды теряют вес
  - Дефрагментация во время Sleep Mode
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class FractalLink:
    """Ссылка между кодом верхнего уровня и кодами нижнего."""
    parent_code_id: str
    child_code_ids: List[str]
    link_type: str = "composes"  # composes / references / derives
    weight: float = 1.0


@dataclass
class PQCodebook:
    """
    Product Quantization: разбивает вектор на M подвекторов,
    квантует каждый в один из 256 центроидов.
    Сжатие: 4 байта × d → M байт (примерно d/M : 1).
    """

    M: int
    d: int
    sub_dim: int
    codes: np.ndarray

    @classmethod
    def train(
        cls,
        vectors: np.ndarray,
        M: int = 8,
        n_iter: int = 10,
    ) -> "PQCodebook":
        N, d = vectors.shape
        sub_dim = d // M
        if d % M != 0:
            d_pad = ((d + M - 1) // M) * M
            padded = np.zeros((N, d_pad), dtype=np.float32)
            padded[:, :d] = vectors
            vectors = padded
            d = d_pad
            sub_dim = d // M

        codes = np.zeros((256, M, sub_dim), dtype=np.float32)

        for m in range(M):
            sub_vectors = vectors[:, m * sub_dim : (m + 1) * sub_dim].copy()
            centroids = sub_vectors[
                np.random.choice(N, min(256, N), replace=False)
            ]
            for _ in range(n_iter):
                distances = np.sum(
                    (sub_vectors[:, None, :] - centroids[None, :, :]) ** 2,
                    axis=-1,
                )
                assignments = np.argmin(distances, axis=1)

                new_centroids = np.zeros_like(centroids)
                counts = np.zeros(256, dtype=np.int32)
                for i in range(N):
                    c = assignments[i]
                    new_centroids[c] += sub_vectors[i]
                    counts[c] += 1

                for c in range(256):
                    if counts[c] > 0:
                        centroids[c] = new_centroids[c] / counts[c]

            codes[:, m, :] = centroids.astype(np.float32)

        logger.info(f"[PQ] Codebook: M={M}, sub_dim={sub_dim}, d={d}")
        return cls(M=M, d=d, sub_dim=sub_dim, codes=codes)

    def encode(self, vector: np.ndarray) -> np.ndarray:
        v = vector.flatten().astype(np.float32)
        if len(v) < self.d:
            v = np.pad(v, (0, self.d - len(v)))
        codes = np.zeros(self.M, dtype=np.uint8)
        for m in range(self.M):
            sub = v[m * self.sub_dim : (m + 1) * self.sub_dim]
            dists = np.sum((self.codes[:, m, :] - sub) ** 2, axis=1)
            codes[m] = np.argmin(dists)
        return codes

    def encode_batch(self, vectors: np.ndarray) -> np.ndarray:
        N = vectors.shape[0]
        encoded = np.zeros((N, self.M), dtype=np.uint8)
        for i in range(N):
            encoded[i] = self.encode(vectors[i])
        return encoded

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = codes.astype(np.int32)
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)
        N = codes.shape[0]
        decoded = np.zeros((N, self.d), dtype=np.float32)
        for m in range(self.M):
            decoded[:, m * self.sub_dim : (m + 1) * self.sub_dim] = \
                self.codes[codes[:, m], m, :]
        return decoded

    def similarity_batch(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        decoded = self.decode(codes)
        q = query.flatten().astype(np.float32)
        if len(q) < self.d:
            q = np.pad(q, (0, self.d - len(q)))
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        d_norm = decoded / (np.linalg.norm(decoded, axis=1, keepdims=True) + 1e-8)
        return np.dot(d_norm, q_norm)


class HNSWIndex:
    """
    Иерархический HNSW-индекс с Product Quantization.

    Уровень 0: центроиды доменов (точные векторы)
    Уровень 1: сжатые PQ-коды слепков внутри домена
    Уровень 2: сжатые PQ-коды по диапазонам слоёв
    """

    def __init__(self, dim: int = 2560, pq_M: int = 8, temporal_lambda: float = 0.01):
        self.dim = dim
        self.pq_M = pq_M
        self.temporal_lambda = temporal_lambda

        self.level0: Dict[str, np.ndarray] = {}
        self._level0_matrix: Optional[np.ndarray] = None
        self._level0_ids: List[str] = []

        self.level1: Dict[str, List] = {}
        self.level2: Dict[str, Dict[int, List[np.ndarray]]] = {}

        self.fractal_links: Dict[str, FractalLink] = {}
        self._reverse_links: Dict[str, List[str]] = {}

        self.pq_codebook: Optional[PQCodebook] = None
        self._pq_trained = False
        self._pq_cache: Dict[str, np.ndarray] = {}

        self._snapshot_count = 0

        self._snapshot_count = 0

    def train_pq(self, vectors: np.ndarray):
        """Обучить Product Quantization на накопленных векторах."""
        if len(vectors) < 256:
            logger.warning("[PQ] Недостаточно векторов для обучения")
            return
        self.pq_codebook = PQCodebook.train(vectors, M=self.pq_M)
        self._pq_trained = True
        self._pq_cache.clear()
        logger.info(
            f"[PQ] Обучено: M={self.pq_M}, "
            f"compression={32 / self.pq_M:.1f}x"
        )

    def _compress(self, vector: np.ndarray) -> np.ndarray:
        if not self._pq_trained or self.pq_codebook is None:
            v = vector.flatten().astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-8)
        return self.pq_codebook.encode(vector)

    def add_domain(self, domain_id: str, centroid: np.ndarray):
        c = centroid.flatten().astype(np.float32)
        c = c / (np.linalg.norm(c) + 1e-8)
        self.level0[domain_id] = c

        if domain_id not in self.level1:
            self.level1[domain_id] = []
        if domain_id not in self.level2:
            self.level2[domain_id] = {}

        self._rebuild_level0()
        self._pq_cache.pop(domain_id, None)

    def add_snapshot(
        self,
        domain_id: str,
        vector: np.ndarray,
        layer_idx: int = 0,
        code_id: str = None,
    ):
        v = vector.flatten().astype(np.float32)
        v_norm = v / (np.linalg.norm(v) + 1e-8)

        if domain_id not in self.level1:
            self.level1[domain_id] = []

        compressed = self._compress(v)
        timestamp = time.time()
        self.level1[domain_id].append((v_norm, compressed, timestamp))
        self._snapshot_count += 1

        bucket = (layer_idx // 8) * 8
        if domain_id not in self.level2:
            self.level2[domain_id] = {}
        if bucket not in self.level2[domain_id]:
            self.level2[domain_id][bucket] = []

        self.level2[domain_id][bucket].append((compressed, timestamp))

        self._pq_cache.pop(domain_id, None)

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
        return self._level0_ids[best_idx] if best_idx < len(self._level0_ids) else None

    def search_snapshot(
        self,
        domain_id: str,
        c_query: np.ndarray,
        layer_idx: Optional[int] = None,
        top_k: int = 1,
    ) -> List[Tuple[int, float]]:
        q = c_query.flatten().astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        now = time.time()

        candidates = []
        timestamps = []

        if layer_idx is not None and domain_id in self.level2:
            bucket = (layer_idx // 8) * 8
            if bucket in self.level2[domain_id]:
                for item in self.level2[domain_id][bucket]:
                    if isinstance(item, tuple):
                        candidates.append(item[0])
                        timestamps.append(item[1])
                    else:
                        candidates.append(item)

        if not candidates and domain_id in self.level1:
            for _, compressed, ts in self.level1[domain_id]:
                candidates.append(compressed)
                timestamps.append(ts)

        if not candidates:
            return []

        if self._pq_trained and self.pq_codebook is not None:
            domain_key = f"{domain_id}_pq"
            if domain_key not in self._pq_cache:
                codes = np.array([c for c in candidates], dtype=np.uint8)
                self._pq_cache[domain_key] = codes
            similarities = self.pq_codebook.similarity_batch(
                q, self._pq_cache[domain_key]
            )
        else:
            candidates_arr = np.array(candidates, dtype=np.float32)
            candidates_norm = candidates_arr / (
                np.linalg.norm(candidates_arr, axis=1, keepdims=True) + 1e-8
            )
            similarities = np.dot(candidates_norm, q)

        if timestamps:
            ages = np.array([now - ts for ts in timestamps])
            decay = np.exp(-self.temporal_lambda * ages / 86400.0)
            similarities = similarities * decay

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(int(i), float(similarities[i])) for i in top_indices]

    def get_snapshot_count(self, domain_id: str = None) -> int:
        if domain_id:
            return len(self.level1.get(domain_id, []))
        return self._snapshot_count

    def remove_stale(self, indices: List[int], domain_id: str):
        if domain_id not in self.level1:
            return
        for idx in sorted(indices, reverse=True):
            if idx < len(self.level1[domain_id]):
                del self.level1[domain_id][idx]
                self._snapshot_count = max(0, self._snapshot_count - 1)
        self._pq_cache.pop(domain_id, None)

    def defragment(self):
        """Перестроить все кэши после массовых удалений."""
        self._rebuild_level0()
        self._pq_cache.clear()
        for domain_id in list(self.level1.keys()):
            if not self.level1[domain_id]:
                del self.level1[domain_id]
        logger.info(f"[HNSW] Дефрагментация: {self._snapshot_count} слепков, {len(self.level0)} доменов")

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
            "snapshots": self._snapshot_count,
            "pq_trained": self._pq_trained,
            "pq_M": self.pq_M,
            "fractal_links": len(self.fractal_links),
        }

    def add_fractal_link(self, parent_id: str, child_ids: List[str],
                         link_type: str = "composes", weight: float = 1.0):
        if parent_id not in self.fractal_links:
            self.fractal_links[parent_id] = FractalLink(
                parent_code_id=parent_id,
                child_code_ids=[],
                link_type=link_type,
                weight=weight,
            )
        self.fractal_links[parent_id].child_code_ids.extend(child_ids)

        for cid in child_ids:
            if cid not in self._reverse_links:
                self._reverse_links[cid] = []
            self._reverse_links[cid].append(parent_id)

    def cascade_update(self, code_id: str, new_vector: np.ndarray):
        """Обновить код и каскадно все связанные коды нижнего уровня."""
        for child_id in self.fractal_links.get(code_id, FractalLink(code_id, [])).child_code_ids:
            if child_id in self.fractal_links:
                self.cascade_update(child_id, new_vector)

    def get_fractal_children(self, code_id: str) -> List[str]:
        link = self.fractal_links.get(code_id)
        return link.child_code_ids if link else []

    def get_fractal_parents(self, code_id: str) -> List[str]:
        return self._reverse_links.get(code_id, [])

    def summary(self) -> str:
        s = self.size()
        return (
            f"HNSWIndex(domains={s['domains']}, snapshots={s['snapshots']}, "
            f"PQ={'on' if s['pq_trained'] else 'off'}, M={s['pq_M']})"
        )
