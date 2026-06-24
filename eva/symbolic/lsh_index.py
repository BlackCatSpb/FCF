"""LSHIndex — locality-sensitive hashing for fast approximate nearest neighbor search."""

import numpy as np
from collections import defaultdict


class LSHIndex:
    """Multi-table LSH with random hyperplane projections.

    Each table independently hashes vectors -> hamming-space bucket.
    Union across all tables gives high-recall candidate set.

    Args:
        dim: vector dimension
        n_tables: number of independent hash tables (more = higher recall)
        n_bits: bits per hash key (more = more selective)
        seed: rng seed for reproducibility
    """

    def __init__(self, dim=768, n_tables=4, n_bits=8, seed=None):
        if seed is None:
            from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
            seed = _R.seed('lsh_index')
        self.dim = dim
        self.n_tables = n_tables
        self.n_bits = n_bits
        rng = np.random.RandomState(seed)
        self._projs = [
            rng.randn(n_bits, dim).astype(np.float32)
            for _ in range(n_tables)
        ]
        self._tables = [defaultdict(set) for _ in range(n_tables)]
        self._vectors = {}

    def _hash(self, vec, table_idx):
        proj = self._projs[table_idx] @ vec
        bits = (proj > 0).astype(np.uint8)
        key = bits.dot(1 << np.arange(self.n_bits))
        return int(key)

    def add(self, cid, vec):
        self._vectors[cid] = vec
        for t in range(self.n_tables):
            key = self._hash(vec, t)
            self._tables[t][key].add(cid)

    def remove(self, cid):
        vec = self._vectors.pop(cid, None)
        if vec is None:
            return
        for t in range(self.n_tables):
            key = self._hash(vec, t)
            self._tables[t][key].discard(cid)
            if not self._tables[t][key]:
                del self._tables[t][key]

    def query(self, vec, k=10):
        candidates = set()
        for t in range(self.n_tables):
            key = self._hash(vec, t)
            candidates |= self._tables[t].get(key, set())
        if not candidates:
            return []
        scored = []
        vn = np.linalg.norm(vec)
        if vn < 1e-10:
            return []
        for cid in candidates:
            other = self._vectors.get(cid)
            if other is None:
                continue
            cos = float(np.dot(vec, other) / (vn * np.linalg.norm(other) + 1e-30))
            scored.append((cid, cos))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def update(self, cid, vec):
        self.remove(cid)
        self.add(cid, vec)


class EntityFieldIndex:
    """LSH-powered index for EntityField: find entities by vector similarity.

    Instead of scanning all entities linearly, uses multi-table LSH.
    sync() rebuilds from ef.entities — call after batch modifications.
    """

    def __init__(self, entity_field, n_tables=4, n_bits=8):
        self.ef = entity_field
        self._dim = entity_field.dim
        self._lsh = LSHIndex(dim=self._dim, n_tables=n_tables, n_bits=n_bits)
        self._key_map = {}
        self._dirty = True

    def sync(self):
        self._lsh = LSHIndex(dim=self._dim, n_tables=self._lsh.n_tables,
                             n_bits=self._lsh.n_bits)
        self._key_map = {}
        for key, vec in self.ef.entities.items():
            cid = hash(key) % (2**31 - 1)
            self._lsh.add(cid, vec)
            self._key_map[cid] = key
        self._dirty = False

    def find_similar(self, vec, k=10):
        if self._dirty:
            self.sync()
        results = self._lsh.query(vec, k=k)
        return [(self._key_map[cid], cos) for cid, cos in results]

    def on_bind(self, key):
        """Call after ef.bind() to invalidate cache for this key."""
        self._dirty = True
