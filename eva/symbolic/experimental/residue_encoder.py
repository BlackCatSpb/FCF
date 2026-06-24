"""ResidueEncoder — Residue Number System encoding for VSA."""

import numpy as np


class ResidueEncoder:
    """RNS encoding via modular residues.
    value → [value % m1, value % m2, ...] → bind(base_vectors).
    """

    def __init__(self, moduli, dim=768, rng=None):
        self.moduli = list(moduli)
        self.dim = dim
        if rng is None:
            from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
            rng = _R.rng('residue')
        self.bases = {}
        for m in self.moduli:
            self.bases[m] = [rng.randn(dim).astype(np.float64) for _ in range(m)]
            for bv in self.bases[m]:
                bn = np.linalg.norm(bv)
                if bn > 1e-10:
                    bv /= bn

    def encode(self, value):
        from eva.symbolic.concept_space import _hybrid_bind
        residues = [int(value) % m for m in self.moduli]
        result = self.bases[self.moduli[0]][residues[0]].copy()
        for m, r in zip(self.moduli[1:], residues[1:]):
            result = _hybrid_bind(result, self.bases[m][r])
        return result

    def decode(self, vec):
        """Recover integer value from RNS via unbind + similarity.

        For each modulus m, unbinds vec with each base[r] and picks
        the residue with highest response norm.
        """
        from eva.symbolic.concept_space import _hybrid_unbind
        v = vec.data if hasattr(vec, 'data') else np.asarray(vec)
        residues = []
        for m in self.moduli:
            best_r = 0
            best_sim = -1.0
            for r in range(m):
                b = self.bases[m][r]
                bv = b.data if hasattr(b, 'data') else np.asarray(b)
                unbound = _hybrid_unbind(v, bv)
                sim = float(np.dot(unbound, unbound)) ** 0.5
                if sim > best_sim:
                    best_sim = sim
                    best_r = r
            residues.append(best_r)
        # CRT reconstruction
        value = 0
        M = 1
        for m, r in zip(self.moduli, residues):
            inv = pow(M % m, -1, m)
            value += r * M * inv
            M *= m
        return value % M

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        from eva.symbolic.concept_space import _hybrid_bind
        return _hybrid_bind(a, b)
