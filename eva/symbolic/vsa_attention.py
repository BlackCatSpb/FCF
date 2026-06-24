"""VSAAttention — VSA-native attention without softmax or matrix multiply.

Replaces transformer attention with:
  1. Cosine similarity query↔key
  2. Discretisation → Zeckendorf weights (Fibonacci 0-7)
  3. Weighted aggregation via scale(value × weight/max) ⊕ bundle

Multi-head: each head binds aggregated output with quasi-orthogonal role.
Position encoding: optional Fibonacci shift per token position.
"""

import numpy as np
from eva.symbolic.fibonacci_utils import FibonacciUtils
from eva.symbolic.concept_space import _hybrid_bind


class VSAAttention:
    """VSA-native weighted aggregation layer.

    Args:
        dim: vector dimension (default 768)
        n_heads: number of attention heads (default 4)
        max_weight: max discrete weight (default 7)
        use_fib_pos: apply Fibonacci position encoding (default True)
    """

    def __init__(self, dim=768, n_heads=4, max_weight=7, use_fib_pos=True):
        self.dim = dim
        self.n_heads = n_heads
        self.max_weight = max_weight
        self.use_fib_pos = use_fib_pos

        from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
        mat = _R.rng('vsa_attention').randn(n_heads, dim).astype(np.float32)
        Q, _ = np.linalg.qr(mat.T, mode='reduced')
        self.head_roles = Q.T.copy()

    def _quantize_weight(self, sim):
        return 0 if sim <= 0 else int(round(self.max_weight * min(sim, 1.0)))

    def _zeckendorf_tree(self, w):
        tree = FibonacciUtils.zeckendorf(w)
        if sum(tree) != w:
            tree.append(w - sum(tree))
        return tree

    def _fib_position_shift(self, vec, t):
        return np.roll(vec, FibonacciUtils.fib_position_shift(t, self.dim))

    def _scale_bundle(self, value, weight):
        """Weighted value: scale by weight/max, decompose via Zeckendorf, bundle."""
        if weight <= 0:
            return None
        parts = self._zeckendorf_tree(weight)
        result = None
        for p in parts:
            scaled = value * (p / self.max_weight)
            sn = np.linalg.norm(scaled)
            if sn > 1e-10:
                scaled /= sn
            result = scaled if result is None else result + scaled
        return result

    def forward(self, query, keys, values, positions=None):
        n = len(keys)
        if n == 0:
            return query.copy()

        if positions is None:
            positions = list(range(n))

        qn = np.linalg.norm(query)
        if qn > 1e-10:
            query = query / qn

        if self.use_fib_pos:
            keys = [self._fib_position_shift(k, t) for k, t in zip(keys, positions)]

        # Compute attention weights (shared across all heads)
        weights = []
        for k in keys:
            dot = float(np.dot(query, k))
            kn = float(np.linalg.norm(k))
            sim = dot / kn if kn > 1e-10 else 0.0
            weights.append(self._quantize_weight(sim))

        all_zero = all(w == 0 for w in weights)
        if all_zero:
            return query.copy()

        # Compute weighted values once (pre-head)
        weighted_vals = []
        for i in range(n):
            w = weights[i]
            wv = self._scale_bundle(values[i], w)
            weighted_vals.append(wv)

        # Multi-head: each head binds its weighted aggregation with a role
        head_outputs = []
        for h in range(self.n_heads):
            agg = None
            for wv in weighted_vals:
                if wv is None:
                    continue
                if self.n_heads > 1:
                    tagged = _hybrid_bind(wv, self.head_roles[h])
                else:
                    tagged = wv
                agg = tagged if agg is None else agg + tagged

            if agg is None:
                continue
            an = np.linalg.norm(agg)
            if an > 1e-10:
                head_outputs.append(agg / an)

        if not head_outputs:
            return query.copy()

        result = sum(head_outputs)
        rn = np.linalg.norm(result)
        return result / rn if rn > 1e-10 else query.copy()
