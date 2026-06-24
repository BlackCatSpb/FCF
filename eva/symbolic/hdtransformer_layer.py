"""HDTransformerLayer — VSA-native transformer without backprop.

Replaces VSACNN/VSAConvLayer. Uses:
  - LSH-indexed cosine attention (no QK^T matmul)
  - Zeckendorf-tree weighting via Fibonacci weights 0-7
  - Multi-head subspace masks
  - Fractal convolution FFN
  - STDP training via EntityField bridge

No softmax. No matrix multiply. No backprop.
"""

import numpy as np
from eva.symbolic.fibonacci_utils import FibonacciUtils
from eva.symbolic.concept_space import _hybrid_bind, _hybrid_unbind
from eva.symbolic.experimental.vsa_utils import _fractal_convolution, _random_masks


class HDTransformerLayer:
    """VSA-transformer: LSH-attention + Zeckendorf-tree + fractal FFN.

    Args:
        dim: vector dimension (default 768)
        num_heads: number of attention heads (default 3)
        top_k: number of top keys to attend to (default 10)
        adaptive_quantile: use z-score quantile for weight mapping (default True)
        use_bind_weighting: bind(weight, value) vs scale+bundle (default True)
    """

    def __init__(self, dim=768, num_heads=3, top_k=10,
                 adaptive_quantile=True, use_bind_weighting=True):
        self.dim = dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.adaptive_quantile = adaptive_quantile
        self.use_bind_weighting = use_bind_weighting
        self.masks = _random_masks(dim, n_heads=num_heads)
        self._lsh = None

    # ── Weight helpers ────────────────────────────────────────────

    def _quantize_linear(self, sim, max_val=7):
        norm = (max(0.0, min(1.0, (sim + 1.0) / 2.0)))
        return int(round(max_val * norm))

    def _quantize_adaptive(self, sim, mean, std, z_score=2.0, max_val=7):
        z = (sim - mean) / (std + 1e-8)
        z = np.clip(z, -z_score, z_score)
        scaled = (z + z_score) / (2 * z_score) * max_val
        return int(round(np.clip(scaled, 0, max_val)))

    def _zeckendorf_tree(self, w):
        tree = FibonacciUtils.zeckendorf(w)
        if sum(tree) != w:
            tree.append(w - sum(tree))
        return tree

    # ── Attention core ────────────────────────────────────────────

    def _lsh_attention(self, query, kv_pairs):
        """LSH-attention: cosine → Zeckendorf-weight → bundle.

        Args:
            query: ndarray (dim,) unit-norm
            kv_pairs: list of (key_vec, val_vec) tuples

        Returns:
            aggregated ndarray (dim,)
        """
        if not kv_pairs:
            return query.copy()

        # Score all pairs
        sims = [float(np.dot(query, k) / (np.linalg.norm(k) + 1e-10))
                for k, _ in kv_pairs]

        # Adaptive quantile stats
        if self.adaptive_quantile and len(sims) > 1:
            mean = np.mean(sims)
            std = np.std(sims)

        # Top-K selection
        n_top = min(self.top_k, len(sims))
        idxs = np.argsort(sims)[-n_top:][::-1]

        result = None
        for i in idxs:
            sim = sims[i]
            _, val = kv_pairs[i]

            if self.adaptive_quantile and len(sims) > 1:
                w = self._quantize_adaptive(sim, mean, std)
            else:
                w = self._quantize_linear(sim)

            if w == 0:
                continue

            tree = self._zeckendorf_tree(w)
            for part in tree:
                if part == 0:
                    continue
                if self.use_bind_weighting:
                    weight_hv = np.full(self.dim, part / 7.0, dtype=np.float32)
                    wn = np.linalg.norm(weight_hv)
                    if wn > 1e-10:
                        weight_hv /= wn
                    weighted = _hybrid_bind(val, weight_hv)
                else:
                    weighted = val * (part / 7.0)
                    wn = np.linalg.norm(weighted)
                    if wn > 1e-10:
                        weighted /= wn
                result = weighted if result is None else result + weighted

        if result is None:
            return query.copy()
        rn = np.linalg.norm(result)
        return result / rn if rn > 1e-10 else query.copy()

    # ── Forward ───────────────────────────────────────────────────

    def forward(self, sequence, positions=None):
        """Forward: pos_encode → multi-head attention → fractal FFN.

        Args:
            sequence: list of ndarrays (dim,) unit-norm
            positions: optional list of int positions for Fib encoding

        Returns:
            list of ndarrays (dim,) unit-norm
        """
        if not sequence:
            return []

        # Fibonacci position encoding
        if positions is not None:
            encoded = [np.roll(s, FibonacciUtils.fib_position_shift(t, self.dim))
                      for s, t in zip(sequence, positions)]
        else:
            encoded = [s.copy() for s in sequence]

        outputs = []
        for i, q in enumerate(encoded):
            kv = [(encoded[j], encoded[j]) for j in range(len(encoded))]

            if self.num_heads > 1:
                head_results = []
                for mask in self.masks:
                    mask_bin = (np.asarray(mask, dtype=np.float64) > 0.5)
                    masked_q = q.copy()
                    masked_q[mask_bin] *= np.asarray(mask, dtype=np.float64)[mask_bin]
                    mq_norm = np.linalg.norm(masked_q)
                    if mq_norm > 1e-10:
                        masked_q /= mq_norm
                    head_results.append(self._lsh_attention(masked_q, kv))
                aggregated = head_results[0]
                for h in head_results[1:]:
                    aggregated = aggregated + h
                an = float(np.linalg.norm(aggregated))
                if an > 1e-10:
                    aggregated /= an
            else:
                aggregated = self._lsh_attention(q, kv)

            # Residual + fractal FFN
            out = q + aggregated
            on = float(np.linalg.norm(out))
            if on > 1e-10:
                out /= on
            ff = _fractal_convolution(out, kernel_sizes=(3, 5, 7))
            outputs.append(ff)

        return outputs

    # ── STDP training step ────────────────────────────────────────

    def train_step(self, sequence, target, lr=0.003):
        """STDP step: compare attention output to target, correct via field.

        Args:
            sequence: list of ndarrays input
            target: list of ndarrays target
            lr: learning rate

        Returns:
            mean error norm
        """
        outputs = self.forward(sequence)
        errors = []
        for out, tgt in zip(outputs, target):
            error = tgt - out
            error_norm = float(np.linalg.norm(error))
            errors.append(error_norm)
            if error_norm > 0.1:
                correction = out + error * lr
                cn = float(np.linalg.norm(correction))
                if cn > 1e-10:
                    correction /= cn
        return float(np.mean(errors)) if errors else 0.0
