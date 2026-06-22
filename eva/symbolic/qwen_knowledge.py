"""QwenKnowledge — load and query precomputed Qwen knowledge for STDP modulation.

Usage:
  qk = QwenKnowledge("real_data/qwen_knowledge.npz")
  factor = qk.get_factor(cid_a, cid_b)  # returns 1.0 if no knowledge
"""

import os
import numpy as np
from typing import Dict


class QwenKnowledge:
    """
    Loads precomputed Qwen pairwise cosine similarities and provides
    modulation factors for STDP training.

    The knowledge is loaded from a .npz file with keys:
      rows: uint32[N] — FCF CID A
      cols: uint32[N] — FCF CID B
      vals: float32[N] — mean cosine similarity
      counts: uint32[N] — observation count
    """

    def __init__(self, path: str = None, factor_strength: float = 0.3,
                 min_threshold: float = 0.15, repulse_threshold: float = 0.2,
                 max_factor: float = 1.5, min_factor: float = 0.85):
        """
        Args:
            path: Path to qwen_knowledge.npz. None = skip (all factors = 1.0).
            factor_strength: How strongly cos_sim modulates lr for boost.
                lr *= 1.0 + cos_sim * factor_strength  (for cos >= repulse_threshold)
            min_threshold: Pairs with cos < this are treated as unknown (factor=1.0).
            repulse_threshold: Pairs with cos in [min_threshold, repulse_threshold)
                get repulsion: linear from min_factor at min_threshold to 1.0 at repulse_threshold.
            max_factor: Cap for boost
            min_factor: Floor for repulsion
        """
        self._map: Dict[int, float] = {}
        self._total_counts = 0
        self.factor_strength = factor_strength
        self.min_threshold = min_threshold
        self.repulse_threshold = repulse_threshold
        self.max_factor = max_factor
        self.min_factor = min_factor

        if path is not None and os.path.exists(path):
            self._load(path)

    def _load(self, path: str):
        data = np.load(path)
        rows = data['rows']
        cols = data['cols']
        vals = data['vals']
        counts = data['counts']
        n = len(rows)
        self._total_counts = int(counts.sum())

        # Build lookup dict: packed key (int64) → float16 (cos_sim)
        # key = (min_cid << 32) | max_cid
        for i in range(n):
            a = int(rows[i])
            b = int(cols[i])
            key = (a << 32) | b
            self._map[key] = float(vals[i])

        print(f"[QwenKnowledge] loaded {len(self._map)} pairs "
              f"({self._total_counts} observations) from {path}")

    @property
    def is_loaded(self) -> bool:
        return len(self._map) > 0

    def get_raw(self, cid_a: int, cid_b: int) -> float | None:
        """Get raw cosine similarity, or None if pair not in knowledge."""
        key = (cid_a << 32) | cid_b if cid_a <= cid_b else (cid_b << 32) | cid_a
        return self._map.get(key)

    def get_factor(self, cid_a: int, cid_b: int) -> float:
        """
        Get LR modulation factor for STDP pair (cid_a, cid_b).
        1.0 = no change, >1 = boost, <1 = reduce.

        Three regimes:
          cos < min_threshold      → 1.0 (unknown, neutral)
          min_threshold ≤ cos < repulse_threshold → linear repel (min_factor → 1.0)
          cos ≥ repulse_threshold  → 1.0 + cos * factor_strength (boost)
        """
        cos = self.get_raw(cid_a, cid_b)
        if cos is None:
            return 1.0
        if cos < self.min_threshold:
            return 1.0
        if cos < self.repulse_threshold:
            # Linear repulsion: min_factor at threshold → 1.0 at repulse_threshold
            t = (cos - self.min_threshold) / (self.repulse_threshold - self.min_threshold)
            return float(self.min_factor + (1.0 - self.min_factor) * t)
        factor = 1.0 + cos * self.factor_strength
        return float(np.clip(factor, 1.0, self.max_factor))

    def __len__(self) -> int:
        return len(self._map)

    def __bool__(self) -> bool:
        return self.is_loaded


# ── Integration helper ──────────────────────────────────────────

def inject_qwen_knowledge(args, kwargs):
    """
    Call this at the start of _build_pairs to inject qwen factor.
    Returns (skip_pair_bool, qwen_factor_float).

    Usage in _build_pairs, after lr computation (line 244):
      qwen_factor = 1.0
      if gen.qwen_knowledge and gen.qwen_knowledge.is_loaded:
          qwen_factor = gen.qwen_knowledge.get_factor(ids[i], ids[j])
      lr = base_lr * max(freq_weight, 0.05) * pmi_w * field_weight * qwen_factor
    """
    pass  # Inline the code above in _build_pairs
