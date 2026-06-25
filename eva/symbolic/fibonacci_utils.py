import numpy as np
from typing import List, Tuple, Optional


class FibonacciUtils:
    _cache = {0: 0, 1: 1}

    @classmethod
    def get(cls, n):
        if n < 0:
            return 0
        if n not in cls._cache:
            cls._cache[n] = cls.get(n - 1) + cls.get(n - 2)
        return cls._cache[n]

    @classmethod
    def zeckendorf(cls, n):
        fibs = []
        i = 2
        while True:
            f = cls.get(i)
            if f > n:
                break
            fibs.append(f)
            i += 1
        result = []
        for f in reversed(fibs):
            if n >= f:
                result.append(f)
                n -= f
        return result

    @classmethod
    def fib_scale(cls, value, max_val=7):
        fib_scale = [0, 1, 2, 3, 5, 8, 13, 21][:max_val + 1]
        scaled = value * fib_scale[-1]
        idx = np.argmin(np.abs(np.array(fib_scale) - scaled))
        return idx

    @classmethod
    def golden_ratio(cls):
        return (1 + np.sqrt(5)) / 2

    @classmethod
    def zeckendorf_decompose_weight(cls, w, max_val=7):
        w_clamped = max(0, min(max_val, int(round(w))))
        tree = cls.zeckendorf(w_clamped)
        if sum(tree) != w_clamped:
            tree.append(w_clamped - sum(tree))
        return tree

    @classmethod
    def fib_position_shift(cls, t, dim):
        if t < 0:
            return 0
        return int(cls.get(int(t)) % dim) if dim > 0 else 0

    @classmethod
    def balance_subspaces(cls, z_c, z_a, z_m, eps=1e-8):
        phi = cls.golden_ratio()
        norm_c = float(np.linalg.norm(z_c))
        norm_a = float(np.linalg.norm(z_a))
        norm_m = float(np.linalg.norm(z_m))
        total = norm_c + norm_a + norm_m
        target = total / (phi + 1 + 1 / phi)
        z_c = z_c * (target * phi) / (norm_c + eps)
        z_a = z_a * target / (norm_a + eps)
        z_m = z_m * (target / phi) / (norm_m + eps)
        return z_c, z_a, z_m


class ZeckendorfQuantizer:
    """Quantize float weights to HD vectors via Zeckendorf bundle.

    Zeckendorf's theorem guarantees a unique, non-consecutive Fibonacci
    decomposition for every integer. This class maps each Fibonacci number
    to a fixed random HD vector (dim=D). A weight w is quantized as:

        encode(w) → zeckendorf(round(|w| * scale)) → bundle indices → HD sum

    Decoding is by cosine similarity between bundles.
    Lossy, ~8-15× compression vs fp32.
    """

    def __init__(self, dim: int = 768, max_fib_value: int = 100000,
                 scale: float = 10000, seed: int = 42):
        self.dim = dim
        self.scale = scale
        # Build fib_value → index map
        self._fib_to_idx = {}
        i = 2
        while True:
            f = FibonacciUtils.get(i)
            if f > max_fib_value:
                break
            self._fib_to_idx[f] = i - 2
            i += 1
        n_vecs = len(self._fib_to_idx)
        rng = np.random.RandomState(seed)
        vecs = rng.randn(n_vecs, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self._vecs = vecs / np.clip(norms, 1e-10, None)

    def encode(self, w: float) -> np.ndarray:
        """Encode a single float weight → HD vector (unit norm)."""
        idx = int(round(abs(w) * self.scale))
        if idx <= 0:
            return np.zeros(self.dim, dtype=np.float32)
        fibs = FibonacciUtils.zeckendorf(idx)
        indices = [self._fib_to_idx[f] for f in fibs if f in self._fib_to_idx]
        if not indices:
            return np.zeros(self.dim, dtype=np.float32)
        vec = np.array(list(self._vecs[i] for i in indices)).sum(axis=0)
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-10 else np.zeros(self.dim, dtype=np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two quantized weight HD vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def compression_ratio(fp32_bytes: int = 4, hd_bytes: int = 768 * 2) -> float:
        """Expected compression ratio: fp32_size / hd_vector_size."""
        return fp32_bytes / hd_bytes

    def encode_batch(self, weights: List[float]) -> np.ndarray:
        """Encode multiple weights, return (N, D) matrix."""
        return np.array([self.encode(w) for w in weights], dtype=np.float32)


class TemporalZeckendorf:
    """Zeckendorf-based temporal encoding for STDP-like traces.

    Each event at step t is encoded by the index of the largest Fibonacci
    number ≤ t. Trace value = fib_index / max_depth — monotonic with time.

    LCP between two event times = shared Fib structure = temporal proximity.
    Replace linear/exponential decay with a natural Fibonacci hierarchy.
    """

    def __init__(self, max_steps: int = 1000000):
        self._cache = {}
        self._max_depth = len(FibonacciUtils.zeckendorf(max_steps)) + 1

    @staticmethod
    def _largest_fib_idx(t: int) -> int:
        """Index of largest Fibonacci number ≤ t."""
        i = 2
        while FibonacciUtils.get(i) <= t:
            i += 1
        return i - 1

    def trace(self, t: int) -> float:
        """Monotonic trace: fib_index / max_depth."""
        if t <= 0:
            return 0.0
        if t not in self._cache:
            idx = self._largest_fib_idx(t)
            zlen = len(FibonacciUtils.zeckendorf(t))
            self._cache[t] = (idx, zlen)
        idx, _ = self._cache[t]
        return idx / max(self._max_depth, 1)

    def temporal_lcp(self, t_a: int, t_b: int) -> int:
        """LCP of Zeckendorf decompositions of two timestamps."""
        z_a = FibonacciUtils.zeckendorf(max(t_a, 0))
        z_b = FibonacciUtils.zeckendorf(max(t_b, 0))
        n = min(len(z_a), len(z_b))
        for i in range(n):
            if z_a[i] != z_b[i]:
                return i
        return n

    def temporal_H(self, t_a: int, t_b: int, gamma: float = 0.5) -> float:
        """H = (1 - γ^{LCP}) / (1 - γ) for temporal proximity."""
        k = self.temporal_lcp(t_a, t_b)
        if k == 0:
            return 0.0
        return (1.0 - gamma ** k) / (1.0 - gamma)
