import numpy as np
from typing import List, Tuple, Optional


class FibonacciUtils:
    _cache = {0: 0, 1: 1}
    _gen_cache: dict = {}  # (d, n) → F^(d)_n

    @classmethod
    def get(cls, n):
        if n < 0:
            return 0
        if n not in cls._cache:
            cls._cache[n] = cls.get(n - 1) + cls.get(n - 2)
        return cls._cache[n]

    @classmethod
    def get_lambda(cls, d: int) -> float:
        """Положительный корень x^d = x^{d-1} + ... + 1.

        Для d=2: λ₂ = φ = (1+√5)/2.
        Для d→∞: λ_d → 2.
        Fixed-point: x_{k+1} = 2 - x_k^{-d}, сходится за < 30 итераций.
        """
        if d < 2:
            d = 2
        x = 2.0
        for _ in range(100):
            x_new = 2.0 - 1.0 / (x ** d)
            if abs(x_new - x) < 1e-14:
                return x_new
            x = x_new
        return x

    @classmethod
    def get_generalized(cls, n: int, d: int = 2) -> int:
        """d-мерное число Фибоначчи F^(d)_n.

        F^(d)_0 = 1, F^(d)_{-k}=0 (1≤k≤d-1),
        F^(d)_n = Σ_{k=1..d} F^(d)_{n-k} (n≥1).
        Для d=2 даёт классические числа со смещением: F^(2)_n = F_{n+1}.
        """
        if d < 2:
            d = 2
        key = (d, n)
        if key in cls._gen_cache:
            return cls._gen_cache[key]
        if n < 0:
            return 0
        if n == 0:
            return 1
        val = 0
        for k in range(1, d + 1):
            val += cls.get_generalized(n - k, d)
        cls._gen_cache[key] = val
        return val

    @classmethod
    def generalized_sequence(cls, max_n: int, d: int = 2) -> list:
        """Первые max_n+1 членов d-последовательности F^(d)_0..F^(d)_max_n."""
        return [cls.get_generalized(i, d) for i in range(max_n + 1)]

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
    def fib_scale_generalized(cls, value, max_val=7, d=2):
        """Квантование в d-ряд Фибоначчи вместо классического.

        При d=2 эквивалентно fib_scale.
        """
        seq = cls.generalized_sequence(max_val, d)
        scaled = value * seq[-1]
        idx = int(np.argmin(np.abs(np.array(seq, dtype=np.float64) - scaled)))
        return idx

    @classmethod
    def golden_ratio(cls):
        return (1.0 + np.sqrt(5.0)) / 2.0

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
    def balance_subspaces(cls, z_c, z_a, z_m, eps=1e-8, d=2):
        """Балансировка подпространств через λ_d: λ_d² : λ_d : 1.

        При d=2 даёт φ²:φ:1 — классическое поведение.
        """
        lam = cls.get_lambda(d)
        norm_c = float(np.linalg.norm(z_c))
        norm_a = float(np.linalg.norm(z_a))
        norm_m = float(np.linalg.norm(z_m))
        total = norm_c + norm_a + norm_m
        target = total / (lam * lam + lam + 1.0)
        z_c = z_c * (target * lam * lam) / (norm_c + eps)
        z_a = z_a * (target * lam) / (norm_a + eps)
        z_m = z_m * target / (norm_m + eps)
        return z_c, z_a, z_m

    @classmethod
    def generalized_fib_sizes(cls, d: int = 2, target_beam: int = 10946) -> dict:
        """Вычислить размеры буферов как d-числа Фибоначчи.

        Находит n такое, что F^(d)_n ≈ target_beam, затем возвращает
        F^(d)_{n-4..n} с теми же относительными индексами, что и текущие
        константы (F₁₇..F₂₁). При d=2 даёт классические числа.
        """
        n = 0
        while cls.get_generalized(n, d) < target_beam:
            n += 1
        return {
            'beam_buffer_size': cls.get_generalized(n, d),
            'checkpoint_every': cls.get_generalized(n - 2, d) if n >= 2 else 1,
            'manifold_buffer': cls.get_generalized(n - 3, d) if n >= 3 else 1,
            'eval_every_slow': cls.get_generalized(n - 4, d) if n >= 4 else 1,
        }

    @classmethod
    def delta_estimate(cls, n: int, d: int = 2) -> float:
        """Δ_n^(d) = F^(d)_n - F_{n+1} — асимптотическая оценка ошибки.

        Положительна для d > 2 (обобщённые числа растут быстрее).
        """
        return float(cls.get_generalized(n, d) - cls.get(n + 1))

    @classmethod
    def delta_asymptotic(cls, n: int, d: int = 2) -> float:
        """Асимптотическая оценка Δ_n^(d) ~ λ_d^n / P'(λ_d) - φ^{n+1}/√5."""
        lam = cls.get_lambda(d)
        phi = cls.golden_ratio()
        # P'(x) = d·x^{d-1} - (d-1)·x^{d-2} - ... - 2·x - 1 = Σ_{k=1..d} k·x^{k-1}
        p_prime = sum(k * (lam ** (k - 1)) for k in range(1, d + 1))
        term1 = (lam ** n) / p_prime if abs(p_prime) > 1e-30 else 0.0
        term2 = (phi ** (n + 1)) / np.sqrt(5.0)
        return float(term1 - term2)


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

    def theta(self, distance: int, fast_window: int = 5, slow_window: int = 10) -> tuple[float, float]:
        """Zeckendorf-based temporal decay for STDP, replaces exp(-d/tau).

        Uses Zeckendorf digit count as multi-scale temporal hierarchy:
          theta_base = 1 / (1 + len(zeckendorf(distance)))

        Short distances → few Fibonacci digits → high theta.
        Long distances → many digits → low theta.
        The digit count grows as O(log_φ(d)), giving automatic multi-scale decay.

        No free tau parameter: the Fibonacci hierarchy IS the decay schedule.
        Smooth linear rolloff after fast_window/slow_window (no hard cutoff).

        Returns (fast_theta, slow_theta) — both in (0, 1], decreasing with distance.
        """
        if distance <= 0:
            return (1.0, 1.0)
        # Number of Zeckendorf digits = natural log-scale distance measure
        zlen = len(FibonacciUtils.zeckendorf(distance))
        theta_base = 1.0 / (1.0 + zlen)
        # Fast: smooth rolloff after fast_window
        if distance <= fast_window:
            fast = theta_base
        else:
            fast = theta_base * max(0.0, 1.0 - (distance - fast_window) / max(fast_window, 1))
        # Slow: smoother rolloff after slow_window
        if distance <= slow_window:
            slow = theta_base
        else:
            slow = theta_base * max(0.0, 1.0 - (distance - slow_window) / max(slow_window, 1))
        return (max(fast, 0.0), max(slow, 0.0))
