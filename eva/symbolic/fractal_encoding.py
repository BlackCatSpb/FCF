"""
Fractal encoding for FCF — Zeckendorf representation from Fibonacci numbers.

Each concept ID is decomposed into a sum of non-consecutive Fibonacci numbers
(Zeckendorf's theorem). This gives a hierarchical path where longer common
prefixes directly correspond to shared Fibonacci structure — semantically
closer than arbitrary base-8 octree paths.

H[i,j] = Σ γ^l over longest common prefix (LCP) of Zeckendorf paths.

Replaces the previous octree (base-8 digit) encoding with a Fibonacci-based
hierarchy that naturally encodes nested structure.
"""

try:
    from eva.symbolic.fcf_config import FCFConfig as _FCFConfig
    __cfg = _FCFConfig()
    LEVELS = __cfg.octree_levels
    GAMMA = __cfg.octree_gamma
except (ImportError, AttributeError):
    LEVELS = 16
    GAMMA = 0.5

from eva.symbolic.fibonacci_utils import FibonacciUtils as _FU


def _zeckendorf(n: int) -> list[int]:
    """Decompose n into sum of non-consecutive Fibonacci numbers (Zeckendorf)."""
    return _FU.zeckendorf(n)


def path(val: int) -> tuple[int, ...]:
    """Zeckendorf path: Fibonacci numbers from largest to smallest,
    padded/truncated to LEVELS length.

    Returns tuple of LEVELS ints (Fib number or 0 for padding).
    """
    z = _zeckendorf(abs(val))
    if len(z) >= LEVELS:
        return tuple(z[:LEVELS])
    return tuple(z + [0] * (LEVELS - len(z)))


def lcp(path_a: tuple, path_b: tuple) -> int:
    """Longest common prefix length between two Zeckendorf paths."""
    n = min(len(path_a), len(path_b))
    for i in range(n):
        if path_a[i] != path_b[i]:
            return i
    return n


def H_weighted(path_a: tuple, path_b: tuple, gamma: float = GAMMA) -> float:
    """H = (1 - γ^{LCP}) / (1 - γ)."""
    k = lcp(path_a, path_b)
    if k == 0:
        return 0.0
    return (1.0 - gamma ** k) / (1.0 - gamma)
