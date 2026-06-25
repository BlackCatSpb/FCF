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

from eva.symbolic.fibonacci_utils import FibonacciUtils as _FU

# ── LEVELS: adaptive — max Zeckendorf depth for vocabulary ───────
# Computed at import time; falls back to config or 16.

_LEVELS = 0
_GAMMA = 0.5

def _compute_levels_from_vocab(vocab_size: int) -> int:
    """Max Zeckendorf decomposition length for any ID < vocab_size, +1."""
    if vocab_size <= 1:
        return 2
    return len(_FU.zeckendorf(vocab_size - 1)) + 1

def _init():
    global _LEVELS, _GAMMA
    if _LEVELS:
        return
    try:
        from eva.symbolic.fcf_config import FCFConfig as _FCFConfig, EnvironmentResolver
        __cfg = _FCFConfig()
        _GAMMA = getattr(__cfg, 'octree_gamma', 0.618)
        import sentencepiece as spm
        _LEVELS = _compute_levels_from_vocab(
            spm.SentencePieceProcessor(
                model_file=EnvironmentResolver().bpe_model_path
            ).vocab_size()
        )
    except Exception:
        try:
            from eva.symbolic.fcf_config import FCFConfig as _FCFConfig
            __cfg = _FCFConfig()
            _LEVELS = getattr(__cfg, 'path_levels', 16)
            _GAMMA = getattr(__cfg, 'octree_gamma', 0.618)
        except Exception:
            _LEVELS = 16
            _GAMMA = 0.5

_init()

# Public API
LEVELS = _LEVELS
GAMMA = _GAMMA


# ── Zeckendorf path functions ──────────────────────────────────────

def zeckendorf(n: int) -> list[int]:
    """Decompose n into sum of non-consecutive Fibonacci numbers."""
    return _FU.zeckendorf(n)


def path(val: int) -> tuple[int, ...]:
    """Zeckendorf path: Fibonacci numbers from largest to smallest,
    padded/truncated to LEVELS length.

    Returns tuple of LEVELS ints (Fib number or 0 for padding).
    """
    z = zeckendorf(abs(val))
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
