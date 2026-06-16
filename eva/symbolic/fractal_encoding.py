"""
Fractal encoding for FCF — nested octree from decimal digits.

Each concept ID is decomposed into decimal digits.
Each digit → octant (0..7) at that level of the octree.
H[i,j] = Σ γ^l over longest common prefix (LCP) of octant paths.

This replaces the PMI-based H matrix with a deterministic
hierarchical encoding that requires no corpus statistics.
"""

try:
    from eva.symbolic.fcf_config import FCFConfig as _FCFConfig
    __cfg = _FCFConfig()
    LEVELS = __cfg.octree_levels
    GAMMA = __cfg.octree_gamma
except (ImportError, AttributeError):
    LEVELS = 16
    GAMMA = 0.5

def _next_digit(val, pos, seed=99991):
    """Deterministic pseudo-random digit for padding short decimal strings."""
    # MMIX LCG — deterministic, no PYTHONHASHSEED dependency
    state = val * 6364136223846793005 + pos * 1442695040888963407 + seed
    state &= 0x7FFFFFFF
    state = (state * 6364136223846793005 + 1442695040888963407) & 0x7FFFFFFF
    return state % 10

def digits(val, n=LEVELS):
    """Extract n decimal digits from val, padding with PRNG digits if needed."""
    s = str(abs(val))
    ds = [int(c) for c in s]
    while len(ds) < n:
        ds.append(_next_digit(val, len(ds)))
    return ds[:n]

def path(val):
    """Octree path: each of LEVELS digits → octant 0..7"""
    return tuple(d % 8 for d in digits(val))

def lcp(path_a, path_b):
    """Longest common prefix length between two octree paths."""
    n = min(len(path_a), len(path_b))
    for i in range(n):
        if path_a[i] != path_b[i]:
            return i
    return n

def H_weighted(path_a, path_b, gamma=GAMMA):
    """H = (1 - γ^{LCP}) / (1 - γ)."""
    k = lcp(path_a, path_b)
    if k == 0:
        return 0.0
    return (1.0 - gamma ** k) / (1.0 - gamma)


