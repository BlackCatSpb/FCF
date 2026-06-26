"""DimensionCoordinator — single source of truth for component dimensions.

Adaptive: vec_dim computed from vocab_size + VRAM limit.
All other dims derived from vec_dim + latent ratio.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np


class DimensionError(ValueError):
    """Inconsistent dimensions between components."""


class AdaptiveDimensionResolver:
    """Compute vec_dim and latent_dim from vocab_size + VRAM.

    vec_dim = power-of-2 in [min_dim, max_dim] bounded by VRAM.
    latent_dim = int(vec_dim * ratio).
    """

    def __init__(self, vocab_size, vram_limit_mb=2048, latent_ratio=2.67):
        self.vocab_size = vocab_size
        self.vram_limit_mb = vram_limit_mb
        self.latent_ratio = latent_ratio

    @property
    def min_vec_dim(self):
        """SNR lower bound: D >= 10*log2(V) for reliable unbind."""
        return max(64, int(10 * math.log2(max(self.vocab_size, 2))))

    @property
    def max_vec_dim(self):
        """VRAM upper bound: vecs(V,D) + codes(V, L) + momentum + field < limit."""
        d = 2048
        L = int(d * self.latent_ratio)
        for _ in range(10):
            total_mb = self._vram_estimate(d, L, self.vocab_size)
            if total_mb < self.vram_limit_mb * 0.9:
                return d
            d = self._prev_power_of_2(d)
            L = int(d * self.latent_ratio)
        return max(self.min_vec_dim, 64)

    @staticmethod
    def _vram_estimate(d, L, V):
        vecs = V * d * 2          # fp16
        codes = V * L * 2         # bf16
        basis = L * d * 4         # fp32
        mom = V * d * 2           # bf16
        fb = V * ((L + 7) // 8)   # uint8
        return (vecs + codes + basis + mom + fb) / 1024**2

    @staticmethod
    def _prev_power_of_2(x):
        return 2 ** int(math.log2(x))

    @staticmethod
    def _next_power_of_2(x):
        return 2 ** int(math.ceil(math.log2(x)))

    @property
    def vec_dim(self):
        low = self.min_vec_dim
        high = self.max_vec_dim
        # Pick largest power-of-2 in [low, high]
        dim = self._prev_power_of_2(high)
        if dim < low:
            dim = self._next_power_of_2(low)
        return dim

    @property
    def latent_dim(self):
        ld = int(self.vec_dim * self.latent_ratio)
        # Align to multiple of 8 for VSAGrid alignment
        return ((ld + 7) // 8) * 8

    @property
    def grid_shape(self):
        from eva.symbolic.experimental.vsa_grid import VSAGrid
        g = VSAGrid(self.vec_dim)
        return g.shape

    @property
    def padded_dim(self):
        from eva.symbolic.experimental.vsa_grid import VSAGrid
        g = VSAGrid(self.vec_dim)
        return getattr(g, '_padded_dim', self.vec_dim)


@dataclass(frozen=True)
class DimensionCoordinator:
    """Single source of truth for all FCF component dimensions.

    vec_dim — concept vector dimension (hyper-sphere).
    latent_dim — fractal code dimension (all internal components).

    EntityField and Harmonizer operate directly at latent_dim.
    No separate entity_dim / harm_dim.
    """
    vec_dim: int = 768
    latent_dim: int = 2048
    fib_dimension: int = 2
    use_fib_generalized: bool = False

    def __post_init__(self):
        if self.vec_dim % 8 != 0:
            raise DimensionError(f"vec_dim={self.vec_dim} must be divisible by 8")
        if self.vec_dim > self.latent_dim:
            raise DimensionError(
                f"vec_dim={self.vec_dim} > latent_dim={self.latent_dim}")

    @classmethod
    def from_vocab(cls, vocab_size, vram_limit_mb=2048, latent_ratio=2.67,
                   fib_dimension=2, use_fib_generalized=False):
        r = AdaptiveDimensionResolver(vocab_size, vram_limit_mb, latent_ratio)
        return cls(vec_dim=r.vec_dim, latent_dim=r.latent_dim,
                   fib_dimension=fib_dimension, use_fib_generalized=use_fib_generalized)

    @property
    def entity_dim(self):
        """EntityField operates in latent space."""
        return self.latent_dim

    @property
    def harm_dim(self):
        """Harmonizer operates in latent space."""
        return self.latent_dim

    def make_projection(self, from_dim: int, to_dim: int, seed: int | None = None):
        """Johnson-Lindenstrauss projection between dimensions.

        Returns a callable that projects vectors from from_dim to to_dim.
        Result is cached via lru_cache on (from_dim, to_dim, seed).
        """
        if from_dim == to_dim:
            return lambda v: v
        if seed is None:
            from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
            seed = _R.seed('jl_projector')
        return _jl_projector(from_dim, to_dim, seed)

    @property
    def subspace(self) -> dict:
        from eva.symbolic.fibonacci_utils import FibonacciUtils
        if self.use_fib_generalized and self.fib_dimension >= 2:
            lam = FibonacciUtils.get_lambda(self.fib_dimension)
        else:
            lam = FibonacciUtils.golden_ratio()
        total = lam * lam + lam + 1.0
        l_c = max(8, int(self.latent_dim * lam * lam / total))
        l_a = max(8, int(self.latent_dim * lam / total))
        return {
            'l_c': l_c,
            'l_a': l_a,
            'l_m': self.latent_dim - l_c - l_a,
        }


@lru_cache(maxsize=16)
def _jl_projector(from_dim: int, to_dim: int, seed: int):
    rng = np.random.RandomState(seed)
    scale = 1.0 / np.sqrt(from_dim)
    proj = rng.randn(to_dim, from_dim).astype(np.float32) * scale
    return lambda v: proj @ v
