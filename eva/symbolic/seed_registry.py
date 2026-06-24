"""SeedRegistry — единый источник seed для всех RandomState.

    seed(name) = master_seed + hash(name) mod 2^31
    rng(name) = RandomState(seed(name))
"""

import numpy as np
from typing import Dict, Optional


class SeedRegistry:
    """Central registry for all RandomState seeds.

    Usage:
        reg = SeedRegistry(master_seed=42)
        reg.rng('basis')  # → RandomState(42 + hash('basis') % 2**31)
    """

    _NAMES: Dict[str, int] = {}  # shared across instances for collision detection

    def __init__(self, master_seed: int = 42):
        self._master = master_seed
        self._cache: Dict[str, np.random.RandomState] = {}

    def seed(self, name: str) -> int:
        """Deterministic seed for name: master + hash(name), clamped to [0, 2^31)."""
        s = (self._master + hash(name)) % (2**31 - 1)
        prev = self._NAMES.get(name, s)
        if prev != s and prev is not None:
            raise RuntimeError(
                f"Seed collision for '{name}': hash collision (prev={prev}, new={s})")
        self._NAMES[name] = s
        return s

    def rng(self, name: str) -> np.random.RandomState:
        """Return a NEW RandomState instance for name. Never cached.

        Each call creates a fresh RandomState seeded deterministically.
        This ensures two objects using the same name get identical RNGs
        that can advance independently.
        """
        return np.random.RandomState(self.seed(name))


# Default global registry — most code uses this
DEFAULT_REGISTRY = SeedRegistry(master_seed=42)
