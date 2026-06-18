"""RNGRegistry — shared thread-local RNG registry for reproducibility."""

import random
import hashlib


class RNGRegistry:
    """Thread-local registry of named RNGs with deterministic sub-seeding.

    Each named RNG is derived from the master seed using a deterministic
    hash of the name, so adding/removing names does not affect other RNGs.

    Args:
        master_seed: base seed for all registered RNGs
        rng_factory: callable that returns a new RNG instance (default random.Random)
    """

    def __init__(self, master_seed=42, rng_factory=None):
        self._master_seed = master_seed
        self._rng_factory = rng_factory or (lambda s: random.Random(s))
        self._rngs = {}

    def get(self, name):
        """Get (or create) the named RNG."""
        if name not in self._rngs:
            sub_seed = self._seed_for(name)
            self._rngs[name] = self._rng_factory(sub_seed)
        return self._rngs[name]

    def _seed_for(self, name):
        """Deterministic integer seed derived from master_seed and name."""
        h = hashlib.sha256(f'{self._master_seed}:{name}'.encode()).hexdigest()
        return int(h[:8], 16)

    def reset_all(self, master_seed=None):
        """Reset all RNGs (optionally with a new master_seed)."""
        if master_seed is not None:
            self._master_seed = master_seed
        self._rngs.clear()

    def reset(self, name):
        """Reset a single named RNG."""
        self._rngs.pop(name, None)

    @property
    def names(self):
        return list(self._rngs.keys())
