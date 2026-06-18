"""AdaptiveErrorTracker — isolated per-concept error EMA for self-paced learning."""

from collections import OrderedDict


class AdaptiveErrorTracker:
    """Per-concept prediction error EMA with FIFO eviction.

    Provides dict-like interface for backward compatibility with
    CrystalGenerator.concept_error usage.

    Args:
        decay: EMA decay factor (default 0.9)
        max_size: maximum number of tracked concepts (default 100000)
    """

    def __init__(self, decay=0.9, max_size=100000):
        self.decay = decay
        self.max_size = max_size
        self._data = OrderedDict()

    def update(self, cid, error):
        """Update EMA for a concept ID, then move to end (LRU)."""
        old = self._data.get(cid, error)
        new = self.decay * old + (1 - self.decay) * error
        self._data[cid] = new
        self._data.move_to_end(cid)
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def get(self, cid, default=None):
        return self._data.get(cid, default)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, cid):
        return self._data[cid]

    def __setitem__(self, cid, error):
        self._data[cid] = error

    def __contains__(self, cid):
        return cid in self._data

    def move_to_end(self, cid):
        self._data.move_to_end(cid)

    def popitem(self, last=True):
        return self._data.popitem(last=last)

    def copy(self):
        return self._data.copy()

    def __bool__(self):
        return bool(self._data)

    def __repr__(self):
        return f'AdaptiveErrorTracker({len(self)} items, decay={self.decay})'
