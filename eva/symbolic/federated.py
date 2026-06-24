"""FederatedAggregator — weighted ensemble of EntityField memories with DP."""

import numpy as np


class FederatedAggregator:
    """Aggregate multiple EntityField memories into one global memory.

    Supports weighted averaging with differential privacy noise.
    """

    @staticmethod
    def aggregate(fields, weights=None, noise_scale=0.01):
        """Weighted average of concept vectors across fields.

        Args:
            fields: list of EntityField instances
            weights: optional list of float weights (default: equal)
            noise_scale: standard deviation of Gaussian DP noise (default 0.01)

        Returns:
            dict {cid: ndarray} of aggregated vectors, or None if no common concepts
        """
        if not fields:
            return None

        if weights is None:
            weights = [1.0] * len(fields)
        w_sum = sum(weights)
        if w_sum > 0:
            weights = [w / w_sum for w in weights]

        # Find common concepts across all fields
        common_ids = set(fields[0].entities.keys())
        for f in fields[1:]:
            common_ids &= set(f.entities.keys())

        if not common_ids:
            # Partial overlap: average over existing only
            common_ids = set(fields[0].entities.keys())
            for f in fields[1:]:
                common_ids |= set(f.entities.keys())
            if not common_ids:
                return None

        dim = fields[0].dim
        result = {}
        for cid in common_ids:
            aggregated = np.zeros(dim, dtype=np.float32)
            total_w = 0.0
            for field, w in zip(fields, weights):
                vec = field.entities.get(cid)
                if vec is not None:
                    v = vec.astype(np.float32) if hasattr(vec, 'astype') else np.asarray(vec, dtype=np.float32)
                    aggregated += w * v
                    total_w += w
            if total_w > 0:
                aggregated /= total_w
            # Differential privacy noise
            aggregated += np.random.normal(0, noise_scale, size=dim).astype(np.float32)
            an = float(np.linalg.norm(aggregated))
            if an > 1e-10:
                aggregated /= an
            result[cid] = aggregated

        return result
