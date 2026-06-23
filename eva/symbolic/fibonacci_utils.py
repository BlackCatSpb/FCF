import numpy as np

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
