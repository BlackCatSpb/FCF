"""VSA utility functions — kernel generation, convolution, analogy, quantization."""

import numpy as np


def _make_kernel(ksize, kernel_type='uniform', sigma=1.0, freq=0.1):
    x = np.arange(ksize) - ksize // 2
    if kernel_type == 'uniform':
        k = np.ones(ksize)
    elif kernel_type == 'gaussian':
        k = np.exp(-0.5 * (x / sigma) ** 2)
    elif kernel_type == 'laplacian':
        g = np.exp(-0.5 * (x / sigma) ** 2)
        lap = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
        k = lap - lap.mean()
    elif kernel_type == 'gabor':
        gauss = np.exp(-0.5 * (x / sigma) ** 2)
        k = gauss * np.cos(2 * np.pi * freq * x)
    elif kernel_type == 'dog':
        g1 = np.exp(-0.5 * (x / (sigma * 0.6)) ** 2)
        g2 = np.exp(-0.5 * (x / (sigma * 1.6)) ** 2)
        k = g1 - g2
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    kn = np.linalg.norm(k)
    return k / (kn + 1e-10) if kn > 0 else k


def _fractal_convolution(vec, kernel_sizes=(3, 5, 7), mode='reflect',
                         kernel_type='uniform', sigma=1.0):
    from scipy.ndimage import convolve1d
    result = None
    for ksize in kernel_sizes:
        kernel = _make_kernel(ksize, kernel_type=kernel_type, sigma=sigma)
        smoothed = convolve1d(vec, kernel.astype(vec.dtype), mode=mode)
        if result is None:
            result = smoothed.copy()
        else:
            result = result + smoothed
    nrm = np.linalg.norm(result)
    return result / (nrm + 1e-10) if nrm > 0 else result


def _compute_dim_importance(vectors, labels):
    from sklearn.feature_selection import mutual_info_classif
    vecs = np.asarray(vectors, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.int64)
    if vecs.ndim != 2 or len(vecs) < 2:
        return np.ones(vecs.shape[-1] if vecs.ndim == 2 else 768)
    return mutual_info_classif(vecs, labs, random_state=42)


def _analogy(a, b, c, alpha=0.7, eps=1e-8):
    from eva.symbolic.concept_space import _hybrid_unbind, _hybrid_bind
    ratio = _hybrid_unbind(b, a, alpha=alpha, eps=eps)
    d = _hybrid_bind(ratio, c, alpha=alpha, eps=eps)
    return d


def _quantize_adaptive(sim, mean, std, z_score=2.0, max_val=7):
    z = (sim - mean) / (std + 1e-8)
    z = np.clip(z, -z_score, z_score)
    scaled = (z + z_score) / (2 * z_score) * max_val
    return int(round(np.clip(scaled, 0, max_val)))


def _random_masks(dim, n_heads=3, rng=None):
    if rng is None:
        from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
        rng = _R.rng('vsa_masks')
    masks = []
    for _ in range(n_heads):
        m = rng.randn(dim).astype(np.float64) * 0.3 + 0.5
        masks.append(m)
    return masks
