"""VSAGrid, VSAConvLayer, VSACNN — Z^8 group algebra prototypes."""

import numpy as np
from eva.symbolic.experimental.vsa_utils import _make_kernel


class VSAGrid:
    """Mapping between flat ℝ^D and mixed-radix grid Z_{s_1} × ... × Z_{s_d}."""

    @staticmethod
    def _factorize(dim, max_radix=8):
        n = dim
        factors = []
        for r in range(max_radix, 1, -1):
            while n % r == 0:
                factors.append(r)
                n //= r
        if n > 1:
            factors.append(n)
        return tuple(sorted(factors, reverse=True)) if factors else (dim,)

    @staticmethod
    def factorize_with_padding(dim, max_radix=8):
        """Find the smallest dim' >= dim that fully factorises with radices <= max_radix."""
        padded = dim
        for _ in range(256):
            factors = VSAGrid._factorize(padded, max_radix)
            if max(factors) <= max_radix:
                return padded, factors
            padded += 1
        return padded, (padded,)

    def __init__(self, dim):
        self.dim = dim
        raw_shape = self._factorize(dim)
        if raw_shape and max(raw_shape) > 8:
            self._padded_dim, self.shape = self.factorize_with_padding(dim)
        else:
            self._padded_dim = dim
            self.shape = raw_shape
        self.ndim = len(self.shape)
        self.strides = [1]
        for s in self.shape[:-1]:
            self.strides.append(self.strides[-1] * s)

    def flat_to_grid(self, idx):
        result = []
        for s, st in zip(self.shape, self.strides):
            result.append((idx // st) % s)
        return tuple(result)

    def grid_to_flat(self, coord):
        idx = 0
        for c, st in zip(coord, self.strides):
            idx += c * st
        return idx

    def fft_along_axis(self, vec, axis=0):
        grid = vec.astype(np.complex128).reshape(self.shape)
        return np.fft.fft(grid, axis=axis).ravel()

    def ifft_along_axis(self, vec, axis=0):
        grid = vec.astype(np.complex128).reshape(self.shape)
        return np.fft.ifft(grid, axis=axis).ravel().real.astype(np.float64)

    def fft_nd(self, vec):
        grid = vec.astype(np.complex128).reshape(self.shape)
        return np.fft.fftn(grid).ravel()

    def ifft_nd(self, vec):
        grid = vec.astype(np.complex128).reshape(self.shape)
        return np.fft.ifftn(grid).ravel().real.astype(np.float64)

    def conv_nd(self, vec, kernel):
        V = self.fft_nd(vec)
        K = self.fft_nd(kernel)
        conv = self.ifft_nd(V * K)
        return conv.real.astype(np.float64)


class VSAConvLayer:
    """One VSA-CNN layer: multi-scale convolution → bundle → normalize."""

    def __init__(self, kx_weights=None, grid=None, dim=768, mode='reflect'):
        if kx_weights is None:
            kx_weights = [(3, 'gaussian', 1.0), (5, 'gaussian', 1.5), (7, 'gaussian', 2.0)]
        self.kx_weights = kx_weights
        self.grid = grid or VSAGrid(dim)
        self.mode = mode

    def forward(self, vec):
        results = []
        for ksize, ktype, sigma in self.kx_weights:
            kernel = _make_kernel(ksize, kernel_type=ktype, sigma=sigma)
            if self.grid.ndim == 1:
                from scipy.ndimage import convolve1d
                conv = convolve1d(vec, kernel.astype(vec.dtype), mode=self.mode)
            else:
                conv = self.grid.conv_nd(vec, np.resize(kernel, self.grid.dim))
            results.append(conv)
        result = results[0].copy()
        for r in results[1:]:
            result = result + r
        nrm = np.linalg.norm(result)
        return result / (nrm + 1e-10) if nrm > 0 else result


class VSACNN:
    """Hierarchical VSA-CNN: stack of VSAConvLayer with increasing scale."""

    def __init__(self, dim=768, n_layers=3):
        self.dim = dim
        self.grid = VSAGrid(dim)
        self.layers = []
        for i in range(n_layers):
            base_ksize = 3 + 2 * i
            kx = [
                (base_ksize, 'gaussian', 0.5 + i * 0.5),
                (base_ksize + 2, 'gaussian', 1.0 + i * 0.5),
                (base_ksize + 4, 'laplacian', 1.0 + i * 0.5),
            ]
            self.layers.append(VSAConvLayer(kx, grid=self.grid))

    def forward(self, vec):
        h = vec.copy()
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def forward_pyramid(self, vec):
        pyramid = [vec.copy()]
        h = vec.copy()
        for layer in self.layers:
            h = layer.forward(h)
            pyramid.append(h.copy())
        return pyramid
