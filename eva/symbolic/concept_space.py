"""ConceptSpace — vector space for BPE-token concepts (FCF).

VSA-операции формализованы как групповая алгебра ℝ[G] на ℤ₈^d:
  - bind = convolution on group (FFT-HRR), hybrid α=0.7
  - permute = сдвиг на элемент группы (циклический)
  - bundle = суперпозиция функций на G
  - conv_nd = многомерное FFT на mixed-radix решётке (768=8×8×6×2)

Ключевые компоненты:
  - ConceptSpace — 146K BPE-концептов на 768D гиперсфере
  - EntityField — рекурсивное семантическое поле char↔morph↔word↔sent↔para
  - Harmonizer — морфемная гармонизация через VSA compose/decompose
  - VSAGrid — flat ↔ ℤ₈^d отображение + FFT по каждой оси
  - VSACNN — иерархическая VSA-свёртка (5 типов ядер)
  - FractalField — латентные коды z ∈ ℝ^{2048} → вектор на сфере

Обучение: STDP + негативная выборка + контрастивная цель + L1.
Никаких трансформеров. Никакого обратного распространения.
"""

import numpy as np
from collections import defaultdict, Counter
import math, json, os, random
import threading
from typing import Dict, List, Optional
from eva.symbolic.dimension_coordinator import DimensionCoordinator
from eva.symbolic.adaptive_controller import AdaptiveArchitectureController
from eva.symbolic.fcf_config import FCFConfig

# ── FFT-HRR VSA primitives ─────────────────────────────────────
# Circular convolution (bind) and circular correlation (unbind)
# via FFT for real-valued unit-norm vectors.
# Unlike Hadamard product a*b, FFT-HRR is (approximately) invertible:
#   unbind(bind(a, b), b) ≈ a   with SNR ~ sqrt(D).

def _hrr_bind(a, b):
    """FFT-HRR bind = circular convolution a ⊛ b."""
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    return np.fft.irfft(fa * fb, n=len(a)).astype(a.dtype)

def _hrr_unbind(c, b):
    """FFT-HRR unbind = circular correlation c ⋆ b."""
    fc = np.fft.rfft(c)
    fb_conj = np.conj(np.fft.rfft(b))
    return np.fft.irfft(fc * fb_conj, n=len(c)).astype(c.dtype)

# ── Hybrid bind/unbind (HRR + element-wise, §5 Training Dynamics V18) ──

# P1.2: α curriculum — starts HRR-heavy for invertibility, decays for expressivity
_ALPHA_EPOCH = 0
_ALPHA_TOTAL = 0
_ALPHA_DECAY = 'exp'  # 'linear' or 'exp'

def _set_alpha_curriculum(epoch, total_epochs, decay='exp'):
    global _ALPHA_EPOCH, _ALPHA_TOTAL, _ALPHA_DECAY
    _ALPHA_EPOCH = epoch
    _ALPHA_TOTAL = total_epochs
    _ALPHA_DECAY = decay

def _alpha_from_curriculum(alpha_max=None, alpha_min=None, decay_rate=None):
    from eva.symbolic.fcf_config import FCFConfig
    _fc = FCFConfig().formula
    alpha_max = alpha_max if alpha_max is not None else _fc.hybrid_alpha_max
    alpha_min = alpha_min if alpha_min is not None else _fc.hybrid_alpha_min
    decay_rate = decay_rate if decay_rate is not None else _fc.hybrid_alpha_decay_rate
    if _ALPHA_TOTAL > 0 and _ALPHA_EPOCH > 0:
        t = _ALPHA_EPOCH / _ALPHA_TOTAL
        if _ALPHA_DECAY == 'exp':
            return alpha_min + (alpha_max - alpha_min) * math.exp(-decay_rate * t)
        else:
            return alpha_max * (1 - t) + alpha_min
    return _fc.hybrid_bind_alpha  # default fallback

def _hybrid_bind(a, b, alpha=None, eps=1e-8):
    """Гибрид HRR ⊛ и element-wise: alpha*hrr + (1-alpha)*ew."""
    if alpha is None:
        alpha = _alpha_from_curriculum()
    A = np.fft.rfft(a)
    B = np.fft.rfft(b)
    hrr = np.fft.irfft(A * B, n=len(a))
    ew = a * b
    combined = alpha * hrr + (1 - alpha) * ew
    nrm = np.linalg.norm(combined)
    return combined / (nrm + eps) if nrm > 0 else combined

def _hybrid_unbind(c, b, alpha=None, eps=1e-8):
    """Гибрид HRR correlation и element-wise unbind."""
    if alpha is None:
        alpha = _alpha_from_curriculum()
    fc = np.fft.rfft(c)
    fb_conj = np.conj(np.fft.rfft(b))
    hrr = np.fft.irfft(fc * fb_conj, n=len(c))
    ew = c * b
    combined = alpha * hrr + (1 - alpha) * ew
    nrm = np.linalg.norm(combined)
    return combined / (nrm + eps) if nrm > 0 else combined

def _bind_weighted_zeckendorf(vec, weight, max_val=7):
    """Структурированное взвешивание через Zeckendorf-разложение веса.
    Вес w ∈ [0, max_val] раскладывается на сумму непоследовательных чисел
    Фибоначчи, каждое применяется как bind(vec, scale(sub_vec)) и bundle.
    """
    from eva.symbolic.fibonacci_utils import FibonacciUtils
    w_clamped = max(0, min(max_val, int(round(weight))))
    tree = FibonacciUtils.zeckendorf(w_clamped)
    if sum(tree) != w_clamped:
        tree.append(w_clamped - sum(tree))
    result = None
    for part in tree:
        scale = part / tree[0] if tree else 1.0
        sub = vec * scale
        sn = np.linalg.norm(sub)
        if sn > 1e-10:
            sub /= sn
        bound = _hybrid_bind(vec, sub)
        result = bound if result is None else result + bound
    if result is None:
        return vec.copy()
    rn = np.linalg.norm(result)
    return result / (rn + 1e-10) if rn > 0 else result


# ── VSA utility functions (from Фибоначчи.txt analysis) ──

def _hybrid_bind_masked(a, b, mask, threshold=0.5, alpha=0.7, eps=1e-8):
    """Селективный hybrid bind: только измерения где mask > threshold."""
    mask_bin = (np.asarray(mask, dtype=np.float64) > threshold).astype(np.float64)
    bound = _hybrid_bind(a, b, alpha=alpha, eps=eps)
    result = a.copy()
    result[mask_bin > 0] = bound[mask_bin > 0]
    nrm = np.linalg.norm(result)
    return result / (nrm + eps) if nrm > 0 else result


# ── Experimental VSA utilities (defined in eva/symbolic/experimental/) ──
# Imported lazily to keep concept_space.py focused on core training pipeline.
try:
    from eva.symbolic.experimental import (  # noqa: F401
        VSAGrid, VSAConvLayer, VSACNN, ResidueEncoder,
        _make_kernel, _fractal_convolution, _compute_dim_importance,
        _analogy, _quantize_adaptive, _random_masks,
    )
except ImportError:
    pass


try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


def _hybrid_bind_torch(a, b, alpha=0.7, eps=1e-8):
    """Batch GPU hybrid bind — 4.4× faster than np FFT per V20."""
    A = torch.fft.rfft(a)
    B = torch.fft.rfft(b)
    hrr = torch.fft.irfft(A * B, n=a.shape[-1])
    ew = a * b
    combined = alpha * hrr + (1 - alpha) * ew
    nrm = combined.norm(dim=-1, keepdim=True)
    return combined / nrm.clamp(min=eps)


class ConceptVectorStore:
    """Dense ndarray-backed store with dict-like interface.

    Supports .get(), .keys(), .items(), .values(), len(), 'in', iteration.
    Internal: self._data[V, dim] float32, self._valid[V] bool.
    """
    __slots__ = ('_data', '_valid', '_V')

    def __init__(self, V, dim):
        self._V = V
        self._data = np.zeros((V, dim), dtype=np.float32)
        self._valid = np.zeros(V, dtype=bool)

    @property
    def data(self):
        return self._data

    @property
    def valid(self):
        return self._valid

    @property
    def size(self):
        return self._V

    def __getitem__(self, cid):
        if self._valid[cid]:
            return self._data[cid]
        return None

    def __setitem__(self, cid, v):
        self._data[cid] = v
        self._valid[cid] = True

    def get(self, cid, default=None):
        if 0 <= cid < self._V and self._valid[cid]:
            return self._data[cid]
        return default

    def keys(self):
        return np.where(self._valid)[0].tolist()

    def items(self):
        valid = np.where(self._valid)[0]
        for cid in valid:
            yield cid, self._data[cid]

    def values(self):
        return self._data[self._valid]

    def __len__(self):
        return int(self._valid.sum())

    def __contains__(self, cid):
        return 0 <= cid < self._V and self._valid[cid]

    def __iter__(self):
        return iter(np.where(self._valid)[0])


class FractalField:
    """Fractal computation matrix for relative concept vector space.

    Code split into subspaces:
        z = [z_c | z_a | z_m]
        z_c: concept identity (slow plasticity)
        z_a: attention/context mask (fast plasticity)
        z_m: meta-plasticity (modulates learning)

    v = normalize(code @ basis) — unchanged.
    """

    def __init__(self, dim=None, latent_dim=None, l_c=None, l_a=None, l_m=None, l1_lambda=None,
                 n_field_bits=None, field_lr=None, max_latent_dim=None, arch_controller=None):
        from eva.symbolic.fcf_config import FCFConfig
        _c = FCFConfig()
        self.dim = dim if dim is not None else _c.dim
        self.latent_dim = latent_dim if latent_dim is not None else _c.latent_dim
        self.max_latent_dim = max_latent_dim or self.latent_dim * _c.fractal_max_latent_dim_mult
        self.l1_lambda = l1_lambda if l1_lambda is not None else _c.fractal_l1_lambda

        # Adaptive architecture controller (reads from FCFConfig.subspace_*)
        self.arch = arch_controller or AdaptiveArchitectureController(latent_dim=self.latent_dim)
        if l_c is not None and l_a is not None and l_m is not None:
            self.arch.l_c_ratio = l_c / self.latent_dim
            self.arch.l_a_ratio = l_a / self.latent_dim
            self.arch.l_m_ratio = l_m / self.latent_dim
        self.l_c = self.arch.l_c
        self.l_a = self.arch.l_a
        self.l_m = self.arch.l_m

        # Fractal basis: (latent_dim, dim) with orthonormal columns
        from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
        mat = _R.rng('basis').randn(self.latent_dim, self.dim).astype(np.float32)
        Q, _ = np.linalg.qr(mat, mode='reduced')
        self.basis = Q.astype(np.float32)
        # Latent codes: cid → (latent_dim,) array
        self.codes = {}

        # Learnable field projection: code @ W_proj → binarized field bits
        self.n_field_bits = n_field_bits if n_field_bits is not None else _c.fractal_n_field_bits
        self.field_lr = field_lr if field_lr is not None else _c.fractal_field_lr
        self.W_proj: Optional[np.ndarray] = None  # [latent_dim, n_field_bits]
        self.field_bits: Dict[int, np.ndarray] = {}
        self._fb_dirty = False

        # HDC n-gram memory: prefix_cids_tuple → bundled latent repr (LRU-capped)
        self.hdc_memory: Dict[tuple, np.ndarray] = {}
        self.hdc_memory_counts: Dict[tuple, int] = {}
        self.hdc_memory_max = FCFConfig().fractal_hdc_memory_max
        self._hdc_access_order: List[tuple] = []  # (unused after P3.6 LFU, kept for compat)
        self._capacity_lock = threading.Lock()

        # Per-concept adaptive L1 lambda (dynamic dimensionality)
        self.l1_lambda_per_cid: Dict[int, float] = {}
        self.l1_target_density = self.arch.l1_target_density
        self.l1_density_window: Dict[int, list] = {}

        # Cache
        self._vector_matrix = None
        self._cid_order = []
        self._matrix_dirty = True

        # Dynamic capacity growth tracking
        self._capacity_growths = 0

        # Sector index for focal search (field-in-field)
        self._sector_W: List[np.ndarray] = []  # per-level W_proj
        self._sector_index: Dict[int, Dict[tuple, list]] = {}  # depth → {prefix → [cids]}
        self._sector_depths: list = self.arch.sector_depths

    def _apply_l1(self, code: np.ndarray, ce: float = 0.0, cid: Optional[int] = None) -> np.ndarray:
        """Soft-threshold z_c subspace: high CE → weak L1 (allows densification).

        Uses per-concept L1 lambda when available (adaptive dimensionality).
        """
        if self.l1_lambda <= 0:
            return code
        if cid is not None and cid in self.l1_lambda_per_cid:
            lmbda = self.l1_lambda_per_cid[cid]
        else:
            lmbda = self.l1_lambda
        strength = lmbda * max(0.0, 1.0 - ce * 2.0)
        z_c = code[:self.l_c]
        active_before = int(np.sum(np.abs(z_c) > 1e-6))
        z_c = np.sign(z_c) * np.maximum(0.0, np.abs(z_c) - strength)
        code[:self.l_c] = z_c
        # Track density for adaptive adjustment
        if cid is not None:
            active_after = int(np.sum(np.abs(z_c) > 1e-6))
            density = active_after / max(self.l_c, 1)
            if cid not in self.l1_density_window:
                self.l1_density_window[cid] = []
            self.l1_density_window[cid].append(density)
            # Keep last 100 measurements
            if len(self.l1_density_window[cid]) > 100:
                self.l1_density_window[cid].pop(0)
        return code

    def _apply_l1_batch(self, codes: np.ndarray, ce_list: list, cid_list: Optional[list] = None) -> np.ndarray:
        """Batched L1 for GPU path."""
        if self.l1_lambda <= 0 or len(codes) == 0:
            return codes
        for i in range(len(codes)):
            cid = cid_list[i] if cid_list is not None and i < len(cid_list) else None
            self._apply_l1(codes[i], ce_list[i] if i < len(ce_list) else 0.0, cid=cid)
        return codes

    def check_basis_health(self):
        """Verify orthogonality, re-orthogonalize if drifted. Returns True if changed."""
        QtQ = self.basis.T @ self.basis
        err = np.max(np.abs(QtQ - np.eye(self.dim, dtype=np.float32)))
        if err > 1e-3:
            Q, _ = np.linalg.qr(self.basis, mode='reduced')
            old_basis = self.basis
            cids = list(self.codes.keys())
            codes_mat = np.stack([self.codes[cid] for cid in cids])
            vecs = codes_mat @ old_basis
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            vecs /= norms
            new_codes = vecs @ Q.T
            for i, cid in enumerate(cids):
                self.codes[cid] = new_codes[i]
            self.basis = Q.astype(np.float32)
            self._matrix_dirty = True
            return True
        return False

    # ── Init ─────────────────────────────────────────────────

    def init_concept(self, cid, rng_seed=None):
        """Initialize a concept with split subspace code.

        z_c: sparse identity pattern (~3% active)
        z_a: small noise (context attention starts neutral)
        z_m: near zero (meta-gates start open)
        """
        from eva.symbolic.fcf_config import FCFConfig
        _fi = FCFConfig()
        seed = rng_seed if rng_seed is not None else cid * 137 + 42
        rng = np.random.RandomState(abs(seed) % (2**31))

        z = np.zeros(self.latent_dim, dtype=np.float32)

        # z_c: sparse identity (~3% active, room to grow via STDP)
        n_active = max(int(self.l_c * _fi.fractal_init_z_c_active_pct), _fi.fractal_init_z_c_active_min)
        idxs = rng.choice(self.l_c, n_active, replace=False)
        vals = rng.randn(n_active).astype(np.float32)
        z[:self.l_c][idxs] = vals

        # z_a: small noise
        z[self.l_c:self.l_c + self.l_a] = rng.randn(self.l_a).astype(np.float32) * _fi.fractal_init_z_a_scale

        # z_m: near zero — meta gates start neutral
        z[self.l_c + self.l_a:] = rng.randn(self.l_m).astype(np.float32) * _fi.fractal_init_z_m_scale

        # Rescale so |code @ basis| = 1
        v_raw = z @ self.basis
        scale = 1.0 / (np.linalg.norm(v_raw) + 1e-10)
        z *= scale

        self.codes[cid] = z
        self._matrix_dirty = True
        return self.compute_vector(cid)

    # ── Field bits ───────────────────────────────────────────

    def init_fields(self, n_anchors=None):
        """Initialize binary field bit arrays for all concepts.

        field_bits[cid] = np.uint8 array of n_anchors/8 bytes.
        Used by Zeckendorf path (build_zeckendorf_fields).
        """
        from eva.symbolic.fcf_config import FCFConfig
        n_anchors = n_anchors or FCFConfig().fractal_init_field_n_anchors
        self.field_bits = {}
        n_bytes = (n_anchors + 7) // 8
        for cid in self.codes:
            self.field_bits[cid] = np.zeros(n_bytes, dtype=np.uint8)
        self._fb_dirty = True

    def init_learned_fields(self, field_bits=512):
        """Initialize W_proj and compute field_bits from latent codes.

        W_proj: random hyperplane projection matrix [latent_dim, field_bits].
        Each field bit = sign(code @ W_proj[:, i]) → LSH-preserving similarity.
        field_bits[cid] packed as np.uint8 array (bitmask).
        """
        from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
        scale = 1.0 / np.sqrt(self.latent_dim)
        self.W_proj = _R.rng('field_bits').randn(self.latent_dim, field_bits).astype(np.float32) * scale
        self.n_field_bits = field_bits
        self._init_sector_fields()
        self._rebuild_field_bits()
        print(f"  Learnable fields: {len(self.field_bits)}/{len(self.codes)} "
              f"concepts, {field_bits} bits ({len(self._sector_depths)} levels)")
        depths_str = ', '.join(str(d) for d in self._sector_depths)
        print(f"  Sector depth bits: [{depths_str}]")

    def _rebuild_field_bits(self):
        """Recompute field_bits from current codes using W_proj."""
        if self.W_proj is None:
            return
        n_bytes = (self.n_field_bits + 7) // 8
        self.field_bits = {}
        for cid, code in self.codes.items():
            raw = code @ self.W_proj  # [n_field_bits]
            bits = (raw > 0).astype(np.uint8)
            packed = np.packbits(bits)[:n_bytes]
            self.field_bits[cid] = packed
        self._rebuild_sector_index()
        self._fb_dirty = True

    def update_learned_fields(self, batches_seen=0, lr_scale=1.0):
        """Periodic field update with Hebbian W_proj adaptation.

        W_proj update: outer product of code with sign(code @ W_proj).
        Strengthens hyperplanes aligned with concept distribution.
        """
        if self.W_proj is None or len(self.codes) < 2:
            return

        codes = np.array(list(self.codes.values()), dtype=np.float32)
        n = len(codes)
        raw = codes @ self.W_proj  # [n, n_field_bits]
        signs = np.sign(raw)  # ±1

        # Hebbian: W += lr * mean(code * sign) over batch
        lr = self.field_lr * lr_scale
        delta = (codes.T @ signs) / max(n, 1)  # [latent_dim, n_field_bits]
        self.W_proj += lr * delta

        # Re-normalize columns to unit length (prevent drift)
        norms = np.linalg.norm(self.W_proj, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.W_proj /= norms

        # P1.3a: QR orthogonalization of W_proj columns — prevents Hebbian collapse
        Q_w, _ = np.linalg.qr(self.W_proj, mode='reduced')
        self.W_proj = Q_w.astype(np.float32) * np.sqrt(float(self.latent_dim))

        # P1.3b: collapse detection — reset degenerate hyperplanes
        if self.field_bits and len(self.field_bits) >= 10:
            try:
                all_bits = np.array([np.unpackbits(self.field_bits[cid])[:self.n_field_bits]
                                     for cid in self.field_bits])
                bit_ratio = all_bits.mean(axis=0)
                collapsed = np.where((bit_ratio > 0.85) | (bit_ratio < 0.15))[0]
                if len(collapsed) > 0:
                    from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
                    rng = _R.rng(f'collapse_reset_{self._capacity_growths}')
                    self.W_proj[:, collapsed] = rng.randn(self.latent_dim, len(collapsed)).astype(np.float32)
                    print(f"  Collapse guard: reset {len(collapsed)}/{self.n_field_bits} degenerate hyperplanes")
            except Exception:
                pass  # non-critical — skip collapse detection on error

        self._rebuild_field_bits()

    def adjust_l1_lambdas(self, lr_scale=1.0):
        """Adjust per-concept L1 lambdas to maintain target density.

        Each concept tracks its active-fraction in z_c over time.
        If density > target: increase L1 (more sparsity pressure).
        If density < target*0.5: decrease L1 (allow densification).
        """
        if not self.l1_density_window:
            return
        n_adjusted = 0
        for cid, densities in self.l1_density_window.items():
            if len(densities) < 10:
                continue
            mean_density = np.mean(densities[-50:])  # trailing window
            current_lambda = self.l1_lambda_per_cid.get(cid, self.l1_lambda)
            if mean_density > self.l1_target_density * 1.5:
                # Too dense — increase sparsity pressure
                new_lambda = current_lambda * (1.0 + 0.1 * lr_scale)
                self.l1_lambda_per_cid[cid] = min(new_lambda, 0.1)  # cap
                n_adjusted += 1
            elif mean_density < self.l1_target_density * 0.5 and current_lambda > 1e-6:
                # Too sparse — relax sparsity pressure
                new_lambda = current_lambda * (1.0 - 0.1 * lr_scale)
                self.l1_lambda_per_cid[cid] = max(new_lambda, 1e-6)
                n_adjusted += 1
        if n_adjusted:
            n_total = len([v for v in self.l1_density_window.values() if len(v) >= 10])
            print(f"  Adaptive L1: adjusted {n_adjusted}/{n_total} concepts")

    # ── Dynamic capacity ─────────────────────────────────────

    def grow_capacity(self, new_latent_dim=None):
        with self._capacity_lock:
            old_dim = self.latent_dim
            if old_dim >= self.max_latent_dim:
                return old_dim
            if new_latent_dim is None:
                new_latent_dim = int(old_dim * self.arch.growth_factor)
            new_latent_dim = min(new_latent_dim, self.max_latent_dim)
            new_latent_dim = max(new_latent_dim, old_dim + 8)
            # Ensure new dim respects subspace alignment
            new_latent_dim = ((new_latent_dim + 7) // 8) * 8

            # Generate new orthogonal basis vectors
            from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
            rng = _R.rng(f'grow_capacity_{self._capacity_growths}')
            n_new = new_latent_dim - old_dim
            mat = rng.randn(n_new, self.dim).astype(np.float32)
            # Orthogonalise against existing basis
            residual = mat - mat @ self.basis.T @ self.basis
            Q_new, _ = np.linalg.qr(residual, mode='reduced')
            self.basis = np.vstack([self.basis, Q_new.astype(np.float32)])

            # Extend all codes with zeros
            for cid in self.codes:
                self.codes[cid] = np.append(self.codes[cid], np.zeros(n_new, dtype=np.float32))

            # Update subspace ratios (φ² : φ : 1)
            old_l_c, old_l_a, old_l_m = self.l_c, self.l_a, self.l_m
            self.latent_dim = new_latent_dim
            phi = (1.0 + 5.0 ** 0.5) / 2.0
            total = phi * phi + phi + 1.0
            self.l_c = max(8, int(new_latent_dim * phi * phi / total))
            self.l_a = max(8, int(new_latent_dim * phi / total))
            self.l_m = new_latent_dim - self.l_c - self.l_a
            # Shift existing code entries to new subspace positions
            for cid in self.codes:
                old_code = self.codes[cid]
                new_code = np.zeros(new_latent_dim, dtype=np.float32)
                # z_c: identity — kept in same relative position, extended
                new_code[:old_l_c] = old_code[:old_l_c]
                # z_a: attention — same
                new_code[old_l_c:old_l_c + old_l_a] = old_code[old_l_c:old_l_c + old_l_a]
                # z_m: meta — same
                new_code[old_l_c + old_l_a:old_l_c + old_l_a + old_l_m] = old_code[old_l_c + old_l_a:]
                self.codes[cid] = new_code

            # Grow field projection matrices
            if self.W_proj is not None:
                pad = np.zeros((n_new, self.n_field_bits), dtype=np.float32)
                self.W_proj = np.vstack([self.W_proj, pad])
            for lvl in range(len(self._sector_W)):
                pad = np.zeros((n_new, self._sector_W[lvl].shape[1]), dtype=np.float32)
                self._sector_W[lvl] = np.vstack([self._sector_W[lvl], pad])

            # Grow L1 lambda dict entries (inherit global)
            if self.l1_lambda_per_cid:
                self.l1_lambda_per_cid = {
                    cid: lmbda for cid, lmbda in self.l1_lambda_per_cid.items()
                    if cid in self.codes
                }

            self._matrix_dirty = True
            self._capacity_growths += 1
            self._fb_dirty = True
            self._rebuild_sector_index()
            print(f"  Grown capacity: {old_dim} -> {new_latent_dim} "
                  f"(l_c={self.l_c} l_a={self.l_a} l_m={self.l_m})")
            return new_latent_dim

    def prune_capacity(self, sparsity_threshold=0.98):
        """Prune near-zero latent dimensions across all codes.

        A dimension is pruned if > 98% of codes have |val| < 1e-4.
        Rare concept protection: prevents pruning dims that are significant
        for concepts with few active dimensions.
        Returns number of pruned dimensions.
        """
        with self._capacity_lock:
            if len(self.codes) < 10:
                return 0
            codes_arr = np.array(list(self.codes.values()), dtype=np.float32)
            active_frac = np.mean(np.abs(codes_arr) > 1e-4, axis=0)
            dead = np.where(active_frac < (1.0 - sparsity_threshold))[0]
            if len(dead) == 0:
                return 0

            # V18: Rare concept protection — identify significant dims per concept
            per_concept_significant = np.abs(codes_arr) > 0.1 * np.max(np.abs(codes_arr), axis=1, keepdims=True)
            rare_mask = per_concept_significant.sum(axis=1) < max(1, self.latent_dim // 20)
            # Candidate dead dims to remove
            to_remove = list(dead)
            if rare_mask.any():
                # Re-check: would removing 'dead' leave any rare concept with < 5 significant dims?
                keep_extra = set()
                for i, is_rare in enumerate(rare_mask):
                    if not is_rare:
                        continue
                    significant_now = np.where(per_concept_significant[i])[0]
                    after_prune = np.array([d for d in significant_now if d not in to_remove])
                    if len(after_prune) < max(5, self.latent_dim // 20):
                        # Protect the most significant dims that were going to be pruned
                        at_risk = [d for d in significant_now if d in to_remove]
                        at_risk.sort(key=lambda d: -abs(codes_arr[i, d]))
                        for d in at_risk[:3]:
                            keep_extra.add(d)
                if keep_extra:
                    n_protected = len(keep_extra)
                    to_remove = [d for d in to_remove if d not in keep_extra]
                    print(f"  Rare concept protection: kept {n_protected} dims for {int(rare_mask.sum())} rare concepts")

            if not to_remove:
                return 0
            dead = np.array(to_remove)
            # Keep only live dimensions
            live_mask = np.ones(self.latent_dim, dtype=bool)
            live_mask[dead] = False
            live = np.where(live_mask)[0]
            old_dim = self.latent_dim

            # Remap: live dims form new code, basis, W_proj
            new_basis_rows = self.basis[live]
            Q, _ = np.linalg.qr(new_basis_rows.T, mode='reduced')
            self.basis = Q.T.astype(np.float32)
            new_latent_dim = len(live)

            for cid in self.codes:
                self.codes[cid] = self.codes[cid][live]

            self.latent_dim = new_latent_dim
            phi = (1.0 + 5.0 ** 0.5) / 2.0
            total = phi * phi + phi + 1.0
            self.l_c = max(8, int(new_latent_dim * phi * phi / total))
            self.l_a = max(8, int(new_latent_dim * phi / total))
            self.l_m = new_latent_dim - self.l_c - self.l_a

            if self.W_proj is not None:
                self.W_proj = self.W_proj[live]

            for lvl in range(len(self._sector_W)):
                self._sector_W[lvl] = self._sector_W[lvl][live]

            self._matrix_dirty = True
            self._fb_dirty = True
            self._rebuild_sector_index()
            print(f"  Pruned capacity: {old_dim}→{new_latent_dim} "
                  f"(removed {len(dead)} dead dimensions)")
            return len(dead)

    def auto_adjust_capacity(self):
        """Automatically grow or shrink capacity based on code density.

        If mean density across concepts exceeds threshold → grow.
        If many dimensions are dead → prune.
        Called periodically during training.
        """
        if len(self.codes) < 5:
            return
        codes_arr = np.array(list(self.codes.values()), dtype=np.float32)
        # Fraction of dimensions with |val| > 1e-4 per concept
        per_concept_density = np.mean(np.abs(codes_arr) > 1e-4, axis=1)
        mean_density = float(np.mean(per_concept_density))
        max_density = float(np.max(per_concept_density))

        if mean_density > self.arch.density_threshold_grow:
            self.grow_capacity()
        elif mean_density < self.arch.density_threshold_prune * 2:
            n_pruned = self.prune_capacity()
            if n_pruned > 0:
                return

        # Per-dimension prunning
        active_frac = np.mean(np.abs(codes_arr) > 1e-4, axis=0)
        dead_pct = float(np.mean(active_frac < 0.02))
        if dead_pct > 0.3:
            self.prune_capacity()

    # ── Sector index (focal search / field-in-field) ──────

    def _init_sector_fields(self, depths=None):
        """Initialise hierarchical sector projections (field-in-field).

        Each level has its own W_proj[lvl] of [latent_dim, n_bits_lvl].
        Level 0 = coarsest (few bits, large buckets).
        Level N = finest (many bits, small buckets).
        """
        if depths is None:
            depths = self._sector_depths
        self._sector_depths = depths
        from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
        _R_sector = _R.rng('field_bits')  # same seed for sector W_proj
        scale = 1.0 / np.sqrt(self.latent_dim)
        self._sector_W = []
        for n_bits in depths:
            W = _R_sector.randn(self.latent_dim, n_bits).astype(np.float32) * scale
            self._sector_W.append(W)
        self._rebuild_sector_index()

    def _rebuild_sector_index(self):
        """Rebuild inverted sector index from current codes.

        sector_index[depth][prefix_tuple] = [cid, ...]
        """
        if not hasattr(self, '_sector_W') or not self._sector_W:
            return
        self._sector_index = {}
        for depth in range(len(self._sector_W)):
            self._sector_index[depth] = {}
            cumulative_bits = sum(self._sector_depths[:depth + 1])
            prefix_bytes = (cumulative_bits + 7) // 8
            for cid, code in self.codes.items():
                # Build prefix bits across all levels up to this depth
                raw = code @ np.hstack(self._sector_W[:depth + 1])
                bits = (raw > 0).astype(np.uint8)
                packed = np.packbits(bits)[:prefix_bytes]
                key = tuple(packed)
                if key not in self._sector_index[depth]:
                    self._sector_index[depth][key] = []
                self._sector_index[depth][key].append(cid)
        self._fb_dirty = True

    def sector_key(self, cid, depth=0):
        """Get sector key for a concept at given depth.

        Returns hashable tuple of packed uint8 bytes for the first
        sum(depths[:depth+1]) bits of the sector projection.
        """
        if not hasattr(self, '_sector_W') or depth >= len(self._sector_W):
            return None
        code = self.codes.get(cid)
        if code is None:
            return None
        cumulative_bits = sum(self._sector_depths[:depth + 1])
        raw = code @ np.hstack(self._sector_W[:depth + 1])
        bits = (raw > 0).astype(np.uint8)
        packed = np.packbits(bits)[:(cumulative_bits + 7) // 8]
        return tuple(packed)

    def search_in_sector(self, query_cid, depth=0, k=10, all_codes=None):
        """Focal search: only score concepts sharing the same sector prefix.

        Searches at given depth (0=coarse, N=fine).
        Falls back to full search if sector is empty or query has no match.
        """
        key = self.sector_key(query_cid, depth)
        if key is None or depth not in self._sector_index:
            return []
        candidates = self._sector_index[depth].get(key, [])
        if len(candidates) < 2:
            return []

        if all_codes is None:
            all_codes = self.codes

        q_code = all_codes.get(query_cid)
        if q_code is None:
            return []
        q_norm = np.linalg.norm(q_code)
        if q_norm < 1e-10:
            return []
        q_code /= q_norm

        sims = []
        for cid in candidates:
            if cid == query_cid:
                continue
            code = all_codes.get(cid)
            if code is None:
                continue
            sim = float(q_code @ code / (np.linalg.norm(code) + 1e-10))
            sims.append((cid, sim))
        sims.sort(key=lambda x: -x[1])
        return sims[:k]

    def focal_refine(self, query_cid, start_depth=0, target_k=5, max_depth=None):
        """Progressive sector refinement: start coarse, narrow to fine.

        At each depth, if enough candidates found, stop.
        Otherwise go one level deeper.
        """
        if max_depth is None:
            max_depth = len(self._sector_depths) - 1 if hasattr(self, '_sector_depths') else 0
        for depth in range(start_depth, max_depth + 1):
            results = self.search_in_sector(query_cid, depth=depth, k=target_k * 3)
            if len(results) >= target_k:
                return results[:target_k]
        return results

    def get_field_bits(self, cid):
        """Get binary field vector for a concept."""
        return self.field_bits.get(cid)

    def field_overlap(self, cid_a, cid_b):
        """Count overlapping field bits between two concepts."""
        ba = self.field_bits.get(cid_a)
        bb = self.field_bits.get(cid_b)
        if ba is None or bb is None or len(ba) != len(bb):
            return 0
        return int(np.unpackbits(np.bitwise_and(ba, bb)).sum())

    # ── HDC/VSA n-gram fallback ────────────────────────────

    def hdc_bind(self, a, b):
        """Hybrid bind: HRR ⊛ + element-wise (α=0.7 for STDP compatibility)."""
        return _hybrid_bind(a, b)

    def hdc_unbind(self, c, b):
        """Hybrid unbind: HRR correlation + element-wise."""
        return _hybrid_unbind(c, b)

    def hdc_permute(self, v, n=1):
        """Circular shift by n positions."""
        return np.roll(v, n)

    def hdc_fib_permute(self, v, t, dim=None):
        """Circular shift by Fibonacci(t) positions (позиционное кодирование Фибоначчи)."""
        from eva.symbolic.fibonacci_utils import FibonacciUtils
        d = dim or len(v)
        shift = FibonacciUtils.fib_position_shift(t, d)
        return np.roll(v, shift)

    def hdc_bundle(self, v, accum, lr=0.1):
        """Bundle v into accumulator (adaptive running average)."""
        return accum * (1.0 - lr) + v * lr

    def hdc_ngram_repr(self, codes):
        """Build HDC representation for an n-gram sequence of codes.

        For (w1, w2, ..., wn): ρ^{n-1}(w1) ⊛ ρ^{n-2}(w2) ⊛ ... ⊛ wn
        where ⊛ is circular convolution and ρ is circular shift.
        """
        n = len(codes)
        if n == 0:
            return None
        result = codes[-1].copy()
        for i in range(n - 1):
            result = self.hdc_bind(self.hdc_permute(codes[i], n - 1 - i), result)
        return result

    def hdc_update_ngram(self, prefix_cids, next_code):
        """Update HDC memory for {prefix_cids → next_token_code}.

        Bundles next_code into hdc_memory[prefix_cids] (running average).
        Evicts LFU (least-frequently-used) entries when over hdc_memory_max.
        """
        key = tuple(prefix_cids)
        if key not in self.hdc_memory:
            if len(self.hdc_memory) >= self.hdc_memory_max:
                # P3.6: LFU eviction — remove entry with lowest access count
                min_count = min(self.hdc_memory_counts.values()) if self.hdc_memory_counts else 0
                evict_candidates = [k for k, v in self.hdc_memory_counts.items() if v == min_count]
                evict_key = evict_candidates[0]
                self.hdc_memory.pop(evict_key, None)
                self.hdc_memory_counts.pop(evict_key, None)
            self.hdc_memory[key] = next_code.copy()
            self.hdc_memory_counts[key] = 1
        else:
            count = self.hdc_memory_counts[key]
            lr = 1.0 / max(count + 1, 1.0)
            self.hdc_memory[key] = self.hdc_bundle(
                next_code, self.hdc_memory[key], lr)
            self.hdc_memory_counts[key] = count + 1

    def hdc_predict(self, context_cids, all_codes, k=20):
        """HDC fallback prediction from prefix context (CID-based key).

        Args:
            context_cids: list of concept IDs for the context (prefix)
            all_codes: dict of {cid: code} for all candidates
            k: number of candidates to return

        Returns:
            [(cid, score), ...] scored by cosine similarity
        """
        key = tuple(context_cids)
        if key in self.hdc_memory:
            mem_repr = self.hdc_memory[key]
        elif len(key) >= 2:
            # No stored repr — use the context itself as a probe
            ctx_codes = [self.codes.get(cid) for cid in key]
            ctx_codes = [c for c in ctx_codes if c is not None]
            if len(ctx_codes) < 2:
                return []
            ctx_repr = self.hdc_ngram_repr(ctx_codes)
            if ctx_repr is None:
                return []
            sims = []
            for cid, code in all_codes.items():
                score = float(ctx_repr @ code) / (np.linalg.norm(ctx_repr) * np.linalg.norm(code) + 1e-10)
                sims.append((int(cid), score))
            sims.sort(key=lambda x: -x[1])
            return sims[:k]
        else:
            return []

        # Query: mem_repr stores next_code directly (bundled average),
        # so use it directly as the query probe.
        qnorm = np.linalg.norm(mem_repr)
        if qnorm < 1e-10:
            return []
        query = mem_repr / qnorm

        sims = []
        for cid, code in all_codes.items():
            score = float(query @ code / (np.linalg.norm(code) + 1e-10))
            sims.append((int(cid), score))
        sims.sort(key=lambda x: -x[1])
        return sims[:k]

    # ── Vector computation ───────────────────────────────────

    def compute_vector(self, cid):
        """Compute full vector from latent coords @ basis + normalize."""
        coords = self.codes.get(cid)
        if coords is None:
            return None
        v = coords @ self.basis
        nv = np.linalg.norm(v)
        if nv > 1e-10:
            v /= nv
        return v

    def fluctuate(self, fluctuation_amp=0.005, decay=0.999):
        """Apply autonomous drift to all latent codes."""
        if not hasattr(self, '_fluct_rng'):
            from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
            self._fluct_rng = _R.rng('fluctuate')
        for cid in list(self.codes.keys()):
            c = self.codes[cid]
            noise = self._fluct_rng.randn(self.latent_dim).astype(np.float32) * fluctuation_amp
            c[:] = c * decay + noise
        self._matrix_dirty = True

    def reinitialize_all(self, cid_list):
        """Reset all latent codes to random initialization."""
        self.codes = {}
        for cid in cid_list:
            self.init_concept(cid)
        self._matrix_dirty = True

    def to_dict(self, binary_path=None):
        """Serialize for saving.

        Args:
            binary_path: if set, save codes+basis as .npz and return lightweight dict.
        """
        with self._capacity_lock:
            if binary_path:
                tmp_path = binary_path.replace('.npz', '.tmp.npz')
                cids = np.array(list(self.codes.keys()), dtype=np.int32)
                codes_arr = np.array([self.codes[cid] for cid in cids], dtype=np.float32)
                kw = dict(codes=codes_arr, cids=cids, basis=self.basis)
                # Save field bits if present
                if hasattr(self, 'field_bits') and self.field_bits:
                    fb_cids = np.array(list(self.field_bits.keys()), dtype=np.int32)
                    fb_arr = np.array([self.field_bits[cid] for cid in fb_cids], dtype=np.uint8)
                    kw['fb_cids'] = fb_cids
                    kw['fb_arr'] = fb_arr
                np.savez_compressed(tmp_path, **kw)
                os.replace(tmp_path, binary_path)
                result = {
                    'dim': self.dim,
                    'latent_dim': self.latent_dim,
                    'binary_codes': os.path.basename(binary_path),
                    'n_codes': len(cids),
                }
                if self.l1_lambda_per_cid:
                    result['l1_lambda_per_cid'] = {str(cid): float(v) for cid, v in self.l1_lambda_per_cid.items()}
                return result
            result = {
                'dim': self.dim,
                'latent_dim': self.latent_dim,
                'basis': self.basis.tolist(),
                'codes': {str(cid): c.tolist() for cid, c in self.codes.items()},
            }
            if self.l1_lambda_per_cid:
                result['l1_lambda_per_cid'] = {str(cid): float(v) for cid, v in self.l1_lambda_per_cid.items()}
            return result

    @classmethod
    def from_dict(cls, data, base_dir=None):
        field = cls(dim=data['dim'], latent_dim=data['latent_dim'])
        binary_file = data.get('binary_codes')
        if binary_file and base_dir:
            path = os.path.join(base_dir, binary_file) if os.path.isdir(base_dir) else binary_file
            npz = np.load(path)
            field.basis = npz['basis'].astype(np.float32)
            cids = npz['cids']
            codes_arr = npz['codes']
            # Pre-extract arrays before dict comprehensions
            # (NpzFile.__getitem__ is slow on repeated access)
            field.codes = {int(cid): codes_arr[i].copy() for i, cid in enumerate(cids)}
            # Backward compat: field bits added in v2
            # Partial checkpoint ok — fb_cids missing → empty field_bits
            if 'fb_cids' in npz.files:
                fb_arr = npz['fb_arr']
                fb_cids_arr = npz['fb_cids']
                field.field_bits = {int(cid): fb_arr[i].copy()
                                     for i, cid in enumerate(fb_cids_arr)}
        else:
            field.basis = np.array(data['basis'], dtype=np.float32)
            field.codes = {int(cid): np.array(c, dtype=np.float32)
                            for cid, c in data['codes'].items()}
        # Verify basis orthogonality — re-orthogonalize if drifted
        QtQ = field.basis.T @ field.basis
        err = np.max(np.abs(QtQ - np.eye(field.dim, dtype=np.float32)))
        if err > 1e-3:
            Q, _ = np.linalg.qr(field.basis, mode='reduced')
            old_basis = field.basis
            # Batched re-encoding: stack all codes into matrix
            cids = list(field.codes.keys())
            codes_mat = np.stack([field.codes[cid] for cid in cids])
            # codes_mat @ old_basis → vectors on sphere
            vecs = codes_mat @ old_basis
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            vecs /= norms
            # Encode under new orthonormal basis: vecs @ Q.T → new codes
            new_codes = vecs @ Q.T
            for i, cid in enumerate(cids):
                field.codes[cid] = new_codes[i]
            field.basis = Q.astype(np.float32)

        raw_l1 = data.get('l1_lambda_per_cid')
        if raw_l1:
            field.l1_lambda_per_cid = {int(cid): float(v) for cid, v in raw_l1.items()}
        field._capacity_lock = threading.Lock()
        field._matrix_dirty = True
        return field


# ═══════════════════════════════════════════════════════════
# Harmonizer — multi-level field agreement via VSA
# ═══════════════════════════════════════════════════════════

class EntityField:
    """Recursive semantic field — every entity (char, word, sent, para) has one
    vector that encodes VSA bindings to its contexts at all levels.

    Roles (cross-level):
      CHAR  — char↔word
      MORPH — morph↔word
      WORD  — word↔sent
      SENT  — sent↔para

    All entities share one dict. Word vectors are synced from ConceptSpace
    (source of truth for STDP). Char/sent/para vectors are stored here.

    V(entity) accumulates bind(V(context), role) * lr for every occurrence.
    query(entity, role) = unbind(V(entity), role) → superposition of contexts.
    """
    LEVEL_ROLES = ['CHAR', 'MORPH', 'WORD', 'SENT', 'PARA']
    ETYPE_TO_ROLE = {'c': 'CHAR', 'm': 'MORPH', 'w': 'WORD', 's': 'SENT', 'p': 'PARA'}

    def __init__(self, dim=None, word_store=None, dim_coord=None):
        _c = FCFConfig()
        self.dim = dim or (dim_coord.latent_dim if dim_coord else _c.latent_dim)
        self.word_store = word_store  # optional reference to ConceptVectorStore
        self.dim_coord = dim_coord

        # All entities: key = (etype_char, id)  e.g. ('c', 97), ('w', 42), ('s', hash)
        self.entities = {}

        # Quasi-orthogonal level roles (Gram-Schmidt)
        from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
        n_roles = len(self.LEVEL_ROLES)
        mat = _R.rng('entity_roles').randn(n_roles, self.dim).astype(np.float32)
        Q, _ = np.linalg.qr(mat.T, mode='reduced')
        self.role_vecs = {role: Q[:, i].copy() for i, role in enumerate(self.LEVEL_ROLES)}

        # Random projection: 768D → dim, for syncing word_store vectors
        self._proj = None  # lazy init
        self._proj_lock = threading.Lock()

        # LRU cache for char↔word bindings — prevents O(corpus_bytes) on repeated chars
        self._char_word_cache = {}
        self._char_word_cache_evict = []

        # P1.7: EntityField cleanup — TTL + size cap
        self._entity_access_time: Dict[tuple, float] = {}
        self._entity_batch_counter = 0
        self._max_entities = _c.entity_field_max_entities  # 50K × 2048 × fp32 ≈ 400MB cap

    # ── Key helpers ──────────────────────────────────────────
    @staticmethod
    def key_char(cp):    return ('c', cp)
    @staticmethod
    def key_morph(mid):  return ('m', mid)
    @staticmethod
    def key_word(cid):   return ('w', cid)
    @staticmethod
    def key_sent(h):     return ('s', h)
    @staticmethod
    def key_para(h):     return ('p', h)

    # ── Dimension bridge ─────────────────────────────────────
    def _to_dim(self, v):
        """Project vector v to self.dim if needed (Johnson-Lindenstrauss style)."""
        if len(v) == self.dim:
            return v
        with self._proj_lock:
            if self._proj is None or self._proj.shape[1] != len(v):
                from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
                scale = 1.0 / np.sqrt(len(v))
                self._proj = _R.rng('entity_proj').randn(self.dim, len(v)).astype(np.float32) * scale
        return self._proj @ v

    # ── VSA primitives (hybrid HRR+element-wise) ─────────────
    def _bind(self, a, b):
        return _hybrid_bind(a, b)
    def _unbind(self, c, b):
        return _hybrid_unbind(c, b)

    # ── Core: ensure, get, set, sync_word ────────────────────
    def ensure(self, key):
        if key not in self.entities:
            from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
            rng = _R.rng(f'entityfield_{key}')
            v = rng.randn(self.dim).astype(np.float32)
            n = float(np.linalg.norm(v))
            self.entities[key] = (v / n).astype(np.float16) if n > 1e-10 else v.astype(np.float16)
        v = self.entities[key]
        return v.astype(np.float32) if hasattr(v, 'astype') else v

    def get(self, key):
        # Word entities: if not in dict, try syncing from word_store
        if key not in self.entities and key[0] == 'w' and self.word_store is not None:
            v = self.word_store.get(key[1])
            if v is not None:
                self.entities[key] = self._to_dim(v.copy().astype(np.float32))
        # P1.7: track access time for cleanup
        if key in self.entities:
            self._entity_access_time[key] = self._entity_batch_counter
        v = self.entities.get(key)
        return v.astype(np.float32) if v is not None and hasattr(v, 'astype') else v

    def set(self, key, v):
        self.entities[key] = v.astype(np.float16) if hasattr(v, 'astype') else v

    def clear_bind_cache(self):
        self._char_word_cache.clear()
        self._char_word_cache_evict.clear()

    def sync_word(self, cid, vec=None):
        """Sync word vector from concept_vectors into entity field."""
        if vec is None and self.word_store is not None:
            vec = self.word_store.get(cid)
        if vec is not None:
            self.entities[('w', cid)] = self._to_dim(vec.copy().astype(np.float32)).astype(np.float16)

    # ── Bind / Query ─────────────────────────────────────────
    def bind(self, etype, eid, ctx_type, ctx_id, lr=0.1):
        key = (etype, eid)
        ctx_key = (ctx_type, ctx_id)
        role = self.ETYPE_TO_ROLE.get(etype)
        if role is None:
            return
        # LRU skip: if this exact (entity, context) pair was bound recently, skip
        cache_tag = (key, ctx_key)
        if cache_tag in self._char_word_cache:
            return
        if len(self._char_word_cache) > FCFConfig().entity_field_max_entities:
            evict = self._char_word_cache_evict.pop(0) if self._char_word_cache_evict else next(iter(self._char_word_cache))
            self._char_word_cache.pop(evict, None)
        self._char_word_cache[cache_tag] = True
        self._char_word_cache_evict.append(cache_tag)
        v_ctx = self.get(ctx_key)
        if v_ctx is None:
            v_ctx = self.ensure(ctx_key)
        v_e = self.get(key)
        if v_e is None:
            v_e = self.ensure(key)
        rv = self.role_vecs.get(role)
        if rv is None:
            return
        bound = self._bind(v_ctx, rv)
        self.entities[key] = v_e + bound * lr
        n = float(np.linalg.norm(self.entities[key]))
        if n > 1e-10:
            self.entities[key] /= n

    def query(self, etype, eid):
        """unbind(V(entity), role) → superposition of bound contexts."""
        key = (etype, eid)
        v = self.get(key)
        if v is None:
            return None
        role = self.ETYPE_TO_ROLE.get(etype)
        if role is None:
            return None
        rv = self.role_vecs.get(role)
        if rv is None:
            return None
        return self._unbind(v, rv)

    # ── Cleanup: remove stale entities ────────────────────────
    def cleanup(self):
        """Удалить entity, превышающие лимит (по access time)."""
        if len(self.entities) <= self._max_entities:
            return
        sorted_keys = sorted(self._entity_access_time.items(), key=lambda x: x[1])
        to_remove = len(self.entities) - self._max_entities
        for key, _ in sorted_keys[:to_remove]:
            self.entities.pop(key, None)
            self._entity_access_time.pop(key, None)

    # ── Serialisation ────────────────────────────────────────
    def to_dict(self):
        keys = []
        vecs = []
        for k, v in self.entities.items():
            keys.append(k)
            vecs.append(v)
        return {
            'ef_keys': keys,
            'ef_vecs': np.array(vecs, dtype=np.float32) if vecs else np.empty((0, self.dim), dtype=np.float32),
            'ef_dim': self.dim,
            'ef_batch_counter': self._entity_batch_counter,
        }

    @classmethod
    def from_dict(cls, data, word_store=None):
        ef = cls(dim=data.get('ef_dim', FCFConfig().latent_dim), word_store=word_store)
        keys = data.get('ef_keys', [])
        vecs = data.get('ef_vecs', np.empty((0, ef.dim), dtype=np.float32))
        for k, v in zip(keys, vecs):
            k = tuple(k)
            ef.entities[k] = v.astype(np.float32)
        ef._entity_batch_counter = data.get('ef_batch_counter', 0)
        return ef

    # ── Decay: fade old bindings → prevent saturation ────────
    def decay(self, factor=0.999):
        for key in self.entities:
            self.entities[key] *= factor
            n = float(np.linalg.norm(self.entities[key]))
            if n > 1e-10:
                self.entities[key] /= n


from eva.symbolic.semantic_piece import CharEnvelope
# CharEnvelope previously was a separate class here; now unified with semantic_piece.
# API compatible: CharEnvelope(dim, max_chars=...) → ensure / word_envelope / modulate / stdp_update


class Harmonizer:
    """Harmonises WordField, MorphemeField, and SupraField via VSA bind/unbind.

    Architecture:
      - Each word = ⊕ bind(morpheme_vec, ROLE) over its morphemes (root+affixes)
      - Decompose = unbind(word_vec, ROLE) to recover individual morphemes
      - Harmonize: pull word toward its composition, backprop error to morphemes
      - Dirty-flag tracking prevents avalanche: dirty means 'harmonise on next focus'

    Role vectors are fixed quasi-orthogonal references initialised once.
    """

    ROLES = ['ROOT', 'PREFIX', 'SUFFIX', 'ENDING', 'WORD_POS', 'WORD_ROLE']

    def __init__(self, dim=None, harm_lr=None, morph_lr=None, n_iter=None, dim_coord=None):
        from eva.symbolic.fcf_config import FCFConfig
        _c = FCFConfig()
        self.dim = dim or _c.latent_dim
        self.harm_lr = harm_lr if harm_lr is not None else _c.harm_lr
        self.morph_lr = morph_lr if morph_lr is not None else _c.morph_lr
        self.n_iter = n_iter if n_iter is not None else _c.n_harm_iterations
        self.damping = _c.harm_damping

        # Initialise quasi-orthogonal role vectors
        from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
        role_mat = _R.rng('harm_roles').randn(len(self.ROLES), self.dim).astype(np.float32)
        # Gram-Schmidt orthonormalisation
        Q, _ = np.linalg.qr(role_mat.T, mode='reduced')
        self.role_vecs = {role: Q[:, i].copy() for i, role in enumerate(self.ROLES)}

        # Backward index: morph_id -> set of word_ids that use it
        self.morph_to_words = defaultdict(set)
        # Dirty flags
        self.word_dirty = set()
        self.morph_dirty = set()
        # Morpheme storage: morph_id -> HD vector (2048,)
        self.morphemes = {}
        # Word -> list of (morph_id, role) mappings
        self.word_morphs = defaultdict(list)

    # ── VSA primitives ────────────────────────────────────────

    def _bind(self, a, b):
        """Hybrid bind: HRR ⊛ + element-wise (α=0.7 for STDP compatibility)."""
        return _hybrid_bind(a, b)

    def _unbind(self, c, b):
        """Hybrid unbind: HRR correlation + element-wise."""
        return _hybrid_unbind(c, b)

    def _bundle(self, vecs):
        """Bundle (superposition) with normalisation."""
        if not vecs:
            return None
        result = sum(vecs)
        n = np.linalg.norm(result)
        return result / n if n > 1e-10 else result

    # ── Compose / Decompose ──────────────────────────────────

    def compose_word(self, morph_parts, ctx_vec=None):
        """Build word vector from morpheme parts.

        If ctx_vec (sentence vector) is provided, the ROOT morpheme is
        contextually modulated: root_effective = root_vec + unbind(ctx_vec, ROLE_POS) * 0.3
        This enables context-dependent disambiguation (e.g. homonyms).

        Args:
            morph_parts: dict of {role_str: vec_or_morph_id} or list of (role, vec)
            ctx_vec: optional sentence-level HD vector for context modulation

        Returns:
            unit-norm HD vector
        """
        bound = []
        if isinstance(morph_parts, dict):
            items = morph_parts.items()
        else:
            items = morph_parts
        for role, vec in items:
            if isinstance(vec, int):
                vec = self.morphemes.get(vec)
            if vec is None:
                continue
            # Context modulation: bias root toward sentence context
            if role == 'ROOT' and ctx_vec is not None:
                ctx_bias = self._unbind(ctx_vec, self.role_vecs['WORD_POS'])
                bn = float(np.linalg.norm(ctx_bias))
                if bn > 1e-10:
                    w = 0.3
                    vec = vec + ctx_bias * w
                    vn = float(np.linalg.norm(vec))
                    if vn > 1e-10:
                        vec /= vn
            role_v = self.role_vecs.get(role)
            if role_v is None:
                continue
            bound.append(self._bind(vec, role_v))
        return self._bundle(bound) if bound else None

    def decompose_word(self, word_vec, roles=None):
        """Extract morpheme vectors from a word vector via unbind.

        Args:
            word_vec: unit-norm word HD vector
            roles: list of role strings (default: all except WORD_POS/WORD_ROLE)

        Returns:
            dict of {role: reconstructed_vec}
        """
        if roles is None:
            roles = ['ROOT', 'PREFIX', 'SUFFIX', 'ENDING']
        result = {}
        for role in roles:
            if role in self.role_vecs:
                result[role] = self._unbind(word_vec, self.role_vecs[role])
        return result

    # ── Register morphology ──────────────────────────────────

    def register_word(self, word_id, morph_map):
        """Register a word's morphological decomposition.

        Args:
            word_id: int CID
            morph_map: dict of {role_str: morph_id} — e.g. {'ROOT': 42, 'ENDING': 7}
        """
        for role, morph_id in morph_map.items():
            self.word_morphs[word_id].append((morph_id, role))
            self.morph_to_words[morph_id].add(word_id)

    def set_morpheme_vec(self, morph_id, vec):
        """Set or update a morpheme's HD vector."""
        self.morphemes[morph_id] = vec.copy() if isinstance(vec, np.ndarray) else vec

    def get_morpheme_vec(self, morph_id):
        return self.morphemes.get(morph_id)

    # ── Dirty tracking ───────────────────────────────────────

    def mark_word_dirty(self, word_id):
        self.word_dirty.add(word_id)

    def mark_morph_dirty(self, morph_id):
        if morph_id not in self.morph_dirty:
            self.morph_dirty.add(morph_id)
            # Cascade: all words containing this morph become dirty
            for wid in self.morph_to_words.get(morph_id, set()):
                self.word_dirty.add(wid)

    def clear_dirty(self):
        self.word_dirty.clear()
        self.morph_dirty.clear()

    # ── Harmonize ─────────────────────────────────────────────

    def harmonize(self, word_id, word_vec, sent_vec=None):
        """Pull a word vector toward its composition, propagate error to morphemes.

        Args:
            word_id: int CID
            word_vec: current word HD vector (will NOT be mutated here)
            sent_vec: optional sentence-level vector for top-down bias

        Returns:
            (new_word_vec, delta_norm) or (None, 0) if not applicable
        """
        if word_id not in self.word_morphs:
            return None, 0.0

        actual = word_vec
        prev_delta = float('inf')
        total_delta = 0.0

        for iteration in range(self.n_iter):
            # Bottom-up prediction: recompose from morphemes
            morph_parts = []
            for morph_id, role in self.word_morphs[word_id]:
                mv = self.morphemes.get(morph_id)
                if mv is not None:
                    morph_parts.append((role, mv))
            if not morph_parts:
                break

            pred_up = self.compose_word(morph_parts, ctx_vec=sent_vec)
            if pred_up is None:
                break

            # Top-down prediction from sentence context
            pred_dn = None
            if sent_vec is not None:
                pred_dn = self._unbind(sent_vec, self.role_vecs['WORD_POS'])

            # Error = weighted combination of bottom-up and top-down
            error = (pred_up - actual) * 0.5
            if pred_dn is not None:
                error += (pred_dn - actual) * 0.3

            delta = float(np.linalg.norm(error))
            total_delta += delta

            # Convergence check
            if delta > prev_delta * 1.5:
                break  # diverging — stop
            if delta < 1e-6:
                break  # converged
            prev_delta = delta

            # Update word vector (clamped)
            update = error * self.harm_lr * self.damping
            un = float(np.linalg.norm(update))
            if un > 0.3:
                update = update / un * 0.3
            actual = actual + update
            an = float(np.linalg.norm(actual))
            if an > 1e-10:
                actual /= an

            # Backpropagate error to morphemes via unbind
            for morph_id, role in self.word_morphs[word_id]:
                mv = self.morphemes.get(morph_id)
                if mv is None:
                    continue
                # Error projection: what would the morpheme need to be to fix the error?
                morph_grad = self._unbind(error, self.role_vecs[role])
                mg_norm = float(np.linalg.norm(morph_grad))
                if mg_norm > 0.3:
                    morph_grad = morph_grad / mg_norm * 0.3
                new_mv = mv + morph_grad * self.morph_lr
                nmv = float(np.linalg.norm(new_mv))
                if nmv > 1e-10:
                    new_mv /= nmv
                self.morphemes[morph_id] = new_mv
                # Mark connected words as dirty (cascade)
                for wid in self.morph_to_words.get(morph_id, set()):
                    if wid != word_id:
                        self.word_dirty.add(wid)

        # Clear own dirty flag
        self.word_dirty.discard(word_id)
        if total_delta > 1e-6:
            return actual, total_delta
        return None, 0.0

    def balance_subspaces(self, z_c, z_a, z_m):
        from eva.symbolic.fibonacci_utils import FibonacciUtils
        return FibonacciUtils.balance_subspaces(z_c, z_a, z_m)

    def to_dict(self):
        """Serialize harmonizer state for npz/json persistence."""
        morpheme_ids = np.array(list(self.morphemes.keys()), dtype=np.int64)
        morpheme_vecs = np.array([self.morphemes[m] for m in morpheme_ids], dtype=np.float32)
        words = np.array(list(self.word_morphs.keys()), dtype=np.int64)
        roles_flat = []
        mids_flat = []
        word_lens = []
        for w in words:
            parts = self.word_morphs[w]
            word_lens.append(len(parts))
            for mid, role in parts:
                mids_flat.append(mid)
                roles_flat.append(role)
        return {
            'harm_morph_ids': morpheme_ids,
            'harm_morph_vecs': morpheme_vecs,
            'harm_words': words,
            'harm_mids_flat': np.array(mids_flat, dtype=np.int64),
            'harm_roles_flat': roles_flat,
            'harm_word_lens': np.array(word_lens, dtype=np.int32),
            'harm_dim': self.dim,
            'harm_lr': self.harm_lr,
            'morph_lr': self.morph_lr,
            'n_iter': self.n_iter,
        }

    @classmethod
    def from_dict(cls, data):
        """Restore harmonizer from serialized dict."""
        from eva.symbolic.fcf_config import FCFConfig
        _c = FCFConfig()
        harm = cls(
            dim=data.get('harm_dim', _c.latent_dim),
            harm_lr=data.get('harm_lr', _c.harm_lr),
            morph_lr=data.get('morph_lr', _c.morph_lr),
            n_iter=data.get('n_iter', _c.n_harm_iterations),
        )
        morph_ids = data.get('harm_morph_ids', np.array([], dtype=np.int64))
        morph_vecs = data.get('harm_morph_vecs', np.empty((0, harm.dim), dtype=np.float32))
        for mid, vec in zip(morph_ids, morph_vecs):
            harm.morphemes[int(mid)] = vec.astype(np.float32)
        words = data.get('harm_words', np.array([], dtype=np.int64))
        mids_flat = data.get('harm_mids_flat', np.array([], dtype=np.int64))
        roles_flat = data.get('harm_roles_flat', [])
        word_lens = data.get('harm_word_lens', np.array([], dtype=np.int32))
        idx = 0
        for w, n_parts in zip(words, word_lens):
            w = int(w)
            for j in range(n_parts):
                if idx < len(mids_flat) and idx < len(roles_flat):
                    mid = int(mids_flat[idx])
                    role = roles_flat[idx]
                    harm.word_morphs[w].append((mid, role))
                    harm.morph_to_words[mid].add(w)
                    idx += 1
        return harm


class ConceptSpace:
    """Vector space for BPE-token concepts.

    Each BPE token is a concept with:
      - cid: token ID (0..vocab_size-1)
      - vector: unit sphere vector from fractal field
      - STDP-learned transitions to other tokens
    """

    def __init__(self, vocab_size=None, dim=768, latent_dim=2048):
        self.vocab_size = vocab_size or 0
        self.dim = dim

        # Dimension coordinator — validates all component dims at construction
        self.dims = DimensionCoordinator(vec_dim=dim, latent_dim=latent_dim)

        # Fractal field: latent codes → full vectors via shared basis
        self.fractal = FractalField(dim=self.dims.vec_dim, latent_dim=self.dims.latent_dim)

        # Concept vectors: dense ndarray[V, dim] with dict-like convenience
        self.concept_vectors = ConceptVectorStore(self.vocab_size, self.dim)

        # Random state
        from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
        self.rng = _R.rng('rng')
        self._item_rng = _R.rng('item_rng')
        self._inhibition_step = 0
        self._inhibit_rng = _R.rng('inhibit_rng')

        # Shift tracking
        self._total_shift = 0.0
        self._update_count = 0
        self._after_update_hook = None

        # ── Morphological harmonizer (levels 1-2: morpheme ↔ word) ──
        _c = FCFConfig()
        self.harmonizer = Harmonizer(dim=self.dims.latent_dim)
        self._morph_conf_threshold = _c.morph_confidence_threshold
        self._morph_vocab = None  # loaded on demand
        self._harm_n_checkpoints = 0
        self._harm_slow_start_epochs = _c.harm_slow_start_epochs

        # ── EntityField: recursive semantic field (char↔word↔sent↔para) ──
        self.entity_field = EntityField(dim=self.dims.latent_dim, word_store=self.concept_vectors, dim_coord=self.dims)

        # ---- Initialization ----

    def init_concepts(self):
        """Initialize all concept vectors (0..vocab_size-1) via fractal field."""
        for cid in range(self.vocab_size):
            v = self.fractal.init_concept(cid)
            if v is not None:
                self.concept_vectors[cid] = v
            else:
                v = self.rng.randn(self.dim).astype(np.float32)
                v /= max(np.linalg.norm(v), 1e-10)
                self.concept_vectors[cid] = v

    def _sync_from_fractal(self):
        """Sync concept_vectors from fractal latent codes."""
        for cid in list(self.fractal.codes.keys()):
            v = self.fractal.compute_vector(cid)
            if v is not None:
                self.concept_vectors[cid] = v

    def get_vec(self, cid):
        """O(1) vector access."""
        return self.concept_vectors[cid] if cid in self.concept_vectors else None

    def set_vec(self, cid, v):
        """Update vector in dense store."""
        self.concept_vectors[cid] = v

    # ── Octree encoding ──────────────────────────────────────

    def reinit_rare(self, freq_map, threshold=3, e5_model=None, sp=None, morph_bundle=False, device='cpu'):
        """Replace rare concept vectors with random unit vectors or e5 embeddings.

        Concepts with freq < threshold get either:
          - random unit vectors (default, no e5_model)
          - e5 embeddings (if e5_model provided)
          - VSA bundle of morpheme e5 embeddings (if morph_bundle=True)

        Returns dict with reinit_count and method used.
        """
        reinit_count = 0
        e5_count = 0
        rare_cids = []
        for cid in range(self.vocab_size):
            freq = freq_map.get(cid, 0)
            if 0 < freq < threshold:
                rare_cids.append(cid)

        if not rare_cids:
            return {'reinit': 0, 'e5': 0, 'method': 'none'}

        if e5_model is not None and sp is not None:
            # Batch encode rare tokens with e5
            batch_size = 512
            for i in range(0, len(rare_cids), batch_size):
                batch_cids = rare_cids[i:i + batch_size]
                batch_texts = []
                valid_cids = []
                for cid in batch_cids:
                    try:
                        token = sp.IdToPiece(cid).replace('\u2581', '').strip()
                    except Exception:
                        token = ''
                    if not token or len(token) < 1:
                        continue
                    if any(c in '.,!?;:()[]{}«»—–-\'\"1234567890' for c in token) and len(token) < 3:
                        continue
                    batch_texts.append(token)
                    valid_cids.append(cid)

                if morph_bundle:
                    for cid, token in zip(valid_cids, batch_texts):
                        parts = self._decompose_word(token)
                        if parts:
                            m_texts = [m for _, m in parts.items()]
                            try:
                                m_embs = e5_model.encode(m_texts, normalize_embeddings=True,
                                                         show_progress_bar=False)
                                bundle = np.mean(m_embs, axis=0).astype(np.float32)
                                bundle /= max(np.linalg.norm(bundle), 1e-10)
                                self.set_vec(cid, bundle)
                                self.fractal.codes.pop(cid, None)
                                e5_count += 1
                            except Exception:
                                v = self._item_rng.randn(self.dim).astype(np.float32)
                                v /= max(np.linalg.norm(v), 1e-10)
                                self.set_vec(cid, v)
                                self.fractal.codes.pop(cid, None)
                        else:
                            v = self._item_rng.randn(self.dim).astype(np.float32)
                            v /= max(np.linalg.norm(v), 1e-10)
                            self.set_vec(cid, v)
                            self.fractal.codes.pop(cid, None)
                        reinit_count += 1
                else:
                    try:
                        embs = e5_model.encode(batch_texts, normalize_embeddings=True,
                                               show_progress_bar=False)
                        for cid, emb in zip(valid_cids, embs):
                            v = np.asarray(emb, dtype=np.float32)
                            v /= max(np.linalg.norm(v), 1e-10)
                            self.set_vec(cid, v)
                            self.fractal.codes.pop(cid, None)
                            e5_count += 1
                            reinit_count += 1
                    except Exception:
                        for cid in valid_cids:
                            v = self._item_rng.randn(self.dim).astype(np.float32)
                            v /= max(np.linalg.norm(v), 1e-10)
                            self.set_vec(cid, v)
                            self.fractal.codes.pop(cid, None)
                            reinit_count += 1
        else:
            for cid in rare_cids:
                v = self._item_rng.randn(self.dim).astype(np.float32)
                v /= max(np.linalg.norm(v), 1e-10)
                self.set_vec(cid, v)
                self.fractal.codes.pop(cid, None)
                reinit_count += 1

        if reinit_count:
            self.fractal._matrix_dirty = True
            # Sync reinitialised vectors to EntityField so word vectors
            # are immediately available as VSA composition.
            ef = getattr(self, 'entity_field', None)
            if ef is not None:
                for cid in rare_cids:
                    v = self.concept_vectors.get(cid)
                    if v is not None:
                        ef.sync_word(cid, v)
        return {'reinit': reinit_count, 'e5': e5_count, 'method': 'morph_bundle' if morph_bundle else 'direct_e5' if e5_model else 'random'}

    def build_zeckendorf_fields(self, lattice, n_anchors=1024, min_lcp=1, gamma=0.5, path_overrides=None):
        """Build H matrix and field_bits from nested Zeckendorf encoding.

        Replaces PMI-based build_anchor_matrix + build_fields_from_lattice.
        Each concept ID → Zeckendorf representation (Fibonacci sum) → path.
        H[i,j] = (1 - γ^{LCP}) / (1 - γ) where LCP = longest common prefix.
        Zeckendorf paths give longer common prefixes for semantically related
        concept IDs than arbitrary base-8 octree paths.

        Uses prefix grouping for O(n_concepts + n_anchors) field_bits construction.

        Args:
            lattice: SyntaxLattice with concept_freq
            n_anchors: number of anchor concepts
            min_lcp: minimum LCP to create field bit
            gamma: Zeckendorf weight decay
            path_overrides: dict {cid: tuple_path} for custom Zeckendorf paths
                           (e.g. from MorphVocab for morphological encoding)

        Sets:
            self.H: scipy.sparse.csr_matrix (n, n) of H values
            self.anchor_ids: list of anchor concept IDs
            self.anchor_idx: dict {cid: index}
        """
        import numpy as np
        from scipy.sparse import csr_matrix
        from collections import defaultdict
        from eva.symbolic.fractal_encoding import path as zeckendorf_path_default, H_weighted

        def get_path(cid):
            if path_overrides and cid in path_overrides:
                return path_overrides[cid]
            return zeckendorf_path_default(cid)

        # 1. Select anchors (top by frequency)
        sorted_cids = sorted(lattice.concept_freq.keys(),
                             key=lambda c: -lattice.concept_freq[c])
        anchor_ids = sorted_cids[:n_anchors]
        self.anchor_ids = anchor_ids
        self.anchor_idx = {cid: i for i, cid in enumerate(anchor_ids)}
        self.n_anchors = n_anchors

        # 2. Precompute octree paths for anchors
        anchor_paths = [get_path(cid) for cid in anchor_ids]

        # 3. Build H matrix (CSR) from H_weighted
        rows, cols, vals = [], [], []
        for i in range(n_anchors):
            pi = anchor_paths[i]
            for j in range(n_anchors):
                if i == j:
                    continue
                h = H_weighted(pi, anchor_paths[j], gamma)
                if h > 0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(h)

        self.H = csr_matrix((vals, (rows, cols)), shape=(n_anchors, n_anchors))
        print(f"  Octree H: {n_anchors}x{n_anchors}, "
              f"{len(vals)} non-zero ({100*len(vals)/n_anchors**2:.1f}%)")

        # 4. Build field_bits via prefix grouping (O(n) instead of O(n²))
        self.fractal.init_fields(n_anchors)
        seen_cids = set(lattice.concept_freq.keys()) & set(self.fractal.codes.keys())
        n_bytes = (n_anchors + 7) // 8

        # Group anchors by their first min_lcp digits (default min_lcp=2:
        # only LCP>=2 anchors are considered, giving ~16 groups from octal digits)
        prefix_to_anchors = defaultdict(list)
        for aidx, ap in enumerate(anchor_paths):
            prefix_to_anchors[ap[:min_lcp]].append(aidx)

        active_counts = []
        for cid in seen_cids:
            cp = get_path(cid)
            prefix = cp[:min_lcp]
            indices = prefix_to_anchors.get(prefix, [])
            bits = bytearray(n_bytes)
            for aidx in indices:
                bits[aidx >> 3] |= 1 << (aidx & 7)
            self.fractal.field_bits[cid] = np.frombuffer(bytes(bits),
                                                         dtype=np.uint8).copy()
            active_counts.append(len(indices))

        if active_counts:
            a = np.array(active_counts)
            print(f"  Octree fields: {len(seen_cids)}/{len(self.fractal.codes)} concepts, "
                  f"sizes: min={a.min()} max={a.max()} mean={a.mean():.1f}")

    def build_learned_fields(self, n_field_bits=512, sp=None):
        """Build field_bits from learned projection instead of octree.

        Initializes W_proj as random hyperplanes, computes field_bits
        from latent codes. Call periodically during training to adapt.

        Also builds morpheme field and initialises word vectors from
        morphological composition if sp is provided.
        """
        self.fractal.init_learned_fields(field_bits=n_field_bits)
        if sp is not None:
            self._build_morphemes(sp=sp)
            # Reinitialise word vectors from morpheme composition
            n_reinit = 0
            for cid, word_morphs in list(self.harmonizer.word_morphs.items()):
                morph_parts = [(r, self.harmonizer.get_morpheme_vec(m))
                               for m, r in word_morphs]
                composed = self.harmonizer.compose_word([(r, v) for r, v in morph_parts if v is not None])
                if composed is not None:
                    latent_code = composed.copy()
                    ln = float(np.linalg.norm(latent_code))
                    if ln > 1e-10:
                        latent_code /= ln
                    # Mix with existing fractal code (if any)
                    existing = self.fractal.codes.get(cid)
                    if existing is not None:
                        mix = latent_code * 0.7 + existing * 0.3
                        mix /= max(np.linalg.norm(mix), 1e-10)
                        self.fractal.codes[cid] = mix
                    else:
                        self.fractal.codes[cid] = latent_code
                    v = self.fractal.compute_vector(cid)
                    if v is not None:
                        self.concept_vectors[cid] = v
                    # Sync to EntityField for VSA-composed word vectors
                    if hasattr(self, 'entity_field') and v is not None:
                        self.entity_field.sync_word(cid, v)
                    n_reinit += 1
            if n_reinit:
                print(f"  [harmonizer] {n_reinit} words initialised from morphology")


    def _load_morph_vocab(self):
        """Load or import MorphVocab for morpheme decomposition."""
        cached = getattr(self, '_morph_vocab', None)
        if cached is not None:
            return cached
        try:
            from eva.symbolic.fcf_config import EnvironmentResolver
            path = EnvironmentResolver().morph_vocab_path
            if os.path.exists(path):
                from eva.symbolic.morph_vocab import MorphVocab
                self._morph_vocab = MorphVocab.load(path)
        except Exception as e:
            print(f"  [harmonizer] MorphVocab load failed: {e}")
        return self._morph_vocab

    def _pymorphy_decompose(self, word):
        """Fallback decomposition using pymorphy3 (Python 3.12+ compatible fork).

        Returns dict of {role: string} or None if parse confidence too low.
        """
        import pymorphy3
        morph = getattr(self, '_pymorphy_analyzer', None)
        if morph is None:
            morph = pymorphy3.MorphAnalyzer()
            self._pymorphy_analyzer = morph

        parsed = morph.parse(word)
        if not parsed or parsed[0].score < 0.3:
            return None

        p = parsed[0]
        nf = p.normal_form.lower()
        w = word.lower().strip()

        if w == nf or len(w) < 4:
            return {'ROOT': nf}

        # Known Russian prefixes (same as rule-based)
        _pfx_list = ['вз', 'воз', 'вос', 'вы', 'до', 'за', 'из', 'ис',
                     'на', 'над', 'наи', 'не', 'недо', 'низ', 'нис',
                     'о', 'об', 'обез', 'обес', 'пере', 'по', 'под',
                     'подо', 'пра', 'пред', 'пре', 'при', 'про',
                     'раз', 'рас', 'со', 'с', 'у', 'без', 'бес']

        # Strip known prefix from word
        rest = w
        pfx = ''
        for p in sorted(_pfx_list, key=len, reverse=True):
            if not rest.startswith(p):
                continue
            min_stem = 3 if len(p) == 1 else 2  # single-letter prefixes need longer stem
            if len(rest) <= len(p) + min_stem:
                continue
            # If normal_form also starts with this prefix, it's part of the root
            if nf.startswith(p) and len(nf) > len(p) + min_stem:
                continue
            pfx = p
            rest = rest[len(p):]
            break

        # Strip same prefix from normal_form (if and only if word had it)
        nf_stem = nf
        if pfx:
            for p in sorted(_pfx_list, key=len, reverse=True):
                if nf_stem.startswith(p):
                    nf_stem = nf_stem[len(p):]
                    break

        result = {}
        if pfx:
            result['PREFIX'] = pfx

        # Align rest with nf_stem via longest common prefix
        i = 0
        while i < min(len(rest), len(nf_stem)) and rest[i] == nf_stem[i]:
            i += 1

        if i >= 2:
            result['ROOT'] = rest[:i]
            if i < len(rest):
                result['ENDING'] = rest[i:]
        elif len(rest) <= len(nf_stem) + 3:
            # Word form similar length to lemma — consonant boundary split
            _cons = set('бвгджзйклмнпрстфхцчшщ')
            split_pos = max(2, len(rest) - 2)
            while split_pos < len(rest) and rest[split_pos] not in _cons:
                split_pos += 1
            if split_pos < len(rest):
                result['ROOT'] = rest[:split_pos]
                result['ENDING'] = rest[split_pos:]
            else:
                result['ROOT'] = rest
        else:
            result['ROOT'] = nf_stem if len(nf_stem) >= 2 else rest[:3]
            suffix = rest[len(result['ROOT']):] if rest.startswith(result['ROOT']) else rest[i:]
            if suffix:
                result['ENDING'] = suffix

        return result

    def _rule_decompose(self, word):
        """Rule-based Russian morpheme decomposition: prefix+stem+ending.

        Returns dict of {role: string} or None if confidence < threshold.
        """
        word = word.lower().strip()
        if len(word) < 3:
            return None

        # Known Russian prefixes (most common)
        prefixes = ['вз', 'воз', 'вос', 'вы', 'до', 'за', 'из', 'ис',
                    'на', 'над', 'наи', 'не', 'недо', 'низ', 'нис',
                    'о', 'об', 'обез', 'обес', 'пере', 'по', 'под',
                    'подо', 'пра', 'пред', 'пре', 'при', 'про',
                    'раз', 'рас', 'со', 'с', 'у', 'без', 'бес',
                    'вне', 'внутри', 'меж', 'между',
                    'после', 'сверх', 'через',
                    'анти', 'архи', 'гипер', 'де', 'дис', 'ин',
                    'контр', 'суб', 'супер', 'ультра', 'экс']

        # Known Russian endings (approximate)
        endings = ['а', 'ы', 'е', 'у', 'ой', 'ую', 'ою',
                   'ей', 'ий', 'ие', 'ия', 'ию', 'ием', 'иях',
                   'ами', 'ях', 'ах', 'ов', 'ев', 'ём', 'ем',
                   'ам', 'ом', 'ею', 'о', 'ых', 'им', 'ими',
                   'ешь', 'ет', 'ем', 'ете', 'ут', 'ют', 'ат', 'ят',
                   'ал', 'ла', 'ло', 'ли', 'ть', 'ти', 'чь',
                   'л', 'на', 'ся', 'сь', 'ого', 'его', 'ому', 'ему',
                   'ым', 'им', 'ыми', 'ими', 'ых', 'их']

        result = {}
        rest = word

        # 1. Split prefix
        pfx = ''
        for p in sorted(prefixes, key=len, reverse=True):
            if not rest.startswith(p):
                continue
            min_stem = 3 if len(p) == 1 else 2
            if len(rest) <= len(p) + min_stem:
                continue
            nxt = rest[len(p)]
            if nxt in 'аеёиоуыэюя':
                continue
            pfx = p
            rest = rest[len(p):]
            result['PREFIX'] = pfx
            break

        # 2. Split ending
        for e in sorted(endings, key=len, reverse=True):
            if len(rest) > len(e) + 1 and rest.endswith(e):
                pre = rest[-(len(e) + 1)]
                if pre in 'бвгджзйклмнпрстфхцчшщ':
                    rest = rest[:-len(e)]
                    result['ENDING'] = e
                    break

        # 3. The remainder is the stem (root + suffix)
        if rest:
            result['ROOT'] = rest

        # Confidence: > 1 morpheme found and rest ≥ 2 chars
        confidence = len(result) / 3.0
        if len(rest) < 2:
            confidence *= 0.5

        threshold = getattr(self, '_morph_conf_threshold', 0.8)
        if len(word) <= 4:
            threshold = 0.4  # lower bar for short words
        if confidence < threshold:
            return None

        return result

    @staticmethod
    def _has_cyrillic(text):
        """Check if text contains at least one Cyrillic letter."""
        return any('\u0400' <= c <= '\u04FF' for c in text)

    def _decompose_word(self, word):
        """Two-level morpheme decomposition:

        1. pymorphy3 (morphological analysis) — high accuracy, 100% coverage
        2. Rule-based (prefix+stem+ending tables) — fallback if pymorphy3 unavailable

        Returns dict of {role: string} or None if all methods fail.
        """
        # Skip tokens without Cyrillic content
        w = word.lower().strip()
        if len(w) < 3 or not self._has_cyrillic(w):
            return None
        # Skip pure punctuation/numbers
        if all(c in '.,!?;:()[]{}«»—–-…\'\"1234567890 ' for c in w):
            return None
        result = self._pymorphy_decompose(word)
        if result is not None:
            return result
        return self._rule_decompose(word)

    def _build_morphemes(self, sp=None):
        """Build morpheme field from MorphVocab: decompose known words and
        initialise morpheme vectors as quasi-orthogonal HD vectors.

        Populates self.harmonizer with morph→word mappings and morpheme vectors.

        Args:
            sp: optional SentencePieceProcessor for CID→text lookup
        """
        from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
        rng = _R.rng('morph_init')
        morph_set = set()
        n_words = 0
        n_skipped = 0

        # Collect all decompositions
        if sp is not None:
            for cid in range(min(self.vocab_size, sp.vocab_size())):
                try:
                    text = sp.IdToPiece(cid).replace('\u2581', '').strip()
                except Exception:
                    continue
                if not text or len(text) < 3:
                    continue
                # Skip pure punctuation/numbers
                if all(c in '.,!?;:()[]{}«»—–-…\'\"1234567890' for c in text):
                    continue
                decomp = self._decompose_word(text)
                if decomp is None:
                    n_skipped += 1
                    continue
                # Assign integer IDs to each unique morpheme
                morph_ids = {}
                for role, morph_str in decomp.items():
                    # Normalise ending/prefix by role to reduce vocabulary
                    key = (role, morph_str)
                    if key not in morph_set:
                        # Convert set to list to track insertion order
                        pass
                    # Use hash as stable ID
                    morph_id = abs(hash(key)) % (2**31 - 1)
                    morph_set.add(key)
                    morph_ids[role] = morph_id
                self.harmonizer.register_word(cid, morph_ids)
                n_words += 1

        # Initialise morpheme vectors: quasi-orthogonal
        morph_list = list(morph_set)
        ids_done = set()
        n_morph = 0
        for key in morph_list:
            morph_id = abs(hash(key)) % (2**31 - 1)
            if morph_id in ids_done:
                continue
            ids_done.add(morph_id)
            harm_dim = self.harmonizer.dim
            v = rng.randn(harm_dim).astype(np.float32)
            v /= max(np.linalg.norm(v), 1e-10)
            self.harmonizer.set_morpheme_vec(morph_id, v)
            n_morph += 1

        print(f"  [harmonizer] {n_morph} unique morphemes, "
              f"{n_words} words decomposed ({n_skipped} skipped)")
        return n_morph

    def update_learned_fields(self, batches_seen=0):
        """Periodic update of learned fields (Hebbian W_proj adaptation)."""
        self.fractal.update_learned_fields(batches_seen=batches_seen)

    def fluctuate_fractal(self, fluctuation_amp=0.003, decay=0.9995, repel_strength=0.0, generator=None, current_cos=None):
        if current_cos is not None and current_cos > 0:
            if current_cos > 0.25:
                cos_factor = 1.0 - (current_cos - 0.25) / 0.15
                cos_factor = max(cos_factor, 0.2)
                fluctuation_amp *= cos_factor
            elif current_cos < 0.05:
                cos_factor = current_cos / 0.05
                fluctuation_amp *= max(cos_factor, 0.3)
        self.fractal.fluctuate(fluctuation_amp=fluctuation_amp, decay=decay)
        self.fractal.hdc_memory.clear()
        self.fractal.hdc_memory_counts.clear()
        self._sync_from_fractal()
        if generator is not None:
            generator._sync_after_fluctuate()
        if repel_strength > 0:
            self._repel_centroid(repel_strength)

    def _repel_centroid(self, strength=0.05):
        """Push all vectors away from global centroid via Riemannian gradient.

        Uses the negative Riemannian gradient of dot(v, cn) on the sphere:
            -grad_R = sim * v - cn
        which is tangent at v (dot(-grad_R, v) = 0) and maximally decreases
        alignment with centroid. Degenerate at v = ±cn → random fallback.

        Uniform strength (no asymmetry threshold) — the gradient |repel| * |sim|
        naturally handles magnitude: largest for mid-similarity vectors, zero
        for v aligned with or opposite to centroid.
        """
        if len(self.concept_vectors) < 2:
            return
        # NOTE: vecs ~225MB temp array (len×768×float32); fine for ~75K vectors
        vecs = self.concept_vectors._data[self.concept_vectors._valid]
        centroid = np.mean(vecs, axis=0)
        cn = centroid / max(np.linalg.norm(centroid), 1e-10)
        for cid, v in self.concept_vectors.items():
            sim = float(np.dot(v, cn))
            repel = sim * v - cn
            nrep = np.linalg.norm(repel)
            if nrep > 1e-10:
                repel /= nrep
                v_new = v + repel * abs(sim) * strength
            else:
                rng = np.random.RandomState(cid * 137 + 42)
                tangent = rng.randn(self.dim).astype(np.float32)
                tangent -= np.dot(tangent, v) * v
                nt = np.linalg.norm(tangent)
                if nt < 1e-10:
                    continue
                tangent /= nt
                v_new = v + tangent * strength
            nv = np.linalg.norm(v_new)
            if nv > 1e-10:
                v_new /= nv
            self._apply_vector_update(cid, v_new)

    # ---- STDP: Spike-Timing-Dependent Plasticity on fractal codes ----

    def _apply_vector_update(self, cid: int, v_new: np.ndarray, max_shift: float = 0.5, ce: float = 0.0) -> None:
        """Set concept_vector[cid] directly, then sync fractal code.

        Vector is the canonical representation — fractal code is
        re-computed as projection: code = normalize(v_new @ basis.T).
        Bypasses the subspace-LR bottleneck (lr_c=0.01 was freezing
        50 % of code capacity).

        Args:
            cid: concept ID
            v_new: new vector (should already be unit-normed)
            max_shift: clamp delta_v norm to prevent explosive updates
        """
        v_old = self.concept_vectors.get(cid)
        code = self.fractal.codes.get(cid)

        if v_old is not None:
            delta_v = v_new - v_old
            shift = float(np.linalg.norm(delta_v))
            if shift > max_shift:
                delta_v = delta_v / shift * max_shift
                v_new = v_old + delta_v
                nv = np.linalg.norm(v_new)
                if nv > 1e-10:
                    v_new /= nv
                shift = max_shift
            self._total_shift += shift
            self._update_count += 1

        # Store the actual vector (canonical representation)
        self.set_vec(cid, v_new)

        # Sync fractal code to match — no subspace LR filtering
        if self.fractal.basis is not None:
            new_code = v_new @ self.fractal.basis.T
            nv_code = np.linalg.norm(new_code @ self.fractal.basis)
            if nv_code > 1e-10:
                new_code /= nv_code
            self.fractal._apply_l1(new_code, ce, cid=cid)
            self.fractal.codes[cid] = new_code
            self.fractal._matrix_dirty = True

        # Notify external hook (e.g. CrystalGenerator _vecs_t sync)
        if hasattr(self, '_after_update_hook') and self._after_update_hook is not None:
            self._after_update_hook(cid, v_new)

    def _apply_subspace_update(self, cid, grad, base_lr_val, subspace_lr):
        v_old = self.concept_vectors.get(cid)
        code = self.fractal.codes.get(cid)
        if code is None or self.fractal.basis is None:
            return
        lr_c, lr_a, lr_m = subspace_lr
        basis = self.fractal.basis
        latent_dim = basis.shape[0]
        mask_c = np.zeros(latent_dim, dtype=np.float32); mask_c[:self.fractal.l_c] = 1.0
        mask_a = np.zeros(latent_dim, dtype=np.float32); mask_a[self.fractal.l_c:self.fractal.l_c + self.fractal.l_a] = 1.0
        mask_m = np.zeros(latent_dim, dtype=np.float32); mask_m[self.fractal.l_c + self.fractal.l_a:] = 1.0
        code_grad = grad @ basis.T
        code_grad *= (lr_c * mask_c + lr_a * mask_a + lr_m * mask_m)
        code_new = code + code_grad * base_lr_val
        v_new = code_new @ basis
        nv = np.linalg.norm(v_new)
        if nv > 1e-10:
            v_new /= nv
            code_new /= np.linalg.norm(code_new)
        if v_old is not None:
            shift = float(np.linalg.norm(v_new - v_old))
            self._total_shift += shift
            self._update_count += 1
        self.set_vec(cid, v_new)
        self.fractal._apply_l1(code_new, self.concept_error.get(cid, 0.0) if hasattr(self, 'concept_error') else 0.0, cid=cid)
        self.fractal.codes[cid] = code_new
        self.fractal._matrix_dirty = True
        if hasattr(self, '_after_update_hook') and self._after_update_hook is not None:
            self._after_update_hook(cid, v_new)

    def _apply_subspace_update_batch(self, cids, grads, base_lr_val, subspace_lr, gen):
        """Batched GPU subspace update for multiple CIDs.
        cids: list[int], grads: np.ndarray (N, D), gen: CrystalGenerator with torch tensors.
        """
        lr_c, lr_a, lr_m = subspace_lr
        latent_dim = self.fractal.latent_dim
        l_c = self.fractal.l_c
        l_a = self.fractal.l_a
        device = gen._torch_device

        mask = torch.zeros(latent_dim, device=device, dtype=torch.float32)
        mask[:l_c] = lr_c
        mask[l_c:l_c + l_a] = lr_a
        mask[l_c + l_a:] = lr_m

        cids_t = torch.tensor(cids, dtype=torch.long, device=device)
        grads_t = torch.from_numpy(grads).to(device, dtype=torch.float32)
        codes = gen._codes_master_t[cids_t]
        basis_t = gen._basis_t

        code_grads = grads_t @ basis_t.T
        code_grads *= mask
        new_codes = codes + code_grads * base_lr_val

        new_vecs = new_codes @ basis_t
        nv = new_vecs.norm(dim=1, keepdim=True)
        nv[nv < 1e-10] = 1.0
        new_vecs /= nv

        nc = new_codes.norm(dim=1, keepdim=True)
        nc[nc < 1e-10] = 1.0
        new_codes /= nc

        new_vecs_np = new_vecs.cpu().numpy()
        new_codes_np = new_codes.cpu().numpy()
        # Apply L1 to z_c subspace (batch)
        if self.fractal.l1_lambda > 0 and hasattr(gen, '_ce_t'):
            ce_vals = gen._ce_t[cids_t].cpu().numpy()
            self.fractal._apply_l1_batch(new_codes_np, ce_vals.tolist(), cid_list=cids)
        gen._codes_master_t[cids_t] = new_codes.to(torch.float32)
        for i, cid in enumerate(cids):
            v_new = new_vecs_np[i]
            code_new = new_codes_np[i]
            v_old = self.concept_vectors.get(cid)
            if v_old is not None:
                shift = float(np.linalg.norm(v_new - v_old))
                self._total_shift += shift
                self._update_count += 1
            self.set_vec(cid, v_new)
            self.fractal.codes[cid] = code_new
            if hasattr(self, '_after_update_hook') and self._after_update_hook is not None:
                self._after_update_hook(cid, v_new)
        self.fractal._matrix_dirty = True

    def _lateral_inhibition_fractal(self, winner_cid, strength=0.01, threshold=0.35, sample_size=None):
        """Lateral inhibition with correct Riemannian gradient, vectorised.

        The negative Riemannian gradient of sim = dot(v, v_win) is:
            -grad_R = sim * v - v_win
        which is tangent at v and maximally decreases alignment with winner.
        The Euclidean chord (v - v_win) used previously has a radial component
        and does not follow the geodesic — fixed.

        Inner loop over sampled concepts is vectorised (numpy batch ops).
        Uses dense ndarray for O(1) gather.
        """
        v_win = self.concept_vectors.get(winner_cid)
        if v_win is None:
            return
        vw_n = v_win / max(np.linalg.norm(v_win), 1e-10)

        if sample_size is None:
            sample_size = min(200, len(self.concept_vectors))

        cids = self.concept_vectors.keys()
        n_cids = len(cids)
        if n_cids <= 1:
            return

        raw = self._inhibit_rng.randint(0, n_cids, size=sample_size + 50)
        u_idxs = np.unique(raw)
        sampled_indices = [i for i in u_idxs if cids[i] != winner_cid][:sample_size]
        if len(sampled_indices) < 1:
            return

        sampled_cids = [cids[i] for i in sampled_indices]
        sampled_vecs = self.concept_vectors.data[sampled_cids]
        sims = np.dot(sampled_vecs, vw_n)
        mask = sims > threshold
        if not np.any(mask):
            return

        affected = sampled_vecs[mask]
        sims_k = sims[mask]
        # Neg Riemannian gradient: sim*affected - vw_n (tangent to sphere)
        inhibit = sims_k[:, None] * affected - vw_n
        norms = np.linalg.norm(inhibit, axis=1)
        norms[norms < 1e-10] = 1.0
        inhibit /= norms[:, None]

        v_new = affected + inhibit * strength
        vnorms = np.linalg.norm(v_new, axis=1)
        vnorms[vnorms < 1e-10] = 1.0
        v_new /= vnorms[:, None]

        for idx, cid in enumerate([sc for sc, m in zip(sampled_cids, mask) if m]):
            self._apply_vector_update(cid, v_new[idx])

    def init_homeostasis(self):
        """Initialize homeostasis tracking for concepts."""
        self.concept_usage = {cid: 0.0 for cid in self.concept_vectors}

        self._hboost_mean_cache = None
        self._hboost_std_cache = 0.0
        self._hboost_cache_step = 0
        self._usage_decay_steps = 0

    def decay_usage(self, decay=0.98, rare_protect=False, rare_threshold=3):
        """Exponential decay of concept usage to prevent homeostatic saturation."""
        for cid in self.concept_usage:
            if rare_protect and cid in self.fractal.codes and np.any(self.fractal.codes[cid] != 0):
                if self.concept_usage[cid] < rare_threshold:
                    continue
            self.concept_usage[cid] *= decay
        self._usage_decay_steps += 1
        self._hboost_mean_cache = None

    def check_code_range(self, bound=10.0):
        """Check if any fractal code exceeds |bound|. Returns (n_outliers, max_abs)."""
        if not self.fractal.codes:
            return 0, 0.0
        codes_list = list(self.fractal.codes.values())
        all_codes = np.array(codes_list, dtype=np.float32)
        abs_codes = np.abs(all_codes)
        max_abs = float(np.max(abs_codes))
        n_out = int(np.sum(np.max(abs_codes, axis=1) > bound))
        return n_out, max_abs

    def validate_vector_norms(self):
        """Check all vectors are unit norm. Returns (ok_count, total, max_deviation)."""
        if not self.concept_vectors:
            return 0, 0, 0.0
        all_vecs = self.concept_vectors._data[self.concept_vectors._valid]
        norms = np.linalg.norm(all_vecs, axis=1)
        devs = np.abs(norms - 1.0)
        ok = int(np.sum(devs < 1e-6))
        max_dev = float(np.max(devs))
        return ok, len(self.concept_vectors), max_dev

    _hboost_mean_cache = None
    _hboost_cache_step = 0

    def homeostatic_boost(self, cid):
        """Get homeostatic boost for a concept.
        Underused -> positive boost (novelty)
        Overused -> negative boost (fatigue)"""
        usage = self.concept_usage.get(cid, 0.0)
        # Refresh cache every 1000 calls
        self._hboost_cache_step += 1
        if self._hboost_cache_step % 1000 == 0 or self._hboost_mean_cache is None:
            vals = list(self.concept_usage.values())
            self._hboost_mean_cache = np.mean(vals) if vals else 1.0
            self._hboost_std_cache = np.std(vals) if vals and len(vals) > 1 else 0.0
        mean_usage = self._hboost_mean_cache
        std_usage = self._hboost_std_cache
        if mean_usage < 0.01:
            return 0.0
        denom = max(std_usage, 0.01 * mean_usage)
        boost = (mean_usage - usage) / denom
        return np.clip(boost, -0.3, 0.3)

    def update_usage(self, cid, delta=1.0):
        """Record concept usage for homeostasis."""
        if cid in self.concept_usage:
            # Exponential moving average
            alpha = 0.1
            self.concept_usage[cid] = (1 - alpha) * self.concept_usage[cid] + alpha * delta

    # ---- Query API ----

    def concept_vector(self, cid):
        """Get concept centroid vector."""
        return self.concept_vectors.get(cid)

    def topk_similar_concepts(self, cid, k=10, sample_size=None):
        """Top-k concepts closest to given concept (dense array NN)."""
        v = self.concept_vectors.get(cid)
        if v is None:
            return []
        if sample_size is not None:
            all_cids = list(self.concept_vectors)
            if len(all_cids) > sample_size:
                rng = random.Random(cid)
                sampled = rng.sample(all_cids, sample_size)
                mask = np.zeros(self.concept_vectors.size, dtype=bool)
                for sc in sampled:
                    if self.concept_vectors.valid[sc]:
                        mask[sc] = True
                valid = mask
            else:
                valid = self.concept_vectors.valid
        else:
            valid = self.concept_vectors.valid
        mat = self.concept_vectors.data[valid]
        order = np.where(valid)[0]
        if mat.shape[0] == 0:
            return []
        vn = v / max(np.linalg.norm(v), 1e-10)
        sims = mat @ vn
        n = len(sims)
        k_actual = min(k + 1, n)
        if k_actual <= 0:
            return []
        idx = np.argpartition(-sims, k_actual - 1)[:k_actual]
        idx = idx[np.argsort(-sims[idx])]
        result = []
        for i in idx:
            c = int(order[i])
            if c == cid:
                continue
            result.append((c, float(sims[i])))
            if len(result) >= k:
                break
        return result[:k]

    def batch_dot(self, ctx_ids, target_id):
        """Batch dot products of context vectors with target vector.

        Args:
            ctx_ids: list of concept IDs
            target_id: target concept ID
        Returns:
            list of float dot products
        """
        tv = self.concept_vectors[target_id]
        return [float(np.dot(self.concept_vectors[c], tv)) for c in ctx_ids]

    def save(self, path: str, use_pq: bool = False) -> None:
        """Save ConceptSpace to disk.

        Args:
            path: file path
            use_pq: if True, save PQ-compressed format (much smaller).
        """
        # Binary .npz for fractal codes
        clean = path[:-4] if path.endswith('.tmp') else path
        binary_path = clean.replace('.json', '.codes.npz')
        data = {
            'dim': self.dim,
            'vocab_size': self.vocab_size,
        }
        data['fractal'] = self.fractal.to_dict(binary_path=binary_path)
        concept_usage = getattr(self, 'concept_usage', None)
        if concept_usage is not None:
            data['concept_usage'] = {str(c): u for c, u in concept_usage.items()}
        data['inhibit_rng_state'] = [s.tolist() if isinstance(s, np.ndarray) else s for s in self._inhibit_rng.get_state()]
        data['inhibition_step'] = self._inhibition_step
        data['total_shift'] = self._total_shift
        data['update_count'] = self._update_count
        data['usage_decay_steps'] = getattr(self, '_usage_decay_steps', 0)

        harmonizer_state = getattr(self, 'harmonizer', None)
        if harmonizer_state is not None and harmonizer_state.morphemes:
            harm_data = harmonizer_state.to_dict()
            data['harm_dim'] = harm_data['harm_dim']
            data['harm_lr'] = harm_data['harm_lr']
            data['morph_lr'] = harm_data['morph_lr']
            data['harm_n_iter'] = harm_data['n_iter']
            data['harm_words'] = harm_data['harm_words'].tolist()
            data['harm_mids_flat'] = harm_data['harm_mids_flat'].tolist()
            data['harm_roles_flat'] = harm_data['harm_roles_flat']
            data['harm_word_lens'] = harm_data['harm_word_lens'].tolist()
            # Morpheme vectors go into npz, not json
            npz_harm = {
                'harm_morph_ids': harm_data['harm_morph_ids'],
                'harm_morph_vecs': harm_data['harm_morph_vecs'],
            }
            npz_path = clean.replace('.json', '.codes.npz')
            if os.path.exists(npz_path):
                # Merge into existing npz
                existing = dict(np.load(npz_path, allow_pickle=True))
                existing.update(npz_harm)
                np.savez_compressed(npz_path, **existing)
            else:
                np.savez_compressed(npz_path, **npz_harm)

        # EntityField save → npz
        ef = getattr(self, 'entity_field', None)
        if ef is not None and len(ef.entities) > 0:
            ef_data = ef.to_dict()
            npz_ef = {
                'ef_keys': np.array(ef_data['ef_keys'], dtype=object),
                'ef_vecs': ef_data['ef_vecs'],
                'ef_dim': ef_data['ef_dim'],
            }
            npz_path = clean.replace('.json', '.codes.npz')
            if os.path.exists(npz_path):
                existing = dict(np.load(npz_path, allow_pickle=True))
                existing.update(npz_ef)
                np.savez_compressed(npz_path, **existing)
            else:
                np.savez_compressed(npz_path, **npz_ef)

        with open(path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(path + '.tmp', path)
        json_kb = os.path.getsize(path) / 1024
        npz_kb = os.path.getsize(binary_path) / 1024 if os.path.exists(binary_path) else 0
        total_mb = (json_kb + npz_kb) / 1024
        note = 'PQ ' if use_pq else ''
        print(f"  Saved ConceptSpace ({note}{total_mb:.0f}MB = {json_kb/1024:.1f}MB json + {npz_kb/1024:.1f}MB npz) to {path}")

    @classmethod
    def load(cls, path):
        """Load ConceptSpace from disk (class method)."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.dim = data['dim']
        obj.vocab_size = data.get('vocab_size', 0)

        obj.concept_vectors = ConceptVectorStore(obj.vocab_size, obj.dim)

        latent_from_data = data.get('fractal', {}).get('latent_dim', None)
        obj.dims = DimensionCoordinator(vec_dim=obj.dim,
                                        latent_dim=latent_from_data or obj.dim * 2)

        from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
        obj.rng = _R.rng('rng')
        obj._item_rng = _R.rng('item_rng')
        rng_state = data.get('inhibit_rng_state')
        if rng_state is not None:
            from eva.symbolic.seed_registry import DEFAULT_REGISTRY as __R
            obj._inhibit_rng = __R.rng('inhibit_rng')
            obj._inhibit_rng.set_state(tuple(rng_state))
        else:
            obj._inhibit_rng = _R.rng('inhibit_rng')
        obj._inhibition_step = data.get('inhibition_step', 0)
        obj._total_shift = data.get('total_shift', 0.0)
        obj._update_count = data.get('update_count', 0)
        obj._usage_decay_steps = data.get('usage_decay_steps', 0)

        if 'fractal' in data:
            base_dir = os.path.dirname(path)
            obj.fractal = FractalField.from_dict(data['fractal'], base_dir=base_dir)
            for cid in list(obj.fractal.codes.keys()):
                v = obj.fractal.compute_vector(cid)
                if v is not None:
                    obj.concept_vectors[cid] = v
        else:
            obj.fractal = FractalField(dim=obj.dim, latent_dim=obj.fractal.latent_dim if hasattr(obj, 'fractal') and hasattr(obj.fractal, 'latent_dim') else obj.dim * 2)
            valid_idxs = np.flatnonzero(obj.concept_vectors._valid)
            for idx in valid_idxs:
                v = obj.concept_vectors._data[idx]
                cid = int(idx)
                code = v @ obj.fractal.basis.T
                nv = np.linalg.norm(code @ obj.fractal.basis)
                if nv > 1e-10:
                    code /= nv
                obj.fractal.codes[cid] = code

        saved_usage = data.get('concept_usage')
        if saved_usage:
            obj.concept_usage = {int(c): u for c, u in saved_usage.items()}
        else:
            obj.init_homeostasis()
        # Ensure all vocab CIDs have entries (P2-10)
        for cid in range(obj.vocab_size):
            if cid not in obj.concept_usage:
                obj.concept_usage[cid] = 0

        # ── Restore Harmonizer ──────────────────────────────────
        if 'harm_dim' in data:
            harm_data = {
                'harm_dim': data['harm_dim'],
                'harm_lr': data.get('harm_lr', 0.05),
                'morph_lr': data.get('morph_lr', 0.03),
                'n_iter': data.get('harm_n_iter', 5),
                'harm_words': np.array(data.get('harm_words', []), dtype=np.int64),
                'harm_mids_flat': np.array(data.get('harm_mids_flat', []), dtype=np.int64),
                'harm_roles_flat': data.get('harm_roles_flat', []),
                'harm_word_lens': np.array(data.get('harm_word_lens', []), dtype=np.int32),
            }
            # Load morpheme vectors from npz
            binary_path = path.replace('.json', '.codes.npz')
            if os.path.exists(binary_path):
                npz = np.load(binary_path, allow_pickle=True)
                harm_data['harm_morph_ids'] = npz.get('harm_morph_ids', np.array([], dtype=np.int64))
                harm_data['harm_morph_vecs'] = npz.get('harm_morph_vecs', np.empty((0, data['harm_dim']), dtype=np.float32))
                npz.close()
            else:
                harm_data['harm_morph_ids'] = np.array([], dtype=np.int64)
                harm_data['harm_morph_vecs'] = np.empty((0, data['harm_dim']), dtype=np.float32)
            obj.harmonizer = Harmonizer.from_dict(harm_data)
            print(f"  Restored Harmonizer: {len(obj.harmonizer.morphemes)} morphemes, {len(obj.harmonizer.word_morphs)} words")
        else:
            harm_dim = getattr(obj, 'latent_dim', getattr(getattr(obj, 'fractal', None), 'latent_dim', None))
            harm_dim = harm_dim or obj.dim
            obj.harmonizer = Harmonizer(dim=harm_dim)

        # ── Restore EntityField ────────────────────────────────
        binary_path = path.replace('.json', '.codes.npz')
        if os.path.exists(binary_path):
            npz = np.load(binary_path, allow_pickle=True)
            ef_keys = npz.get('ef_keys', None)
            ef_vecs = npz.get('ef_vecs', None)
            ef_dim = npz.get('ef_dim', None)
            if ef_keys is not None and ef_vecs is not None:
                ef_data = {
                    'ef_keys': list(ef_keys),
                    'ef_vecs': np.array(ef_vecs, dtype=np.float32),
                    'ef_dim': int(ef_dim) if ef_dim is not None else obj.fractal.latent_dim,
                }
                obj.entity_field = EntityField.from_dict(ef_data, word_store=obj.concept_vectors)
                print(f"  Restored EntityField: {len(obj.entity_field.entities)} entities")
            else:
                ef_dim = obj.dims.latent_dim
                obj.entity_field = EntityField(dim=ef_dim, word_store=obj.concept_vectors)
            npz.close()
        else:
            obj.entity_field = EntityField(dim=obj.dims.latent_dim, word_store=obj.concept_vectors)

        print(f"  Loaded ConceptSpace: {len(obj.concept_vectors)} concepts @ {obj.dim}D")
        return obj


if __name__ == '__main__':
    from eva.symbolic.fcf_config import EnvironmentResolver
    _env = EnvironmentResolver()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=_env.bpe_model_path)

    print("Initializing ConceptSpace with BPE vocabulary...")
    cs = ConceptSpace(vocab_size=sp.vocab_size())
    cs.init_concepts()
    cs.init_homeostasis()

    print(f"\nConcepts: {cs.vocab_size}")
    print(f"Vector dim: {cs.dim}")

    # Test vector properties
    sample_tokens = [0, 100, 1000, 5000, 9999]
    for tid in sample_tokens:
        tok_text = sp.IdToPiece(tid) if tid < sp.vocab_size() else '?'
        v = cs.concept_vector(tid)
        norm = np.linalg.norm(v) if v is not None else 0.0
        print(f"  CID {tid:5d} ({tok_text:12s}) -> norm={norm:.4f}")

    # Test similarity of related tokens
    pairs = [('▁соба', 'ка'), ('▁ко', 'шка'), ('▁человек', 'а')]
    for a, b in pairs:
        id_a = sp.PieceToId(a)
        id_b = sp.PieceToId(b)
        if id_a >= 0 and id_b >= 0:
            va = cs.concept_vector(id_a)
            vb = cs.concept_vector(id_b)
            sim = float(va @ vb) if va is not None and vb is not None else -99
            print(f"  sim({a:12s} [{id_a}], {b:12s} [{id_b}]) = {sim:.4f}")

    # Top-k from one token
    cid = sp.PieceToId('▁человек')
    if cid >= 0:
        top = cs.topk_similar_concepts(cid, k=10)
        print(f"\nTop-10 similar to {sp.IdToPiece(cid)} (CID {cid}):")
        for c, s in top:
            print(f"  {sp.IdToPiece(c):20s} (CID {c:5d}) sim={s:.4f}")

    cs.save(_env.cs_path)
