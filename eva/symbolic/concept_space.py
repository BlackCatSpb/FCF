"""ConceptSpace — vector space for BPE-token concepts.

Architecture:
  - Each BPE token is a concept with a vector on the unit sphere
  - Vectors are computed from a fractal field: v = code @ basis
  - Concept transitions learned via STDP from corpus (token_i → token_j)
  - Generation = concept navigation -> token sequence -> SentencePiece decode

The concept vocabulary is defined by a SentencePiece model trained
on the corpus. No external knowledge bases (ConceptNet) needed.
"""

import numpy as np
from collections import defaultdict, Counter
import math, json, os, random
from typing import Dict, List, Optional
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


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

    def __init__(self, dim=768, latent_dim=2048, l_c=None, l_a=None, l_m=None, l1_lambda=0.001,
                 n_field_bits=512, field_lr=0.01):
        self.dim = dim
        self.latent_dim = latent_dim
        self.l1_lambda = l1_lambda
        if l_c is not None and l_a is not None and l_m is not None:
            self.l_c, self.l_a, self.l_m = l_c, l_a, l_m
        else:
            self.l_c = latent_dim * 3 // 5      # ~60% — identity
            self.l_a = latent_dim // 4          # 25%  — attention
            self.l_m = latent_dim - self.l_c - self.l_a  # ~15% — meta

        # Fractal basis: (latent_dim, dim) with orthonormal columns
        rng = np.random.RandomState(42)
        mat = rng.randn(latent_dim, dim).astype(np.float32)
        Q, _ = np.linalg.qr(mat, mode='reduced')
        self.basis = Q.astype(np.float32)
        # Latent codes: cid → (latent_dim,) array
        self.codes = {}

        # Learnable field projection: code @ W_proj → binarized field bits
        self.n_field_bits = n_field_bits
        self.field_lr = field_lr
        self.W_proj: Optional[np.ndarray] = None  # [latent_dim, n_field_bits]
        self.field_bits: Dict[int, np.ndarray] = {}
        self._fb_dirty = False

        # HDC n-gram memory: prefix_cids_tuple → bundled latent repr (LRU-capped)
        self.hdc_memory: Dict[tuple, np.ndarray] = {}
        self.hdc_memory_counts: Dict[tuple, int] = {}
        self.hdc_memory_max = 50000  # evict oldest when exceeding
        self._hdc_access_order: List[tuple] = []  # simple FIFO eviction queue

        # Per-concept adaptive L1 lambda (dynamic dimensionality)
        self.l1_lambda_per_cid: Dict[int, float] = {}
        self.l1_target_density = 0.08  # target 8% active in z_c
        self.l1_density_window: Dict[int, list] = {}

        # Cache
        self._vector_matrix = None
        self._cid_order = []
        self._matrix_dirty = True

        # Dynamic capacity growth tracking
        self._capacity_growths = 0
        self._density_threshold_grow = 0.15   # grow if mean density exceeds 15%
        self._density_threshold_prune = 0.01  # prune if dimension sparse across all codes
        self._growth_factor = 1.5  # multiply latent_dim by this when growing

        # Sector index for focal search (field-in-field)
        self._sector_W: List[np.ndarray] = []  # per-level W_proj
        self._sector_index: Dict[int, Dict[tuple, list]] = {}  # depth → {prefix → [cids]}
        self._sector_depths: list = [4, 10, 20]  # bits at each depth level

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

        z_c: sparse identity pattern (~12% active)
        z_a: small noise (context attention starts neutral)
        z_m: near zero (meta-gates start open)
        """
        seed = rng_seed if rng_seed is not None else cid * 137 + 42
        rng = np.random.RandomState(abs(seed) % (2**31))

        z = np.zeros(self.latent_dim, dtype=np.float32)

        # z_c: sparse identity (~3% active, room to grow via STDP)
        n_active = max(int(self.l_c * 0.03), 8)
        idxs = rng.choice(self.l_c, n_active, replace=False)
        vals = rng.randn(n_active).astype(np.float32)
        z[:self.l_c][idxs] = vals

        # z_a: small noise
        z[self.l_c:self.l_c + self.l_a] = rng.randn(self.l_a).astype(np.float32) * 0.01

        # z_m: near zero — meta gates start neutral
        z[self.l_c + self.l_a:] = rng.randn(self.l_m).astype(np.float32) * 0.001

        # Rescale so |code @ basis| = 1
        v_raw = z @ self.basis
        scale = 1.0 / (np.linalg.norm(v_raw) + 1e-10)
        z *= scale

        self.codes[cid] = z
        self._matrix_dirty = True
        return self.compute_vector(cid)

    # ── Field bits ───────────────────────────────────────────

    def init_fields(self, n_anchors=1024):
        """Initialize binary field bit arrays for all concepts.

        field_bits[cid] = np.uint8 array of n_anchors/8 bytes.
        Used by octree encoding path (build_octree_fields).
        """
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
        rng = np.random.RandomState(42)
        scale = 1.0 / np.sqrt(self.latent_dim)
        self.W_proj = rng.randn(self.latent_dim, field_bits).astype(np.float32) * scale
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
            elif mean_density < self.l1_target_density * 0.3 and current_lambda > 1e-6:
                # Too sparse — relax sparsity pressure
                new_lambda = current_lambda * (1.0 - 0.1 * lr_scale)
                self.l1_lambda_per_cid[cid] = max(new_lambda, 1e-6)
                n_adjusted += 1
        if n_adjusted:
            n_total = len([v for v in self.l1_density_window.values() if len(v) >= 10])
            print(f"  Adaptive L1: adjusted {n_adjusted}/{n_total} concepts")

    # ── Dynamic capacity ─────────────────────────────────────

    def grow_capacity(self, new_latent_dim=None):
        """Grow latent_dim by adding orthogonal basis vectors.

        Preserves existing subspace structure. Extends all codes with zeros
        in new dimensions (STDP will populate them).
        """
        old_dim = self.latent_dim
        if new_latent_dim is None:
            new_latent_dim = int(old_dim * self._growth_factor)
        new_latent_dim = max(new_latent_dim, old_dim + 8)  # at least 8 new dims
        # Ensure new dim respects subspace alignment
        new_latent_dim = ((new_latent_dim + 7) // 8) * 8

        # Generate new orthogonal basis vectors
        rng = np.random.RandomState(42 + self._capacity_growths)
        n_new = new_latent_dim - old_dim
        mat = rng.randn(n_new, self.dim).astype(np.float32)
        # Orthogonalise against existing basis
        residual = mat - mat @ self.basis.T @ self.basis
        Q_new, _ = np.linalg.qr(residual, mode='reduced')
        self.basis = np.vstack([self.basis, Q_new.astype(np.float32)])

        # Extend all codes with zeros
        for cid in self.codes:
            self.codes[cid] = np.append(self.codes[cid], np.zeros(n_new, dtype=np.float32))

        # Update subspace ratios
        old_l_c, old_l_a, old_l_m = self.l_c, self.l_a, self.l_m
        self.latent_dim = new_latent_dim
        self.l_c = new_latent_dim * 3 // 5
        self.l_a = new_latent_dim // 4
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
        print(f"  Grown capacity: {old_dim} -> {new_latent_dim} "
              f"(l_c={self.l_c} l_a={self.l_a} l_m={self.l_m})")
        return new_latent_dim

    def prune_capacity(self, sparsity_threshold=0.98):
        """Prune near-zero latent dimensions across all codes.

        A dimension is pruned if > 98% of codes have |val| < 1e-4.
        Returns number of pruned dimensions.
        """
        if len(self.codes) < 10:
            return 0
        codes_arr = np.array(list(self.codes.values()), dtype=np.float32)
        active_frac = np.mean(np.abs(codes_arr) > 1e-4, axis=0)
        dead = np.where(active_frac < (1.0 - sparsity_threshold))[0]
        if len(dead) == 0:
            return 0

        # Keep only live dimensions
        live = np.where(active_frac >= (1.0 - sparsity_threshold))[0]
        live_set = set(live)
        old_dim = self.latent_dim

        # Remap: live dims form new code, basis, W_proj
        new_basis_rows = self.basis[live]
        # Re-orthogonalise to maintain orthonormal basis
        Q, _ = np.linalg.qr(new_basis_rows.T, mode='reduced')
        self.basis = Q.T.astype(np.float32)
        new_latent_dim = len(live)

        for cid in self.codes:
            self.codes[cid] = self.codes[cid][live]

        self.latent_dim = new_latent_dim
        self.l_c = new_latent_dim * 3 // 5
        self.l_a = new_latent_dim // 4
        self.l_m = new_latent_dim - self.l_c - self.l_a

        if self.W_proj is not None:
            self.W_proj = self.W_proj[live]

        for lvl in range(len(self._sector_W)):
            self._sector_W[lvl] = self._sector_W[lvl][live]

        self._matrix_dirty = True
        self._fb_dirty = True
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

        if mean_density > self._density_threshold_grow:
            self.grow_capacity()
        elif mean_density < self._density_threshold_prune * 2:
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
        rng = np.random.RandomState(42)
        scale = 1.0 / np.sqrt(self.latent_dim)
        self._sector_W = []
        for n_bits in depths:
            W = rng.randn(self.latent_dim, n_bits).astype(np.float32) * scale
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
        """Element-wise multiply (real-valued VSA binding)."""
        return a * b

    def hdc_permute(self, v, n=1):
        """Circular shift by n positions."""
        return np.roll(v, n)

    def hdc_bundle(self, v, accum, lr=0.1):
        """Bundle v into accumulator (adaptive running average)."""
        return accum * (1.0 - lr) + v * lr

    def hdc_ngram_repr(self, codes):
        """Build HDC representation for an n-gram sequence of codes.

        For (w1, w2, ..., wn): ρ^{n-1}(w1) ⊙ ρ^{n-2}(w2) ⊙ ... ⊙ wn
        """
        n = len(codes)
        if n == 0:
            return None
        result = codes[-1].copy()
        for i in range(n - 1):
            result = self.hdc_bind(self.hdc_permute(codes[i], n - 1 - i), result)
        return result

    def hdc_unbind(self, context_codes, memory_repr):
        """Given context (prefix) and n-gram memory, unbind to find next token.

        query = hdc_ngram_repr(context) ⊙ memory_repr ≈ next_token_code
        Context = all but last token of the n-gram.
        """
        ctx_repr = self.hdc_ngram_repr(context_codes)
        if ctx_repr is None:
            return None
        return self.hdc_bind(ctx_repr, memory_repr)

    def hdc_update_ngram(self, prefix_cids, next_code):
        """Update HDC memory for {prefix_cids → next_token_code}.

        Bundles next_code into hdc_memory[prefix_cids] (running average).
        Evicts oldest entries when over hdc_memory_max.
        """
        key = tuple(prefix_cids)
        if key not in self.hdc_memory:
            if len(self.hdc_memory) >= self.hdc_memory_max and self._hdc_access_order:
                # FIFO eviction
                evict_key = self._hdc_access_order.pop(0)
                self.hdc_memory.pop(evict_key, None)
                self.hdc_memory_counts.pop(evict_key, None)
            self.hdc_memory[key] = next_code.copy()
            self.hdc_memory_counts[key] = 1
            self._hdc_access_order.append(key)
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
            # Mark as recently accessed (move to end of FIFO queue)
            if key in self._hdc_access_order:
                self._hdc_access_order.remove(key)
                self._hdc_access_order.append(key)
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
                sims.append((cid, score))
            sims.sort(key=lambda x: -x[1])
            return sims[:k]
        else:
            return []

        # Query: unbind memory repr with context codes to find next token
        ctx_codes = [self.codes.get(cid) for cid in key]
        ctx_codes = [c for c in ctx_codes if c is not None]
        if len(ctx_codes) < 1:
            return []
        query = self.hdc_unbind(ctx_codes, mem_repr)
        if query is None:
            return []
        qnorm = np.linalg.norm(query)
        if qnorm < 1e-10:
            return []
        query /= qnorm

        sims = []
        for cid, code in all_codes.items():
            score = float(query @ code / (np.linalg.norm(code) + 1e-10))
            sims.append((cid, score))
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
            self._fluct_rng = np.random.RandomState(42)
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
            return {
                'dim': self.dim,
                'latent_dim': self.latent_dim,
                'binary_codes': os.path.basename(binary_path),
                'n_codes': len(cids),
            }
        return {
            'dim': self.dim,
            'latent_dim': self.latent_dim,
            'basis': self.basis.tolist(),
            'codes': {str(cid): c.tolist() for cid, c in self.codes.items()},
        }

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

        field._matrix_dirty = True
        return field


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

        # Fractal field: latent codes → full vectors via shared basis
        self.fractal = FractalField(dim=self.dim, latent_dim=latent_dim)

        # Concept vectors: dense ndarray[V, dim] with dict-like convenience
        self.concept_vectors = ConceptVectorStore(self.vocab_size, self.dim)

        # Random state
        self.rng = np.random.RandomState(42)
        self._item_rng = np.random.RandomState(42)
        self._inhibition_step = 0
        self._inhibit_rng = np.random.RandomState(42)

        # Shift tracking
        self._total_shift = 0.0
        self._update_count = 0
        self._after_update_hook = None

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

    def reinit_rare(self, freq_map, threshold=3):
        """Replace rare concept vectors with random unit vectors (item memory).
        Concepts with freq < threshold get random vectors, naturally orthogonal.
        """
        reinit_count = 0
        for cid in range(self.vocab_size):
            freq = freq_map.get(cid, 0)
            if 0 < freq < threshold:
                v = self._item_rng.randn(self.dim).astype(np.float32)
                v /= max(np.linalg.norm(v), 1e-10)
                self.set_vec(cid, v)
                self.fractal.codes.pop(cid, None)
                reinit_count += 1
        if reinit_count:
            self.fractal._matrix_dirty = True
        return reinit_count

    def build_octree_fields(self, lattice, n_anchors=1024, min_lcp=1, gamma=0.5, path_overrides=None):
        """Build H matrix and field_bits from nested octree encoding.

        Replaces PMI-based build_anchor_matrix + build_fields_from_lattice.
        Each concept ID → decimal digits → octant path (0..7 per level).
        H[i,j] = (1 - γ^{LCP}) / (1 - γ) where LCP = longest common prefix.

        Uses prefix grouping for O(n_concepts + n_anchors) field_bits construction.

        Args:
            lattice: SyntaxLattice instance (needed for concept_freq)
            n_anchors: number of anchor concepts (top by frequency)
            min_lcp: minimum LCP for field_bits (2 → only LCP≥2 anchors)
            gamma: octree weight decay
            path_overrides: dict {cid: tuple_path} for custom octree paths
                           (e.g. from MorphVocab for morphological encoding)

        Sets:
            self.H: scipy.sparse.csr_matrix (n, n) of H values
            self.anchor_ids: list of anchor concept IDs
            self.anchor_idx: dict {cid: index}
        """
        import numpy as np
        from scipy.sparse import csr_matrix
        from collections import defaultdict
        from eva.symbolic.fractal_encoding import path as octree_path_default, H_weighted

        def get_path(cid):
            if path_overrides and cid in path_overrides:
                return path_overrides[cid]
            return octree_path_default(cid)

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

    def build_learned_fields(self, n_field_bits=512):
        """Build field_bits from learned projection instead of octree.

        Initializes W_proj as random hyperplanes, computes field_bits
        from latent codes. Call periodically during training to adapt.
        """
        self.fractal.init_learned_fields(field_bits=n_field_bits)

    def update_learned_fields(self, batches_seen=0):
        """Periodic update of learned fields (Hebbian W_proj adaptation)."""
        self.fractal.update_learned_fields(batches_seen=batches_seen)

    def fluctuate_fractal(self, fluctuation_amp=0.003, decay=0.9995, repel_strength=0.0, generator=None, current_cos=None):
        """Autonomous drift + optional centroid repulsion.

        Args:
            generator: Optional CrystalGenerator instance whose GPU tensors
                       to invalidate after the drift.
            current_cos: Optional float — mean cosine similarity. Used to
                         modulate drift: high cos → reduce amp (prevent collapse),
                         low cos (<0.05) → reduce amp (too sparse).
        """
        if current_cos is not None and current_cos > 0:
            if current_cos > 0.25:
                cos_factor = 1.0 - (current_cos - 0.25) / 0.15
                cos_factor = max(cos_factor, 0.2)
                fluctuation_amp *= cos_factor
            elif current_cos < 0.05:
                cos_factor = current_cos / 0.05
                fluctuation_amp *= max(cos_factor, 0.3)
        self.fractal.fluctuate(fluctuation_amp=fluctuation_amp, decay=decay)
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
        codes = gen._codes_t[cids_t]
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
        gen._codes_t[cids_t] = new_codes.to(torch.float16)
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

        obj.rng = np.random.RandomState(42)
        rng_state = data.get('inhibit_rng_state')
        if rng_state is not None:
            obj._inhibit_rng = np.random.RandomState()
            obj._inhibit_rng.set_state(tuple(rng_state))
        else:
            obj._inhibit_rng = np.random.RandomState(42)
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
        print(f"  Loaded ConceptSpace: {len(obj.concept_vectors)} concepts @ {obj.dim}D")
        return obj


if __name__ == '__main__':
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(        model_file=os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'bpe_ru_146k.model'))

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

    cs.save(os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'concept_space.json'))
