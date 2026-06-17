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
from typing import Dict


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

    def __init__(self, dim=384, latent_dim=512, l_c=None, l_a=None, l_m=None):
        self.dim = dim
        self.latent_dim = latent_dim
        if l_c is not None and l_a is not None and l_m is not None:
            self.l_c, self.l_a, self.l_m = l_c, l_a, l_m
        else:
            self.l_c = latent_dim // 2      # 256 — identity
            self.l_a = latent_dim // 4      # 128 — attention
            self.l_m = latent_dim - self.l_c - self.l_a  # 128 — meta

        # Fractal basis: (latent_dim, dim) with orthonormal columns
        rng = np.random.RandomState(42)
        mat = rng.randn(latent_dim, dim).astype(np.float32)
        Q, _ = np.linalg.qr(mat, mode='reduced')
        self.basis = Q.astype(np.float32)
        # Latent codes: cid → (latent_dim,) array
        self.codes = {}

        # Field bits (lazy init via init_fields())
        self.field_bits: Dict[int, np.ndarray] = {}
        self._fb_dirty = False

        # Cache
        self._vector_matrix = None
        self._cid_order = []
        self._matrix_dirty = True

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

        # z_c: sparse identity
        n_active = max(self.l_c // 8, 16)
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
        """
        self.field_bits = {}
        n_bytes = (n_anchors + 7) // 8
        for cid in self.codes:
            self.field_bits[cid] = np.zeros(n_bytes, dtype=np.uint8)
        self._fb_dirty = True

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

    def fluctuate(self, noise_scale=0.005, decay=0.999):
        """Apply autonomous drift to all latent codes."""
        if not hasattr(self, '_fluct_rng'):
            self._fluct_rng = np.random.RandomState(42)
        for cid in list(self.codes.keys()):
            c = self.codes[cid]
            noise = self._fluct_rng.randn(self.latent_dim).astype(np.float32) * noise_scale
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
            npz = np.load(path, allow_pickle=False)
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

    def __init__(self, vocab_size=None, dim=384):
        self.vocab_size = vocab_size or 0
        self.dim = dim

        # Fractal field: latent codes → full vectors via shared basis
        self.fractal = FractalField(dim=self.dim, latent_dim=512)

        # Concept vectors: dense ndarray[V, dim] with dict-like convenience
        self.concept_vectors = ConceptVectorStore(self.vocab_size, self.dim)

        # Random state
        self.rng = np.random.RandomState(42)
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

    def fluctuate_fractal(self, noise_scale=0.003, decay=0.9995, repel_strength=0.0, generator=None):
        """Autonomous drift + optional centroid repulsion.

        Args:
            generator: Optional CrystalGenerator instance whose GPU tensors
                       to invalidate after the drift.
        """
        self.fractal.fluctuate(noise_scale=noise_scale, decay=decay)
        self._sync_from_fractal()
        if generator is not None:
            generator._invalidate_torch()
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

    def _apply_vector_update(self, cid: int, v_new: np.ndarray, max_shift: float = 0.5) -> None:
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
            self.fractal.codes[cid] = new_code
            self.fractal._matrix_dirty = True

        # Notify external hook (e.g. CrystalGenerator _vecs_t sync)
        if hasattr(self, '_after_update_hook') and self._after_update_hook is not None:
            self._after_update_hook(cid, v_new)

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

    def save(self, path: str, use_pq: bool = False) -> None:
        """Save ConceptSpace to disk.

        Args:
            path: file path
            use_pq: if True, save PQ-compressed format (much smaller).
        """
        # Binary .npz for fractal codes
        binary_path = path.replace('.json', '.codes.npz')
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

        if 'fractal' in data:
            base_dir = os.path.dirname(path)
            obj.fractal = FractalField.from_dict(data['fractal'], base_dir=base_dir)
            for cid in list(obj.fractal.codes.keys()):
                v = obj.fractal.compute_vector(cid)
                if v is not None:
                    obj.concept_vectors[cid] = v
        else:
            obj.fractal = FractalField(dim=obj.dim, latent_dim=512)
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
    cs = ConceptSpace(vocab_size=sp.vocab_size(), dim=384)
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
