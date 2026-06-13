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
import math, json, os


class FractalField:
    """Fractal computation matrix for relative concept vector space.

    Code split into subspaces:
        z = [z_c | z_a | z_m]
        z_c: concept identity (slow plasticity)
        z_a: attention/context mask (fast plasticity)
        z_m: meta-plasticity (modulates learning)

    v = normalize(code @ basis) — unchanged.
    """

    def __init__(self, dim=384, latent_dim=512):
        self.dim = dim
        self.latent_dim = latent_dim
        self.l_c = latent_dim // 2      # 256 — identity
        self.l_a = latent_dim // 4      # 128 — attention
        self.l_m = latent_dim - self.l_c - self.l_a  # 128 — meta

        # Fractal basis: (latent_dim, dim) with orthonormal columns
        rng = np.random.RandomState(42)
        mat = rng.randn(latent_dim, dim).astype(np.float32)
        Q, _ = np.linalg.qr(mat, mode='reduced')
        self.basis = Q.astype(np.float32)
        self._fluctuation_step = 0

        # Latent codes: cid → (latent_dim,) array
        self.codes = {}

        # Meta-weights (trainable, modulate plasticity per concept)
        self.meta_w_lr = np.zeros(self.l_m, dtype=np.float32)
        self.meta_w_th = np.zeros(self.l_m, dtype=np.float32)
        self.meta_b_lr = np.float32(0.0)
        self.meta_b_th = np.float32(0.0)

        # Cache
        self._vector_matrix = None
        self._cid_order = []
        self._matrix_dirty = True

    # ── Subspace ops ─────────────────────────────────────────

    def split_code(self, z):
        """Разделить код на подпространства: (z_c, z_a, z_m)."""
        return (z[:self.l_c],
                z[self.l_c:self.l_c + self.l_a],
                z[self.l_c + self.l_a:])

    def merge_code(self, z_c, z_a, z_m):
        """Собрать код из подпространств."""
        return np.concatenate([z_c, z_a, z_m])

    def meta_gate(self, z_m):
        """Вычислить мета-ворота из meta-подпространства.

        Returns:
            lr_mod: множитель learning rate [0, 1]
            th_mod: сдвиг inhibition threshold [-1, 1]
        """
        lr_mod = 1.0 / (1.0 + np.exp(-(np.dot(z_m, self.meta_w_lr) + self.meta_b_lr)))
        th_mod = np.tanh(np.dot(z_m, self.meta_w_th) + self.meta_b_th)
        return lr_mod, th_mod

    # ── Init ─────────────────────────────────────────────────

    def init_concept(self, cid, rng_seed=None):
        """Initialize a concept with split subspace code.

        z_c: sparse identity pattern (~12% active)
        z_a: small noise (context attention starts neutral)
        z_m: near zero (meta-gates start open)
        """
        seed = rng_seed if rng_seed is not None else cid * 137 + 42
        rng = np.random.RandomState(seed % (2**31))

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

    def get_field_bits(self, cid):
        """Get binary field vector for a concept."""
        return self.field_bits.get(cid)

    def set_field_bit(self, cid, anchor_idx, value=1):
        """Set a single bit in the concept's field vector."""
        bits = self.field_bits.get(cid)
        if bits is None:
            return
        byte_idx = anchor_idx // 8
        bit_idx = anchor_idx % 8
        if value:
            bits[byte_idx] |= (1 << bit_idx)
        else:
            bits[byte_idx] &= ~(1 << bit_idx)

    def check_field_bit(self, cid, anchor_idx):
        """Test a single bit in the concept's field vector."""
        bits = self.field_bits.get(cid)
        if bits is None:
            return False
        byte_idx = anchor_idx // 8
        bit_idx = anchor_idx % 8
        return bool(bits[byte_idx] & (1 << bit_idx))

    def field_overlap(self, cid_a, cid_b):
        """Count overlapping field bits between two concepts."""
        ba = self.field_bits.get(cid_a)
        bb = self.field_bits.get(cid_b)
        if ba is None or bb is None or len(ba) != len(bb):
            return 0
        return int(np.bitwise_and(ba, bb).sum())

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

    def ensure_matrix(self):
        """Build (N, D) normalized vector matrix from all latent codes."""
        if not self._matrix_dirty and self._vector_matrix is not None:
            return self._vector_matrix, self._cid_order

        self._cid_order = list(self.codes.keys())
        n = len(self._cid_order)
        if n == 0:
            self._vector_matrix = np.empty((0, self.dim), dtype=np.float32)
            return self._vector_matrix, self._cid_order

        C = np.array([self.codes[cid] for cid in self._cid_order], dtype=np.float32)
        V = C @ self.basis
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        V /= norms
        self._vector_matrix = V
        self._matrix_dirty = False
        return self._vector_matrix, self._cid_order

    # ── Concept updates ──────────────────────────────────────

    def apply_code_update(self, cid, delta_code, lr_c=0.01, lr_a=1.0, lr_m=0.1):
        """Apply subspace-aware STDP update to a latent code.

        Args:
            cid: concept ID
            delta_code: full-dimensional delta (projected from vector delta)
            lr_c: learning rate for identity subspace
            lr_a: learning rate for attention subspace
            lr_m: learning rate for meta subspace
        """
        code = self.codes.get(cid)
        if code is None:
            return

        z_c, z_a, z_m = self.split_code(code)
        d_c, d_a, d_m = self.split_code(delta_code)
        lr_mod, th_mod = self.meta_gate(z_m)

        # Apply updates with subspace-specific rates
        z_c_new = z_c + d_c * lr_c * lr_mod
        z_a_new = z_a + d_a * lr_a * lr_mod
        z_m_new = z_m + d_m * lr_m * lr_mod

        code_new = self.merge_code(z_c_new, z_a_new, z_m_new)

        # Normalize so |code @ basis| = 1
        v_raw = code_new @ self.basis
        norm = np.linalg.norm(v_raw)
        if norm > 1e-10:
            code_new /= norm

        self.codes[cid] = code_new.astype(np.float32)
        self._matrix_dirty = True

    # ── Attention shift ──────────────────────────────────────

    def shift_attention(self, cid, context_code_deltas, weights):
        """Вычислить сдвиг z_a под влиянием контекста.

        Args:
            cid: target concept ID
            context_code_deltas: list of (ctx_cid, direction_hint) or None
            weights: list of scalar weights (PMI x dist_decay)

        Returns:
            z_a_shifted: новый z_a вектор или None
        """
        code = self.codes.get(cid)
        if code is None:
            return None
        z_c, z_a, z_m = self.split_code(code)
        lr_mod, _ = self.meta_gate(z_m)

        shift = np.zeros_like(z_a)
        total_w = 0.0
        for ctx_z, w in zip(context_code_deltas, weights):
            if ctx_z is None:
                continue
            _, ctx_za, _ = self.split_code(ctx_z)
            shift += w * (ctx_za - z_a)
            total_w += w

        if total_w > 1e-10:
            shift /= total_w
            z_a_shifted = z_a + shift * lr_mod
            return z_a_shifted
        return z_a

    # ── Fluctuation ──────────────────────────────────────────

    def fluctuate(self, noise_scale=0.005, decay=0.999):
        """Autonomous fluctuation: all latent codes drift.
        Noise scaled by subspace: more for z_a, less for z_c, minimal for z_m."""
        if not hasattr(self, '_fluct_rng'):
            self._fluct_rng = np.random.RandomState(42)
        for cid in list(self.codes.keys()):
            c = self.codes[cid]
            z_c, z_a, z_m = self.split_code(c)
            noise = self._fluct_rng.randn(self.latent_dim).astype(np.float32) * noise_scale
            # Subspace-specific noise scaling
            noise[:self.l_c] *= 0.3           # identity: low drift
            noise[self.l_c:self.l_c + self.l_a] *= 2.0   # attention: high drift
            noise[self.l_c + self.l_a:] *= 0.1           # meta: minimal
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
            np.savez_compressed(tmp_path,
                                codes=codes_arr, cids=cids, basis=self.basis,
                                meta_w_lr=self.meta_w_lr,
                                meta_w_th=self.meta_w_th)
            # Save field bits if present
            if hasattr(self, 'field_bits') and self.field_bits:
                fb_cids = np.array(list(self.field_bits.keys()), dtype=np.int32)
                fb_arr = np.array([self.field_bits[cid] for cid in fb_cids], dtype=np.uint8)
                # Add to existing npz
                with np.load(tmp_path) as f:
                    kw = dict(f)
                kw['fb_cids'] = fb_cids
                kw['fb_arr'] = fb_arr
                np.savez_compressed(tmp_path, **kw)
            os.replace(tmp_path, binary_path)
            return {
                'dim': self.dim,
                'latent_dim': self.latent_dim,
                'binary_codes': os.path.basename(binary_path),
                'n_codes': len(cids),
                'meta_b_lr': float(self.meta_b_lr),
                'meta_b_th': float(self.meta_b_th),
            }
        return {
            'dim': self.dim,
            'latent_dim': self.latent_dim,
            'basis': self.basis.tolist(),
            'codes': {str(cid): c.tolist() for cid, c in self.codes.items()},
            'meta_w_lr': self.meta_w_lr.tolist(),
            'meta_w_th': self.meta_w_th.tolist(),
            'meta_b_lr': float(self.meta_b_lr),
            'meta_b_th': float(self.meta_b_th),
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
            # Backward compat: meta-weights added in v2
            field.meta_w_lr = npz.get('meta_w_lr', np.zeros(field.l_m, dtype=np.float32))
            field.meta_w_th = npz.get('meta_w_th', np.zeros(field.l_m, dtype=np.float32))
            # Backward compat: field bits added in v2
            if 'fb_cids' in npz.files:
                fb_arr = npz['fb_arr']
                fb_cids_arr = npz['fb_cids']
                field.field_bits = {int(cid): fb_arr[i].copy()
                                     for i, cid in enumerate(fb_cids_arr)}
        else:
            field.basis = np.array(data['basis'], dtype=np.float32)
            field.codes = {int(cid): np.array(c, dtype=np.float32)
                            for cid, c in data['codes'].items()}
            field.meta_w_lr = np.array(data.get('meta_w_lr', np.zeros(field.l_m)), dtype=np.float32)
            field.meta_w_th = np.array(data.get('meta_w_th', np.zeros(field.l_m)), dtype=np.float32)

        field.meta_b_lr = np.float32(data.get('meta_b_lr', 0.0))
        field.meta_b_th = np.float32(data.get('meta_b_th', 0.0))

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

        # Concept vectors (cid → np.array)
        self.concept_vectors = {}

        # Array-backed vector access for O(1) lookups
        self._vec_array = None      # (N, dim) float32
        self._cids = []             # list of cids matching _vec_array rows
        self._cid_to_idx = {}       # cid → row index in _vec_array

        # Random state
        self.rng = np.random.RandomState(42)
        self._inhibition_step = 0

        # Shift tracking
        self._total_shift = 0.0
        self._update_count = 0

        # Precomputed vector matrix for fast nearest-neighbor
        self._vector_matrix = None
        self._cid_order = []
        self._matrix_dirty = True

        # Product Quantization (storage compression)
        self.pq_codebooks = None
        self.pq_codes = None
        self.pq_cid_order = []
        self.pq_n_subvectors = 0
        self.pq_n_centroids = 0

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
        self._all_cids = list(self.concept_vectors.keys())
        self.sync_vec_array()
        print(f"  Initialized {len(self.concept_vectors)} concepts via fractal")

    def _sync_concept_vectors_from_fractal(self):
        """Rebuild concept_vectors dict from fractal latent codes."""
        for cid in list(self.fractal.codes.keys()):
            v = self.fractal.compute_vector(cid)
            if v is not None:
                self.concept_vectors[cid] = v
        self.mark_matrix_dirty()

    def sync_vec_array(self):
        """Build _vec_array from concept_vectors dict for O(1) array access."""
        self._cids = list(self.concept_vectors.keys())
        self._cid_to_idx = {cid: i for i, cid in enumerate(self._cids)}
        self._vec_array = np.array([self.concept_vectors[cid] for cid in self._cids], dtype=np.float32)

    def get_vec(self, cid):
        """O(1) vector access via array index, fall back to dict."""
        idx = self._cid_to_idx.get(cid)
        if idx is not None:
            return self._vec_array[idx]
        return self.concept_vectors.get(cid)

    def set_vec(self, cid, v):
        """Update vector in both dict and array."""
        self.concept_vectors[cid] = v
        idx = self._cid_to_idx.get(cid)
        if idx is not None:
            self._vec_array[idx] = v

    def reinit_fractal(self, cid_list=None):
        """Reinitialize all fractal latent codes (resets all vectors)."""
        cids = cid_list if cid_list is not None else list(self.concept_vectors.keys())
        self.fractal.reinitialize_all(cids)
        self._sync_concept_vectors_from_fractal()
        print(f"  Reinitialized {len(cids)} concepts via fractal field")

    # ── H matrix + BMSSP ────────────────────────────────────

    def build_anchor_matrix(self, lattice, n_anchors=1024, min_pmi=0.1):
        """Build anchor matrix H from SyntaxLattice PMI and store anchors.

        Args:
            lattice: SyntaxLattice instance
            n_anchors: number of anchor concepts
            min_pmi: minimum PMI threshold

        Sets:
            self.H: scipy.sparse.csr_matrix (n, n) of PMI values
            self.anchor_ids: list of concept IDs
            self.anchor_idx: dict {cid: index}
        """
        self.H, self.anchor_ids = lattice.build_anchor_matrix(n_anchors, min_pmi)
        self.anchor_idx = {cid: i for i, cid in enumerate(self.anchor_ids)}
        self.n_anchors = len(self.anchor_ids)
        print(f"  H matrix: {self.n_anchors}x{self.n_anchors} anchors, "
              f"{self.H.nnz} non-zero ({100*self.H.nnz/self.n_anchors**2:.1f}%)")

    def build_fields_from_lattice(self, lattice, min_pmi=4.5):
        """Compute binary field bits via direct PMI connections.

        Each concept's field = itself + anchors connected by PMI > threshold.

        Args:
            lattice: SyntaxLattice instance
            min_pmi: minimum PMI for anchor connection
        """
        if not hasattr(self, 'H'):
            raise ValueError("Call build_anchor_matrix() first")

        self.fractal.init_fields(self.n_anchors)
        seen_cids = set(lattice.concept_freq.keys()) & set(self.fractal.codes.keys())

        # Build concept→anchor co-occurrence index from 2-grams
        # For each bigram (a→b), record b in cooc[a] and a in cooc[b]
        import math
        from collections import defaultdict, Counter
        n2 = lattice.ngrams.get(2, {})
        total_freq = max(sum(lattice.concept_freq.values()), 1)
        anchor_set = set(self.anchor_ids)
        cooc = defaultdict(Counter)  # cid → {anchor_cid: total_count}

        for prefix, counter in n2.items():
            a = prefix[0]
            for b, cnt in counter.items():
                if a in anchor_set:
                    cooc[b][a] = cooc[b].get(a, 0) + cnt
                if b in anchor_set:
                    cooc[a][b] = cooc[a].get(b, 0) + cnt

        for cid in seen_cids:
            bits = self._compute_pmi_field_fast(cid, lattice, total_freq,
                                                cooc, min_pmi)
            self.fractal.field_bits[cid] = bits

        active_counts = []
        for bs in self.fractal.field_bits.values():
            c = int(sum(1 for b in bs for i in range(8) if b & (1 << i)))
            active_counts.append(c)
        if active_counts:
            import numpy as np
            a = np.array(active_counts)
            print(f"  PMI fields: {len(seen_cids)}/{len(self.fractal.codes)} concepts, "
                  f"sizes: min={a.min()} max={a.max()} mean={a.mean():.1f}")

    def _compute_pmi_field_fast(self, cid, lattice, total_freq,
                                 cooc, min_pmi=4.5):
        """Field = self + anchors with PMI > threshold.

        Uses precomputed concept→anchor co-occurrence index.

        Returns packed uint8 array of n_anchors bits.
        """
        import math
        n_bytes = (self.n_anchors + 7) // 8
        bits = bytearray(n_bytes)

        def set_bit(ai):
            bits[ai >> 3] |= 1 << (ai & 7)

        if cid in self.anchor_idx:
            start_idx = self.anchor_idx[cid]
            set_bit(start_idx)
            row = self.H.getrow(start_idx)
            for dst, pmi in zip(row.indices, row.data):
                if pmi > min_pmi:
                    set_bit(dst)
            return np.frombuffer(bytes(bits), dtype=np.uint8).copy()

        # Non-anchor: use precomputed co-occurrence index
        count_c = lattice.concept_freq.get(cid, 0)
        if count_c < 2:
            return np.frombuffer(bytes(bits), dtype=np.uint8).copy()
        p_c = count_c / total_freq
        cid_cooc = cooc.get(cid, {})

        for aidx, anchor_id in enumerate(self.anchor_ids):
            count_anchor = lattice.concept_freq.get(anchor_id, 0)
            if count_anchor < 1:
                continue
            count_pair = cid_cooc.get(anchor_id, 0)
            if count_pair < 2:
                continue
            p_a = count_anchor / total_freq
            p_pair = count_pair / total_freq
            pmi = math.log(p_pair / max(p_c * p_a, 1e-10))
            if pmi > min_pmi:
                set_bit(aidx)

        return np.frombuffer(bytes(bits), dtype=np.uint8).copy()

    # ── Octree encoding ──────────────────────────────────────

    def build_octree_fields(self, lattice, n_anchors=1024, min_lcp=2, gamma=0.5, path_overrides=None):
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

        # Group anchors by their first min_lcp digits
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

    def fluctuate_fractal(self, noise_scale=0.003, decay=0.9995, repel_strength=0.0):
        """Autonomous drift + optional centroid repulsion."""
        self.fractal.fluctuate(noise_scale=noise_scale, decay=decay)
        self._sync_concept_vectors_from_fractal()
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
        vecs = np.array(list(self.concept_vectors.values()), dtype=np.float32)
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
        self._matrix_dirty = True

    # ---- STDP: Spike-Timing-Dependent Plasticity on fractal codes ----

    def _apply_vector_update(self, cid, v_new, max_shift=0.5):
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
        if code is not None or v_old is not None:
            new_code = v_new @ self.fractal.basis.T
            nv_code = np.linalg.norm(new_code @ self.fractal.basis)
            if nv_code > 1e-10:
                new_code /= nv_code
            self.fractal.codes[cid] = new_code

        self._matrix_dirty = True

    def fractal_stdp(self, prev_cid, gen_cid, expected_cid=None, lr=0.1, word_num=0,
                     inh_strength=0.05, inh_threshold=0.35):
        """STDP on fractal latent codes — self-organisation through code projection.

        Same Riemannian geometry as svd_shift(), but the vector update
        is projected back into fractal code space via basis.T.

        Args:
            prev_cid: context concept (pre-synaptic)
            gen_cid: generated concept (post-synaptic)
            expected_cid: target concept (from training data), or None
            lr: learning rate
            word_num: position in sentence (theta-rhythm modulates learning)
            inh_strength: lateral inhibition strength multiplier
            inh_threshold: cosine threshold for lateral inhibition
        """
        theta_gate = math.exp(-word_num / 15.0)
        effective_lr = lr * max(theta_gate, 0.1)

        expected = expected_cid or gen_cid
        is_match = (gen_cid == expected)

        v_ctx = self.concept_vectors.get(prev_cid)
        v_gen = self.concept_vectors.get(gen_cid)
        if v_ctx is None or v_gen is None:
            return

        if is_match:
            scale = 1.0 * effective_lr
        else:
            scale = -0.05 * effective_lr
            v_exp = self.concept_vectors.get(expected)
            if v_exp is not None:
                y_exp = float(np.dot(v_gen, v_exp))
                y_exp = max(y_exp, 0.05)
                corr = (v_exp - y_exp * v_gen) * effective_lr
                v_corrected = v_gen + corr
                cn = np.linalg.norm(v_corrected)
                if cn > 1e-10:
                    v_corrected /= cn
                self._apply_vector_update(gen_cid, v_corrected)
                v_gen = self.concept_vectors[gen_cid]

        y = float(np.dot(v_gen, v_ctx))
        y = max(y, 0.05)
        shift = (v_ctx - y * v_gen) * scale
        v_new = v_gen + shift
        nv = np.linalg.norm(v_new)
        if nv > 1e-10:
            v_new /= nv
        self._apply_vector_update(gen_cid, v_new)

        self._lateral_inhibition_fractal(gen_cid, strength=inh_strength * effective_lr, threshold=inh_threshold)
        self.mark_matrix_dirty()

    def _lateral_inhibition_fractal(self, winner_cid, strength=0.01, threshold=0.35, sample_size=None):
        """Lateral inhibition with correct Riemannian gradient, vectorised.

        The negative Riemannian gradient of sim = dot(v, v_win) is:
            -grad_R = sim * v - v_win
        which is tangent at v and maximally decreases alignment with winner.
        The Euclidean chord (v - v_win) used previously has a radial component
        and does not follow the geodesic — fixed.

        Inner loop over sampled concepts is vectorised (numpy batch ops).
        Uses _vec_array for O(1) array-backed gather.
        """
        v_win = self.concept_vectors.get(winner_cid)
        if v_win is None:
            return
        vw_n = v_win / max(np.linalg.norm(v_win), 1e-10)

        if sample_size is None:
            sample_size = min(200, len(self.concept_vectors))

        cids = self._cids if hasattr(self, '_cids') and self._cids else self._all_cids
        n_cids = len(cids)
        if n_cids <= 1:
            return

        if not hasattr(self, '_inhibit_rng'):
            self._inhibit_rng = np.random.RandomState(42)
        # Fast: randint + unique avoids full permutation (30x faster)
        raw = self._inhibit_rng.randint(0, n_cids, size=sample_size + 50)
        u_idxs = np.unique(raw)
        sampled_indices = [i for i in u_idxs if cids[i] != winner_cid][:sample_size]
        if len(sampled_indices) < 1:
            return

        # Array-backed gather: O(sample_size) instead of dict lookups + array creation
        sampled_vecs = self._vec_array[sampled_indices]  # (S, dim)
        sampled_cids = [cids[i] for i in sampled_indices]
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

        v_new = affected + inhibit * strength * sims_k[:, None]
        vnorms = np.linalg.norm(v_new, axis=1)
        vnorms[vnorms < 1e-10] = 1.0
        v_new /= vnorms[:, None]

        for idx, cid in enumerate([sc for sc, m in zip(sampled_cids, mask) if m]):
            self._apply_vector_update(cid, v_new[idx])
        self._matrix_dirty = True

    def init_homeostasis(self):
        """Initialize homeostasis tracking for concepts."""
        self.concept_usage = {cid: 0.0 for cid in self.concept_vectors}
        self.concept_fitness = {cid: 1.0 for cid in self.concept_vectors}
        self.concept_momentum = {cid: np.zeros(self.dim, dtype=np.float32)
                                 for cid in self.concept_vectors}
        self._hboost_mean_cache = None
        self._hboost_cache_step = 0
        self._usage_decay_steps = 0

    def decay_usage(self, decay=0.98):
        """Exponential decay of concept usage to prevent homeostatic saturation."""
        for cid in self.concept_usage:
            self.concept_usage[cid] *= decay
        self._usage_decay_steps += 1
        self._hboost_mean_cache = None

    def check_code_range(self, bound=10.0):
        """Check if any fractal code exceeds |bound|. Returns (n_outliers, max_abs)."""
        if not self.fractal.codes:
            return 0, 0.0
        all_codes = np.array(list(self.fractal.codes.values()), dtype=np.float32)
        max_abs = float(np.max(np.abs(all_codes)))
        n_out = int(np.sum(np.max(np.abs(all_codes), axis=1) > bound))
        return n_out, max_abs

    def validate_vector_norms(self):
        """Check all vectors are unit norm. Returns (ok_count, total, max_deviation)."""
        if not self.concept_vectors:
            return 0, 0, 0.0
        all_vecs = np.array(list(self.concept_vectors.values()), dtype=np.float32)
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
        if self._hboost_cache_step % 1000 == 1 or self._hboost_mean_cache is None:
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

    def ensure_matrix(self):
        """Build or rebuild the precomputed vector matrix for fast NN search."""
        if not self._matrix_dirty and self._vector_matrix is not None:
            return
        cids = []
        vecs = []
        for cid, v in self.concept_vectors.items():
            cids.append(cid)
            vecs.append(v)
        if not vecs:
            self._vector_matrix = np.empty((0, self.dim), dtype=np.float32)
            self._cid_order = []
            self._matrix_dirty = False
            return
        mat = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        self._vector_matrix = mat / norms
        self._cid_order = cids
        self._matrix_dirty = False

    def mark_matrix_dirty(self):
        """Call after vector changes (STDP, new concepts)."""
        self._matrix_dirty = True

    def concept_vector(self, cid):
        """Get concept centroid vector."""
        return self.concept_vectors.get(cid)

    def topk_similar_concepts(self, cid, k=10, sample_size=500):
        """Top-k concepts closest to given concept (batched matrix NN)."""
        v = self.concept_vectors.get(cid)
        if v is None:
            return []
        self.ensure_matrix()
        mat = self._vector_matrix
        if mat.shape[0] == 0:
            return []
        vn = v / max(np.linalg.norm(v), 1e-10)
        sims = mat @ vn
        n = len(sims)
        k_actual = min(k + 1, n)  # +1 to skip self
        if k_actual <= 0:
            return []
        idx = np.argpartition(-sims, k_actual - 1)[:k_actual]
        idx = idx[np.argsort(-sims[idx])]
        result = []
        for i in idx:
            c = self._cid_order[i]
            if c == cid:
                continue
            result.append((c, float(sims[i])))
            if len(result) >= k:
                break
        return result[:k]

    # ── Product Quantization — storage compression ─────────────────

    def pq_train(self, n_subvectors=32, n_centroids=256):
        """Train PQ codebooks from current concept vectors.

        Args:
            n_subvectors: number of sub-vectors to split each D-dim vector into
            n_centroids: centroids per subspace (256 = 8-bit index)

        Splits each D-dim vector into n_subvectors subspaces of dim D/n_subvectors.
        Performs k-means in each subspace to learn n_centroids.

        PQ compression ratio:
            before: N x D x float32 (4 bytes)
            after:  N x n_subvectors x uint8 (1 byte) + n_subvectors x n_centroids x subdim x float32
            typical for 128D → 32x8 = 32 bytes vs 512 bytes = 16x compression
        """
        vecs = list(self.concept_vectors.values())
        if not vecs:
            return
        N = len(vecs)
        D = self.dim
        assert D % n_subvectors == 0, f'Dim {D} must be divisible by {n_subvectors}'
        subdim = D // n_subvectors

        mat = np.array(vecs, dtype=np.float32)
        # Normalize each vector
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        mat /= norms

        codebooks = []
        for m in range(n_subvectors):
            subvecs = mat[:, m * subdim:(m + 1) * subdim]
            kmeans = KMeans(n_clusters=n_centroids, random_state=42 + m,
                            n_init=1, max_iter=20)
            kmeans.fit(subvecs)
            cb = kmeans.cluster_centers_.astype(np.float32)
            # Normalize centroids so sim = 1 - ||q-cb||²/2 is valid
            cb_norms = np.linalg.norm(cb, axis=1, keepdims=True)
            cb_norms[cb_norms < 1e-10] = 1.0
            cb /= cb_norms
            codebooks.append(cb)

        self.pq_codebooks = codebooks
        self.pq_n_subvectors = n_subvectors
        self.pq_n_centroids = n_centroids
        self.pq_cid_order = list(self.concept_vectors.keys())
        self.pq_codes = None  # not encoded yet
        return codebooks

    def pq_encode(self):
        """Encode all concept vectors using trained codebooks.

        Returns:
            pq_codes: (N, n_subvectors) uint8 array
        """
        if self.pq_codebooks is None:
            raise ValueError('Call pq_train() first')
        vecs = [self.concept_vectors[cid] for cid in self.pq_cid_order]
        mat = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        mat /= norms

        N = len(mat)
        n_sub = self.pq_n_subvectors
        subdim = self.dim // n_sub
        codes = np.zeros((N, n_sub), dtype=np.uint8)

        for m in range(n_sub):
            sub = mat[:, m * subdim:(m + 1) * subdim]  # (N, subdim)
            cb = self.pq_codebooks[m]  # (n_centroids, subdim)
            dists = np.sum((sub[:, None, :] - cb[None, :, :]) ** 2, axis=2)  # (N, n_centroids)
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

        self.pq_codes = codes
        return codes

    def pq_decode_all(self):
        """Reconstruct full vectors from PQ codes.

        Updates concept_vectors in-place with decoded (approximate) vectors.
        """
        if self.pq_codes is None or self.pq_codebooks is None:
            return
        n_sub = self.pq_n_subvectors
        subdim = self.dim // n_sub
        N = len(self.pq_cid_order)
        decoded = np.zeros((N, self.dim), dtype=np.float32)
        for m in range(n_sub):
            cb = self.pq_codebooks[m]
            codes_m = self.pq_codes[:, m]
            decoded[:, m * subdim:(m + 1) * subdim] = cb[codes_m]
        # Renormalize
        norms = np.linalg.norm(decoded, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        decoded /= norms
        # Write back
        for i, cid in enumerate(self.pq_cid_order):
            self.concept_vectors[cid] = decoded[i]
        self.mark_matrix_dirty()

    def pq_decode(self, cid):
        """Decode a single concept vector from PQ codes."""
        if self.pq_codes is None or self.pq_codebooks is None:
            return self.concept_vectors.get(cid)
        try:
            idx = self.pq_cid_order.index(cid)
        except ValueError:
            return self.concept_vectors.get(cid)
        n_sub = self.pq_n_subvectors
        subdim = self.dim // n_sub
        v = np.zeros(self.dim, dtype=np.float32)
        for m in range(n_sub):
            cb = self.pq_codebooks[m]
            v[m * subdim:(m + 1) * subdim] = cb[self.pq_codes[idx, m]]
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v /= norm
        return v

    def pq_adc_search(self, query_vec, k=10):
        """Approximate nearest neighbor via Asymmetric Distance Computation.

        Query is kept in full float32 (not encoded).
        Database distances computed by summing subspace distances
        from pre-computed lookup tables.

        Args:
            query_vec: (D,) float32 query vector (NOT normalized — done internally)
            k: number of nearest neighbors

        Returns:
            [(cid, similarity), ...]  (cosine similarity, higher = closer)
        """
        if self.pq_codes is None or self.pq_codebooks is None:
            return []
        vn = query_vec / max(np.linalg.norm(query_vec), 1e-10)
        n_sub = self.pq_n_subvectors
        subdim = self.dim // n_sub
        N = len(self.pq_cid_order)

        # Build distance table: for each subspace, distance from query to each centroid
        dist_tables = []
        for m in range(n_sub):
            q_sub = vn[m * subdim:(m + 1) * subdim]
            cb = self.pq_codebooks[m]
            diffs = cb - q_sub  # (n_centroids, subdim)
            dists = np.sum(diffs ** 2, axis=1)  # (n_centroids,)
            dist_tables.append(dists)

        # For each database vector, sum subspace distances via lookup
        # Optimized: precompute full distance matrix
        total_dists = np.zeros(N, dtype=np.float32)
        for m in range(n_sub):
            codes_m = self.pq_codes[:, m].astype(np.int32)
            total_dists += dist_tables[m][codes_m]

        # Convert distance to cosine similarity: sim = 1 - dist²/2
        sims = 1.0 - total_dists / 2.0
        sims = np.clip(sims, -1.0, 1.0)

        # Top-k
        k_actual = min(k, N)
        idx = np.argpartition(-sims, k_actual - 1)[:k_actual]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.pq_cid_order[i], float(sims[i])) for i in idx[:k]]

    def pq_compression_ratio(self):
        """Report storage savings from PQ compression."""
        if self.pq_codes is None:
            return 0
        orig = len(self.pq_cid_order) * self.dim * 4  # float32
        codes_size = self.pq_codes.nbytes
        cb_size = sum(cb.nbytes for cb in self.pq_codebooks)
        return orig / (codes_size + cb_size)

    def expand_dim(self, target_dim):
        """Expand vector space dimension (e.g. 128 → 384).

        Extends existing basis with orthogonal new columns (Schur complement).
        Existing fractal codes are preserved by appending zero coefficients
        for the new dimensions — existing concept vectors remain unchanged.

        Args:
            target_dim: new dimension (must be > current dim)
        """
        if target_dim <= self.dim:
            return
        old_dim = self.dim
        print(f'  Expanding dimension: {old_dim} -> {target_dim}')
        new_dim = target_dim
        n_new = new_dim - old_dim

        # Extend existing basis with orthogonal new columns
        rng = np.random.RandomState(42)
        extension = rng.randn(self.fractal.latent_dim, n_new).astype(np.float32)
        # Orthogonalize against existing basis columns
        extension = extension - self.fractal.basis @ (self.fractal.basis.T @ extension)
        Q_ext, _ = np.linalg.qr(extension, mode='reduced')
        self.fractal.basis = np.concatenate([self.fractal.basis, Q_ext], axis=1).astype(np.float32)

        # Extend existing codes with zeros for new dimensions
        for cid in self.fractal.codes:
            code = self.fractal.codes[cid]
            ext = np.zeros(n_new, dtype=np.float32)
            self.fractal.codes[cid] = np.concatenate([code, ext])

        self.dim = new_dim
        self.fractal._matrix_dirty = True
        self._sync_concept_vectors_from_fractal()

        # Invalidate caches
        self.pq_codebooks = None
        self.pq_codes = None
        self.pq_cid_order = []
        self._vector_matrix = None
        self._cid_order = []
        self._matrix_dirty = True
        print(f'  Done: {len(self.concept_vectors)} concepts @ {self.dim}D')

    def normalize_vectors(self):
        """Center and normalize all concept vectors onto the unit sphere.

        Current vectors are clustered (mean pair sim > 0.3, all pointing
        toward centroid). This spreads them: subtract global centroid,
        then L2-normalize each vector. Call after load or training shift.
        """
        if not self.concept_vectors:
            return
        vecs = np.array(list(self.concept_vectors.values()), dtype=np.float32)
        centroid = np.mean(vecs, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        print(f'  Centroid norm before: {centroid_norm:.4f} (0 = centered)')

        # Center: subtract centroid
        centered = vecs - centroid

        # Normalize each to unit sphere
        norms = np.linalg.norm(centered, axis=1)
        norms[norms < 1e-10] = 1.0
        centered /= norms[:, np.newaxis]

        for i, cid in enumerate(self.concept_vectors):
            self.concept_vectors[cid] = centered[i]

        # Rebuild fractal codes to maintain invariant normalize(code @ basis) == v
        for cid, v in self.concept_vectors.items():
            code_new = v @ self.fractal.basis.T
            self.fractal.codes[cid] = code_new
        self.fractal._matrix_dirty = True

        # Invalidate caches
        self._vector_matrix = None
        self._cid_order = []
        self._matrix_dirty = True
        self.pq_codebooks = None
        self.pq_codes = None
        self.pq_cid_order = []

        new_norms = np.linalg.norm(centered, axis=1)
        print(f'  All vectors normalized: mean_norm={np.mean(new_norms):.4f}')
        print(f'  New centroid norm: {np.linalg.norm(np.mean(centered, axis=0)):.4f}')

    def contrastive_spread(self, target_sim=0.5, lr=0.1, epochs=10):
        """Push over-clustered vectors apart via targeted repulsion.

        Unlike random-pair sampling, this finds each vector's nearest
        neighbor and pushes THAT pair apart — directly attacking the
        most egregious clustering.

        Args:
            target_sim: push if sim > this value
            lr: learning rate
            epochs: full passes
        """
        cids = list(self.concept_vectors.keys())
        n = len(cids)
        rng = np.random.RandomState(42)

        from scipy.spatial.distance import cdist
        print(f'  Contrastive spread: {n} concepts, target_sim={target_sim}, lr={lr}, epochs={epochs}')

        for epoch in range(epochs):
            vecs = np.array([self.concept_vectors[c] for c in cids], dtype=np.float32)
            subset_size = min(n, 3000)
            idxs = rng.choice(n, subset_size, replace=False)
            n_pushed = 0
            for idx in idxs:
                vi = vecs[idx]
                sims = np.dot(vecs, vi)
                sims[idx] = -1
                max_sim = sims.max()
                if max_sim > target_sim:
                    j = sims.argmax()
                    vj = vecs[j]
                    # Correct Riemannian gradient: push vi away from vj
                    # ∇_R sim(vi, vj) = vj - sim*vi  → negative = vi - sim*vj
                    grad = vi - max_sim * vj
                    new_vi = vi + lr * grad
                    nvi = np.linalg.norm(new_vi)
                    if nvi > 1e-10:
                        self.concept_vectors[cids[idx]] = new_vi / nvi
                    # Symmetric push for vj away from vi
                    grad2 = vj - max_sim * vi
                    new_vj = vj + lr * grad2
                    nvj = np.linalg.norm(new_vj)
                    if nvj > 1e-10:
                        self.concept_vectors[cids[j]] = new_vj / nvj
                    n_pushed += 1

            # Verify
            check = np.array([self.concept_vectors[c] for c in cids], dtype=np.float32) if epoch == epochs-1 or epoch % 3 == 2 else None
            if check is not None:
                sim_vals = []
                for _ in range(10000):
                    i = rng.randint(0, n)
                    j = rng.randint(0, n)
                    if i != j:
                        sim_vals.append(float(np.dot(check[i], check[j])))
                p50 = np.percentile(sim_vals, 50)
                p99 = np.percentile(sim_vals, 99)
                print(f'    epoch {epoch+1}: pushed {n_pushed}, mean_sim={np.mean(sim_vals):.3f}, p99={p99:.3f}')

        self._vector_matrix = None
        self._cid_order = []
        self._matrix_dirty = True
        print(f'  Done')

    def save(self, path, use_pq=False):
        """Save ConceptSpace to disk.

        Args:
            path: file path
            use_pq: if True, save PQ-compressed format (much smaller).
        """
        # Binary .npz for fractal codes
        binary_path = path.replace('.json', '.codes.npz')
        if use_pq and self.pq_codes is not None:
            data = {
                'dim': self.dim,
                'vocab_size': self.vocab_size,
                'pq': True,
                'n_subvectors': self.pq_n_subvectors,
                'n_centroids': self.pq_n_centroids,
                'pq_cid_order': self.pq_cid_order,
                'pq_codes': self.pq_codes.tolist(),
                'codebooks': [cb.tolist() for cb in self.pq_codebooks],
            }
        else:
            data = {
                'dim': self.dim,
                'vocab_size': self.vocab_size,
                'pq': False,
            }
        data['fractal'] = self.fractal.to_dict(binary_path=binary_path)
        concept_usage = getattr(self, 'concept_usage', None)
        if concept_usage is not None:
            data['concept_usage'] = {str(c): u for c, u in concept_usage.items()}
            data['concept_fitness'] = {str(c): f for c, f in
                                       getattr(self, 'concept_fitness', {}).items()}

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

        obj.pq_codebooks = None
        obj.pq_codes = None
        obj.pq_cid_order = []
        obj.pq_n_subvectors = 0
        obj.pq_n_centroids = 0

        if data.get('pq'):
            n_sub = data['n_subvectors']
            n_cen = data['n_centroids']
            obj.pq_n_subvectors = n_sub
            obj.pq_n_centroids = n_cen
            obj.pq_cid_order = data['pq_cid_order']
            obj.pq_codebooks = [np.array(cb, dtype=np.float32) for cb in data['codebooks']]
            obj.pq_codes = np.array(data['pq_codes'], dtype=np.uint8)
            subdim = obj.dim // n_sub
            N = len(obj.pq_cid_order)
            decoded = np.zeros((N, obj.dim), dtype=np.float32)
            for m in range(n_sub):
                cb = obj.pq_codebooks[m]
                codes_m = obj.pq_codes[:, m]
                decoded[:, m * subdim:(m + 1) * subdim] = cb[codes_m]
            norms = np.linalg.norm(decoded, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            decoded /= norms
            obj.concept_vectors = {}
            for i, cid in enumerate(obj.pq_cid_order):
                obj.concept_vectors[cid] = decoded[i]
        else:
            obj.concept_vectors = {}

        obj.rng = np.random.RandomState(42)
        obj._inhibition_step = 0
        obj._total_shift = 0.0
        obj._update_count = 0
        obj._vector_matrix = None
        obj._cid_order = []
        obj._matrix_dirty = True

        if 'fractal' in data:
            base_dir = os.path.dirname(path)
            obj.fractal = FractalField.from_dict(data['fractal'], base_dir=base_dir)
            obj.concept_vectors = {}
            for cid in list(obj.fractal.codes.keys()):
                v = obj.fractal.compute_vector(cid)
                if v is not None:
                    obj.concept_vectors[cid] = v
        else:
            obj.fractal = FractalField(dim=obj.dim, latent_dim=512)

        saved_usage = data.get('concept_usage')
        saved_fitness = data.get('concept_fitness')
        if saved_usage:
            obj.concept_usage = {int(c): u for c, u in saved_usage.items()}
            obj.concept_fitness = {int(c): f for c, f in saved_fitness.items()}
        else:
            obj.init_homeostasis()
        obj._all_cids = list(obj.concept_vectors.keys())
        obj.sync_vec_array()
        pq_note = ' (PQ)' if data.get('pq') else ''
        print(f"  Loaded ConceptSpace: {len(obj.concept_vectors)} concepts @ {obj.dim}D{pq_note}")
        return obj


if __name__ == '__main__':
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=r'C:\Users\black\OneDrive\Desktop\FCF\real_data\bpe_ru.model')

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

    cs.save(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space.json')
