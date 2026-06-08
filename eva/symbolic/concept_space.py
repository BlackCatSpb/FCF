"""ConceptSpace — vector space organized by ConceptNet concepts.

Architecture:
  - Each concept has a VECTOR (centroid of its semantic field)
  - Words within a concept are SATELLITES around the centroid
  - Concept transitions learned from corpus (concept_i → concept_j)
  - ConceptNet relations = HARD CONSTRAINTS on vector positions
  - Generation = concept navigation → word selection within concept

Levels:
  L3: Meta-concepts (Louvain clusters of concept vectors)
  L2: Concepts (from ConceptNet + corpus transitions)
  L1: Words (concept anchor + morphological satellites)
  L0: Characters (BPE subword tokens within words)
"""

import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import math, json, os, pickle


class ConceptSpace:
    """Vector space organized by ConceptNet concepts.

    Each concept has:
      - cid: concept ID
      - anchor: the lemma/root word
      - vector: centroid vector in semantic space
      - word_vectors: dict {word: offset_from_centroid}
      - transitions: concept → concept transition probabilities
    """

    def __init__(self, skeleton, dim=256):
        self.skeleton = skeleton
        self.dim = dim

        # Concept vectors
        self.concept_vectors = {}     # cid → np.array(dim,)
        self.concept_info = {}        # cid → concept dict from skeleton

        # Word → concept mapping
        self.word_to_cid = {}         # word → cid (from skeleton)
        self.cid_to_words = {}        # cid → [words]
        self.word_to_morph = {}       # word → {normal_form, prefix, suffix, ending, pos}

        # Transition matrix (concept → concept)
        self.concept_transitions = None  # CSR matrix (n_concepts × n_concepts)
        self.cid_list = []               # ordered list of concept IDs
        self.cid_to_idx = {}             # cid → row/col index

        # Concept → concept vectors (via SVD on transitions)
        self.cid_vectors = {}         # cid → concept-level vector

        # Random state
        self.rng = np.random.RandomState(42)
        self._inhibition_step = 0  # counter for lateral inhibition seed

        # Precomputed vector matrix for fast nearest-neighbor
        # (N, D) normalized array + CID ordering
        self._vector_matrix = None
        self._cid_order = []
        self._matrix_dirty = True

        # Product Quantization (storage compression)
        # PQ replaces full float32 vectors with compact codes:
        #   n_subvectors × uint8 per vector instead of D × float32
        #   typical ratio: 32 bytes vs 512 bytes (128D) → 16× compression
        self.pq_codebooks = None    # list of (n_centroids, subdim) arrays
        self.pq_codes = None        # (N, n_subvectors) uint8 array
        self.pq_cid_order = []      # CID order matching pq_codes rows
        self.pq_n_subvectors = 0
        self.pq_n_centroids = 0

        # Affix shift vectors (morphological modifiers)
        # affix → dim-D shift vector, initialized as random unit vectors
        self.affix_shifts = {}
        self._init_affix_shifts()

    def _init_affix_shifts(self):
        """Initialize affix shift vectors as random unit vectors.

        These represent grammatical modifications to root concepts:
        - prefix: directional/aspectual modification
        - suffix: derivational modification
        - ending: grammatical role (case, number, gender, person)
        """
        rng = np.random.RandomState(42)
        common_prefixes = ['по', 'за', 'на', 'вы', 'от', 'при', 'пере', 'про',
                           'раз', 'рас', 'вз', 'воз', 'вос', 'из', 'ис',
                           'под', 'над', 'об', 'о', 'у', 'с', 'со', 'в', 'во',
                           'до', 'без', 'бес', 'пре', 'пред']
        common_suffixes = ['к', 'ок', 'ек', 'ик', 'ник', 'тель', 'чик', 'щик',
                           'ств', 'ость', 'ени', 'ани', 'изм', 'ист',
                           'лив', 'чив', 'оват', 'ну', 'а']
        common_endings =  ['а', 'я', 'о', 'е', 'ы', 'и', 'у', 'ю',
                           'ой', 'ей', 'ых', 'их', 'ам', 'ям',
                           'ого', 'его', 'ому', 'ему', 'ым', 'им',
                           'ую', 'юю', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие']

        for affix in common_prefixes + common_suffixes + common_endings:
            v = rng.randn(self.dim).astype(np.float32)
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v /= norm
            self.affix_shifts[affix] = v

    def get_affix_shift(self, affix):
        """Get shift vector for a morphological affix.

        Returns 128D unit vector (or zero vector if unknown).
        """
        v = self.affix_shifts.get(affix)
        if v is not None:
            return v
        # Create on first use
        rng = np.random.RandomState(hash(affix) % (2**31))
        v = rng.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v /= norm
        self.affix_shifts[affix] = v
        return v

    def build(self, corpus_path=None, tok=None):
        """Build ConceptSpace from skeleton and corpus.

        Args:
            corpus_path: path to corpus text file
            tok: ConceptTokenizer instance (for encoding corpus)
        """
        # Copy skeleton data
        for cid, concept in self.skeleton.concepts.items():
            self.concept_info[cid] = concept
            self.cid_to_words[cid] = self.skeleton.cid_to_words.get(cid, [concept['anchor']])
            for w in self.cid_to_words[cid]:
                self.word_to_cid[w] = cid

        self.cid_list = sorted(self.skeleton.concepts.keys())
        self.cid_to_idx = {cid: i for i, cid in enumerate(self.cid_list)}
        n_concepts = len(self.cid_list)

        # Build concept transitions from corpus
        if corpus_path and tok:
            self._build_concept_transitions(corpus_path, tok, n_concepts)

        # Compute concept vectors via SVD on concept transitions
        self._compute_concept_vectors(n_concepts)

        # Apply ConceptNet constraints (pull synonyms, push antonyms)
        self._apply_conceptnet_constraints()

        return self

    def _build_concept_transitions(self, corpus_path, tok, n_concepts):
        """Build concept→concept transition matrix from corpus."""
        trans_count = np.zeros((n_concepts, n_concepts), dtype=np.float32)
        word_count = Counter()

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Encode line with tokenizer
                ids = tok.encode(line)
                meta = tok.metadata_from_ids(ids)

                # Extract concept sequence from word starts
                prev_cid = None
                for m in meta:
                    if m['flags'] & 1:  # word_start
                        cid = m.get('concept_id')
                        if cid is not None and prev_cid is not None and cid != prev_cid:
                            ci = self.cid_to_idx.get(prev_cid)
                            cj = self.cid_to_idx.get(cid)
                            if ci is not None and cj is not None:
                                trans_count[ci, cj] += 1
                        if cid is not None:
                            prev_cid = cid

        # Build CSR matrix
        n = n_concepts
        rows, cols, data = [], [], []
        for i in range(n):
            for j in range(n):
                if trans_count[i, j] > 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(trans_count[i, j])

        if rows:
            self.concept_transitions = csr_matrix(
                (data, (rows, cols)), shape=(n, n), dtype=np.float32
            )
            print(f"  Concept transitions: {len(rows)} edges")
        else:
            print("  WARNING: No concept transitions found!")
            self.concept_transitions = csr_matrix((n, n), dtype=np.float32)

    def _compute_concept_vectors(self, n_concepts):
        """Compute concept vectors via SVD on transition matrix."""
        if self.concept_transitions is None or self.concept_transitions.nnz == 0:
            # Fallback: random vectors
            print("  WARNING: No transitions, using random concept vectors")
            for cid in self.cid_list:
                v = self.rng.randn(self.dim).astype(np.float32)
                v /= max(np.linalg.norm(v), 1e-10)
                self.concept_vectors[cid] = v
            return

        ndim = min(self.dim, n_concepts - 1, self.concept_transitions.shape[0] - 1)
        ndim = max(ndim, 2)

        svd = TruncatedSVD(n_components=ndim, random_state=42)
        emb = svd.fit_transform(self.concept_transitions)

        # Normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb /= norms

        # If ndim < self.dim, pad with zeros
        if ndim < self.dim:
            padded = np.zeros((emb.shape[0], self.dim), dtype=np.float32)
            padded[:, :ndim] = emb
            emb = padded

        # Store
        for i, cid in enumerate(self.cid_list):
            self.concept_vectors[cid] = emb[i].astype(np.float32)

        # Reinitialize zero-norm concepts using ConceptNet neighbors
        zero_count = 0
        for cid in self.cid_list:
            v = self.concept_vectors[cid]
            if np.linalg.norm(v) < 0.01:
                zero_count += 1
                # Try to use ConceptNet neighbors
                neighbor_vecs = []
                for (ci, cj), rels in self.skeleton.relations.items():
                    other = cj if ci == cid else (ci if cj == cid else None)
                    if other is not None:
                        other_v = self.concept_vectors.get(other)
                        if other_v is not None and np.linalg.norm(other_v) > 0.01:
                            neighbor_vecs.append(other_v)
                if neighbor_vecs:
                    new_v = np.mean(neighbor_vecs, axis=0).astype(np.float32)
                    new_v /= max(np.linalg.norm(new_v), 1e-10)
                    self.concept_vectors[cid] = new_v
                else:
                    # Random unit vector
                    new_v = self.rng.randn(self.dim).astype(np.float32)
                    new_v /= np.linalg.norm(new_v)
                    self.concept_vectors[cid] = new_v

        if zero_count > 0:
            print(f"  Reinitialized {zero_count} zero-norm concepts")

        print(f"  SVD concept vectors: {len(self.cid_list)} @ {self.dim}D (effective {ndim}D)")

    def _apply_conceptnet_constraints(self):
        """Adjust concept vectors based on ConceptNet relations.

        Synonyms: pulled closer
        Antonyms: pushed apart
        is_a: sub-concept pulled toward parent
        """
        lr = 0.05
        for (ci, cj), rels in self.skeleton.relations.items():
            vi = self.concept_vectors.get(ci)
            vj = self.concept_vectors.get(cj)
            if vi is None or vj is None:
                continue

            if 'synonym' in rels or 'similar_to' in rels:
                # Pull together
                mid = (vi + vj) / 2
                self.concept_vectors[ci] = vi + lr * (mid - vi)
                self.concept_vectors[cj] = vj + lr * (mid - vj)
            elif 'antonym' in rels or 'distinct_from' in rels:
                # Push apart
                away = vj - vi
                d = np.linalg.norm(away)
                if d > 0:
                    away /= d
                    self.concept_vectors[ci] = vi - lr * away * 0.5
                    self.concept_vectors[cj] = vj + lr * away * 0.5

            if 'is_a' in rels:
                # Sub-concept pulled toward parent
                # ci is_a cj → ci pulled toward cj
                diff = vj - vi
                self.concept_vectors[ci] = vi + lr * diff * 0.5

        # Renormalize all
        for cid in self.concept_vectors:
            v = self.concept_vectors[cid]
            n = np.linalg.norm(v)
            if n > 1e-10:
                self.concept_vectors[cid] = v / n

        print(f"  ConceptNet constraints applied to {len(self.skeleton.relations)} relations")

    # ---- STDP: Spike-Timing-Dependent Plasticity on concept vectors ----

    def svd_shift(self, prev_cid, gen_cid, expected_cid=None, lr=0.1, word_num=0):
        """Concept-level STDP on sphere.

        Riemannian gradient descent on concept vectors:
        - Match (gen == expected or no expected): LTP — pull gen toward prev
        - Mismatch (gen != expected): LTD — push gen away from prev, pull toward expected

        Args:
            prev_cid: context concept (pre-synaptic)
            gen_cid: generated concept (post-synaptic)
            expected_cid: target concept (from training data), or None
            lr: learning rate
            word_num: position in sentence (theta-rhythm modulates learning)
        """
        # Theta rhythm: early words learn more, later less (STDP window)
        theta_gate = math.exp(-word_num / 7.0)  # τ=7 words
        effective_lr = lr * max(theta_gate, 0.1)

        expected = expected_cid or gen_cid
        is_match = (gen_cid == expected)

        v_ctx = self.concept_vectors.get(prev_cid)
        v_gen = self.concept_vectors.get(gen_cid)
        if v_ctx is None or v_gen is None:
            return

        # STDP: LTP for match, LTD for mismatch
        if is_match:
            # LTP: pull generated toward context (pre→post association)
            scale = 1.0 * effective_lr
        else:
            # LTD: push generated away from context
            scale = -0.05 * effective_lr
            # Correction: pull generated toward expected (target)
            v_exp = self.concept_vectors.get(expected)
            if v_exp is not None:
                y_exp = float(np.dot(v_gen, v_exp))
                y_exp = max(y_exp, 0.05)
                corr = (v_exp - y_exp * v_gen) * effective_lr
                self.concept_vectors[gen_cid] = v_gen + corr
                n = np.linalg.norm(self.concept_vectors[gen_cid])
                if n > 0:
                    self.concept_vectors[gen_cid] /= n
                v_gen = self.concept_vectors[gen_cid]

        # Riemannian gradient: ∇ = v_ctx - (v_gen·v_ctx) × v_gen
        y = float(np.dot(v_gen, v_ctx))
        y = max(y, 0.05)
        shift = (v_ctx - y * v_gen) * scale

        self.concept_vectors[gen_cid] = v_gen + shift
        n = np.linalg.norm(self.concept_vectors[gen_cid])
        if n > 0:
            self.concept_vectors[gen_cid] /= n

        # Lateral inhibition: dampen concepts similar to generated one
        self._lateral_inhibition(gen_cid, strength=0.02 * effective_lr)

        self.mark_matrix_dirty()

    def _lateral_inhibition(self, winner_cid, strength=0.01):
        """Suppress concepts similar to the winner.
        In cortex: winner suppresses neighbors via inhibitory interneurons."""
        v_win = self.concept_vectors.get(winner_cid)
        if v_win is None:
            return
        vw_n = v_win / max(np.linalg.norm(v_win), 1e-10)

        # To avoid O(n²), sample ~500 random concepts
        n_total = len(self.concept_vectors)
        sample_size = min(500, n_total)

        rng = np.random.RandomState(winner_cid + self._inhibition_step)
        self._inhibition_step += 1
        candidates = list(self.concept_vectors.keys())
        sampled = rng.choice(candidates, size=min(sample_size, len(candidates)), replace=False)

        for cid in sampled:
            if cid == winner_cid:
                continue
            v = self.concept_vectors[cid]
            vn = v / max(np.linalg.norm(v), 1e-10)
            sim = float(np.dot(vw_n, vn))
            if sim > 0.3:  # only inhibit if similar enough
                # Push away from winner
                away = v - v_win
                d = np.linalg.norm(away)
                if d > 1e-10:
                    away /= d
                    self.concept_vectors[cid] = v - away * strength * sim
                    n = np.linalg.norm(self.concept_vectors[cid])
                    if n > 0:
                        self.concept_vectors[cid] /= n

    # ---- Homeostatic plasticity ----

    def init_homeostasis(self):
        """Initialize homeostasis tracking for concepts."""
        self.concept_usage = {cid: 0.0 for cid in self.concept_vectors}
        self.concept_fitness = {cid: 1.0 for cid in self.concept_vectors}
        self.concept_momentum = {cid: np.zeros(self.dim, dtype=np.float32)
                                 for cid in self.concept_vectors}
        self._hboost_mean_cache = None
        self._hboost_cache_step = 0

    _hboost_mean_cache = None
    _hboost_cache_step = 0

    def homeostatic_boost(self, cid):
        """Get homeostatic boost for a concept.
        Underused -> positive boost (novelty)
        Overused -> negative boost (fatigue)"""
        usage = self.concept_usage.get(cid, 0.0)
        # Refresh mean cache every 1000 calls
        self._hboost_cache_step += 1
        if self._hboost_cache_step % 1000 == 1 or self._hboost_mean_cache is None:
            vals = list(self.concept_usage.values())
            self._hboost_mean_cache = np.mean(vals) if vals else 1.0
        mean_usage = self._hboost_mean_cache
        if mean_usage < 0.01:
            return 0.0
        boost = (mean_usage - usage) / max(mean_usage, 0.01)
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

    def word_vector(self, word):
        """Get vector for a word (concept centroid + offset)."""
        cid = self.word_to_cid.get(word.lower())
        if cid is None:
            return None
        cv = self.concept_vectors.get(cid)
        if cv is None:
            return None
        return cv.copy()

    def similarity(self, word_a, word_b):
        """Cosine similarity between two words (via concept vectors)."""
        va = self.word_vector(word_a)
        vb = self.word_vector(word_b)
        if va is None or vb is None:
            return 0.0
        return float(np.dot(va, vb) / (
            max(np.linalg.norm(va), 1e-10) * max(np.linalg.norm(vb), 1e-10)
        ))

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
            result.append((c, self.concept_info[c]['anchor'], float(sims[i])))
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
            before: N × D × float32 (4 bytes)
            after:  N × n_subvectors × uint8 (1 byte) + n_subvectors × n_centroids × subdim × float32
            typical for 128D → 32×8 = 32 bytes vs 512 bytes = 16× compression
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
            codebooks.append(kmeans.cluster_centers_.astype(np.float32))

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
            subvecs = mat[:, m * subdim:(m + 1) * subdim]
            cb = self.pq_codebooks[m]  # (n_centroids, subdim)
            # Compute distances from each subvec to each centroid
            # Use argmin across centroids
            for i in range(N):
                diffs = subvecs[i] - cb  # (n_centroids, subdim)
                dists = np.sum(diffs ** 2, axis=1)
                codes[i, m] = np.argmin(dists).astype(np.uint8)

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

        Existing vectors are kept in the first `self.dim` coordinates.
        New dimensions are initialized from a random subspace
        (orthogonal complement) to preserve existing relationships.

        Args:
            target_dim: new dimension (must be > current dim)
        """
        if target_dim <= self.dim:
            return
        old_dim = self.dim
        print(f'  Expanding dimension: {old_dim} -> {target_dim}')
        new_dim = target_dim

        # Generate random orthonormal basis for new dimensions
        rng = np.random.RandomState(42)
        # Use QR decomposition to get orthogonal basis
        rand_mat = rng.randn(new_dim, new_dim).astype(np.float32)
        q, _ = np.linalg.qr(rand_mat)
        # Take projection matrix: old_dim rows of the orthogonal matrix
        # This preserves old vectors and adds orthogonal new dimensions
        proj = q[:old_dim, :new_dim]  # (old_dim, new_dim)

        new_vectors = {}
        for cid, v in self.concept_vectors.items():
            # Project old vector to new space via random orthogonal matrix
            v_new = v @ proj  # (new_dim,) — v is (old_dim,)
            norm = np.linalg.norm(v_new)
            if norm > 1e-10:
                v_new /= norm
            new_vectors[cid] = v_new

        self.concept_vectors = new_vectors
        self.dim = new_dim

        # Expand affix shifts to new dimension
        new_affix = {}
        for affix, shift in self.affix_shifts.items():
            if len(shift) == old_dim:
                new_shift = shift @ proj
                n = np.linalg.norm(new_shift)
                if n > 1e-10:
                    new_shift /= n
                new_affix[affix] = new_shift
            else:
                new_affix[affix] = shift
        self.affix_shifts = new_affix

        # Invalidate caches
        self.pq_codebooks = None
        self.pq_codes = None
        self.pq_cid_order = []
        self._vector_matrix = None
        self._cid_order = []
        self._matrix_dirty = True
        print(f'  Done: {len(new_vectors)} concepts @ {self.dim}D')

    def predict_next_concept(self, prev_cid, top_k=20):
        """Predict next concept from transition matrix."""
        if self.concept_transitions is None:
            return []
        idx = self.cid_to_idx.get(prev_cid)
        if idx is None:
            return []
        row = self.concept_transitions[idx].toarray().flatten()
        if row.sum() == 0:
            return []
        probs = row / row.sum()
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(self.cid_list[i], probs[i]) for i in top_indices if probs[i] > 0]

    def words_in_concept(self, cid, top_k=10):
        """Get words belonging to a concept."""
        return self.cid_to_words.get(cid, [])[:top_k]

    def concept_anchor(self, word):
        """Get anchor/lemma for a word."""
        cid = self.word_to_cid.get(word.lower())
        if cid is not None:
            return self.concept_info[cid]['anchor']
        return None

    def save(self, path, use_pq=False, include_morph=True):
        """Save ConceptSpace to disk.

        Args:
            path: file path
            use_pq: if True, save PQ-compressed format (much smaller).
                    Requires pq_codes to be computed (call pq_encode() first).
            include_morph: if False, omit word_to_morph (for live checkpoints).
        """
        if use_pq and self.pq_codes is not None:
            # PQ format: codes + codebooks, no full vectors
            data = {
                'dim': self.dim,
                'pq': True,
                'n_subvectors': self.pq_n_subvectors,
                'n_centroids': self.pq_n_centroids,
                'pq_cid_order': self.pq_cid_order,
                'pq_codes': self.pq_codes.tolist(),
                'codebooks': [cb.tolist() for cb in self.pq_codebooks],
                'cid_list': self.cid_list,
                'word_to_cid': self.word_to_cid,
                'word_to_morph': self.word_to_morph if include_morph else {},
                'cid_to_words': {str(c): ws for c, ws in self.cid_to_words.items()},
                'concept_info_keys': {str(c): {'anchor': info['anchor'], 'size': info['size']}
                                      for c, info in self.concept_info.items()},
            }
        else:
            data = {
                'dim': self.dim,
                'pq': False,
                'cid_list': self.cid_list,
                'word_to_cid': self.word_to_cid,
                'word_to_morph': self.word_to_morph if include_morph else {},
                'cid_to_words': {str(c): ws for c, ws in self.cid_to_words.items()},
                'concept_vectors': {str(c): v.tolist() for c, v in self.concept_vectors.items()},
                'concept_info_keys': {str(c): {'anchor': info['anchor'], 'size': info['size']}
                                      for c, info in self.concept_info.items()},
            }
        # Save homeostatic state if initialized
        if hasattr(self, 'concept_usage'):
            data['concept_usage'] = {str(c): u for c, u in self.concept_usage.items()}
            data['concept_fitness'] = {str(c): f for c, f in self.concept_fitness.items()}

        with open(path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(path + '.tmp', path)
        size_kb = os.path.getsize(path) / 1024
        print(f"  Saved ConceptSpace ({'PQ ' if use_pq else ''}{size_kb:.0f}KB) to {path}")

    @classmethod
    def load(cls, path):
        """Load ConceptSpace from disk (class method).

        Handles both regular and PQ-compressed formats automatically.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.skeleton = None
        obj.dim = data['dim']
        obj.cid_list = data['cid_list']
        obj.cid_to_idx = {c: i for i, c in enumerate(obj.cid_list)}

        # PQ attributes (set defaults before potential override)
        obj.pq_codebooks = None
        obj.pq_codes = None
        obj.pq_cid_order = []
        obj.pq_n_subvectors = 0
        obj.pq_n_centroids = 0

        # Always load mapping data
        obj.word_to_cid = data.get('word_to_cid') or {}
        obj.cid_to_words = {int(c): ws for c, ws in data.get('cid_to_words', {}).items()}
        obj.word_to_morph = data.get('word_to_morph') or {}

        if data.get('pq'):
            # PQ-compressed format: decode back to full vectors
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
            obj.concept_vectors = {int(c): np.array(v, dtype=np.float32)
                                     for c, v in data['concept_vectors'].items()}

        obj.cid_vectors = {}
        obj.concept_transitions = None
        obj.concept_info = {}
        for c_str, info in data['concept_info_keys'].items():
            c = int(c_str)
            obj.concept_info[c] = {
                'cid': c,
                'anchor': info['anchor'],
                'satellites': [],
                'relations': defaultdict(list),
                'vector': obj.concept_vectors.get(c),
                'size': info['size'],
            }
        obj.rng = np.random.RandomState(42)
        obj._concept_usage = Counter()
        obj._inhibition_step = 0
        obj._vector_matrix = None
        obj._cid_order = []
        obj._matrix_dirty = True
        obj.affix_shifts = {}
        obj._init_affix_shifts()
        pq_note = ' (PQ)' if data.get('pq') else ''
        print(f"  Loaded ConceptSpace: {len(obj.cid_list)} concepts @ {obj.dim}D{pq_note}")
        return obj


# ---- Training: concept-level SVD shift ----

def concept_svd_shift(cs, prev_cid, next_cid, is_match, lr=0.1):
    """Riemannian gradient descent on concept vectors.

    Args:
        cs: ConceptSpace
        prev_cid: previous concept ID (context anchor)
        next_cid: predicted/expected concept ID
        is_match: whether next_cid matches the expected concept
        lr: learning rate
    """
    v_prev = cs.concept_vectors.get(prev_cid)
    v_next = cs.concept_vectors.get(next_cid)
    if v_prev is None or v_next is None:
        return

    scale = 1.0 if is_match else 0.05
    y = float(np.dot(v_next, v_prev))
    y = max(y, 0.05)
    # Riemannian gradient on sphere: ∇ = v_prev - (v_next·v_prev) × v_next
    shift = (v_prev - y * v_next) * lr * scale

    cs.concept_vectors[next_cid] = v_next + shift
    n = np.linalg.norm(cs.concept_vectors[next_cid])
    if n > 0:
        cs.concept_vectors[next_cid] /= n


if __name__ == '__main__':
    import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
    from eva.symbolic.concept_net import ConceptSkeleton
    from eva.symbolic.concept_tokenizer import ConceptTokenizer

    print("Loading ConceptNet skeleton...")
    skeleton = ConceptSkeleton()
    skeleton.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_skeleton.json')

    print("Initializing tokenizer...")
    tok = ConceptTokenizer()
    tok.initialize()

    print("Building ConceptSpace...")
    cs = ConceptSpace(skeleton, dim=128)
    cs.build(corpus_path=r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru.txt', tok=tok)

    print(f"\nConcepts: {len(cs.cid_list)}")
    print(f"Vector dim: {cs.dim}")

    # Test: get vectors and similarities
    test_words = ['собака', 'армия', 'война', 'человек', 'князь', 'сказал']
    for w in test_words:
        cid = cs.word_to_cid.get(w)
        v = cs.concept_vector(cid) if cid is not None else None
        anchor = cs.concept_anchor(w)
        print(f"  {w:12s} -> concept [{cid:4d}] anchor='{anchor}' vector_norm={np.linalg.norm(v):.4f}" if v is not None else f"  {w:12s} -> NO VECTOR")

    # Test concept similarity
    pairs = [('собака', 'собаки'), ('война', 'армия'), ('человек', 'князь'),
             ('сказал', 'говорить'), ('большой', 'маленький')]
    for a, b in pairs:
        sim = cs.similarity(a, b)
        print(f"  sim({a}, {b}) = {sim:.4f}")

    # Test concept prediction
    test_cid = cs.word_to_cid.get('князь')
    if test_cid is not None:
        next_c = cs.predict_next_concept(test_cid, top_k=5)
        print(f"\nConcepts following '{'князь'}' (concept {test_cid}):")
        for cid, prob in next_c:
            print(f"  {cs.concept_info[cid]['anchor']:20s} p={prob:.4f}")

    cs.save(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space.json')
