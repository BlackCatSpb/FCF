"""SemanticGate — semantic sieve: extracts core concept from query.

Not a neural network. Not a transformer.
Pure symbolic geometry + accumulated role statistics + morphology.

Flow:
  words → morph parse → project roots to concept space →
  core_score each word (root-weighted) → weighted centroid →
  semantic attractor → modifier field

The gate's single output to the generator is:
  "The query is about concept X. These are its modifiers."
"""

import numpy as np
from eva.symbolic.concept_tokenizer import ConceptTokenizer
from eva.symbolic.pos_tagger import get_pos


class SemanticGate:
    """Semantic sieve: filters query words → core concept + modifier field.

    Role memory accumulates per-word statistics across training:
      - core_count: how often this word was the semantic attractor
      - mod_count: how often it modified another core
      - noise_count: how often it had no significant connection

    The gate is query-dependent: connection strength between
    query words determines who modifies whom.
    """

    def __init__(self, cs, lattice, resolve_anchor_fn, closest_concept_fn):
        self.cs = cs
        self.lattice = lattice
        self._resolve = resolve_anchor_fn   # (word) → (cid, confidence)
        self._closest = closest_concept_fn  # (vec, k) → [(cid, sim)]

        # word → {core_count, mod_count, noise_count}
        self.role_memory = {}

        self.attractor_threshold = 0.35
        self.noise_threshold = 0.10

    # ── Public API ────────────────────────────────────────────────

    def extract_core(self, words):
        """Extract core concept and modifier field from query words.

        Returns:
            core_cid: concept ID of the semantic attractor
            modifier_field: {cid: {word, strength, relation}}
            core_centroid: weighted centroid vector (query-aligned)
            noise_words: list of words filtered out as noise
        """
        if not words:
            return self._neutral_cid(), {}, None, []

        # 1. Resolve all words → anchors
        anchors = self._resolve_all(words)
        if not anchors:
            return self._neutral_cid(), {}, None, list(words)

        # 2. Score each word for core potential
        scored = {}
        for w, info in anchors.items():
            scored[w] = self._core_score(w, info['cid'], anchors)

        # 3. Find the core word (clear winner required)
        ranked = sorted(scored.items(), key=lambda x: -x[1])
        top_word, top_score = ranked[0]

        if top_score < 0.15:
            return self._neutral_cid(), {}, None, list(words)

        top_cid = anchors[top_word]['cid']
        top_vec = anchors[top_word]['vec']

        # 4. Weighted centroid (core-weighted, not mean)
        centroid = self._weighted_centroid(anchors, scored)

        # 5. Semantic attractor: nearest concept to centroid
        attractor_cid = self._find_attractor(centroid, top_cid)

        # 6. Build modifier field
        modifier_field = {}
        noise_words = []

        for w, info in anchors.items():
            cid = info['cid']
            if cid == attractor_cid:
                continue
            conn = self._connection_strength(attractor_cid, cid)
            if conn > self.noise_threshold:
                modifier_field[cid] = {
                    'word': w,
                    'strength': conn,
                    'relation': self._infer_relation(w, info['vec'], top_vec),
                }
            else:
                noise_words.append(w)

        # 7. Update role memory (silent, no I/O)
        for w, info in anchors.items():
            cid = info['cid']
            if cid == attractor_cid:
                self._remember(w, 'core')
            elif cid in modifier_field:
                self._remember(w, 'mod')
            else:
                self._remember(w, 'noise')

        return attractor_cid, modifier_field, centroid, noise_words

    def update_role_memory(self, words, core_cid, modifier_cids):
        """Externally update role memory with known role assignments."""
        for w in words:
            cid, _ = self._resolve(w)
            if cid == core_cid:
                self._remember(w, 'core')
            elif cid in modifier_cids:
                self._remember(w, 'mod')
            else:
                self._remember(w, 'noise')

    def core_score(self, word, cid):
        """Public: compute core score for a single word (used by tests)."""
        return self._core_score(word, cid, None)

    # ── Internals ────────────────────────────────────────────────

    def _resolve_all(self, words):
        anchors = {}
        for w in words:
            # Use morphological root first
            morph = ConceptTokenizer.morph_parse(w)
            root_word = morph['normal_form'] if morph else w

            cid, conf = self._resolve(root_word)
            v = self.cs.concept_vector(cid)
            if v is not None and conf > 0:
                anchors[w] = {
                    'cid': cid,
                    'conf': conf,
                    'vec': v.copy(),
                    'root': root_word,
                    'pos': morph['pos'] if morph else 'UNK',
                    'has_affixes': bool(morph and (morph['prefix'] or morph['suffix'] or morph['ending'])),
                }
            else:
                # Fallback: resolve original word
                cid, conf = self._resolve(w)
                v = self.cs.concept_vector(cid)
                if v is not None and conf > 0:
                    anchors[w] = {
                        'cid': cid, 'conf': conf, 'vec': v.copy(),
                        'root': w, 'pos': 'UNK', 'has_affixes': False,
                    }
        return anchors

    def _core_score(self, word, cid, anchors=None):
        """Score how likely this word is the semantic core.

        Factors:
          - role_memory: how often this word was core vs modifier vs noise
          - POS type: nouns and verbs are more likely cores
          - morphological complexity: words with affixes are modifiers
          - frequency: how often the concept appears independently in corpus
          - breadth: how many morphological satellites the concept has
          - query_connections: average similarity to other query words
        """
        mem = self.role_memory.get(word, {'core': 1, 'mod': 1, 'noise': 1})
        total = mem['core'] + mem['mod'] + mem['noise']
        role_ratio = mem['core'] / max(total, 1)

        # POS-based score (learned, not hardcoded — but initialized with priors)
        pos = 'UNK'
        if anchors and word in anchors:
            pos = anchors[word].get('pos', 'UNK')
        pos_boost = {'NOUN': 0.15, 'VERB': 0.10, 'ADJ': 0.05, 'ADV': 0.03}.get(pos, 0.0)

        # Morphological complexity penalty: affixes suggest modifier role
        has_affixes = anchors and word in anchors and anchors[word].get('has_affixes', False)
        morph_penalty = -0.05 if has_affixes else 0.0

        freq = self.lattice.concept_freq.get(cid, 0)
        freq_norm = min(freq / 100, 1.0)

        breadth = len(self.cs.cid_to_words.get(cid, []))
        breadth_norm = min(breadth / 10, 1.0)

        # Query-dependent: connection strength to other query words
        query_conn = 0.0
        if anchors:
            others = [info for w, info in anchors.items()
                      if w != word and info['cid'] != cid]
            if others:
                v = self.cs.concept_vector(cid)
                if v is not None:
                    vn = v / max(np.linalg.norm(v), 1e-10)
                    sims = []
                    for info in others:
                        ov = info['vec']
                        s = float(np.dot(vn, ov / max(np.linalg.norm(ov), 1e-10)))
                        sims.append(max(s, 0))
                    query_conn = np.mean(sims) if sims else 0.0

        return (role_ratio * 0.4 + freq_norm * 0.15 +
                breadth_norm * 0.1 + query_conn * 0.2 +
                pos_boost + morph_penalty)

    def _weighted_centroid(self, anchors, scored):
        weights = {w: max(scored.get(w, 0.1), 0.01) for w in anchors}
        total_w = sum(weights.values())
        if total_w < 1e-10:
            vecs = [info['vec'] for info in anchors.values()]
            return np.mean(vecs, axis=0)
        centroid = np.sum([
            info['vec'] * (weights[w] / total_w)
            for w, info in anchors.items()
        ], axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid /= norm
        return centroid

    def _find_attractor(self, centroid, fallback_cid):
        nearest = self._closest(centroid, k=3)
        if nearest and nearest[0][1] > self.attractor_threshold:
            return nearest[0][0]

        # Centroid too far from all existing concepts → create new one
        if centroid is not None and nearest and nearest[0][1] < 0.2:
            new_cid = self._create_concept(centroid, fallback_cid)
            if new_cid is not None:
                return new_cid

        return fallback_cid

    def _create_concept(self, centroid, hint_cid):
        """Create a new concept from a centroid vector.

        Adds it to the ConceptSpace dynamically.
        """
        cs = self.cs
        # Find next available CID
        existing = set(cs.concept_vectors.keys())
        if not existing:
            return None
        new_cid = max(existing) + 1

        # Store vector
        cs.concept_vectors[new_cid] = centroid.copy()
        cs.cid_list.append(new_cid)

        # Store info
        anchor_hint = cs.concept_info.get(hint_cid, {}).get('anchor', 'auto')
        cs.concept_info[new_cid] = {
            'cid': new_cid,
            'anchor': f'{anchor_hint}_auto_{new_cid}',
            'satellites': [],
            'vector': centroid.copy(),
            'size': 1,
        }
        cs.cid_to_words[new_cid] = [f'auto_{new_cid}']
        cs.mark_matrix_dirty()

        return new_cid

    def _connection_strength(self, cid_a, cid_b):
        va = self.cs.concept_vector(cid_a)
        vb = self.cs.concept_vector(cid_b)
        if va is None or vb is None:
            return 0.0
        cos = float(np.dot(va, vb) / max(
            np.linalg.norm(va) * np.linalg.norm(vb), 1e-10))
        return max(cos, 0)

    def _infer_relation(self, word, word_vec, core_vec):
        """Infer relation type from modifier word's morphology and POS.

        Uses morphology to determine semantic relation:
        - ADJ: has_quality (хорошая погода)
        - ADV: has_manner (быстро бежать)
        - NOUN (genitive): has_possession (дом человека)
        - VERB: performs_action (человек идёт)
        - PREP: located_at / temporal_at (на улице)
        """
        morph = ConceptTokenizer.morph_parse(word)
        pos = morph['pos'] if morph else get_pos(word)

        relation_map = {
            'ADJ': 'has_quality',
            'ADV': 'has_manner',
            'VERB': 'performs_action',
            'PREP': 'located_at',
            'PRON': 'related',
            'NUM': 'has_quantity',
            'CONJ': 'connects',
            'PART': 'modifies',
            'NOUN': 'related_to',
        }
        return relation_map.get(pos, 'related')

    def _remember(self, word, role):
        if word not in self.role_memory:
            self.role_memory[word] = {'core': 0, 'mod': 0, 'noise': 0}
        self.role_memory[word][role] += 1

    def _neutral_cid(self):
        if hasattr(self.cs, '_neutral_cid'):
            return self.cs._neutral_cid
        if self.cs.cid_list:
            return self.cs.cid_list[0]
        return 0
