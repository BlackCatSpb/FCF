import re

"""EVA Concept Generator v3 — intent-driven concept navigation.

Architecture:
  - Every input word resolves to a concept ANCHOR (no fallbacks)
  - Query words form an INTENT CLOUD in concept space
  - The model projects the intent onto the closest coherent region
  - Generation EXPANDS from the intent through imagery potential
  - Semantic distance between query and response drives exploration

Core principles:
  - No fallback: every word has a concept (by nearest-neighbor if needed)
  - Intent understanding: what did the query MEAN? (not just pattern match)
  - Distance awareness: model knows how far it's strayed from query
  - Imagery expansion: generation explores the semantic field around intent
"""

import math, random
import numpy as np
from collections import defaultdict, Counter

from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.hormonal_system import HormonalSystem
from eva.symbolic.concept_inductor import ConceptInductor
from eva.symbolic.semantic_gate import SemanticGate
from eva.symbolic.concept_tokenizer import ConceptTokenizer


class CrystalGenerator:
    """Generation as semantic navigation through concept space.

    Each generation step:
      1. Anchor resolution: query words -> concept anchors (always)
      2. Intent projection: anchors -> coherent concept region
      3. Branch: K alternative next concepts (beam search)
      4. Score: syntax + semantics + distance check + homeostasis
      5. Morph: select word form within concept (nucleus -> electron)
      6. Hormonal update: reward/punishment modulates next step
      7. Learn: STDP on each branch (modulated by hormones)
    """

    def __init__(self, cs, tok, lattice, config=None):
        self.cs = cs
        self.tok = tok
        self.lattice = lattice
        self.config = config or {}

        # Beam parameters
        self.beam_width = self.config.get('beam_width', 5)
        self.max_words = self.config.get('max_words', 20)
        self.min_words = self.config.get('min_words', 3)

        # Base temperature (theta-rhythm + hormones modulate these)
        self.base_concept_temp = self.config.get('concept_temp', 0.5)
        self.base_word_temp = self.config.get('word_temp', 0.3)
        self.theta_tau = self.config.get('theta_tau', 12.0)

        # Base STDP learning rate (modulated by ACh + DA)
        self.base_learning_rate = self.config.get('learning_rate', 0.1)

        # Random state
        self.main_rng = random.Random(42)

        # Homeostasis
        self.cs.init_homeostasis()

        # Hormonal system (intrinsic motivation)
        self.hormones = HormonalSystem()

        # Per-branch RNGs
        self.branch_rngs = {}

        # Merge potential cache (precomputed on first use)
        self._merge_cache = None

        # Resolve cache: word → (cid, confidence)
        self._resolve_cache = {}

        # --- Semantic Gate (core concept extraction) ---
        self.gate = SemanticGate(
            cs, lattice,
            resolve_anchor_fn=self.resolve_anchor,
            closest_concept_fn=self._closest_concept,
        )

        # --- Concept Induction (semantic resonance) ---
        self.inductor = ConceptInductor(self.config)

        # --- Intent & Distance tracking ---
        self._query_anchors = []    # concept IDs from query
        self._query_centroid = None # mean vector of query concepts
        self._query_confidence = 1.0  # how well we understood the query
        self._response_centroid = None # mean vector of response so far

        # --- Core concept & modifier field (from gate) ---
        self._core_cid = None
        self._modifier_field = {}
        self._core_aspects = []  # decomposed aspects of current core

    def _theta_temp(self, word_num):
        """Theta-rhythm temperature: high early (explore), low late (exploit)."""
        t = self.base_concept_temp * math.exp(-word_num / self.theta_tau)
        return max(t, self.base_concept_temp * 0.15)  # floor at 15% of base

    def _get_branch_rng(self, branch_id):
        """Get a seeded RNG for a specific beam branch."""
        if branch_id not in self.branch_rngs:
            self.branch_rngs[branch_id] = random.Random(branch_id * 137)
        return self.branch_rngs[branch_id]

    # ── Anchor resolution (no fallbacks) ──────────────────────────

    def _is_noise(self, w):
        """Detect pure noise input: mixed scripts, abnormal composition."""
        if not w:
            return True
        has_cyrillic = bool(re.search(r'[а-яё]', w))
        has_latin = bool(re.search(r'[a-z]', w))
        if has_cyrillic and has_latin:
            return True
        digit_ratio = sum(c.isdigit() for c in w) / len(w)
        if digit_ratio > 0.3:
            return True
        non_alpha_ratio = sum(not c.isalpha() for c in w) / len(w)
        if non_alpha_ratio > 0.3:
            return True
        return False

    def resolve_anchor(self, word):
        """Resolve word to concept anchor with confidence.

        Plastic morph-aware resolution:
          1. morph root → direct CID (fast, O(1))
          2. affixed root → apply affix shifts → nearest concept
          3. unknown root → orthographic/BPE (slow, cached)
          4. nothing → neutral (signal: unknown)

        The resolve cache is NOT a hardcoded dict — it's a learned associative
        memory that grows with training. When the vector space shifts (STDP),
        the cache remains valid because it maps words to CID (stable),
        not words to vectors (plastic).
        """
        w = word.lower().strip()
        if not w or self._is_noise(w):
            return self._neutral_anchor(), 0.0

        # Fast reject: words with replacement chars are encoding artifacts
        if '\ufffd' in w:
            result = (self._neutral_anchor(), 0.0)
            self._resolve_cache[w] = result
            return result

        # Cache check
        cached = self._resolve_cache.get(w)
        if cached is not None:
            return cached

        # 1. Direct lookup (exact dictionary match, fastest)
        cid = self.cs.word_to_cid.get(w)
        if cid is not None:
            self._resolve_cache[w] = (cid, 1.0)
            return cid, 1.0

        # 2. Morph root resolution (fast, O(1) with cache)
        # Root → same CID regardless of affixes.
        # Affix shifts apply to VECTORS (for gate/centroid computation),
        # not to CID resolution — same concept, different word form.
        morph = self._lookup_morph(w)
        root = morph['normal_form'] if morph else w
        root_cid = self.cs.word_to_cid.get(root)
        if root_cid is not None:
            has_affixes = bool(morph and (morph.get('prefix') or morph.get('suffix') or morph.get('ending')))
            conf = 0.9 if has_affixes else 1.0
            result = (root_cid, conf)
            self._resolve_cache[w] = result
            return result

        # 2b. Root not in dict but the word itself might be (if morph failed)
        if not morph and root != w:
            cid = self.cs.word_to_cid.get(w)
            if cid is not None:
                result = (cid, 0.9)
                self._resolve_cache[w] = result
                return result

        # 3. Compound word decomposition: split into known subwords
        #    E.g., "резервуардляводы" → "резервуар" + "для" + "воды"
        subwords = self._decompose_word(w)
        if subwords:
            main = subwords[-1]
            cid = self.cs.word_to_cid.get(main)
            if cid is not None:
                result = (cid, 0.7)
                self._resolve_cache[w] = result
                return result

        # 4. Word not in dictionary → unknown.
        #    No forced fallback (no orthographic/BPE scan).
        #    Unknown = signal: "I have no data for this word."
        #    The gate will still see its vector position via centroid,
        #    and adaptive concept creation handles new cores.
        result = (self._neutral_anchor(), 0.0)
        self._resolve_cache[w] = result
        return result

    def _decompose_word(self, word):
        """Split concatenated word into known subwords via longest-prefix match.
        Uses the model's own vocabulary — no external data.

        E.g., 'резервуардляводы' → ['резервуар', 'для', 'воды']
        Returns list of subwords or None if no decomposition found.
        """
        if not word or len(word) < 4:
            return None
        parts = []
        remaining = word
        while remaining:
            found = False
            for end in range(len(remaining), 2, -1):
                prefix = remaining[:end]
                if prefix in self.cs.word_to_cid:
                    parts.append(prefix)
                    remaining = remaining[end:]
                    found = True
                    break
            if not found:
                # Partial match: if we already found at least one subword,
                # accept what we have (the remainder is likely a suffix)
                if parts:
                    return parts
                return None
        return parts if len(parts) > 1 else None

    def _lookup_morph(self, word):
        """Look up morphological parse: first from in-model cache, then live parse."""
        cached = self.cs.word_to_morph.get(word)
        if cached is not None:
            return cached
        # Fallback to live parse (for words not in corpus)
        morph = ConceptTokenizer.morph_parse(word)
        return morph

    def _neutral_anchor(self):
        """The semantic 'center of mass' — represents complete uncertainty.
        
        This concept sits at the normalized centroid of all concept vectors.
        It means: "this input doesn't match any specific knowledge."
        The model should not generate confidently from this anchor.
        """
        if hasattr(self, '_neutral_anchor_cid'):
            return self._neutral_anchor_cid

        # Compute centroid of all concept vectors (semantic origin)
        all_vs = list(self.cs.concept_vectors.values())
        if not all_vs:
            return self.cs.cid_list[0] if self.cs.cid_list else 0

        centroid = np.mean(all_vs, axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid /= norm

        # Find closest existing concept to centroid
        self._neutral_anchor_cid = self._closest_concept(centroid, k=1)[0][0]

        # Ensure it's stored
        return self._neutral_anchor_cid

    def _edit_similarity(self, a, b):
        """Normalized edit similarity (Dice coefficient on bigrams)."""
        if not a or not b:
            return 0.0
        bigrams_a = set(a[i:i+2] for i in range(len(a)-1))
        bigrams_b = set(b[i:i+2] for i in range(len(b)-1))
        if not bigrams_a or not bigrams_b:
            return 0.0
        overlap = len(bigrams_a & bigrams_b)
        return 2.0 * overlap / (len(bigrams_a) + len(bigrams_b))

    def _closest_concept(self, vec, k=5, exclude=None):
        """Find k closest concepts to a given vector (batched matrix NN)."""
        if vec is None:
            return [(0, 0.0)]
        self.cs.ensure_matrix()
        mat = self.cs._vector_matrix
        if mat.shape[0] == 0:
            return [(0, 0.0)]
        vn = vec / max(np.linalg.norm(vec), 1e-10)
        sims = mat @ vn  # (N,) — single matrix multiply
        exclude_set = set(exclude or [])
        # Argpartition for top k
        n = len(sims)
        k_actual = min(k, n)
        if k_actual <= 0:
            return [(0, 0.0)]
        idx = np.argpartition(-sims, k_actual - 1)[:k_actual]
        idx = idx[np.argsort(-sims[idx])]
        result = []
        for i in idx:
            cid = self.cs._cid_order[i]
            if cid in exclude_set:
                continue
            result.append((cid, float(sims[i])))
            if len(result) >= k:
                break
        # Fallback: if all excluded, pad with any
        while len(result) < k:
            for cid in self.cs.cid_list:
                if cid not in exclude_set and (cid, 0.0) not in result:
                    result.append((cid, 0.0))
                    break
            else:
                break
        return result[:k]

    # ── Intent projection ────────────────────────────────────────

    def project_intent(self, query_words):
        """Project query words into concept space to find INTENT anchor.

        Given query words, finds the concept region that best represents
        the overall INTENT — not just individual words, but the semantic
        field they collectively point to.

        Returns:
            intent_cid: the anchor concept for the response
            intent_vec: centroid vector of the intent region
            delta: semantic spread of the query concepts (diversity measure)
        """
        anchors = []
        vectors = []
        query_confidence = []
        for w in query_words:
            cid, conf = self.resolve_anchor(w)
            v = self.cs.concept_vector(cid)
            if v is not None:
                anchors.append(cid)
                vectors.append(v)
                query_confidence.append(conf)

        self._query_anchors = anchors
        self._query_confidence = np.mean(query_confidence) if query_confidence else 0.0

        if not vectors:
            # Should never happen since resolve_anchor always returns
            cid = self.cs.cid_list[0]
            return cid, self.cs.concept_vector(cid), 0.0

        # Intent centroid: mean of query concept vectors
        intent_vec = np.mean(vectors, axis=0).astype(np.float32)
        intent_vec /= max(np.linalg.norm(intent_vec), 1e-10)

        # Semantic spread: average pairwise distance in query cloud
        if len(vectors) > 1:
            spreads = []
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    d = 1.0 - float(np.dot(vectors[i], vectors[j]))
                    spreads.append(d)
            delta = np.mean(spreads)
        else:
            delta = 0.0

        # Intent anchor = concept closest to centroid
        # Prefer non-query concepts, but don't exclude if no good alternative
        candidates = self._closest_concept(intent_vec, k=5)
        non_query = [(c, s) for c, s in candidates if c not in anchors]
        intent_cid = non_query[0][0] if non_query else candidates[0][0]

        return intent_cid, intent_vec, delta

    # ── Core decomposition (field exploration) ─────────────────

    def decompose_core(self, core_cid, top_k=10):
        """Decompose core concept into aspects learned from corpus.

        Aspects are what the model LEARNED about the core from training:
        - Connected concepts from connection graph (highest priority: real data)
        - N-gram successors from lattice (real sequences from corpus)
        - Modifier field from gate (query-specific)
        - Vector neighbors only as fallback (weakest signal)

        No hardcoded rules — everything comes from training data.
        """
        aspects = {}

        # 1. Connection graph: what concepts co-occur with core in corpus
        connected = self.lattice.connections_of(core_cid, top_k=top_k)
        for cid, conn in connected:
            aspects[cid] = {
                'cid': cid,
                'strength': conn['strength'] * 1.5,  # boost: real data
                'relation': conn['type'],
                'type': 'connection',
            }

        # 2. N-gram lattice: what concepts follow core in real texts
        ngram_preds = self.lattice.predict([core_cid])
        for cid, score in ngram_preds[:top_k]:
            if cid not in aspects or score > aspects[cid]['strength']:
                aspects[cid] = {
                    'cid': cid,
                    'strength': score * 1.3,  # boost: real sequences
                    'relation': 'follows',
                    'type': 'ngram',
                }

        # 3. Modifier field (query-specific from gate)
        for cid, mod_info in self._modifier_field.items():
            if cid not in aspects or mod_info['strength'] > aspects[cid]['strength']:
                aspects[cid] = {
                    'cid': cid,
                    'strength': mod_info.get('strength', 0.5),
                    'relation': mod_info.get('relation', 'modifies'),
                    'type': 'modifier',
                }

        # 4. Vector neighbors (weakest — only as novelty fallback)
        v_core = self.cs.concept_vector(core_cid)
        if v_core is not None:
            similar = self.cs.topk_similar_concepts(core_cid, k=top_k)
            for cid, anchor, sim in similar:
                if cid not in aspects:
                    aspects[cid] = {
                        'cid': cid, 'strength': sim * 0.3,  # discounted
                        'relation': 'similar_to', 'type': 'novelty',
                    }

        result = sorted(aspects.values(), key=lambda x: -x['strength'])[:top_k]
        self._core_aspects = result
        return result

    def _gate_verification(self, cid):
        """Verify that a generated concept is connected to the core.

        Returns:
            (is_connected, connection_strength)
        """
        if self._core_cid is None or cid == self._core_cid:
            return True, 1.0

        # Direct connection from lattice
        conn = self.lattice.get_connection(self._core_cid, cid)
        if conn is not None and conn['strength'] > 0.05:
            return True, conn['strength']

        # Vector similarity
        v_core = self.cs.concept_vector(self._core_cid)
        v_cand = self.cs.concept_vector(cid)
        if v_core is not None and v_cand is not None:
            cos = float(np.dot(v_core, v_cand) / max(
                np.linalg.norm(v_core) * np.linalg.norm(v_cand), 1e-10))
            if cos > 0.2:
                return True, max(cos, 0)

        # No connection found
        return False, 0.0

    # ── Semantic distance tracking ───────────────────────────────

    def _semantic_delta(self, query_vec, response_path, window=5):
        """Distance between query intent and generated response so far.

        Returns:
            delta ∈ [0, 2]: 0 = identical, 1 = orthogonal, 2 = opposite
            The model uses this to stay relevant: not too close (parroting),
            not too far (topic drift).
        """
        if query_vec is None or not response_path:
            return 0.5  # moderate distance by default

        recent = response_path[-window:]
        recent_vecs = [self.cs.concept_vector(c) for c in recent if self.cs.concept_vector(c) is not None]
        if not recent_vecs:
            return self._last_delta if hasattr(self, '_last_delta') else 0.5

        response_vec = np.mean(recent_vecs, axis=0).astype(np.float32)
        response_vec /= max(np.linalg.norm(response_vec), 1e-10)

        qn = query_vec / max(np.linalg.norm(query_vec), 1e-10)
        delta = 1.0 - float(np.dot(qn, response_vec))
        self._last_delta = delta
        return delta

    # ── Generation ───────────────────────────────────────────────

    def generate(self, seed_word=None, seed_cid=None, target_text=None,
                 query_words=None, max_words=None):
        """Generate response anchored to intent.

        Every query word resolves to a concept (no fallbacks).
        The model projects intent, then expands through imagery potential,
        constantly checking semantic distance from the query.

        Args:
            seed_word: starting word (if None, uses first query word)
            seed_cid: starting concept (overrides seed_word)
            target_text: target for supervised training
            query_words: list of words from the query/source

        Returns:
            dict with response text, concept path, intent info
        """
        # ── Extract core concept via semantic gate ──
        src_words = query_words or ([seed_word] if seed_word else [])
        core_cid, modifier_field, core_centroid, noise_words = self.gate.extract_core(src_words)

        if core_centroid is not None:
            self._query_centroid = core_centroid
        self._core_cid = core_cid
        self._modifier_field = modifier_field

        # Confidence = how cleanly the gate found a core
        gate_noise_ratio = len(noise_words) / max(len(src_words), 1)
        self._query_confidence = max(0.1, 1.0 - gate_noise_ratio)

        # ── Decompose core concept into aspects (field exploration) ──
        if core_cid is not None:
            self.decompose_core(core_cid)

        # Use core as seed if no explicit seed given
        if seed_cid is None and seed_word is None:
            seed_cid = core_cid
        elif seed_cid is None and seed_word is not None:
            seed_cid, seed_conf = self.resolve_anchor(seed_word)
            self._query_confidence = min(self._query_confidence, seed_conf)

        # Feed query confidence into hormonal system
        # Low confidence -> NA up, 5HT up, ACh down (cautious mode)
        if self._query_confidence < 0.3:
            self.hormones.noradrenaline = min(1.0, self.hormones.noradrenaline + 0.2)
            self.hormones.serotonin = min(1.0, self.hormones.serotonin + 0.1)

        target_concepts = self._target_concepts(target_text)
        effective_max = max_words if max_words is not None else self.max_words

        # Beam: list of (concept_sequence, score, tokens, branch_id)
        beam = [([seed_cid], 0.0, [self.tok.BOS, self.tok.SENT_OPEN], 0)]

        all_chains = []
        finished = []
        next_branch_id = 1
        self._intent_drift = 0.0  # cumulative drift from intent

        for wn in range(effective_max):
            new_beam = []

            # ---- Hormonal modulation ----
            theta_temp = self._theta_temp(wn)
            h_temp = self.hormones.modulate_temperature(theta_temp)
            h_lr = self.hormones.modulate_stdp_lr(self.base_learning_rate)
            h_beam = self.hormones.modulate_beam_width(self.beam_width)

            # ---- Semantic distance check ----
            qv = self._query_centroid
            delta = self._semantic_delta(qv, [b[0][-1] for b in beam])
            # If drifting too far, increase temperature (explore back)
            # If too close, also increase temperature (explore away from parroting)
            ideal_delta = 0.3 + 0.4 * (1.0 - math.exp(-wn / 5.0))  # increases with wn
            drift_penalty = abs(delta - ideal_delta) * 0.5
            self._intent_drift += drift_penalty

            for seq, score, tokens, branch_id in beam:
                prev_cid = seq[-1]

                expected_idx = wn + 1
                expected_cid = target_concepts[expected_idx] if expected_idx < len(target_concepts) else None

                # ---- Branch: get candidates (with distance awareness) ----
                candidates = self._branch(seq, wn, h_temp, expected_cid, self._query_centroid)
                if not candidates:
                    self.hormones.update(confidence=0.0, is_match=False,
                        novelty=0.0, surprise=0.5, expected_cid=expected_cid)
                    finished.append((seq, score, tokens, wn))
                    continue

                for ci, (cid, cand_score) in enumerate(candidates):
                    new_seq = seq + [cid]

                    # ---- Gate verification: is this candidate connected to core? ----
                    is_verified, gate_strength = self._gate_verification(cid)
                    if not is_verified and core_cid is not None:
                        # Not connected to core — if confidence low, skip entirely
                        if self._query_confidence > 0.5:
                            continue

                    # Distance penalty: penalize candidates that increase drift
                    dist_penalty = drift_penalty * (1.0 if ci == 0 else 1.0 + 0.1 * ci)
                    new_score = score + cand_score - dist_penalty

                    # Gate verification bonus: prefer concepts connected to core
                    if is_verified:
                        new_score += gate_strength * 0.1

                    # ---- Word form selection ----
                    word_text = self._select_word(cid, h_temp)
                    if word_text is None:
                        continue

                    # ---- Build tokens ----
                    new_tokens = tokens.copy()
                    new_tokens.append(self.tok.WORD_OPEN)
                    word_ids = self.tok.bpe.encode(word_text).ids
                    for wid in word_ids:
                        new_tokens.append(wid + self.tok.N_SPECIAL)
                    new_tokens.append(self.tok.WORD_CLOSE)

                    conf = 1.0 / (1.0 + ci * 0.5)
                    is_match = (expected_cid is not None and cid == expected_cid)

                    # ---- STDP ----
                    self.cs.svd_shift(prev_cid, cid, expected_cid=expected_cid,
                                       lr=h_lr, word_num=wn)
                    self.cs.update_usage(cid)

                    # ---- Lattice learning ----
                    if is_match:
                        self.lattice.update([prev_cid, cid])

                    # ---- Hormonal ----
                    novelty = 1.0 - min(self.lattice.concept_freq.get(cid, 0) / 50, 1.0)
                    surprise = abs(delta - ideal_delta) * 0.5 if not is_match else 0.1
                    self.hormones.update(confidence=conf, is_match=is_match,
                        novelty=novelty, surprise=surprise,
                        expected_cid=expected_cid, gen_cid=cid)

                    # ---- Concept induction (resonance-based) ----
                    induced = self.inductor.observe(
                        [prev_cid, cid], self.cs, self.lattice, self.hormones)
                    for new_cid in induced:
                        # New concept was born! Boost it in current beam
                        cand_score += 0.5  # novelty bonus

                    merge_score = self._check_merge(seq, cid, new_seq)

                    new_branch_id = next_branch_id
                    next_branch_id += 1

                    new_beam.append((
                        new_seq, new_score + merge_score,
                        new_tokens, new_branch_id
                    ))

            # ---- Prune beam ----
            new_beam.sort(key=lambda x: -x[1])
            beam = new_beam[:max(1, h_beam)]
            all_chains.extend([(s, sc) for s, sc, _, _ in new_beam])

            # EOS
            for seq, score, tokens, bid in beam:
                if wn >= self.min_words:
                    last_w = self._word_of_concept(seq[-1])
                    if last_w in '.!?\u2026':
                        finished.append((seq, score, tokens, wn))
                        beam = [b for b in beam if b[0] != seq]

            if not beam:
                break

        # Return best
        if finished:
            best_seq, best_score, best_tokens, wn = max(finished, key=lambda x: x[1])
        elif beam:
            best_seq, best_score, best_tokens, _ = beam[0]
        else:
            return {'text': '', 'chains': all_chains}

        best_tokens.append(self.tok.SENT_CLOSE)
        best_tokens.append(self.tok.EOS)
        text = self.tok.decode(best_tokens)

        return {
            'text': text,
            'concept_path': best_seq,
            'score': best_score,
            'word_count': len(best_seq) - 1,
            'max_words': effective_max,
            'chains': all_chains,
            'core_cid': core_cid,
            'modifier_field': modifier_field,
            'noise_words': noise_words,
            'intent_drift': self._intent_drift,
        }

    def _branch(self, seq, word_num, theta_temp=0.3, target_cid=None, intent_vec=None):
        """Generate diverse branching candidates via field exploration.

        Uses Reciprocal Rank Fusion (RRF) combined with core aspect
        decomposition to generate candidates within the core's field.

        Components:
        - Core aspects (decomposed sub-topics of core concept)
        - Core modifier field (query-specific modifiers)
        - Core connection strength (any concept connected to core)
        - Syntax lattice (n-gram) ~ syntactic structure
        - Vector similarity (semantic) ~ semantic proximity

        Gate verification filters candidates not connected to core.

        Returns:
            [(cid, log_score), ...]
        """
        cids = seq[-3:] if len(seq) >= 3 else seq
        prev_cid = seq[-1]
        K = 3  # RRF constant
        core_cid = self._core_cid if hasattr(self, '_core_cid') else None

        # POS context
        from eva.symbolic.pos_tagger import get_pos, pos_transition_score
        prev_anchor = self.cs.concept_info.get(prev_cid, {}).get('anchor', '')
        prev_pos = get_pos(prev_anchor) if prev_anchor else 'UNK'

        # 1. Core aspects (decomposed field — learned from training data)
        aspect_cids = {}
        if core_cid is not None and self._core_aspects:
            for asp in self._core_aspects:
                aspect_cids[asp['cid']] = asp['strength']

        # 2. Connection graph: what's connected to PREV concept (learned co-occurrence)
        connected_prev = {}
        prev_connections = self.lattice.connections_of(prev_cid, top_k=15)
        for cid, conn in prev_connections:
            connected_prev[cid] = conn['strength']

        # 3. N-gram syntax: what follows in real corpus
        syn_preds = self.lattice.predict(cids)
        syn_ranked = {cid: i+1 for i, (cid, _) in enumerate(syn_preds[:30])}

        # 4. All candidate cids (learned > query > novelty)
        modifier_cids = set(self._modifier_field.keys())
        all_cids = (set(aspect_cids.keys()) | set(connected_prev.keys()) |
                    set(syn_ranked.keys()) | modifier_cids)

        if not all_cids:
            return []

        # 5. RRF scoring — learned patterns dominate
        combined = {}
        for cid in all_cids:
            rrf = 0.0

            # Learned: connection from previous concept (real co-occurrence)
            if cid in connected_prev:
                rrf += 0.5 * connected_prev[cid] / (K + 1)

            # Learned: n-gram sequence (real corpus patterns)
            if cid in syn_ranked:
                rrf += 0.4 / (K + syn_ranked[cid])

            # Learned: core aspect (decomposed from connections + ngrams)
            if cid in aspect_cids:
                rrf += 0.3 * aspect_cids[cid] / (K + 1)

            # Query-specific: modifier field from gate
            if cid in modifier_cids:
                mod_strength = self._modifier_field[cid].get('strength', 0.5)
                rrf += 0.2 * mod_strength / (K + 1)

            # Novelty prior: rare concepts get tiny boost
            freq = self.lattice.concept_freq.get(cid, 0)
            prior = 0.02 / (K + 1) * (1.0 - min(freq / 1000, 1.0))
            rrf += prior

            combined[cid] = rrf

        # 6. Homeostatic boost + content word bonus
        for cid in list(combined.keys()):
            h_boost = self.cs.homeostatic_boost(cid)
            combined[cid] *= (1.0 + h_boost * 0.3)

            # Content word bonus: prefer concepts with meaningful anchors
            info = self.cs.concept_info.get(cid, {})
            anchor = info.get('anchor', '')
            sat_count = len(self.cs.cid_to_words.get(cid, []))
            if len(anchor) >= 3 and anchor not in (
                'и', 'в', 'на', 'с', 'к', 'у', 'о', 'от', 'из', 'по',
                'за', 'для', 'до', 'без', 'через', 'при', 'об', 'про',
                'а', 'но', 'да', 'или', 'что', 'как', 'не', 'ни',
                'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
                'тот', 'этот', 'весь', 'сам', 'свой', 'каждый', 'другой',
                'быть', 'мочь', 'хотеть', 'стать',
            ):
                # Bonus proportional to morphological richness
                combined[cid] *= (1.0 + 0.15 * min(sat_count / 3, 1.0))
            else:
                # Small penalty for function words
                combined[cid] *= 0.85

            # POS transition score: prefer syntactically compatible sequences
            cand_anchor = info.get('anchor', '')
            cand_pos = get_pos(cand_anchor) if cand_anchor else 'UNK'
            pos_score = pos_transition_score(prev_pos, cand_pos)
            combined[cid] *= (1.0 + pos_score * 2.0)  # up to 3x boost for good transitions

            # Core connection penalty: concepts unrelated to core get penalized
            if core_cid is not None and cid != core_cid and cid not in modifier_cids:
                conn = self.lattice.connection_strength(core_cid, cid, self.cs)
                if conn < 0.05:
                    combined[cid] *= 0.5  # strong penalty for drifting

        # 7. Avoid repetition and loops
        recent = seq[-6:] if len(seq) >= 6 else seq
        for cid in list(combined.keys()):
            # Exact concept repetition in recent window
            if cid in recent:
                count = recent.count(cid)
                combined[cid] *= (0.05 ** count)

            # Bigram loop detection: A→B→A→B pattern
            if len(seq) >= 4 and cid == seq[-2] and seq[-1] == seq[-3]:
                combined[cid] *= 0.01  # strong loop penalty

        # 8. Apply temperature (softmax with theta_temp)
        result = [(cid, max(s, 1e-10)) for cid, s in combined.items()]
        result.sort(key=lambda x: -x[1])

        scores = np.array([s for _, s in result], dtype=np.float64)
        scores = scores - scores.max()
        scores = np.clip(scores, -50, 50)
        temp = max(theta_temp, 0.01)
        probs = np.exp(scores / temp)
        probs /= probs.sum()

        # ---- Target boosting (training mode) ----
        if target_cid is not None and target_cid in self.cs.concept_vectors:
            for i, (cid, _) in enumerate(result):
                if cid == target_cid:
                    boost = 5.0 * (1.0 - theta_temp * 0.5)
                    probs[i] *= boost
                    break
            probs /= probs.sum()

        n_candidates = min(15 + int(15 * theta_temp), len(result))
        scored = [(result[i][0], math.log(probs[i] + 1e-10))
                  for i in range(n_candidates)]
        return scored

    def _check_merge(self, context_seq, next_cid, new_seq):
        """Check if this chain converges with another known pattern.
        Uses precomputed merge potential for efficiency.
        """
        # Get precomputed merge score
        if self._merge_cache is None:
            self._precompute_merge_potential()
        return self._merge_cache.get(next_cid, 0.0)

    def _precompute_merge_potential(self):
        """Precompute merge potential for all concepts.
        Merge potential = how many different n-gram prefixes lead to this concept.
        High merge potential = syntactic/semantic hub."""
        print("  Precomputing merge potential...")
        prefix_counts = Counter()
        for n in [2, 3]:
            ngrams = self.lattice.ngrams.get(n, {})
            seen = set()
            for prefix, counter in ngrams.items():
                for cid in counter:
                    key = (n, cid, prefix)
                    if key not in seen:
                        seen.add(key)
                        prefix_counts[cid] += 1

        n_concepts = len(prefix_counts)
        if n_concepts > 0:
            max_count = max(prefix_counts.values())
        else:
            max_count = 1

        # Normalize to [0, 0.3]
        self._merge_cache = {
            cid: min(c / max_count * 0.3, 0.3)
            for cid, c in prefix_counts.items()
        }
        print(f"    {len(self._merge_cache)} concepts with merge potential")

    def _select_word(self, cid, theta_temp=0.2, prev_word=None):
        """Select word from concept's electron cloud.
        Anchor = nucleus (preferred, stable).
        Satellites = electron cloud (morphological variety, volatile).

        Args:
            cid: concept ID
            theta_temp: temperature for word selection
            prev_word: previous word (for agreement checking)
        """
        from eva.symbolic.pos_tagger import get_pos, check_agreement

        words = self.cs.words_in_concept(cid, top_k=15)
        if not words:
            return None

        info = self.cs.concept_info.get(cid, {})
        anchor = info.get('anchor', words[0])

        if theta_temp <= 0.01:
            return anchor

        # Filter words by agreement with prev_word
        if prev_word:
            compatible = [w for w in words if check_agreement(prev_word, w)]
            if compatible:
                words = compatible

        scored = []
        for w in words:
            is_anchor = (w == anchor)
            base_score = 3.0 if is_anchor else 1.0
            base_score *= math.exp(-0.03 * len(w))

            # Bonus for agreement with prev_word
            if prev_word and check_agreement(prev_word, w):
                base_score *= 1.2

            scored.append((w, base_score))

        scores = np.array([s for _, s in scored], dtype=np.float64)
        scores = scores - scores.max()
        scores = np.clip(scores, -50, 50)
        probs = np.exp(scores / max(theta_temp, 0.01))
        probs /= probs.sum()

        rng = self.main_rng
        idx = rng.choices(range(len(scored)), weights=probs)[0]
        return scored[idx][0]

    def _word_of_concept(self, cid):
        info = self.cs.concept_info.get(cid, {})
        return info.get('anchor', '')

    def _resolve_seed(self, seed_word, seed_cid):
        if seed_cid is not None:
            return seed_cid
        if seed_word:
            cid = self.cs.word_to_cid.get(seed_word.lower())
            if cid is not None:
                return cid
        if self.cs.cid_list:
            return self.main_rng.choice(self.cs.cid_list)
        return None

    def _target_concepts(self, target_text):
        if not target_text:
            return []
        ids = self.tok.encode(target_text)
        meta = self.tok.metadata_from_ids(ids)
        return [m.get('concept_id') for m in meta
                if m['flags'] & 1 and m.get('concept_id') is not None]

    # ---- External Training API ----

    def train_from_text(self, text):
        """Train model from external text (decode → metadata → organize connections).

        Training principle:
        1. Decode text to metadata: extract core concepts, modifiers, connections
        2. Organize semantic connections: update lattice, role_memory, concept space
        3. Accumulate structure: every text enriches the model's semantic map

        This is NOT gradient descent. It's structure extraction and accumulation.

        Args:
            text: input text (any Russian text)
        """
        # Clean text: normalize whitespace, replace bad chars
        text = self._clean_text(text)

        sentences = self._split_into_sentences(text)

        for sentence in sentences:
            raw_words = sentence.split()
            if len(raw_words) < 2:
                continue

            # Clean punctuation from words before any processing
            strip_chars = '.,!?;:()[]{}«»—–-…\'\"\u00a0\ufffd'
            words = []
            for w in raw_words:
                clean = w.strip(strip_chars)
                if clean:
                    words.append(clean)
            if len(words) < 2:
                continue

            # 1. Extract core concept via gate
            core_cid, modifier_field, centroid, noise = self.gate.extract_core(words)
            if core_cid is None:
                continue

            # 2. Update role memory
            mod_cids = set(modifier_field.keys())
            self.gate.update_role_memory(words, core_cid, mod_cids)

            # 3. Build connection graph from sentence structure
            from eva.symbolic.pos_tagger import get_pos
            word_data = []
            for w in words:
                clean = w.strip(strip_chars)
                if not clean:
                    continue
                morph = ConceptTokenizer.morph_parse(clean)
                root = morph['normal_form'] if morph else clean
                cid = self.cs.word_to_cid.get(root) or self.cs.word_to_cid.get(clean)
                pos = morph['pos'] if morph else get_pos(clean)
                word_data.append({
                    'word': clean, 'cid': cid, 'pos': pos, 'root': root,
                })

            # Core→core connections (NOUN→VERB, etc.)
            cores = [d for d in word_data if d['pos'] in ('NOUN', 'VERB') and d['cid'] is not None]
            for i in range(len(cores)):
                for j in range(i + 1, len(cores)):
                    idx_a = word_data.index(cores[i])
                    idx_b = word_data.index(cores[j])
                    env_words = [word_data[k]['word'] for k in range(idx_a + 1, idx_b)]
                    relation = self.lattice.infer_relation(env_words)
                    self.lattice.add_connection(cores[i]['cid'], cores[j]['cid'], relation)

            # 4. Update lattice n-grams
            concept_seq = [d['cid'] for d in word_data if d['cid'] is not None]
            if len(concept_seq) >= 2:
                self.lattice.update(concept_seq)

            # 5. STDP on core→modifier transitions
            for d in word_data:
                if d['cid'] is not None and d['cid'] != core_cid:
                    self.cs.svd_shift(core_cid, d['cid'],
                                      expected_cid=core_cid, lr=0.05)

        return len(sentences)

    @staticmethod
    def _clean_text(text):
        """Normalize text: standardize whitespace, remove encoding artifacts."""
        import re
        # Replace non-breaking spaces, em/en dashes
        text = text.replace('\u00a0', ' ')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2014', ' - ')
        # Remove replacement characters (encoding artifacts)
        text = text.replace('\ufffd', '')
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _split_into_sentences(text):
        import re
        sents = re.split(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z])', text.strip())
        return [s.strip() for s in sents if s.strip()]

    # ---- Training ----

    def train_sentence(self, target_text, seed_word=None, epochs=12):
        """Train with beam search + temperature annealing + hormonal drive."""
        # Reset
        self.cs.init_homeostasis()
        self._merge_cache = None
        self.hormones = HormonalSystem()

        saved_max_words = self.max_words
        self.max_words = min(self.max_words, 8)
        saved_beam_width = self.beam_width
        self.beam_width = max(self.beam_width, 5)

        target_c = self._target_concepts(target_text)
        n_target = len(target_c)

        for epoch in range(epochs):
            e_factor = 1.0 - epoch / (epochs + 1)
            self.base_concept_temp = 0.6 * e_factor
            self.base_word_temp = 0.3 * e_factor
            self.branch_rngs.clear()
            self.main_rng = random.Random(epoch * 42)

            result = self.generate(seed_word=seed_word, target_text=target_text)

            # Update gate role memory from this generation
            query_words = (seed_word or '').split() + target_text.split()
            core_cid = self._core_cid
            mod_cids = set(self._modifier_field.keys())
            if core_cid is not None and query_words:
                self.gate.update_role_memory(query_words, core_cid, mod_cids)

            gen_c = result.get('concept_path', [])
            matches = sum(1 for i, c in enumerate(gen_c)
                         if i < len(target_c) and c == target_c[i])
            total = min(len(gen_c), n_target)

            pct = 100 * matches / max(total, 1)
            reward = self.hormones.dopamine
            print(f"  ep {epoch}: temp={self.base_concept_temp:.3f} "
                  f"match={matches}/{total} ({pct:.1f}%) "
                  f"DA={reward:.2f} "
                  f"text={result['text'][:60]}")

            if matches == n_target:
                print(f"  -> Target matched @ epoch {epoch}")
                break

        self.max_words = saved_max_words
        self.beam_width = saved_beam_width
        self.cs.save(self.config.get('cs_path',
            r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space.json'))


if __name__ == '__main__':
    import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
    from eva.symbolic.concept_space import ConceptSpace
    from eva.symbolic.concept_tokenizer import ConceptTokenizer
    from eva.symbolic.syntax_lattice import SyntaxLattice

    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("Loading...")
    tok = ConceptTokenizer()
    tok.initialize()
    cs = ConceptSpace(None, dim=128)
    cs.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space.json')
    lattice = SyntaxLattice()
    lattice.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\syntax_lattice.json')

    gen = CrystalGenerator(cs, tok, lattice)

    print("\n--- Before training ---")
    for seed in ['князь', 'человек', 'война']:
        result = gen.generate(seed_word=seed)
        print(f"[{seed}] {result['text'][:80]}")
        print(f"  DA={gen.hormones.dopamine:.2f} ACh={gen.hormones.acetylcholine:.2f}")

    # Multi-sentence training
    training_data = [
        ("Князь Андрей вышел на крыльцо.", "князь"),
        ("Человек должен быть свободен.", "человек"),
        ("Война началась неожиданно для всех.", "война"),
        ("Старый князь сидел в кресле.", "князь"),
        ("Наташа любила танцевать на балах.", "Наташа"),
        ("Пьер смотрел на звёздное небо.", "Пьер"),
    ]

    print("\n--- Multi-sentence training (hormonal self-improvement) ---")
    stats = {'total_matches': 0, 'total_concepts': 0, 'da_trend': []}

    for epoch in range(5):
        epoch_matches = 0
        epoch_total = 0
        for sent, seed in training_data:
            gen.train_sentence(sent, seed_word=seed, epochs=1)

            target_c = gen._target_concepts(sent)
            gen_c = gen.generate(seed_word=seed)['concept_path']
            m = sum(1 for i, c in enumerate(gen_c)
                    if i < len(target_c) and c == target_c[i])
            epoch_matches += m
            epoch_total += min(len(gen_c), len(target_c))

        pct = 100 * epoch_matches / max(epoch_total, 1)
        stats['da_trend'].append(gen.hormones.dopamine)
        print(f"  epoch {epoch}: match={epoch_matches}/{epoch_total} "
              f"({pct:.1f}%) DA={gen.hormones.dopamine:.2f} "
              f"5HT={gen.hormones.serotonin:.2f} NA={gen.hormones.noradrenaline:.2f}")

    print(f"\n  DA trend: {[f'{d:.2f}' for d in stats['da_trend']]}")
    da_delta = stats['da_trend'][-1] - stats['da_trend'][0]
    print(f"  DA delta: {da_delta:+.2f} "
          f"{'(self-improving)' if da_delta > 0 else '(degrading)'}")

    print("\n--- After training ---")
    for seed in ['князь', 'человек', 'война']:
        result = gen.generate(seed_word=seed)
        print(f"[{seed}] {result['text'][:80]}")
        print(f"  DA={gen.hormones.dopamine:.2f}")

    # Save trained state
    cs.save(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space_trained.json')
    lattice.save(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\syntax_lattice.json')
    print("\nTrained state saved.")
