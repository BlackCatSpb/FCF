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
        self.beam_width = self.config.get('beam_width', 3)
        self.max_words = self.config.get('max_words', 20)
        self.min_words = self.config.get('min_words', 3)

        # Base temperature (theta-rhythm + hormones modulate these)
        self.base_concept_temp = self.config.get('concept_temp', 0.5)
        self.base_word_temp = self.config.get('word_temp', 0.3)
        self.theta_tau = self.config.get('theta_tau', 5.0)

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

        # --- Concept Induction (semantic resonance) ---
        self.inductor = ConceptInductor(self.config)

        # --- Intent & Distance tracking ---
        self._query_anchors = []    # concept IDs from query
        self._query_centroid = None # mean vector of query concepts
        self._response_centroid = None # mean vector of response so far

    def _theta_temp(self, word_num):
        """Theta-rhythm temperature: high early (explore), low late (exploit)."""
        return self.base_concept_temp * math.exp(-word_num / self.theta_tau)

    def _get_branch_rng(self, branch_id):
        """Get a seeded RNG for a specific beam branch."""
        if branch_id not in self.branch_rngs:
            self.branch_rngs[branch_id] = random.Random(branch_id * 137)
        return self.branch_rngs[branch_id]

    # ── Anchor resolution (no fallbacks) ──────────────────────────

    def resolve_anchor(self, word):
        """Resolve ANY word to a concept anchor. Never returns None.

        Strategy (cascading):
        1. Direct lookup: word -> cid
        2. Orthographic: closest known word by edit distance
        3. Vector projection: encode via BPE, find closest concept vector
        4. Semantic centroid: if all fails, return most connected concept
        """
        w = word.lower().strip()

        # 1. Direct lookup
        cid = self.cs.word_to_cid.get(w)
        if cid is not None:
            return cid

        # 2. Orthographic (Dice bigram similarity, threshold > 0.4)
        best_cid, best_score = None, 0.0
        for known_w, known_cid in self.cs.word_to_cid.items():
            if abs(len(known_w) - len(w)) > 3:
                continue
            score = self._edit_similarity(w, known_w)
            if score > best_score and score > 0.4:
                best_score, best_cid = score, known_cid

        if best_cid is not None:
            return best_cid

        # 3. BPE token overlap (Jaccard similarity, not raw count).
        #    Only consider matches with >10% Jaccard overlap.
        #    Weighted centroid of overlapping words -> closest concept.
        bpe_ids = set(self.tok.bpe.encode(w).ids) if self.tok.bpe.encode(w).ids else set()
        if bpe_ids:
            candidate_vectors = []
            # Speed: limit scan to first 8000 known words (most frequent)
            # Full scan is too slow for 37k words on each unknown input
            scan_limit = min(8000, len(self.cs.word_to_cid))
            for known_w, known_cid in list(self.cs.word_to_cid.items())[:scan_limit]:
                known_ids = set(self.tok.bpe.encode(known_w).ids)
                if not known_ids:
                    continue
                jaccard = len(bpe_ids & known_ids) / len(bpe_ids | known_ids)
                if jaccard > 0.1:  # at least 10% token set overlap
                    v = self.cs.concept_vector(known_cid)
                    if v is not None:
                        candidate_vectors.append((known_cid, v, jaccard))

            if candidate_vectors:
                # Weight by Jaccard similarity
                total_w = sum(o for _, _, o in candidate_vectors)
                weights = np.array([o / total_w for _, _, o in candidate_vectors], dtype=np.float64)
                avg_v = np.average(
                    [v for _, v, _ in candidate_vectors],
                    axis=0, weights=weights
                )
                # Find closest concept to this averaged vector
                best_cid = self._closest_concept(avg_v, k=1)[0][0]
                return best_cid

        # 4. Ultimate anchor: most connected concept in the lattice
        if self.cs.cid_list:
            # Find concept with highest merge potential (most connected)
            if self._merge_cache is None:
                self._precompute_merge_potential()
            best_cid = max(self._merge_cache, key=self._merge_cache.get)
            return best_cid

        # 5. Absolute last resort: first concept
        return self.cs.cid_list[0] if self.cs.cid_list else 0

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
        """Find k closest concepts to a given vector."""
        if vec is None:
            return [(0, 0.0)]
        vn = vec / max(np.linalg.norm(vec), 1e-10)
        exclude_set = set(exclude or [])
        sims = []
        for cid, cv in self.cs.concept_vectors.items():
            if cid in exclude_set:
                continue
            s = float(np.dot(vn, cv / max(np.linalg.norm(cv), 1e-10)))
            sims.append((cid, s))
        sims.sort(key=lambda x: -x[1])
        return sims[:k]

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
        for w in query_words:
            cid = self.resolve_anchor(w)
            v = self.cs.concept_vector(cid)
            if v is not None:
                anchors.append(cid)
                vectors.append(v)

        self._query_anchors = anchors

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

        # Intent anchor = concept closest to centroid, but NOT an exact query match
        # (model should GENERATE from the intent, not repeat the query)
        intent_cid = self._closest_concept(intent_vec, k=1, exclude=anchors)[0][0]

        return intent_cid, intent_vec, delta

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
        # ── Resolve intent from query ──
        src_words = query_words or ([seed_word] if seed_word else [])
        intent_cid, intent_vec, intent_delta = self.project_intent(src_words)
        self._query_centroid = intent_vec

        # Use intent as seed if no explicit seed given
        if seed_cid is None and seed_word is None:
            seed_cid = intent_cid
        elif seed_cid is None and seed_word is not None:
            seed_cid = self.resolve_anchor(seed_word)

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
            delta = self._semantic_delta(intent_vec, [b[0][-1] for b in beam])
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
                candidates = self._branch(seq, wn, h_temp, expected_cid, intent_vec)
                if not candidates:
                    self.hormones.update(confidence=0.0, is_match=False,
                        novelty=0.0, surprise=0.5, expected_cid=expected_cid)
                    finished.append((seq, score, tokens, wn))
                    continue

                for ci, (cid, cand_score) in enumerate(candidates):
                    new_seq = seq + [cid]
                    # Distance penalty: penalize candidates that increase drift
                    dist_penalty = drift_penalty * (1.0 if ci == 0 else 1.0 + 0.1 * ci)
                    new_score = score + cand_score - dist_penalty

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
            top_beams = new_beam[:max(1, h_beam)]

            if len(new_beam) > h_beam:
                runner_ups = new_beam[h_beam:]
                pick = self.main_rng.randint(0, min(3, len(runner_ups) - 1))
                top_beams.append(runner_ups[pick])

            beam = top_beams[:max(1, h_beam)]
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
            'intent_cid': intent_cid,
            'intent_delta': intent_delta,
            'intent_drift': self._intent_drift,
        }

    def _branch(self, seq, word_num, theta_temp=0.3, target_cid=None, intent_vec=None):
        """Generate diverse branching candidates.

        Uses Reciprocal Rank Fusion (RRF) to combine:
        - Syntax lattice (n-gram) ~ syntactic crystal field
        - Vector similarity (semantic) ~ semantic electron field
        - Intent relevance (distance from query) ~ what was MEANT
        - Content word bias ~ imagery potential
        - Target boosting (training) ~ supervised alignment

        The intent relevance ensures the model always considers
        whether a candidate moves TOWARD or AWAY from what the
        query meant — semantic proximity AND distance awareness.

        Returns:
            [(cid, log_score), ...]
        """
        cids = seq[-3:] if len(seq) >= 3 else seq
        prev_cid = seq[-1]
        K = 60  # RRF constant (dampens top-rank advantage)

        # 1. N-gram syntax candidates with ranks
        syn_preds = self.lattice.predict(cids)
        syn_ranked = {cid: i+1 for i, (cid, _) in enumerate(syn_preds[:30])}

        # 2. Vector similarity candidates (semantic)
        v_prev = self.cs.concept_vector(prev_cid)
        vec_candidates = []
        if v_prev is not None:
            similar = self.cs.topk_similar_concepts(prev_cid, k=20)
            vec_candidates = [(cid, sim) for cid, anchor, sim in similar]
        vec_ranked = {cid: i+1 for i, (cid, _) in enumerate(vec_candidates[:20])}

        # 3. All candidate cids (union of syntax + vector + corpus)
        all_cids = set(syn_ranked.keys()) | set(vec_ranked.keys())

        if not all_cids:
            return []

        # 4. RRF scoring
        combined = {}
        for cid in all_cids:
            rrf = 0.0

            # Syntax component: 1/(K + rank_in_ngrams)
            if cid in syn_ranked:
                rrf += 0.5 / (K + syn_ranked[cid])

            # Vector component: 1/(K + rank_in_vectors)
            if cid in vec_ranked:
                rrf += 0.3 / (K + vec_ranked[cid])

            # Novelty prior: rare concepts get a small boost
            freq = self.lattice.concept_freq.get(cid, 0)
            prior = 0.1 / (K + 1) * (1.0 - min(freq / 1000, 1.0))
            rrf += prior

            combined[cid] = rrf

        # 5. Homeostatic boost + content word bonus
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

        # 6. Intent relevance: how close is each candidate to what was MEANT?
        #    Concepts that are semantically relevant to the query get a boost.
        #    Concepts too far from the intent get penalized.
        #    This implements "what did the interlocutor mean?"
        if intent_vec is not None:
            iv_n = intent_vec / max(np.linalg.norm(intent_vec), 1e-10)
            for cid in list(combined.keys()):
                cv = self.cs.concept_vector(cid)
                if cv is not None:
                    cv_n = cv / max(np.linalg.norm(cv), 1e-10)
                    rel = float(np.dot(iv_n, cv_n))  # cosine similarity to intent
                    # Boost near intent (rel > 0.3), penalize far (rel < 0)
                    if rel > 0.3:
                        combined[cid] *= (1.0 + 0.2 * rel)
                    elif rel < 0.0:
                        combined[cid] *= max(0.5, 1.0 + rel * 0.5)

        # 7. Avoid immediate repetition
        if len(seq) > 1:
            for cid in list(combined.keys()):
                if cid == seq[-1]:
                    combined[cid] *= 0.05
                elif len(seq) > 2 and cid == seq[-2]:
                    combined[cid] *= 0.2

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

    def _select_word(self, cid, theta_temp=0.2):
        """Select word from concept's electron cloud.
        Anchor = nucleus (preferred, stable).
        Satellites = electron cloud (morphological variety, volatile)."""
        words = self.cs.words_in_concept(cid, top_k=15)
        if not words:
            return None

        info = self.cs.concept_info.get(cid, {})
        anchor = info.get('anchor', words[0])

        if theta_temp <= 0.01:
            return anchor

        scored = []
        for w in words:
            is_anchor = (w == anchor)
            base_score = 3.0 if is_anchor else 1.0
            base_score *= math.exp(-0.03 * len(w))
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
