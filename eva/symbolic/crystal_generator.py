"""CrystalGenerator — BPE-token concept navigation generator.

Generation as semantic navigation through a concept space where
each BPE token IS a concept. Input text is tokenized via SentencePiece,
producing a sequence of concept IDs (0..vocab_size-1). Generation
picks the next token ID via STDP-guided beam search over the fractal
concept field. Output is decoded back to text via SentencePiece.

Key simplifications vs old architecture:
  - No word->CID resolution: BPE tokens ARE concepts
  - No word-form selection: each CID has exactly one BPE token text
  - No special token stream: raw token IDs with SentencePiece BOS/EOS
"""

import math, random
import numpy as np
from collections import Counter

from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.hormonal_system import HormonalSystem


_BOS_ID = 1
_EOS_ID = 2


class CrystalGenerator:
    """Generation as semantic navigation through BPE-token concept space."""

    def __init__(self, cs, sp, lattice, config=None):
        self.cs = cs
        self.sp = sp
        self.lattice = lattice
        self.config = config or {}

        self.beam_width = self.config.get('beam_width', 5)
        self.max_words = self.config.get('max_words', 30)
        self.min_words = self.config.get('min_words', 3)
        self._graph_cache = {}
        self.base_concept_temp = self.config.get('concept_temp', 0.5)
        self.theta_tau = self.config.get('theta_tau', 12.0)
        self.base_learning_rate = self.config.get('learning_rate', 0.1)

        self.main_rng = random.Random(42)
        self.cs.init_homeostasis()
        self.branch_rngs = {}
        self.hormones = HormonalSystem()

    # ── Temperature ────────────────────────────────────────────

    def _theta_temp(self, word_num):
        t = self.base_concept_temp * math.exp(-word_num / self.theta_tau)
        return max(t, self.base_concept_temp * 0.15)

    # ── Semantic distance ──────────────────────────────────────

    def _semantic_delta(self, query_vec, response_path, window=5):
        if query_vec is None or not response_path:
            return 0.5
        recent = response_path[-window:]
        recent_vecs = [self.cs.concept_vector(c)
                       for c in recent if self.cs.concept_vector(c) is not None]
        if not recent_vecs:
            return 0.5
        response_vec = np.mean(recent_vecs, axis=0).astype(np.float32)
        response_vec /= max(np.linalg.norm(response_vec), 1e-10)
        qn = query_vec / max(np.linalg.norm(query_vec), 1e-10)
        return 1.0 - float(np.dot(qn, response_vec))

    # ── Encode / Decode ────────────────────────────────────────

    def _encode_input(self, text):
        return self.sp.encode(text)

    def _decode_tokens(self, token_ids):
        return self.sp.decode(token_ids)

    def _token_text(self, cid):
        return self.sp.IdToPiece(cid)

    def _is_semantic_token(self, cid):
        """Filter function words and punctuation that dominate graph connections."""
        text = self._token_text(cid).strip()
        if not text:
            return False
        # Punctuation and single non-letter characters
        if len(text) == 1 and not ('а' <= text.lower() <= 'я' or text.isalpha()):
            return False
        # Pure punctuation tokens
        if all(c in '.,!?;:()[]{}""''…—–«»' for c in text):
            return False
        return True

    # ── Generation ─────────────────────────────────────────────

    def generate(self, seed_word=None, seed_cid=None, target_text=None,
                 query_words=None, max_words=None, beam_width=3):
        """Generate a token sequence via beam search over concept IDs.

        Args:
            seed_word: starting word -> tokenized to seed CID
            seed_cid: starting concept ID (overrides seed_word)
            target_text: target for supervised training
            query_words: list of words from the query

        Returns:
            dict with response text, concept path, score, etc.
        """
        # Encode target if provided
        target_ids = self._encode_input(target_text) if target_text else []

        # Determine seed CID
        if seed_cid is None:
            if seed_word:
                token_ids = self._encode_input(seed_word)
                seed_cid = token_ids[0] if token_ids else _BOS_ID
            else:
                seed_cid = _BOS_ID

        # Encode query words to centroid vector
        src_ids = self._encode_input(' '.join(query_words)) if query_words else [seed_cid]
        query_vecs = [self.cs.concept_vector(cid)
                      for cid in src_ids if self.cs.concept_vector(cid) is not None]
        self._centroid = np.mean(query_vecs, axis=0).astype(np.float32) if query_vecs else None
        if self._centroid is not None:
            n = np.linalg.norm(self._centroid)
            if n > 1e-10:
                self._centroid /= n

        effective_max = max_words or self.max_words

        # Beam: list of (concept_sequence, score, branch_id)
        beam = [([seed_cid], 0.0, 0)]
        all_chains = []
        finished = []
        next_branch_id = 1

        for wn in range(effective_max):
            new_beam = []

            theta_temp = self._theta_temp(wn)
            h_temp = self.hormones.modulate_temperature(theta_temp)
            h_lr = self.hormones.modulate_stdp_lr(self.base_learning_rate)
            effective_beam = max(1, beam_width)

            for seq, score, branch_id in beam:
                prev_cid = seq[-1]
                expected_cid = target_ids[wn] if wn < len(target_ids) else None

                candidates = self._branch(seq, wn, h_temp, expected_cid, self._centroid)
                if not candidates:
                    self.hormones.update(confidence=0.0, is_match=False,
                        novelty=0.0, surprise=0.5, expected_cid=expected_cid)
                    finished.append((seq, score, wn))
                    continue

                for ci, (cid, cand_score) in enumerate(candidates):
                    new_seq = seq + [cid]
                    new_score = score + cand_score

                    # Anti-repetition
                    recent = seq[-6:] if len(seq) >= 6 else seq
                    count = recent.count(cid)
                    if count > 0:
                        new_score += -0.3 * count

                    conf = 1.0 / (1.0 + ci * 0.5)
                    is_match = (expected_cid is not None and cid == expected_cid)

                    self.cs.update_usage(cid)

                    novelty = 1.0 - min(self.lattice.concept_freq.get(cid, 0) / 50, 1.0)
                    surprise = 0.1 if is_match else 0.5
                    self.hormones.update(confidence=conf, is_match=is_match,
                        novelty=novelty, surprise=surprise,
                        expected_cid=expected_cid, gen_cid=cid)

                    new_beam.append((new_seq, new_score, next_branch_id))
                    next_branch_id += 1

            new_beam.sort(key=lambda x: -x[1])
            beam = new_beam[:effective_beam]
            all_chains.extend([(s, sc) for s, sc, _ in new_beam])

            # EOS
            for item in list(beam):
                seq, score, bid = item
                if wn >= self.min_words:
                    token_text = self._token_text(seq[-1])
                    if token_text in ('.', '!', '?', '…', '...'):
                        finished.append((seq, score, wn))
                        beam.remove(item)

            if not beam:
                break

        if finished:
            best_seq, best_score, wn = max(finished, key=lambda x: x[1])
        elif beam:
            best_seq, best_score, _ = beam[0]
        else:
            return {'text': '', 'chains': all_chains}

        text = self._decode_tokens(best_seq)

        return {
            'text': text,
            'concept_path': best_seq,
            'score': best_score,
            'word_count': len(best_seq),
            'max_words': effective_max,
            'chains': all_chains,
        }

    # ── Graph-based semantic search ──────────────────────────────

    def _graph_search(self, sources, B=2.0, max_candidates=30, max_depth=5):
        """BMSSP-EVA: single multi-source BFS for semantic paths.

        Args:
            sources: seed concept IDs
            B: distance budget (path cost threshold)
            max_candidates: max results to return
            max_depth: max BFS steps (safety bound, B is the primary limiter)
        """
        if not sources:
            return {}
        sources = list(set(sources))
        # Keep only semantic sources (no punctuation / function words)
        sources = [s for s in sources if self._is_semantic_token(s)]
        if not sources:
            return {}

        d = {}
        visited = set()
        # Track which source(s) reached each node
        origins = {}
        frontier = []

        for s in sources:
            d[s] = 0.0
            visited.add(s)
            origins[s] = {sources.index(s)}
            frontier.append(s)

        step = 0
        while frontier and step < max_depth:
            step += 1
            next_frontier = []
            for u in frontier:
                conns = self.lattice.connections_of(u, top_k=8, use_ppmi=True)
                for v, conn_info in conns:
                    if not self._is_semantic_token(v):
                        continue
                    # Edge weight from PPMI: high PPMI = specific connection = low weight (short path)
                    ppmi = conn_info.get('ppmi', 0.0)
                    w = max(0.20, 1.0 - min(ppmi / 8.0, 1.0) * 0.7)
                    dv = d[u] + w
                    if dv >= B:
                        continue
                    if v not in visited:
                        d[v] = dv
                        visited.add(v)
                        origins[v] = origins.get(u, set())
                        next_frontier.append(v)
                    elif dv < d[v] - 0.01:
                        d[v] = dv
                        origins[v] |= origins.get(u, set())
            frontier = next_frontier

        for s in sources:
            d.pop(s, None)

        if not d:
            return {}

        # RRF: for each candidate, sum over unique sources reached
        total_src = len(sources)
        for cid, dist in d.items():
            n_src = len(origins.get(cid, {1}))
            rrf = (n_src / total_src) / (B + dist)
            d[cid] = rrf

        ranked = sorted(d.items(), key=lambda x: -x[1])
        return dict(ranked[:max_candidates])

    # ── Branch ─────────────────────────────────────────────────

    def _branch(self, seq, word_num, theta_temp=0.3, target_cid=None, centroid=None):
        """Generate diverse branching candidates via RRF over multiple signals."""
        prev_cid = seq[-1]
        cids = seq[-3:] if len(seq) >= 3 else seq
        K = 3

        # 1. Graph-based semantic paths (BMSSP-EVA, replaces single-hop connections)
        sources = list(set(cids))  # unique context tokens
        sources_key = tuple(sorted(set(sources)))
        if sources_key not in self._graph_cache:
            self._graph_cache[sources_key] = self._graph_search(sources, B=1.2, max_candidates=30)
        graph_candidates = self._graph_cache[sources_key]

        # 2. N-gram syntax (filter to semantic tokens only)
        syn_preds = self.lattice.predict(cids)
        syn_ranked = {cid: i + 1 for i, (cid, _) in enumerate(syn_preds[:80])
                      if self._is_semantic_token(cid)}

        # 3. All candidates from learned signals
        all_cids = set(graph_candidates.keys()) | set(syn_ranked.keys())

        # 4. Vector similarity fallback
        v_prev = self.cs.concept_vector(prev_cid)
        vector_sim = {}
        if v_prev is not None:
            sim_candidates = self.cs.topk_similar_concepts(prev_cid, k=20, sample_size=500)
            for cid, sim in sim_candidates:
                if cid not in all_cids and sim > 0.05:
                    all_cids.add(cid)
                vector_sim[cid] = sim

        if not all_cids:
            return []

        # 5. RRF scoring
        combined = {}
        for cid in all_cids:
            rrf = 0.0
            if cid in graph_candidates:
                rrf += 0.7 * graph_candidates[cid]
            if cid in syn_ranked:
                rrf += 0.15 / (K + syn_ranked[cid])
            if cid in vector_sim:
                rrf += 0.15 * vector_sim[cid] / (K + 1)
            freq = self.lattice.concept_freq.get(cid, 0)
            prior = 0.02 / (K + 1) * (1.0 - min(freq / 1000, 1.0))
            rrf += prior
            combined[cid] = rrf

        # 5. Homeostatic boost
        for cid in list(combined.keys()):
            h_boost = self.cs.homeostatic_boost(cid)
            combined[cid] *= (1.0 + h_boost * 0.3)

        # 6. Intent centroid bonus: prefer candidates near the query centroid
        if centroid is not None and np.linalg.norm(centroid) > 1e-10:
            cn = centroid / np.linalg.norm(centroid)
            for cid in list(combined.keys()):
                v = self.cs.concept_vector(cid)
                if v is not None:
                    sim_to_query = float(np.dot(v, cn))
                    # ideal: not too close (parroting), not too far (drift)
                    # bonus = 0.0 at sim=0, peaking at sim=0.5, then decays
                    intent_bonus = max(0, sim_to_query * (1.0 - sim_to_query)) * 0.3
                    combined[cid] *= (1.0 + intent_bonus)

        # 7. Anti-repetition
        recent = seq[-6:] if len(seq) >= 6 else seq
        trigram_set = set()
        if len(seq) >= 3:
            for i in range(len(seq) - 2):
                trigram_set.add((seq[i], seq[i + 1], seq[i + 2]))
        for cid in list(combined.keys()):
            count = recent.count(cid)
            if count > 0:
                combined[cid] *= math.exp(-0.3 * count)
            if len(seq) >= 2:
                candidate_trigram = (seq[-2], seq[-1], cid)
                if candidate_trigram in trigram_set:
                    combined.pop(cid, None)
            if len(seq) >= 4 and cid == seq[-2] and seq[-1] == seq[-3]:
                combined.pop(cid, None)

        if not combined:
            return []

        # 7. Temperature softmax
        result = [(cid, max(s, 1e-10)) for cid, s in combined.items()]
        result.sort(key=lambda x: -x[1])
        scores = np.array([s for _, s in result], dtype=np.float64)
        scores = scores - scores.max()
        scores = np.clip(scores, -50, 50)
        temp = max(theta_temp, 0.01)
        probs = np.exp(scores / temp)
        probs /= probs.sum()

        # Target boosting
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

    # ── PMI-gated STDP ─────────────────────────────────────────

    def _pmi_weight(self, prev_cid, next_cid, distance=1, total_freq=None, min_weight=0.05):
        """Pointwise Mutual Information weight for STDP pull strength.

        PMI = log(P(next|prev) / P(next))
        High PMI = specific, statistically surprising pair (e.g. князь→великий)
        Low PMI  = generic transition (e.g. а→также→в→качестве)
        Negative PMI = they avoid each other

        Uses adjacent ngrams for |i-j|=1, skip2 dict for |i-j|=2.

        Maps to [min_weight, 2.0] multiplier on learning rate.

        Args:
            total_freq: cached sum(concept_freq.values()), computed once per line
            min_weight: floor for the PMI multiplier (tunable via pmi_gate_min)
        """
        if total_freq is None:
            total_freq = sum(self.lattice.concept_freq.values())
        if total_freq < 1:
            return 0.1

        if distance == 1:
            n2 = self.lattice.ngrams[2]
            prefix_counter = n2.get((prev_cid,))
            if not prefix_counter:
                return 0.1
            count_pair = prefix_counter.get(next_cid, 0)
            count_prev = sum(prefix_counter.values())
        elif distance == 2:
            skip2 = self.lattice.skip2.get(prev_cid)
            if not skip2:
                return 0.1
            count_pair = skip2.get(next_cid, 0)
            count_prev = sum(skip2.values())
        else:
            return 0.1

        count_next = self.lattice.concept_freq.get(next_cid, 0)
        if count_pair < 1 or count_prev < 1 or count_next < 1:
            return 0.1

        p_next_given_prev = count_pair / count_prev
        p_next = count_next / total_freq
        pmi = math.log(p_next_given_prev / max(p_next, 1e-10))

        # PMI=0 → 0.2, PMI=2 → 1.0, PMI=5 → 2.0, negative → min_weight
        return max(min(pmi / 2.0 + 0.2, 2.0), min_weight)

    # ── Training ───────────────────────────────────────────────

    def train_from_text(self, text, base_lr=None, context_window=2, pmi_gate=True, pmi_gate_min=0.05, neg_samples=0,
                        inh_strength=0.05, inh_threshold=0.35):
        """Train via PMI-gated context-window STDP, batched by unique gen_cid.

        Batch optimisation: groups all STDP updates for the same generator
        concept into a single combined gradient + code projection + lateral
        inhibition call. Preserves directional forward-only STDP and PMI
        gating; only the per-cid update batching changes from sequential
        (online) to summed (mini-batch) within each line — O(lr^2) difference.

        Each token pulls toward nearby tokens weighted by exponential
        distance decay AND pointwise mutual information. Generic
        transitions (low PMI) get minimal pull — only specific,
        statistically surprising co-occurrences shape the vectors.
        """
        ids = self._encode_input(text)
        if len(ids) < 2:
            return 0

        base_lr = base_lr if base_lr is not None else getattr(self, 'train_lr', 0.01)
        vocab_size = cs.vocab_size
        total_freq = sum(self.lattice.concept_freq.values())
        cs = self.cs
        T = len(ids)

        # ── Build pairs, group by gen_cid ──
        from collections import defaultdict
        gen_updates = defaultdict(list)  # gen_cid -> [(prev_cid, lr), ...]

        for i in range(T):
            start = max(0, i - context_window)
            end = min(T, i + context_window + 1)
            for j in range(start, end):
                if j <= i:
                    continue
                dist = abs(j - i)
                dist_weight = math.exp(-dist / 2.0)

                fa = self.lattice.concept_freq.get(ids[i], 0)
                fb = self.lattice.concept_freq.get(ids[j], 0)
                freq_weight = 1.0 / (1.0 + math.log(max(max(fa, fb), 1)) * 0.15)

                pmi_w = self._pmi_weight(ids[i], ids[j], distance=dist, total_freq=total_freq, min_weight=pmi_gate_min) if pmi_gate else 1.0

                lr = base_lr * max(freq_weight, 0.05) * dist_weight * pmi_w

                # Theta rhythm modulates by position (word_num=j)
                theta_gate = math.exp(-j / max(self.theta_tau, 1.0))
                gen_updates[ids[j]].append((ids[i], lr * max(theta_gate, 0.1)))

        # ── One combined STDP update per unique gen_cid ──
        for gen_cid, updates in gen_updates.items():
            v_gen = cs.concept_vectors.get(gen_cid)
            if v_gen is None:
                continue

            total_delta = np.zeros(cs.dim, dtype=np.float32)
            total_elr = 0.0  # sum of effective lr for lateral inhibition scaling
            for prev_cid, elr in updates:
                v_ctx = cs.concept_vectors.get(prev_cid)
                if v_ctx is None:
                    continue
                y = max(float(np.dot(v_gen, v_ctx)), 0.05)
                total_delta += (v_ctx - y * v_gen) * elr
                total_elr += elr

            v_new = v_gen + total_delta
            nv = np.linalg.norm(v_new)
            if nv > 1e-10:
                v_new /= nv

            cs._apply_vector_update(gen_cid, v_new)
            # Lateral inhibition once per gen_cid, with total cumulative
            # strength matching original per-pair calls: inh_strength * sum(elr)
            cs._lateral_inhibition_fractal(
                gen_cid,
                strength=inh_strength * total_elr,
                threshold=inh_threshold,
                sample_size=min(200 * min(len(updates), 5), len(cs.concept_vectors)),
            )
            cs.mark_matrix_dirty()

        # ── Negative sampling (fixed: push random token AWAY from context) ──
        # Original fractal_stdp had a bug: Step 1 pulled neg_cid toward ctx at 20x
        # the strength of Step 2's push-away — net pull at 95%.  Now does one
        # clean push-away via the negative Riemannian gradient.
        if neg_samples > 0:
            for gen_cid, updates in gen_updates.items():
                for prev_cid, elr in updates:
                    neg_elr = elr * 0.1
                    v_ctx = cs.concept_vectors.get(prev_cid)
                    if v_ctx is None:
                        continue
                    for _ in range(neg_samples):
                        neg_cid = cs.rng.randint(0, vocab_size)
                        v_neg = cs.concept_vectors.get(neg_cid)
                        if v_neg is None:
                            continue
                        # Push away from context via negative Riemannian gradient
                        y = max(float(np.dot(v_neg, v_ctx)), 0.05)
                        shift = (y * v_neg - v_ctx) * neg_elr  # -∇_R = -(v_ctx - y*v_neg)
                        v_new = v_neg + shift
                        nv = np.linalg.norm(v_new)
                        if nv > 1e-10:
                            v_new /= nv
                        cs._apply_vector_update(neg_cid, v_new)

        self.lattice.update(ids)
        return 1

    # ── Evaluation ────────────────────────────────────────────

    def evaluate(self, corpus_path, max_lines=None, batch_size=500):
        """Compute perplexity and accuracy on held-out corpus.

        Full softmax over vocabulary at each position.  Scoring uses
        the same components as _branch():
          - 0.5  × connection PMI weight
          - 0.25 × ngram probability
          - 0.15 × vector cosine similarity
          - 0.02 × frequency prior

        Optimised: ngram+PMI boost precomputed once per eval, not per position.
        """
        import time

        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        if max_lines:
            lines = lines[:max_lines]

        all_ids = []
        for line in lines:
            ids = self._encode_input(line)
            if len(ids) >= 2:
                all_ids.extend(ids)

        n_positions = len(all_ids) - 1
        if n_positions < 1:
            return {'perplexity': float('inf'), 'accuracy_top1': 0.0,
                    'accuracy_top5': 0.0, 'n_tokens': 0}

        cids = sorted(self.cs.concept_vectors.keys())
        cid_to_idx = {c: i for i, c in enumerate(cids)}
        V = np.array([self.cs.concept_vectors[c] for c in cids], dtype=np.float32)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        V /= norms
        vocab_size = len(cids)
        K = 3

        # Precompute prior array for all tokens (shared across positions)
        total_freq = sum(self.lattice.concept_freq.values()) or 1.0
        prior_arr = np.zeros(vocab_size, dtype=np.float32)
        for i, c in enumerate(cids):
            freq = self.lattice.concept_freq.get(c, 0)
            prior_arr[i] = 0.02 / (K + 1) * (1.0 - min(freq / 1000, 1.0))

        # ── Precompute ngram+PMI boost per prev_cid ──
        # Inlines PMI to avoid redundant dict lookups on 2M entries
        ngram_boost = {}
        for (prev_cid,), counter in self.lattice.ngrams[2].items():
            total_ng = sum(counter.values())
            if total_ng < 1:
                continue
            boost_map = {}
            for ncid, ncount in counter.items():
                idx = cid_to_idx.get(ncid)
                if idx is None:
                    continue
                prob = ncount / total_ng
                # Inline _pmi_weight with precomputed count_prev/count_pair
                count_next = self.lattice.concept_freq.get(ncid, 0)
                if count_next < 1:
                    pmi_w = 0.1
                else:
                    p_next_given_prev = ncount / total_ng
                    p_next = count_next / total_freq
                    pmi = math.log(p_next_given_prev / max(p_next, 1e-10))
                    pmi_w = max(min(pmi / 2.0 + 0.2, 2.0), 0.05)
                boost_map[ncid] = (0.25 * prob + 0.5 * pmi_w) / (K + 1)
            ngram_boost[prev_cid] = boost_map

        total_log_prob = 0.0
        vec_log_prob = 0.0
        correct_top1 = 0
        correct_top5 = 0
        vec_correct_top1 = 0
        n_eval = 0
        t0 = time.time()

        for start in range(0, n_positions, batch_size):
            end = min(start + batch_size, n_positions)
            batch_prev = all_ids[start:end]
            batch_next = all_ids[start + 1:end + 1]
            batch_n = len(batch_prev)

            # Batch vector similarities: each context × all vocab
            prev_vecs = np.array([
                self.cs.concept_vectors.get(c, np.zeros(self.cs.dim, dtype=np.float32))
                for c in batch_prev
            ], dtype=np.float32)
            pn = np.linalg.norm(prev_vecs, axis=1, keepdims=True)
            pn[pn < 1e-10] = 1.0
            prev_vecs /= pn
            sims = prev_vecs @ V.T  # (batch, vocab_size)
            sims = np.maximum(sims, 0)

            for pos in range(batch_n):
                prev_cid = batch_prev[pos]
                next_cid = batch_next[pos]

                scores = prior_arr.copy()
                scores += 0.15 * sims[pos] / (K + 1)

                # Ngram+PMI boost: fast lookup of precomputed values
                boost = ngram_boost.get(prev_cid)
                if boost:
                    for ncid, bval in boost.items():
                        idx = cid_to_idx.get(ncid)
                        if idx is not None:
                            scores[idx] += bval

                # Softmax
                scores -= scores.max()
                scores = np.clip(scores, -50, 50)
                exp_s = np.exp(scores)
                probs = exp_s / exp_s.sum()

                actual_idx = cid_to_idx.get(next_cid)
                if actual_idx is not None:
                    lp = np.log(max(probs[actual_idx], 1e-30))
                    total_log_prob += lp
                    if cids[np.argmax(scores)] == next_cid:
                        correct_top1 += 1
                    if next_cid in {cids[i] for i in np.argsort(-scores)[:5]}:
                        correct_top5 += 1

                    # Vector-only PPL (cosine similarity, no ngram/PMI/prior)
                    vec_scores = sims[pos]  # already clamped to ≥ 0
                    vec_scores -= vec_scores.max()
                    vec_scores = np.clip(vec_scores, -50, 50)
                    exp_v = np.exp(vec_scores)
                    vprobs = exp_v / exp_v.sum()
                    vlp = np.log(max(vprobs[actual_idx], 1e-30))
                    vec_log_prob += vlp
                    if cids[np.argmax(vec_scores)] == next_cid:
                        vec_correct_top1 += 1

                    n_eval += 1

            if start % 500 == 0 and n_eval > 0:
                elapsed = time.time() - t0
                rate = end / max(elapsed, 1)
                ppl = np.exp(-total_log_prob / n_eval)
                vppl = np.exp(-vec_log_prob / n_eval)
                acc1 = correct_top1 / n_eval
                vacc1 = vec_correct_top1 / n_eval
                print(f"  eval {end}/{n_positions} | {rate:.0f} tok/s | "
                      f"PPL={ppl:.1f} acc@1={acc1:.3f} | "
                      f"vecPPL={vppl:.1f} vacc@1={vacc1:.3f}")

        elapsed = time.time() - t0
        perplexity = np.exp(-total_log_prob / max(n_eval, 1))
        vec_perplexity = np.exp(-vec_log_prob / max(n_eval, 1))
        return {
            'perplexity': float(perplexity),
            'vec_perplexity': float(vec_perplexity),
            'accuracy_top1': correct_top1 / max(n_eval, 1),
            'accuracy_top5': correct_top5 / max(n_eval, 1),
            'vec_accuracy_top1': vec_correct_top1 / max(n_eval, 1),
            'n_tokens': n_eval,
            'total_log_prob': float(total_log_prob),
            'vec_log_prob': float(vec_log_prob),
            'elapsed_s': float(elapsed),
        }


if __name__ == '__main__':
    import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
    import sentencepiece as spm
    from eva.symbolic.concept_space import ConceptSpace
    from eva.symbolic.syntax_lattice import SyntaxLattice

    sp = spm.SentencePieceProcessor(
        model_file=r'C:\Users\black\OneDrive\Desktop\FCF\real_data\bpe_ru_32k.model')

    print("Initializing ConceptSpace (32K)...")
    cs = ConceptSpace(vocab_size=sp.vocab_size(), dim=384)
    cs.init_concepts()
    cs.init_homeostasis()

    print("Initializing lattice...")
    lattice = SyntaxLattice()
    gen = CrystalGenerator(cs, sp, lattice)

    print("\n--- Generation tests ---")
    for seed in ['князь', 'человек', 'война']:
        result = gen.generate(seed_word=seed)
        print(f"  [{seed}] path_len={len(result['concept_path'])} score={result['score']:.2f}")

    print("\n--- Training on sample ---")
    for sent in ["Князь Андрей вышел на крыльцо.", "Человек должен быть свободен."]:
        n = gen.train_from_text(sent)
        print(f"  trained: {n}")

    print("\n--- After training ---")
    for seed in ['князь', 'человек', 'война']:
        result = gen.generate(seed_word=seed)
        print(f"  [{seed}] path_len={len(result['concept_path'])} score={result['score']:.2f}")

    print("OK")
