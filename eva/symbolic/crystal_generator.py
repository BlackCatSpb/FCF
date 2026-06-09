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

    # ── Generation ─────────────────────────────────────────────

    def generate(self, seed_word=None, seed_cid=None, target_text=None,
                 query_words=None, max_words=None):
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
            h_beam = self.hormones.modulate_beam_width(self.beam_width)

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

                    # STDP
                    self.cs.fractal_stdp(prev_cid, cid,
                        expected_cid=expected_cid, lr=h_lr, word_num=wn)
                    self.cs.update_usage(cid)

                    # Lattice learning
                    self.lattice.update([prev_cid, cid])

                    novelty = 1.0 - min(self.lattice.concept_freq.get(cid, 0) / 50, 1.0)
                    surprise = 0.1 if is_match else 0.5
                    self.hormones.update(confidence=conf, is_match=is_match,
                        novelty=novelty, surprise=surprise,
                        expected_cid=expected_cid, gen_cid=cid)

                    new_beam.append((new_seq, new_score, next_branch_id))
                    next_branch_id += 1

            new_beam.sort(key=lambda x: -x[1])
            beam = new_beam[:max(1, h_beam)]
            all_chains.extend([(s, sc) for s, sc, _ in new_beam])

            # EOS
            for item in list(beam):
                seq, score, bid = item
                if wn >= self.min_words:
                    token_text = self._token_text(seq[-1])
                    if token_text in ('.', '!', '?', '…', '!', '?', '.', '...'):
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

    # ── Branch ─────────────────────────────────────────────────

    def _branch(self, seq, word_num, theta_temp=0.3, target_cid=None, centroid=None):
        """Generate diverse branching candidates via RRF over multiple signals."""
        prev_cid = seq[-1]
        cids = seq[-3:] if len(seq) >= 3 else seq
        K = 3

        # 1. Connection graph
        connected_prev = {}
        for cid, conn in self.lattice.connections_of(prev_cid, top_k=15):
            connected_prev[cid] = conn['strength']

        # 2. N-gram syntax
        syn_preds = self.lattice.predict(cids)
        syn_ranked = {cid: i + 1 for i, (cid, _) in enumerate(syn_preds[:30])}

        # 3. All candidates from learned signals
        all_cids = set(connected_prev.keys()) | set(syn_ranked.keys())

        # 4. Vector similarity fallback (when lattice is cold)
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
            if cid in connected_prev:
                rrf += 0.5 * connected_prev[cid] / (K + 1)
            if cid in syn_ranked:
                rrf += 0.25 / (K + syn_ranked[cid])
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

    def _pmi_weight(self, prev_cid, next_cid):
        """Pointwise Mutual Information weight for STDP pull strength.

        PMI = log(P(next|prev) / P(next))
        High PMI = specific, statistically surprising pair (e.g. князь→великий)
        Low PMI  = generic transition (e.g. а→также→в→качестве)
        Negative PMI = they avoid each other

        Maps to [0.05, 2.0] multiplier on learning rate.
        """
        n2 = self.lattice.ngrams[2]
        prefix_counter = n2.get((prev_cid,))
        if not prefix_counter:
            return 0.1  # unseen pair → minimal pull

        count_pair = prefix_counter.get(next_cid, 0)
        count_prev = sum(prefix_counter.values())
        count_next = self.lattice.concept_freq.get(next_cid, 0)
        total = sum(self.lattice.concept_freq.values())

        if count_pair < 1 or count_prev < 1 or count_next < 1 or total < 1:
            return 0.1

        p_next_given_prev = count_pair / count_prev
        p_next = count_next / total
        pmi = math.log(p_next_given_prev / max(p_next, 1e-10))

        # PMI=0 → 0.2, PMI=2 → 1.0, PMI=5 → 2.0, negative → 0.05
        return max(min(pmi / 2.0 + 0.2, 2.0), 0.05)

    # ── Training ───────────────────────────────────────────────

    def train_from_text(self, text, base_lr=None, context_window=2, pmi_gate=True):
        """Train via PMI-gated context-window STDP.

        Each token pulls toward nearby tokens weighted by exponential
        distance decay AND pointwise mutual information. Generic
        transitions (low PMI) get minimal pull — only specific,
        statistically surprising co-occurrences shape the vectors.

        Args:
            text: input line
            base_lr: learning rate override
            context_window: max distance for STDP pull (1 = adjacent only)
            pmi_gate: if True, weight STDP by PMI
        """
        ids = self._encode_input(text)
        if len(ids) < 2:
            return 0

        base_lr = base_lr if base_lr is not None else getattr(self, 'train_lr', 0.01)

        for i in range(len(ids)):
            start = max(0, i - context_window)
            end = min(len(ids), i + context_window + 1)
            for j in range(start, end):
                if j <= i:  # only forward: later tokens learn from earlier context
                    continue
                dist = abs(j - i)
                dist_weight = math.exp(-dist / 2.0)

                freq_a = self.lattice.concept_freq.get(ids[i], 0)
                freq_b = self.lattice.concept_freq.get(ids[j], 0)
                freq = max(freq_a, freq_b)
                freq_weight = 1.0 / (1.0 + math.log(max(freq, 1)) * 0.15)

                pmi_w = self._pmi_weight(ids[i], ids[j]) if pmi_gate else 1.0

                lr = base_lr * max(freq_weight, 0.05) * dist_weight * pmi_w

                self.cs.fractal_stdp(ids[i], ids[j],
                    expected_cid=ids[j], lr=lr)

        self.lattice.update(ids)
        return 1


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
