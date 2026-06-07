"""ConceptGenerator — neuro-inspired generation via concept navigation.

Architecture based on cortical processing principles:

  1. Theta-rhythm gating: word_num controls temperature (early=explore, late=exploit)
  2. STDP learning: every generation step trains concept vectors
  3. Lateral inhibition: winner concept suppresses similar competitors
  4. Homeostatic plasticity: underused concepts boosted, overused dampened
  5. Continuous learning: no train/eval split — each act is both

Levels:
  L3: Meta-concepts (clustered concept groups)
  L2: Concept navigation (predict next concept via transitions + vectors)
  L1: Word selection within concept (anchor preference + temperature)
  L0: BPE token generation within word (character-level)
"""

import math, random
import numpy as np
from collections import defaultdict
from typing import Optional


class ConceptGenerator:
    """Neuro-inspired generation via concept navigation.

    Each generation step is simultaneously a learning step (STDP).
    Temperature is theta-rhythm gated by word position.
    """

    def __init__(self, cs, tok, config=None):
        self.cs = cs
        self.tok = tok
        self.config = config or {}

        # Theta-rhythm parameters
        self.base_temp = self.config.get('base_temp', 0.5)
        self.theta_tau = self.config.get('theta_tau', 5.0)  # words before settling

        # Concept selection temperature at word=0 (theta peak)
        self.concept_temp_start = self.config.get('concept_temp_start', 0.5)
        self.concept_temp_end = self.config.get('concept_temp_end', 0.05)

        # Word selection temperature
        self.word_temp_start = self.config.get('word_temp_start', 0.3)
        self.word_temp_end = self.config.get('word_temp_end', 0.05)

        # STDP learning rate
        self.learning_rate = self.config.get('learning_rate', 0.1)

        # Sentence level
        self.max_words = self.config.get('max_words', 30)
        self.min_words = self.config.get('min_words', 3)

        # State
        self.rng = random.Random()

        # Initialize homeostasis tracking in ConceptSpace
        self.cs.init_homeostasis()

    # ── Theta rhythm ──

    def _theta_temp(self, word_num, start_temp, end_temp):
        """Theta-rhythm gating: temperature decays with word position.
        Early words: high temp (exploration / binding).
        Late words: low temp (exploitation / convergence)."""
        return end_temp + (start_temp - end_temp) * math.exp(-word_num / self.theta_tau)

    # ── Generation ──

    def generate(self, seed_word=None, seed_cid=None, target_text=None,
                 max_words=None, temperature=None):
        """Generate a sentence with continuous learning.

        Args:
            seed_word: starting word
            seed_cid: starting concept ID (alternative)
            target_text: if provided, compare generation vs target for STDP
            max_words: max words to generate

        Returns:
            dict: text, tokens, concept_path, word_path, matches
        """
        max_words = max_words or self.max_words

        seed_cid = self._resolve_seed(seed_word, seed_cid)
        if seed_cid is None:
            return {'text': '', 'tokens': [], 'concept_path': [],
                    'word_path': [], 'matches': 0, 'total': 0}

        # Target concept sequence (for STDP comparison)
        target_concepts = self._extract_target_concepts(target_text)

        tokens = [self.tok.BOS, self.tok.SENT_OPEN]
        concept_path = [(seed_cid, self._anchor_of(seed_cid))]
        word_path = []
        prev_cid = seed_cid
        matches = 0
        total = 0

        for wn in range(max_words):
            # ── Theta-rhythm temperature ──
            c_temp = self._theta_temp(wn, self.concept_temp_start, self.concept_temp_end)
            w_temp = self._theta_temp(wn, self.word_temp_start, self.word_temp_end)

            # ── Predict next concept ──
            next_cid = self._predict_concept(prev_cid, temperature=c_temp, word_num=wn)
            if next_cid is None:
                break

            # ── Homeostatic novelty boost ──
            h_boost = self.cs.homeostatic_boost(next_cid)
            if h_boost > 0 and self.rng.random() < h_boost * 0.5:
                # Occasionally override with an underused concept
                underused = self._sample_underused_concept(prev_cid)
                if underused is not None:
                    next_cid = underused

            # ── Select word within concept ──
            word_text = self._select_word(next_cid, temperature=w_temp)
            if word_text is None:
                break

            # ── Encode word to tokens ──
            tokens.append(self.tok.WORD_OPEN)
            word_ids = self.tok.bpe.encode(word_text).ids
            for wid in word_ids:
                tokens.append(wid + self.tok.N_SPECIAL)
            tokens.append(self.tok.WORD_CLOSE)

            concept_path.append((next_cid, word_text))
            word_path.append(word_text)

            # ── STDP: learn from this transition ──
            expected_cid = target_concepts[wn] if wn < len(target_concepts) else None
            self.cs.svd_shift(prev_cid, next_cid, expected_cid=expected_cid,
                               lr=self.learning_rate, word_num=wn)

            # Track usage for homeostasis
            self.cs.update_usage(next_cid)

            # Track match (for eval)
            if expected_cid is not None:
                total += 1
                if next_cid == expected_cid:
                    matches += 1

            prev_cid = next_cid

            # ── EOS check ──
            if wn >= self.min_words and word_text in '.!?…':
                break

        tokens.append(self.tok.SENT_CLOSE)
        tokens.append(self.tok.EOS)
        text = self.tok.decode(tokens)

        # Also train backward (post→pre) on the first word
        if len(concept_path) > 1:
            self.cs.svd_shift(concept_path[1][0], seed_cid,
                               expected_cid=seed_cid, lr=self.learning_rate * 0.3,
                               word_num=0)

        return {
            'text': text,
            'tokens': tokens,
            'concept_path': concept_path[1:],  # exclude seed
            'word_path': word_path,
            'matches': matches,
            'total': total,
            'concept_temp': c_temp,
            'word_temp': w_temp,
        }

    def _resolve_seed(self, seed_word, seed_cid):
        if seed_cid is not None:
            return seed_cid
        if seed_word:
            cid = self.cs.word_to_cid.get(seed_word.lower())
            if cid is not None:
                return cid
            # Try finding via skeleton
            cid = self.cs.skeleton.concept_of(seed_word.lower())
            if cid is not None:
                return cid
            print(f"WARNING: '{seed_word}' not in concepts, using random")
        if self.cs.cid_list:
            return self.rng.choice(self.cs.cid_list)
        return None

    def _extract_target_concepts(self, target_text):
        """Extract concept IDs from target text for STDP comparison."""
        if not target_text:
            return []
        ids = self.tok.encode(target_text)
        meta = self.tok.metadata_from_ids(ids)
        return [m.get('concept_id') for m in meta
                if m['flags'] & 1 and m.get('concept_id') is not None]

    # ── Concept prediction with lateral inhibition ──

    def _predict_concept(self, prev_cid, temperature=0.3, word_num=0):
        """Predict next concept.

        Combines:
        - Transition probability (corpus statistics)
        - Vector similarity (semantic coherence)
        - Lateral inhibition (diversity)
        - Homeostatic boost (novelty)
        """
        candidates = self._get_candidates(prev_cid)
        if not candidates:
            similar = self.cs.topk_similar_concepts(prev_cid, k=5)
            if not similar:
                return self.rng.choice(list(self.cs.cid_list))
            candidates = [(s[0], 0.5) for s in similar]

        # Score candidates
        scored = self._score_candidates(candidates, prev_cid)
        if not scored:
            return None

        if temperature <= 0:
            return max(scored, key=lambda x: x[1])[0]

        # Temperature sampling
        scores = np.array([s for _, s in scored], dtype=np.float64)
        scores = scores - scores.max()
        # Clip to avoid extreme values
        scores = np.clip(scores, -50, 50)
        probs = np.exp(scores / max(temperature, 0.01))
        probs /= probs.sum()

        idx = self.rng.choices(range(len(scored)), weights=probs)[0]
        return scored[idx][0]

    def _get_candidates(self, prev_cid):
        """Get candidate concepts from transition matrix + vector neighbors."""
        candidates = {}
        # From transitions
        trans = self.cs.predict_next_concept(prev_cid, top_k=20)
        for cid, prob in trans:
            candidates[cid] = prob
        # From vector similarity
        similar = self.cs.topk_similar_concepts(prev_cid, k=10)
        for cid, anchor, sim in similar:
            if cid not in candidates:
                candidates[cid] = sim * 0.3
            else:
                candidates[cid] += sim * 0.1
        return list(candidates.items())

    def _score_candidates(self, candidates, prev_cid):
        """Score candidate concepts with lateral inhibition and homeostasis."""
        v_prev = self.cs.concept_vector(prev_cid)
        scored = []

        for cid, base_prob in candidates:
            # Vector similarity to context
            v_c = self.cs.concept_vector(cid)
            vec_sim = 0.0
            if v_prev is not None and v_c is not None:
                vp_n = v_prev / max(np.linalg.norm(v_prev), 1e-10)
                vc_n = v_c / max(np.linalg.norm(v_c), 1e-10)
                vec_sim = float(np.dot(vp_n, vc_n))

            # Homeostatic novelty
            h_boost = self.cs.homeostatic_boost(cid)

            # Combined: transition prob + vector sim + homeostasis
            score = base_prob + 0.3 * max(vec_sim, 0) + h_boost * 0.5
            scored.append((cid, score))

        # Lateral inhibition: if a concept candidates are too similar to each other,
        # suppress redundant ones
        if len(scored) > 3:
            scored.sort(key=lambda x: -x[1])
            unique = [scored[0]]
            for cid, s in scored[1:]:
                # Check similarity to already selected top concepts
                v_c = self.cs.concept_vector(cid)
                too_similar = False
                for sel_cid, _ in unique[:3]:
                    v_sel = self.cs.concept_vector(sel_cid)
                    if v_c is not None and v_sel is not None:
                        vc_n = v_c / max(np.linalg.norm(v_c), 1e-10)
                        vs_n = v_sel / max(np.linalg.norm(v_sel), 1e-10)
                        sim = float(np.dot(vc_n, vs_n))
                        if sim > 0.85:
                            too_similar = True
                            break
                if not too_similar or len(unique) < 5:
                    unique.append((cid, s * 0.9 if too_similar else s))
            scored = unique

        return scored

    def _sample_underused_concept(self, prev_cid, threshold=0.3):
        """Sample a concept with low usage (novelty drive)."""
        underused = [cid for cid, usage in self.cs.concept_usage.items()
                     if usage < threshold and cid != prev_cid]
        if not underused:
            return None
        # Prefer concepts similar to context
        v_prev = self.cs.concept_vector(prev_cid)
        if v_prev is not None:
            vp_n = v_prev / max(np.linalg.norm(v_prev), 1e-10)
            underused.sort(key=lambda c: float(np.dot(vp_n,
                self.cs.concept_vector(c) / max(np.linalg.norm(self.cs.concept_vector(c)), 1e-10)
                if self.cs.concept_vector(c) is not None else 0)), reverse=True)
        return underused[0] if underused else None

    # ── Word selection ──

    def _select_word(self, cid, temperature=0.2):
        """Select word form within concept.
        Anchor preferred, satellites for variety."""
        words = self.cs.words_in_concept(cid, top_k=20)
        if not words:
            return None

        info = self.cs.concept_info.get(cid, {})
        anchor = info.get('anchor', words[0])

        if temperature <= 0:
            return anchor

        scored = []
        for w in words:
            is_anchor = (w == anchor)
            # Anchor gets highest weight, satellites vary by edit distance
            score = 3.0 if is_anchor else 1.0
            # Prefer shorter words (less morphological over-generation)
            score *= math.exp(-0.08 * len(w))
            scored.append((w, score))

        scores = np.array([s for _, s in scored], dtype=np.float64)
        scores = scores - scores.max()
        scores = np.clip(scores, -50, 50)
        probs = np.exp(scores / max(temperature, 0.01))
        probs /= probs.sum()

        idx = self.rng.choices(range(len(scored)), weights=probs)[0]
        return scored[idx][0]

    def _anchor_of(self, cid):
        """Get anchor word for a concept."""
        info = self.cs.concept_info.get(cid, {})
        return info.get('anchor', f'C{cid}')

    # ── Training loop ──

    def train_sentence(self, target_text, seed_word=None, epochs=3):
        """Train on a single sentence.

        Each epoch: generate sentence, compare to target, apply STDP.
        Temperature decreases per epoch (annealing).
        """
        results = []
        for epoch in range(epochs):
            # Annealing: less random over time
            e_factor = 1.0 - epoch / (epochs + 1)
            temp = 0.5 * e_factor

            result = self.generate(
                seed_word=seed_word,
                target_text=target_text,
                temperature=temp,
            )
            results.append(result)

            # Report
            if result['total'] > 0:
                acc = result['matches'] / result['total'] * 100
            else:
                acc = 0
            print(f"  epoch {epoch}: temp={temp:.3f}, "
                  f"match={result['matches']}/{result['total']} ({acc:.1f}%), "
                  f"text={result['text'][:60]}")

        return results

    def generate_from_text(self, seed_text, max_words=None):
        """Generate continuation from seed text."""
        ids = self.tok.encode(seed_text)
        meta = self.tok.metadata_from_ids(ids)
        last_cid = None
        for m in reversed(meta):
            cid = m.get('concept_id')
            if cid is not None:
                last_cid = cid
                break
        if last_cid is None:
            return self.generate(seed_word=seed_text.split()[-1] if seed_text else None)
        # Train on seed for context
        self._extract_target_concepts(seed_text)
        return self.generate(seed_cid=last_cid, max_words=max_words,
                              target_text=seed_text)


if __name__ == '__main__':
    import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
    from eva.symbolic.concept_space import ConceptSpace
    from eva.symbolic.concept_tokenizer import ConceptTokenizer

    print("Loading...")
    tok = ConceptTokenizer()
    tok.initialize()
    cs = ConceptSpace(None, dim=128)
    cs.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space.json')

    gen = ConceptGenerator(cs, tok, {
        'base_temp': 0.5,
        'theta_tau': 5.0,
        'concept_temp_start': 0.5,
        'concept_temp_end': 0.05,
        'word_temp_start': 0.3,
        'word_temp_end': 0.05,
        'learning_rate': 0.1,
        'max_words': 12,
        'min_words': 2,
    })

    print("\n--- Free generation (no target) ---")
    for seed in ['князь', 'война', 'человек']:
        r = gen.generate(seed_word=seed)
        temp_str = f"theta_c={r['concept_temp']:.3f} theta_w={r['word_temp']:.3f}"
        print(f"[{seed}] {temp_str} -> {r['text']}")

    print("\n--- Training on target sentences ---")
    targets = [
        "Князь Андрей вышел на крыльцо.",
        "Война и мир это великое произведение.",
    ]
    for target in targets:
        seed = target.split()[0].lower().strip('.,!?')
        print(f"\nTarget: {target}")
        gen.train_sentence(target, seed_word=seed, epochs=3)
