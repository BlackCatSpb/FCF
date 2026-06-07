"""ConceptInductor — active concept induction via semantic resonance.

Mechanism (not hardcoded):
  Every observed concept sequence creates a RESONANCE PATTERN in the concept
  space. When a pattern resonates strongly (frequent + surprising + rewarding),
  a NEW meta-concept emerges at the pattern's semantic centroid.

  Anchor extraction (from noisy data):
    Words project into concept space. Function words cancel out via vector
    interference. Content words amplify. The REMAINING SIGNAL is the anchor —
    the eigen-concept of the input.

  This is NOT rule-based. It's an emergent property of:
    - Vector space geometry (dot products = resonance)
    - Frequency thresholds (resonance builds over time)
    - Hormonal gating (DA modulates induction rate)
"""

import numpy as np
from collections import defaultdict, Counter
import math


class ConceptInductor:
    """Self-organizing concept induction from observed patterns.

    The inductor monitors concept sequences, detects resonant patterns,
    and induces new meta-concepts at the semantic centroid of each pattern.

    Attributes:
        resonance_counter: tracks (n, pattern_tuple) -> (count, surprise_sum)
        meta_concepts: cid -> {pattern, vector, constituents, birth_step}
        induction_threshold: min resonance to trigger induction
    """

    def __init__(self, config=None):
        self.config = config or {}

        # Resonance tracking
        self.resonance = {}  # (n, prefix, next_c) -> [count, total_surprise]
        self.induction_threshold = self.config.get('induction_threshold', 15)
        self.min_pattern_freq = self.config.get('min_pattern_freq', 5)

        # Induced meta-concepts
        self.meta_concepts = {}  # new_cid -> info dict
        self.next_meta_id = 10**8  # high range to avoid collision

        # Curiosity state
        self.curiosity_targets = []  # regions to explore

        # Step counter
        self.step = 0

    def observe(self, concept_sequence, cs, lattice, hormones):
        """Observe a concept sequence, update resonance, induce if ready.

        Args:
            concept_sequence: list of concept IDs
            cs: ConceptSpace
            lattice: SyntaxLattice
            hormones: HormonalSystem (for gating)

        Returns:
            list of newly induced concept IDs (may be empty)
        """
        self.step += 1
        induced = []

        if len(concept_sequence) < 2:
            return induced

        # 1. Detect resonant patterns
        for n in [2, 3, 4]:
            if len(concept_sequence) < n:
                continue
            for i in range(len(concept_sequence) - n + 1):
                gram = concept_sequence[i:i+n]
                prefix = tuple(gram[:-1])
                next_c = gram[-1]

                # Compute surprise: prediction confidence from lattice
                preds = lattice.predict(list(prefix))
                pred_dict = {cid: s for cid, s in preds}
                expected_prob = pred_dict.get(next_c, 0.0)
                if expected_prob > 0:
                    surprise = -math.log(max(expected_prob, 1e-10))
                else:
                    surprise = 3.0  # max surprise for unseen transitions

                # Update resonance: count + cumulative surprise
                key = (n, prefix, next_c)
                if key not in self.resonance:
                    self.resonance[key] = [0, 0.0]
                self.resonance[key][0] += 1
                self.resonance[key][1] += surprise

                # 2. Check induction threshold
                count, total_surprise = self.resonance[key]
                avg_surprise = total_surprise / max(count, 1)

                resonance_strength = count * avg_surprise * (0.5 + 0.5 * hormones.dopamine)
                min_strength = self.induction_threshold * (2.0 - hormones.acetylcholine)

                if (count >= self.min_pattern_freq
                    and resonance_strength > min_strength
                    and avg_surprise > 1.0):
                    new_cid = self._induce_meta(gram, cs)
                    if new_cid is not None:
                        induced.append(new_cid)
                        # Reset resonance to avoid double-induction
                        del self.resonance[key]

        # 3. Update lattice with observed sequence
        lattice.update(concept_sequence)

        # 4. Curiosity-driven exploration if DA is low
        if hormones.dopamine < 0.25 and self.step % 5 == 0:
            self._curious_explore(cs, lattice, hormones)

        return induced

    def _induce_meta(self, gram, cs):
        """Induce a new meta-concept at the semantic centroid of a pattern.

        The new concept represents the PATTERN as a whole — it's a compressed
        higher-order representation of the constituent concepts.

        The vector is the centroid of the pattern's semantic field.
        This means the new concept is SIMILAR TO all concepts in the pattern,
        but not identical to any one of them — it's the "pattern essence".
        """
        # Compute pattern centroid
        vectors = []
        constituents = []
        for c in gram:
            v = cs.concept_vector(c)
            if v is not None:
                vectors.append(v)
                constituents.append(c)

        if len(vectors) < 2:
            return None

        centroid = np.mean(vectors, axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid /= norm
        else:
            return None

        # Create new concept
        new_cid = self.next_meta_id
        self.next_meta_id += 1

        # Generate a descriptive anchor from pattern
        anchors = []
        for c in constituents:
            info = cs.concept_info.get(c, {})
            a = info.get('anchor', '')
            if a and len(a) >= 3:
                anchors.append(a)
        if anchors:
            # Take first 3 anchors joined
            pattern_name = '_'.join(anchors[:3])
        else:
            pattern_name = f'pat{new_cid}'

        # Register in concept space
        cs.concept_vectors[new_cid] = centroid
        if new_cid not in cs.cid_list:
            cs.cid_list.append(new_cid)
        cs.concept_info[new_cid] = {
            'cid': new_cid,
            'anchor': pattern_name,
            'satellites': anchors,
            'relations': defaultdict(list),
            'vector': centroid,
            'size': len(constituents),
        }
        # Also register in cid_to_words
        cs.cid_to_words[new_cid] = anchors

        # Store meta info
        self.meta_concepts[new_cid] = {
            'pattern': gram,
            'constituents': constituents,
            'birth_step': self.step,
            'anchor': pattern_name,
        }

        return new_cid

    def _curious_explore(self, cs, lattice, hormones):
        """Curiosity-driven exploration: when bored, seek novel patterns.

        The model identifies concept regions with HIGH UNCERTAINTY
        (few n-gram transitions, low frequency) and generates
        pseudo-observations to explore them.
        """
        # Find concepts with low n-gram coverage (high uncertainty)
        low_freq_concepts = [
            c for c in cs.cid_list
            if lattice.concept_freq.get(c, 0) < 3
            and c < 10**8  # exclude already-induced meta-concepts
        ]

        if not low_freq_concepts:
            return

        # Pick a random low-frequency concept
        rng = np.random.RandomState(self.step)
        target = rng.choice(low_freq_concepts[:min(500, len(low_freq_concepts))])

        # Find nearest neighbors of this concept
        v = cs.concept_vector(target)
        if v is None:
            return

        similar = cs.topk_similar_concepts(target, k=10, sample_size=300)
        if not similar:
            return

        # Create a pseudo-observation: target -> random neighbor
        # This is "what if" exploration — the model tries to understand
        # how this rare concept connects to familiar ones
        for cid, anchor, sim in similar[:3]:
            if sim > 0.3 and cid != target:
                # Register this as a weak observation
                key = (2, (target,), cid)
                if key not in self.resonance:
                    self.resonance[key] = [0, 0.0]
                self.resonance[key][0] += 0.5  # weak count
                self.resonance[key][1] += 2.0  # high surprise (exploratory)

    def extract_anchor(self, words, cs, tok):
        """Extract semantic anchor from noisy word sequence.

        NO HARDCODING. The mechanism:
          1. Project each word into concept space (resolve_anchor)
          2. Function words produce scattered vectors that cancel
          3. Content words produce coherent vectors that amplify
          4. The EIGENVECTOR of the word cloud = the semantic anchor

        Args:
            words: list of word strings (any language)
            cs: ConceptSpace
            tok: ConceptTokenizer

        Returns:
            anchor_cid: concept closest to the semantic essence
            confidence: how well-defined the anchor is (0..1)
        """
        from eva.symbolic.crystal_generator import CrystalGenerator
        # Temporary: use generator's anchor resolver
        # (We need a reference to a generator, but we don't have one here.
        #  The caller should provide resolve_anchor function.)

        # Fallback: direct word->cid mapping
        vectors = []
        matched = []
        for w in words:
            cid = cs.word_to_cid.get(w.lower().strip())
            if cid is not None:
                v = cs.concept_vector(cid)
                if v is not None:
                    vectors.append(v)
                    matched.append(cid)

        if not vectors:
            return None, 0.0

        # Compute weighted centroid
        # Longer words = more semantic content (heuristic, not hardcoded)
        weights = []
        for w in words:
            w_clean = w.lower().strip()
            cid = cs.word_to_cid.get(w_clean)
            if cid is not None:
                wgt = 1.0 + 0.1 * len(w_clean)  # weak content bias
                # Reduce weight for very common words
                freq = cs.concept_usage.get(cid, 0) if hasattr(cs, 'concept_usage') else 0
                if freq > 0.5:
                    wgt *= 0.3  # common words contribute less (they're noise)
                weights.append(wgt)

        if not weights:
            weights = [1.0] * len(vectors)

        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()

        centroid = np.average(vectors, axis=0, weights=weights).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid /= norm
        else:
            return None, 0.0

        # Find closest concept to centroid
        best_cid, best_sim = None, -1.0
        for cid, cv in cs.concept_vectors.items():
            cv_n = cv / max(np.linalg.norm(cv), 1e-10)
            sim = float(np.dot(centroid, cv_n))
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        # Confidence: how concentrated is the vector cloud?
        # High variance = noisy input, low variance = clear anchor
        if len(vectors) > 1:
            var = np.var([np.dot(v, centroid) for v in vectors])
            confidence = max(0.0, 1.0 - var * 2.0)
        else:
            confidence = 0.5

        return best_cid, confidence

    def save(self):
        """Save resonance state (for continuity across sessions)."""
        # Only save patterns that might lead to induction
        res_data = {}
        for key, (count, surprise) in self.resonance.items():
            if count >= 2:
                n, prefix, next_c = key
                key_str = f'{n}:{" ".join(str(c) for c in prefix)}:{next_c}'
                res_data[key_str] = [count, surprise]

        return {
            'resonance': res_data,
            'meta_concepts': {
                str(cid): info
                for cid, info in self.meta_concepts.items()
            },
            'step': self.step,
        }

    def load(self, data):
        """Load resonance state."""
        self.resonance = {}
        for key_str, val in data.get('resonance', {}).items():
            parts = key_str.split(':')
            n = int(parts[0])
            prefix = tuple(int(c) for c in parts[1].split()) if parts[1] else ()
            next_c = int(parts[2])
            self.resonance[(n, prefix, next_c)] = val
        self.meta_concepts = {
            int(cid): info
            for cid, info in data.get('meta_concepts', {}).items()
        }
        self.step = data.get('step', 0)


if __name__ == '__main__':
    print("Testing ConceptInductor...")

    import sys; sys.path.insert(0, 'C:/Users/black/OneDrive/Desktop/FCF')
    from eva.symbolic.concept_space import ConceptSpace
    from eva.symbolic.syntax_lattice import SyntaxLattice
    from eva.symbolic.hormonal_system import HormonalSystem

    cs = ConceptSpace(None, dim=128)
    cs.load('C:/Users/black/OneDrive/Desktop/FCF/real_data/concept_space.json')
    lattice = SyntaxLattice()
    lattice.load('C:/Users/black/OneDrive/Desktop/FCF/real_data/syntax_lattice.json')
    hormones = HormonalSystem()

    inductor = ConceptInductor({'induction_threshold': 3, 'min_pattern_freq': 2})

    # Simulate: feed the same pattern 5 times -> should induce a meta-concept
    pattern = [11597, 5716, 32716, 12574]  # князь -> выйти -> на -> крыльцо
    print(f"Feeding pattern: {pattern}")

    for i in range(10):
        induced = inductor.observe(pattern, cs, lattice, hormones)
        if induced:
            meta = inductor.meta_concepts[induced[0]]
            print(f"  Step {i}: INDUCED meta-concept {induced[0]}: "
                  f"anchor='{meta['anchor']}' "
                  f"pattern={meta['pattern']}")

    print(f"\nTotal meta-concepts induced: {len(inductor.meta_concepts)}")
    print("Done")
