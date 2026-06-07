"""SyntaxLattice — higher-order concept n-gram patterns from corpus.

Captures natural syntax as concept-sequence templates:
  2-grams: князь->сказать, война->начаться
  3-grams: князь->выйти->на, сказать->что->весь
  4-grams: князь->Андрей->выйти->на

Not rules — probability distributions over concept chains.
Higher n = more syntactic specificity.
Lower n = fallback (semantic coherence).
"""

import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
import json, math, os
from typing import List


class SyntaxLattice:
    """Higher-order concept n-gram models forming the syntactic skeleton.

    The lattice is hierarchical:
      Level 1 (bigram): concept->concept transition matrix [already in ConceptSpace]
      Level 2 (trigram): concept->concept->concept transitions
      Level 3 (4-gram): concept->concept->concept->concept transitions

    Generation interpolates between levels: prefer higher n when data exists,
    fall back to lower n for flexibility.
    """

    def __init__(self):
        # n-gram stores: prefix_tuple -> [(next_concept, count)]
        self.ngrams = {2: {}, 3: {}, 4: {}}  # n -> dict of prefix_key -> Counter

        # Total counts for smoothing
        self.total_ngrams = {}  # n -> total count

        # Concept frequency (for backoff)
        self.concept_freq = Counter()

        # Max n-gram order
        self.max_n = 4

    def build(self, corpus_path, tok, max_n=4, min_count=2):
        """Build n-gram model from corpus concept sequences.

        Args:
            corpus_path: path to text corpus
            tok: ConceptTokenizer (for encoding)
            max_n: maximum n-gram order (4 = look back 3 concepts)
            min_count: minimum occurrences to keep an n-gram
        """
        self.max_n = max_n
        # Initialize per-order containers
        for n in range(2, max_n + 1):
            self.ngrams[n] = defaultdict(Counter)

        # Extract concept sequences from corpus
        n_concepts = [0, 0, 0, 0]  # counts for n=1,2,3,4
        line_count = 0

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line_count += 1

                ids = tok.encode(line)
                meta = tok.metadata_from_ids(ids)

                # Extract concept sequence from word starts
                concepts = [m.get('concept_id') for m in meta
                            if m['flags'] & 1 and m.get('concept_id') is not None]

                if len(concepts) < 2:
                    continue

                # Count concept frequencies (unigram)
                for c in concepts:
                    self.concept_freq[c] += 1
                    n_concepts[0] += 1

                # Build n-grams for orders 2..max_n
                for n in range(2, max_n + 1):
                    for i in range(len(concepts) - n + 1):
                        prefix = tuple(concepts[i:i + n - 1])
                        next_c = concepts[i + n - 1]
                        self.ngrams[n][prefix][next_c] += 1
                        n_concepts[n - 1] += 1

        print(f"  SyntaxLattice: {line_count} lines, {n_concepts[0]} concept tokens")
        for n in range(2, max_n + 1):
            n_unique = sum(len(counter) for counter in self.ngrams[n].values())
            print(f"    {n}-grams: {len(self.ngrams[n])} prefixes, {n_unique} unique transitions")

        return self

    def predict(self, context_concepts: List[int], n_orders=None) -> List:
        """Predict next concept from n-gram lattice.

        Interpolates between all available n-gram orders.
        Higher n = more syntactic specificity -> higher weight when data exists.

        Args:
            context_concepts: list of recent concept IDs (last 1-3)
            n_orders: which n-gram orders to use (default: all available)

        Returns:
            [(concept_id, score), ...] scored by interpolated probability
        """
        if n_orders is None:
            n_orders = sorted(self.ngrams.keys(), reverse=True)

        # Collect predictions from each order
        predictions = defaultdict(float)
        weights = []

        for n in n_orders:
            if len(context_concepts) < n - 1:
                continue

            prefix = tuple(context_concepts[-(n - 1):])
            counter = self.ngrams.get(n, {}).get(prefix)

            if not counter:
                continue

            total = sum(counter.values())
            if total <= 0:
                continue

            # Weight: higher n has more weight when data exists
            # but also discount for statistical noise
            weight = total / (total + 5.0)  # Bayesian: confident when count >> 5
            weights.append(weight)

            for cid, count in counter.items():
                prob = count / total
                predictions[cid] += prob * weight

        if not predictions:
            return []

        # Normalize by total weight
        total_weight = sum(weights)
        if total_weight <= 0:
            return []

        result = [(cid, score / total_weight) for cid, score in predictions.items()]
        result.sort(key=lambda x: -x[1])
        return result

    def predict_with_context(self, concept_sequence: List[int], concept_space,
                              temperature=0.3, top_k=20):
        """Predict next concept using both n-gram lattice and vector space.

        The lattice provides syntactic structure (what concepts typically follow).
        The vector space provides semantic coherence (what's similar to context).

        Args:
            concept_sequence: full concept path so far
            concept_space: ConceptSpace instance for vector similarity
            temperature: sampling temperature
            top_k: max candidates to consider

        Returns:
            [(cid, score), ...]
        """
        # 1. Get n-gram predictions (syntax)
        context = concept_sequence[-3:] if len(concept_sequence) >= 3 else concept_sequence
        syn_preds = self.predict(context)

        # 2. Get vector similarity predictions (semantics)
        if concept_sequence:
            prev_cid = concept_sequence[-1]
            v_prev = concept_space.concept_vector(prev_cid)
            vec_preds = []
            if v_prev is not None:
                vp_n = v_prev / max(np.linalg.norm(v_prev), 1e-10)
                similar = concept_space.topk_similar_concepts(prev_cid, k=top_k)
                for cid, anchor, sim in similar:
                    vec_preds.append((cid, sim))
        else:
            vec_preds = []

        # 3. Combine: syntax + semantics
        combined = {}
        syn_weight = 0.6
        vec_weight = 0.2
        prior_weight = 0.2

        # Syntax score
        syn_dict = {cid: score for cid, score in syn_preds}
        if syn_dict:
            max_syn = max(syn_dict.values())
            for cid, score in syn_dict.items():
                combined[cid] = syn_weight * (score / max_syn)

        # Vector similarity score
        for cid, sim in vec_preds:
            combined[cid] = combined.get(cid, 0) + vec_weight * max(sim, 0)

        # Prior for ALL candidates seen so far
        if self.concept_freq and combined:
            max_freq = max(self.concept_freq.values())
            for cid in list(combined.keys()):
                freq = self.concept_freq.get(cid, 0)
                prior = 1.0 - min(freq / max_freq, 1.0)
                combined[cid] += prior_weight * prior * 0.1

        if not combined:
            return []

        # Temperature sampling
        result = [(cid, score) for cid, score in combined.items() if score > 0]
        result.sort(key=lambda x: -x[1])

        if temperature <= 0:
            return result[:top_k]

        scores = np.array([s for _, s in result])
        scores = scores - scores.max()
        scores = np.clip(scores, -50, 50)
        probs = np.exp(scores / max(temperature, 0.01))
        probs /= probs.sum()

        scored = [(result[i][0], probs[i]) for i in range(min(top_k, len(result)))]
        return scored

    def update(self, concept_sequence):
        """Increment n-gram counts from a concept sequence (online learning).

        Args:
            concept_sequence: list of concept IDs (e.g., [князь, выйти, на, крыльцо])
        """
        for n in range(2, self.max_n + 1):
            for i in range(len(concept_sequence) - n + 1):
                prefix = tuple(concept_sequence[i:i + n - 1])
                next_c = concept_sequence[i + n - 1]
                # Auto-create missing prefixes (new patterns)
                if prefix not in self.ngrams[n]:
                    self.ngrams[n][prefix] = Counter()
                self.ngrams[n][prefix][next_c] += 1
                self.concept_freq[next_c] += 1

    def save(self, path):
        """Save to JSON."""
        data = {
            'ngrams': {
                str(n): {
                    ' '.join(str(c) for c in prefix): dict(counter)
                    for prefix, counter in ngrams.items()
                }
                for n, ngrams in self.ngrams.items()
            },
            'concept_freq': dict(self.concept_freq),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"  Saved SyntaxLattice to {path}")

    def load(self, path):
        """Load from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.ngrams = {}
        for n_str, ngrams_data in data['ngrams'].items():
            n = int(n_str)
            self.ngrams[n] = {}
            for prefix_key, counter_data in ngrams_data.items():
                prefix = tuple(int(c) for c in prefix_key.split())
                self.ngrams[n][prefix] = Counter({int(k): v for k, v in counter_data.items()})
        self.concept_freq = Counter({int(k): v for k, v in data['concept_freq'].items()})
        print(f"  Loaded SyntaxLattice: {[len(v) for v in self.ngrams.values()]} prefixes")
        return self


if __name__ == '__main__':
    import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
    from eva.symbolic.concept_tokenizer import ConceptTokenizer
    from eva.symbolic.concept_space import ConceptSpace

    print("Loading...")
    tok = ConceptTokenizer()
    tok.initialize()
    cs = ConceptSpace(None, dim=128)
    cs.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space.json')

    print("Building SyntaxLattice...")
    lattice = SyntaxLattice()
    lattice.build(
        r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru.txt',
        tok,
        max_n=4,
        min_count=2,
    )

    # Test predictions
    for test_concept_chain in [
        ['князь', 'Андрей'],
        ['война'],
        ['сказать'],
        ['князь', 'быть'],
    ]:
        cids = [cs.word_to_cid.get(w) for w in test_concept_chain if w in cs.word_to_cid]
        if cids:
            preds = lattice.predict_with_context(cids, cs, temperature=0)
            chain_str = ' -> '.join(test_concept_chain)
            print(f"\nAfter {chain_str}:")
            for cid, score in preds[:8]:
                anchor = cs.concept_info.get(cid, {}).get('anchor', '?')[:20]
                print(f"  [{cid:5d}] {anchor:20s} score={score:.4f}")

    lattice.save(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\syntax_lattice.json')
