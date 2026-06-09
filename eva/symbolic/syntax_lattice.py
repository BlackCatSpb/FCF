"""SyntaxLattice — connection strength graph + n-gram patterns.

Captures semantic connections between core concepts, not just sequences:
   connection(core_A, core_B) → {type, strength, context}
   
Relation types are inferred from environment words between concepts:
   "на улице погода" → located_at
   "хорошая погода" → has_quality
   "погода испортилась" → state_change

N-grams remain for syntactic sequence information.
Connection graph adds semantic relation layer.
"""

import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
import json, math, os
from typing import List, Optional, Dict, Tuple


# Relation types between concepts
RELATION_TYPES = {
    'has_quality': 0,       # ADJ→NOUN (хорошая погода)
    'located_at': 1,        # PREP→NOUN (на улице)
    'has_possession': 2,    # NOUN→NOUN genitive (дом человека)
    'performs_action': 3,   # NOUN→VERB (человек идёт)
    'has_manner': 4,        # ADV→VERB (быстро бежать)
    'has_quantity': 5,      # NUM→NOUN (три дома)
    'state_change': 6,      # VERB→state (погода испортилась)
    'contrast': 7,          # но, а
    'connects': 8,          # и, да
    'time_at': 9,           # temporal (вчера, сегодня)
    'related_to': 10,       # generic
}

# Environment words that signal relation types
ENV_TO_RELATION = {
    'в': 'located_at', 'на': 'located_at', 'у': 'located_at',
    'под': 'located_at', 'над': 'located_at', 'между': 'located_at',
    'за': 'located_at', 'перед': 'located_at', 'около': 'located_at',
    'возле': 'located_at', 'среди': 'located_at',
    'и': 'connects', 'да': 'connects',
    'но': 'contrast', 'а': 'contrast', 'однако': 'contrast',
    'вчера': 'time_at', 'сегодня': 'time_at', 'завтра': 'time_at',
    'потом': 'time_at', 'затем': 'time_at', 'сначала': 'time_at',
    'после': 'time_at', 'до': 'time_at', 'во время': 'time_at',
}


class SyntaxLattice:
    """Higher-order concept n-gram models + connection strength graph.

    The lattice has two views:
      Sequential: n-gram concept→concept→concept transitions
      Semantic: connection(core_A, core_B) → {type, strength}

    Generation uses both: syntax for ordering, connections for relevance.
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

        # Connection graph: (cid_a, cid_b) → {count, type_counter}
        self.connections = defaultdict(lambda: {'count': 0, 'types': Counter()})

    def build(self, corpus_path, sp, max_n=4, min_count=2):
        """Build n-gram model from corpus via SentencePiece.

        Args:
            corpus_path: path to text corpus
            sp: SentencePieceProcessor
            max_n: maximum n-gram order
            min_count: minimum occurrences to keep an n-gram (unused, kept for compat)
        """
        self.max_n = max_n
        for n in range(2, max_n + 1):
            self.ngrams[n] = defaultdict(Counter)

        n_concepts = [0, 0, 0, 0]
        line_count = 0

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                ids = sp.encode(line)
                if len(ids) < 2:
                    continue
                for c in ids:
                    self.concept_freq[c] += 1
                    n_concepts[0] += 1
                for n in range(2, max_n + 1):
                    for i in range(len(ids) - n + 1):
                        prefix = tuple(ids[i:i + n - 1])
                        next_c = ids[i + n - 1]
                        self.ngrams[n][prefix][next_c] += 1
                        n_concepts[n - 1] += 1
                for i in range(len(ids) - 1):
                    self.add_connection(ids[i], ids[i + 1])

        print(f"  SyntaxLattice: {line_count} lines, {n_concepts[0]} concept tokens")
        for n in range(2, max_n + 1):
            n_unique = sum(len(counter) for counter in self.ngrams[n].values())
            print(f"    {n}-grams: {len(self.ngrams[n])} prefixes, {n_unique} unique transitions")
        print(f"    connections: {len(self.connections)}")
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
            n_orders = [2, 3]  # exclude 4-grams (statistically empty)

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
            if total < 3:  # minimum count threshold (was 0)
                continue

            # Weight: higher n has more weight when data exists
            # but also discount for statistical noise
            weight = total / (total + 10.0)  # Bayesian: confident when count >> 10
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
        """Predict next concept using both n-gram lattice and vector space."""
        context = concept_sequence[-3:] if len(concept_sequence) >= 3 else concept_sequence
        syn_preds = self.predict(context)

        if concept_sequence:
            prev_cid = concept_sequence[-1]
            v_prev = concept_space.concept_vector(prev_cid)
            vec_preds = []
            if v_prev is not None:
                vp_n = v_prev / max(np.linalg.norm(v_prev), 1e-10)
                similar = concept_space.topk_similar_concepts(prev_cid, k=top_k)
                for cid, sim in similar:
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
        """Increment n-gram counts + connections from a concept sequence.

        Args:
            concept_sequence: list of concept IDs (e.g., [князь, выйти, на, крыльцо])
        """
        for n in range(2, self.max_n + 1):
            for i in range(len(concept_sequence) - n + 1):
                prefix = tuple(concept_sequence[i:i + n - 1])
                next_c = concept_sequence[i + n - 1]
                if prefix not in self.ngrams[n]:
                    self.ngrams[n][prefix] = Counter()
                self.ngrams[n][prefix][next_c] += 1
                self.concept_freq[next_c] += 1

        for i in range(len(concept_sequence) - 1):
            self.add_connection(concept_sequence[i], concept_sequence[i + 1])

    # ── Connection Strength Graph ────────────────────────────

    @staticmethod
    def infer_relation(env_words):
        """Infer relation type from environment words between concepts.

        Args:
            env_words: list of words (prepositions, conjunctions) between cores

        Returns:
            relation type string
        """
        for w in env_words:
            wl = w.lower().strip('.,!?;:()[]{}«»—–-…\'\"')
            if wl in ENV_TO_RELATION:
                return ENV_TO_RELATION[wl]
        return 'related_to'

    def add_connection(self, cid_a, cid_b, relation='related_to', count=1):
        """Record a semantic connection between two concepts.

        Args:
            cid_a, cid_b: concept IDs
            relation: relation type string
            count: increment amount
        """
        key = (min(cid_a, cid_b), max(cid_a, cid_b))
        self.connections[key]['count'] += count
        if relation in RELATION_TYPES:
            self.connections[key]['types'][relation] += count

    def get_connection(self, cid_a, cid_b):
        """Get connection info between two concepts.

        Returns:
            dict with: strength (0..1), type, context
            or None if no connection recorded.
        """
        key = (min(cid_a, cid_b), max(cid_a, cid_b))
        conn = self.connections.get(key)
        if conn is None or conn['count'] == 0:
            return None

        # Connection strength: normalized co-occurrence
        max_count = max(self.concept_freq.get(cid_a, 1),
                        self.concept_freq.get(cid_b, 1))
        strength = min(conn['count'] / max(max_count, 1), 1.0)

        # Dominant relation type
        types = conn['types']
        dom_type = types.most_common(1)[0][0] if types else 'related_to'

        return {
            'strength': strength,
            'type': dom_type,
            'count': conn['count'],
        }

    def connection_strength(self, cid_a, cid_b, cs=None):
        """Compute connection strength between two concepts.

        Combines:
          - co-occurrence count from connection graph
          - cosine similarity from vector space (if cs provided)

        Returns: float 0..1
        """
        conn = self.get_connection(cid_a, cid_b)
        cooc_strength = conn['strength'] if conn else 0.0

        cos_strength = 0.0
        if cs is not None:
            va = cs.concept_vector(cid_a)
            vb = cs.concept_vector(cid_b)
            if va is not None and vb is not None:
                cos = float(np.dot(va, vb) / max(
                    np.linalg.norm(va) * np.linalg.norm(vb), 1e-10))
                cos_strength = max(cos, 0)

        return 0.6 * cooc_strength + 0.4 * cos_strength

    def connections_of(self, cid, top_k=20):
        """Get all concepts connected to a given concept.

        Returns:
            [(connected_cid, {strength, type}), ...] sorted by strength
        """
        results = []
        for (a, b), conn in self.connections.items():
            if a == cid:
                other = b
            elif b == cid:
                other = a
            else:
                continue
            max_c = max(self.concept_freq.get(cid, 1),
                        self.concept_freq.get(other, 1))
            strength = min(conn['count'] / max(max_c, 1), 1.0)
            dom_type = conn['types'].most_common(1)[0][0] if conn['types'] else 'related_to'
            results.append((other, {'strength': strength, 'type': dom_type}))

        results.sort(key=lambda x: -x[1]['strength'])
        return results[:top_k]

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
            'connections': {
                f'{a},{b}': {
                    'count': conn['count'],
                    'types': dict(conn['types']),
                }
                for (a, b), conn in self.connections.items()
            },
        }
        with open(path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(path + '.tmp', path)
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
        # Load connection graph if present
        self.connections = defaultdict(lambda: {'count': 0, 'types': Counter()})
        if 'connections' in data:
            for key_str, conn_data in data['connections'].items():
                a, b = int(key_str.split(',')[0]), int(key_str.split(',')[1])
                self.connections[(a, b)] = {
                    'count': conn_data['count'],
                    'types': Counter({k: v for k, v in conn_data['types'].items()}),
                }
        print(f"  Loaded SyntaxLattice: {[len(v) for v in self.ngrams.values()]} prefixes, "
              f"{len(self.connections)} connections")
        return self


if __name__ == '__main__':
    import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
    import sentencepiece as spm
    from eva.symbolic.concept_space import ConceptSpace

    sp = spm.SentencePieceProcessor(
        model_file=r'C:\Users\black\OneDrive\Desktop\FCF\real_data\bpe_ru.model')

    cs = ConceptSpace.load(
        r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space.json')

    print("Building SyntaxLattice from full corpus via SentencePiece...")
    lattice = SyntaxLattice()
    lattice.build(
        r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru.txt',
        sp,
        max_n=4,
    )

    lattice.save(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\syntax_lattice.json')
    print("Done.")
