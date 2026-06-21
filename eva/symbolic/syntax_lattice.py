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
RELATION_NAMES = {v: k for k, v in RELATION_TYPES.items()}  # reverse lookup

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

    Decay: concept_freq uses exponential moving average during online
    update() to forget old statistics.  build() uses raw counts (full
    corpus).  Call decay_all() periodically to sweep unseen tokens.
    """

    def __init__(self, decay=0.999):
        # n-gram stores: prefix_tuple -> [(next_concept, count)]
        self.ngrams = {}  # n -> dict of prefix_key -> Counter (built dynamically)

        # Total counts for smoothing
        self.total_ngrams = {}  # n -> total count

        # Concept frequency — defaultdict(float) for EMA frequencies
        self.decay = decay
        self.concept_freq = defaultdict(float)

        # Max n-gram order
        self.max_n = 4

        # Connection graph: (cid_a, cid_b) → {count, type_counter}
        self.connections = defaultdict(lambda: {'count': 0, 'types': Counter()})

        # Per-CID index for O(1) lookup (maintained by add_connection)
        self._connections_index = defaultdict(dict)  # cid -> {other_cid: connection_dict}

        # Skip-2 co-occurrences: prev_cid -> Counter[next_cid at distance 2]
        self.skip2 = defaultdict(Counter)

        # PPMI cache (lazy, built on first use_ppmi=True call)
        self._ppmi_cache = None

        # Prefix total caches: sum(prefix_counter.values()) O(1) instead of O(K)
        self._prefix_total = {}  # prefix_tuple -> int (total count)
        self._skip2_total = {}   # prev_cid -> int (total count)

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
            self.ngrams[n] = defaultdict(lambda: defaultdict(float))

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
                    self.concept_freq[c] = self.concept_freq.get(c, 0) * self.decay + 1.0
                    n_concepts[0] += 1
                for n in range(2, max_n + 1):
                    for i in range(len(ids) - n + 1):
                        prefix = tuple(ids[i:i + n - 1])
                        next_c = ids[i + n - 1]
                        self.ngrams[n][prefix][next_c] += 1
                        n_concepts[n - 1] += 1
                for i in range(len(ids) - 1):
                    self.add_connection(ids[i], ids[i + 1])
                for i in range(len(ids) - 2):
                    self.skip2[ids[i]][ids[i + 2]] += 1

        print(f"  SyntaxLattice: {line_count} lines, {n_concepts[0]} concept tokens")
        for n in range(2, max_n + 1):
            n_unique = sum(len(counter) for counter in self.ngrams[n].values())
            print(f"    {n}-grams: {len(self.ngrams[n])} prefixes, {n_unique} unique transitions")
        print(f"    connections: {len(self.connections)}")
        self._refresh_prefix_totals()
        return self

    def _refresh_prefix_totals(self):
        """Precompute sum(counter.values()) for each prefix for O(1) PMI denominator."""
        self._prefix_total = {}
        for n, ngram_dict in self.ngrams.items():
            for prefix, counter in ngram_dict.items():
                self._prefix_total[prefix] = sum(counter.values())
        self._skip2_total = {}
        for cid, counter in self.skip2.items():
            self._skip2_total[cid] = sum(counter.values())

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
                self.concept_freq[next_c] = self.concept_freq.get(next_c, 0) * self.decay + 1.0
                # Incremental prefix total (avoids O(all_ngrams) refresh on every line)
                self._prefix_total[prefix] = self._prefix_total.get(prefix, 0) + 1

        for i in range(len(concept_sequence) - 1):
            self.add_connection(concept_sequence[i], concept_sequence[i + 1])
        for i in range(len(concept_sequence) - 2):
            self.skip2[concept_sequence[i]][concept_sequence[i + 2]] += 1
            # Incremental skip2 total
            cid = concept_sequence[i]
            self._skip2_total[cid] = self._skip2_total.get(cid, 0) + 1

    def decay_all(self, min_freq=0.01, rare_concept_protect=False, rare_threshold=3):
        """Sweep all ngrams, concept frequencies, and connections with decay factor.

        Periodic call during training to forget old statistics while
        preserving a floor to prevent total decay to zero.

        Args:
            min_freq: floor for concept frequency
            rare_concept_protect: if True, skip decay for concepts with freq < rare_threshold
            rare_threshold: max frequency to consider a concept "rare"
        """
        # Decay ngram counts (float)
        for order in self.ngrams:
            for prefix in self.ngrams[order]:
                counter = self.ngrams[order][prefix]
                for ncid in list(counter.keys()):
                    counter[ncid] = counter[ncid] * self.decay
                    if counter[ncid] < 1e-6:
                        del counter[ncid]

        # Decay concept frequency (skip rare concepts if protection enabled)
        for c in list(self.concept_freq.keys()):
            freq = self.concept_freq[c]
            if rare_concept_protect and freq < rare_threshold:
                continue
            self.concept_freq[c] = max(freq * self.decay, min_freq)

        # Decay skip2 (Counter of Counter)
        for k in list(self.skip2.keys()):
            inner = self.skip2[k]
            for tgt in list(inner.keys()):
                inner[tgt] = inner[tgt] * self.decay
                if inner[tgt] < 1e-6:
                    del inner[tgt]

        # Invalidate PPMI cache after ngram decay
        self._ppmi_cache = None
        self._prefix_total = {}
        self._skip2_total = {}
        self._refresh_prefix_totals()

    def decay_connections(self, cutoff=0.1):
        """Decay and prune connections (call periodically)."""
        to_del = []
        for (a, b), conn in self.connections.items():
            conn['count'] = max(conn['count'] * self.decay, cutoff)
            if conn['count'] <= cutoff:
                to_del.append((a, b))
                self._connections_index[a].pop(b, None)
                self._connections_index[b].pop(a, None)
        for k in to_del:
            del self.connections[k]
        # Invalidate PPMI cache after decay
        self._ppmi_cache = None

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

        # Maintain per-CID index
        self._connections_index[cid_a][cid_b] = self.connections[key]
        self._connections_index[cid_b][cid_a] = self.connections[key]

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
        max_count = max(self.concept_freq.get(cid_a, 0),
                        self.concept_freq.get(cid_b, 0))
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

    def _ensure_ppmi(self):
        """Lazily build PPMI cache for all connection pairs."""
        if hasattr(self, '_ppmi_cache') and self._ppmi_cache is not None:
            return
        total_freq = max(sum(self.concept_freq.values()), 1)
        total_pairs = max(sum(c['count'] for c in self.connections.values()), 1)
        self._ppmi_cache = {}
        for (cid_a, cid_b), conn in self.connections.items():
            pair_p = conn['count'] / total_pairs
            marg_p = (self.concept_freq.get(cid_a, 1) / total_freq *
                      self.concept_freq.get(cid_b, 1) / total_freq)
            ppmi = max(math.log2(max(pair_p / max(marg_p, 1e-10), 1.0)), 0)
            self._ppmi_cache[(cid_a, cid_b)] = ppmi
            self._ppmi_cache[(cid_b, cid_a)] = ppmi

    def connections_of(self, cid, top_k=20, use_ppmi=False):
        """Get all concepts connected to a given concept.

        O(1) via per-CID index (not O(N) scan over all edges).

        Args:
            cid: concept ID
            top_k: max results
            use_ppmi: if True, sort by PPMI instead of raw connection strength

        Returns:
            [(connected_cid, {strength, type, ppmi?}), ...] sorted by strength
        """
        conns = self._connections_index.get(cid)
        if not conns:
            return []
        results = []
        if use_ppmi:
            self._ensure_ppmi()
        for other, conn in conns.items():
            max_c = max(self.concept_freq.get(cid, 1),
                        self.concept_freq.get(other, 1))
            strength = min(conn['count'] / max(max_c, 1), 1.0)
            dom_type = conn['types'].most_common(1)[0][0] if conn['types'] else 'related_to'
            entry = {'strength': strength, 'type': dom_type}
            if use_ppmi:
                entry['ppmi'] = self._ppmi_cache.get((cid, other), 0.0)
            results.append((other, entry))

        if use_ppmi:
            results.sort(key=lambda x: -x[1].get('ppmi', 0))
        else:
            results.sort(key=lambda x: -x[1]['strength'])
        return results[:top_k]

    def save(self, path):
        """Save to hybrid binary+JSON format."""
        clean = path[:-4] if path.endswith('.tmp') else path
        binary_path = clean.replace('.json', '.lattice.npz')
        # N-grams → npz jagged arrays
        npz_data = {}
        for n in sorted(self.ngrams.keys()):
            ng = self.ngrams[n]
            N = len(ng)
            if N == 0:
                continue
            prefixes = list(ng.keys())
            n_prefix_len = n - 1
            p_arr = np.zeros((N, n_prefix_len), dtype=np.int32)
            next_list = []
            count_list = []
            splits = [0]
            for i, pref in enumerate(prefixes):
                p_arr[i] = list(pref)
                for nxt, cnt in ng[pref].items():
                    next_list.append(nxt)
                    count_list.append(cnt)
                splits.append(len(next_list))
            npz_data[f'prefixes_{n}'] = p_arr
            npz_data[f'nexts_{n}'] = np.array(next_list, dtype=np.int32)
            npz_data[f'counts_{n}'] = np.array(count_list, dtype=np.int32)
            npz_data[f'splits_{n}'] = np.array(splits, dtype=np.int64)
        # concept_freq → npz
        cf_items = list(self.concept_freq.items())
        if cf_items:
            npz_data['cf_cids'] = np.array([c for c, _ in cf_items], dtype=np.int32)
            npz_data['cf_counts'] = np.array([v for _, v in cf_items], dtype=np.float32)
        # skip2 → npz jagged
        sk_items = list(self.skip2.items())
        if sk_items:
            sk_a, sk_n, sk_c = [], [], []
            sk_splits = [0]
            for a, counter in sk_items:
                sk_a.append(a)
                for b, cnt in counter.items():
                    sk_n.append(b)
                    sk_c.append(cnt)
                sk_splits.append(len(sk_n))
            npz_data['sk_a'] = np.array(sk_a, dtype=np.int32)
            npz_data['sk_n'] = np.array(sk_n, dtype=np.int32)
            npz_data['sk_c'] = np.array(sk_c, dtype=np.int32)
            npz_data['sk_splits'] = np.array(sk_splits, dtype=np.int64)
        # Connections → npz with type indices
        conn_items = list(self.connections.items())
        if conn_items:
            conn_a, conn_b, conn_cnt = [], [], []
            conn_typ_idx, conn_typ_cnt = [], []
            conn_typ_splits = [0]
            for (a, b), conn in conn_items:
                conn_a.append(a)
                conn_b.append(b)
                conn_cnt.append(conn['count'])
                for tname, tcnt in conn['types'].items():
                    tidx = RELATION_TYPES.get(tname, 10)  # default: related_to
                    conn_typ_idx.append(tidx)
                    conn_typ_cnt.append(tcnt)
                conn_typ_splits.append(len(conn_typ_idx))
            npz_data['conn_a'] = np.array(conn_a, dtype=np.int32)
            npz_data['conn_b'] = np.array(conn_b, dtype=np.int32)
            npz_data['conn_cnt'] = np.array(conn_cnt, dtype=np.int32)
            npz_data['conn_typ_idx'] = np.array(conn_typ_idx, dtype=np.uint8)
            npz_data['conn_typ_cnt'] = np.array(conn_typ_cnt, dtype=np.int32)
            npz_data['conn_typ_splits'] = np.array(conn_typ_splits, dtype=np.int64)
        tmp_bin = binary_path.replace('.npz', '.tmp.npz')
        np.savez_compressed(tmp_bin, **npz_data)
        os.replace(tmp_bin, binary_path)

        # Minimal metadata → compact JSON (also write to path for resume checks)
        meta = {'decay': self.decay, 'max_n': self.max_n}
        meta_path = path.replace('.json', '.meta.json')
        for p in (meta_path, path):
            with open(p + '.tmp', 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, separators=(',', ':'))
            os.replace(p + '.tmp', p)

        npz_kb = os.path.getsize(binary_path) / 1024
        meta_kb = os.path.getsize(path) / 1024
        print(f"  Saved SyntaxLattice ({npz_kb/1024:.0f}MB npz + {meta_kb/1024:.1f}MB meta) to {path}")

    def load(self, path, load_ngrams=True):
        """Load from hybrid binary+JSON format (also reads old monolithic JSON).

        Args:
            path: path to .json meta file
            load_ngrams: if False, skip n-gram/skip2 loading (faster, for generation-only use)
        """
        binary_path = path.replace('.json', '.lattice.npz')
        meta_path = path.replace('.json', '.meta.json')

        self.ngrams = {}
        self.connections = defaultdict(lambda: {'count': 0, 'types': Counter()})
        self._connections_index = defaultdict(dict)
        self.skip2 = defaultdict(Counter)
        self.concept_freq = defaultdict(float)
        self.decay = 0.999
        self.max_n = 4

        # Backward compatibility: load old monolithic JSON
        if not os.path.exists(binary_path) and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'ngrams' in data:
                for n_str, ngrams_data in data['ngrams'].items():
                    n = int(n_str)
                    ng = {}
                    for prefix_key, counter_data in ngrams_data.items():
                        prefix = tuple(int(c) for c in prefix_key.split())
                        ng[prefix] = defaultdict(float, {int(k): v for k, v in counter_data.items()})
                    self.ngrams[n] = ng
            if 'concept_freq' in data:
                self.concept_freq = defaultdict(float, {int(k): v for k, v in data['concept_freq'].items()})
            if 'skip2' in data:
                for k, v in data['skip2'].items():
                    self.skip2[int(k)] = Counter({int(ck): cv for ck, cv in v.items()})
            if 'connections' in data:
                for key_str, conn_data in data['connections'].items():
                    a, b = int(key_str.split(',')[0]), int(key_str.split(',')[1])
                    conn = {
                        'count': conn_data['count'],
                        'types': Counter(conn_data.get('types', {})),
                    }
                    self.connections[(a, b)] = conn
                    self._connections_index[a][b] = conn
                    self._connections_index[b][a] = conn
            print(f"  Loaded SyntaxLattice ({os.path.getsize(path)/1024**2:.0f}MB JSON legacy): "
                  f"{[len(v) for v in self.ngrams.values()]} prefixes, {len(self.connections)} connections")
            return self

        # New binary format
        if os.path.exists(binary_path):
            npz = np.load(binary_path, allow_pickle=False)
            # N-grams (skip for generation-only mode)
            if load_ngrams:
                for n_str in [k for k in npz.files if k.startswith('prefixes_')]:
                    n = int(n_str.split('_')[1])
                    p_arr = npz[f'prefixes_{n}']
                    nexts = npz[f'nexts_{n}']
                    counts = npz[f'counts_{n}']
                    splits = npz[f'splits_{n}']
                    N = len(p_arr)
                    ng = {}
                    for i in range(N):
                        pref = tuple(p_arr[i].tolist())
                        start, end = splits[i], splits[i + 1]
                        if end > start:
                            ng[pref] = Counter(dict(zip(nexts[start:end].tolist(),
                                                         counts[start:end].tolist())))
                        else:
                            ng[pref] = Counter()
                    self.ngrams[n] = ng
            # concept_freq
            if 'cf_cids' in npz.files:
                self.concept_freq = defaultdict(float, zip(npz['cf_cids'].tolist(),
                                                            npz['cf_counts'].tolist()))
            # skip2
            if load_ngrams and 'sk_a' in npz.files:
                sk_a = npz['sk_a']
                sk_n = npz['sk_n']
                sk_c = npz['sk_c']
                sk_splits = npz['sk_splits']
                for i in range(len(sk_a)):
                    a = int(sk_a[i])
                    start, end = sk_splits[i], sk_splits[i + 1]
                    if end > start:
                        self.skip2[a] = Counter(dict(zip(sk_n[start:end].tolist(),
                                                         sk_c[start:end].tolist())))
            # Connections
            if 'conn_a' in npz.files:
                conn_a = npz['conn_a']
                conn_b = npz['conn_b']
                conn_cnt = npz['conn_cnt']
                conn_typ_idx = npz.get('conn_typ_idx')
                conn_typ_cnt = npz.get('conn_typ_cnt')
                conn_typ_splits = npz.get('conn_typ_splits')
                K = len(conn_a)
                for i in range(K):
                    a, b = int(conn_a[i]), int(conn_b[i])
                    types = Counter()
                    if conn_typ_splits is not None:
                        start, end = int(conn_typ_splits[i]), int(conn_typ_splits[i + 1])
                        if end > start:
                            tidxs = conn_typ_idx[start:end]
                            tcnts = conn_typ_cnt[start:end]
                            for tidx, tcnt in zip(tidxs.tolist(), tcnts.tolist()):
                                tname = RELATION_NAMES.get(int(tidx), 'related_to')
                                types[tname] = int(tcnt)
                    conn = {'count': int(conn_cnt[i]), 'types': types}
                    self.connections[(a, b)] = conn
                    self._connections_index[a][b] = conn
                    self._connections_index[b][a] = conn

        # Load metadata from JSON
        self.decay = 0.999
        self.max_n = 4
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.decay = meta.get('decay', 0.999)
            self.max_n = meta.get('max_n', 4)

        print(f"  Loaded SyntaxLattice: {[len(v) for v in self.ngrams.values()]} prefixes, "
              f"{len(self.connections)} connections")
        self._refresh_prefix_totals()
        return self


if __name__ == '__main__':
    import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    import sentencepiece as spm
    from eva.symbolic.concept_space import ConceptSpace

    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'bpe_ru_146k.model'))

    cs = ConceptSpace.load(
        os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'concept_space.json'))

    print("Building SyntaxLattice from full corpus via SentencePiece...")
    lattice = SyntaxLattice()
    lattice.build(
        os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'full_corpus_ru_clean.txt'),
        sp,
        max_n=4,
    )

    lattice.save(os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'syntax_lattice.json'))
    print("Done.")
