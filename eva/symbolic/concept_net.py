"""ConceptNet Skeleton — builds concept hierarchy from ConceptNet data.

Each concept is a semantic field:
  - anchor: корень/лемма (главное слово концепта)
  - satellites: морф. варианты, синонимы, related terms
  - relations: связи с другими концептами (is_a, antonym, etc.)

Концепты образуют "скелет" — объективную семантическую структуру,
от которой модель отталкивается при генерации.
"""

import re, math, json, os, pickle
import numpy as np
from collections import defaultdict, Counter

CN_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'conceptnet', 'conceptnet_ru.txt')
SKELETON_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'concept_skeleton.pkl')

REL_MAP = {
    'это': 'is_a',
    'то же, что': 'synonym',
    'форма слова': 'form_of',
    'противоположно': 'antonym',
    'отличается от': 'distinct_from',
    'похоже на': 'similar_to',
    'часть': 'part_of',
    'способ': 'manner_of',
    'связан с': 'related_to',
    'относится к': 'related_to',
    'происходит от': 'derived_from',
    'находится в': 'located_at',
    'используется для': 'used_for',
    'может': 'capable_of',
    'имеет': 'has_a',
    'сделан из': 'made_of',
    'вызывает': 'causes',
    'символизирует': 'symbol_of',
}
# longest first for prefix matching
RU_RELS = sorted(REL_MAP.keys(), key=len, reverse=True)


class ConceptSkeleton:
    """Скелет концептов — загружает ConceptNet, строит иерархию.

    Architecture:
      ConceptNode {
          cid: int           — уникальный ID концепта
          anchor: str        — лемма/корень (главное слово)
          satellites: [str]  — все формы, синонимы этого концепта
          relations: {rel_type: [target_cid]}  — связи с другими концептами
          vector: np.array   — центроид концепта (усреднение векторов слов)
      }

    Levels:
      L0: слова (word → cid mapping)
      L1: концепты (семантические поля из ConceptNet)
      L2: мета-концепты (кластеры концептов через граф связей)
    """

    def __init__(self):
        self.concepts = {}       # cid → ConceptNode
        self.word_to_cid = {}    # word → cid (каждое слово в одном концепте)
        self.cid_to_words = {}   # cid → [word, ...]
        self.relations = defaultdict(list)  # (cid_i, cid_j) → [rel_types]

        self.n_concepts = 0
        self.meta_labels = {}    # cid → meta_id (L2 кластеризация)
        self.cid_to_metas = {}   # meta_id → [cids]

        # Иерархия: is_a chains
        self.parents = {}        # cid → parent_cid (ближайший is_a родитель)
        self.children = {}       # cid → [child_cids]

    def build(self, cn_path=None):
        """Build concept skeleton from ConceptNet file."""
        cn_path = cn_path or CN_PATH
        print("Parsing ConceptNet...")
        triples = self._parse_conceptnet(cn_path)
        print(f"  Parsed {len(triples)} triples")

        # Step 1: Build concept groups from form_of (морфологические группы)
        self._build_form_groups(triples)
        print(f"  Built {self.n_concepts} form groups")

        # Step 2: Build concept graph — synonym/is_a create CONNECTIONS,
        # NOT mergers. Each concept retains its identity.
        self._build_concept_graph(triples)

        # Step 3: Build is_a hierarchy (parent/child chains)
        self._build_hierarchy(triples)

        return self

    def _parse_conceptnet(self, path):
        triples = []
        stats = Counter()

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = re.match(r'^(.+?)\s*[\u2014\u2013\-]\s+(.+?)\.\s*$', line)
                if not m:
                    stats['no_match'] += 1
                    continue
                start = m.group(1).strip().lower()
                rest = m.group(2).strip()

                matched = False
                for ru_rel in RU_RELS:
                    if rest.startswith(ru_rel + ' '):
                        end = rest[len(ru_rel)+1:].strip().lower()
                        en_rel = REL_MAP[ru_rel]
                        triples.append((start, en_rel, end))
                        stats[en_rel] += 1
                        matched = True
                        break
                if not matched:
                    stats['unparsed_rest'] += 1

        return triples

    def _build_form_groups(self, triples):
        """Build initial concept groups from form_of relations.

        Each form_of(X, Y) means X is a grammatical form of Y.
        Y is the lemma (canonical form), X is a morphological variant.

        Strategy: DIRECT mapping only — no transitive chain resolution.
        Each word maps to exactly one lemma (its direct form_of parent).
        Words sharing the same lemma = same concept.

        If a word has NO form_of entry, its concept is the word itself
        (singleton, added later via _add_orphan_word).
        """

        # Direct lemma map: word → its immediate lemma
        # form_of(X, Y) → X is a form of Y → lemma_map[X] = Y
        lemma_map = {}
        lemma_counter = Counter()

        for s, rel, e in triples:
            if rel == 'form_of':
                lemma_map[s] = e
                lemma_counter[e] += 1

        # Group words by their direct lemma
        lemma_to_words = defaultdict(set)
        for word, lemma in lemma_map.items():
            lemma_to_words[lemma].add(word)

        # Also include lemmas that appear as targets but never as sources
        # These are concept anchors even if they have no "is a form of" entry
        for lemma in lemma_counter:
            lemma_to_words[lemma].add(lemma)

        # Create concepts
        next_cid = 0
        for lemma, words in lemma_to_words.items():
            words_list = sorted(words)
            cid = next_cid
            next_cid += 1
            self.concepts[cid] = {
                'cid': cid,
                'anchor': lemma,
                'satellites': [w for w in words_list if w != lemma],
                'relations': defaultdict(list),
                'vector': None,
                'size': len(words_list),
            }
            for w in words_list:
                self.word_to_cid[w] = cid
            self.cid_to_words[cid] = words_list

        self.n_concepts = next_cid

    def _add_orphan_word(self, word):
        """Create a singleton concept for a word with no form_of entry."""
        cid = self.n_concepts
        self.n_concepts += 1
        self.concepts[cid] = {
            'cid': cid,
            'anchor': word,
            'satellites': [],
            'relations': defaultdict(list),
            'vector': None,
            'size': 1,
        }
        self.word_to_cid[word] = cid
        self.cid_to_words[cid] = [word]
        return cid

    def _ensure_word_has_concept(self, word):
        """Ensure word has a concept (add orphan if needed)."""
        if word not in self.word_to_cid:
            self._add_orphan_word(word)
        return self.word_to_cid[word]

    def _build_concept_graph(self, triples):
        """Build relations between concepts from all triples.
        Words without concept groups get singleton concepts.
        All relation types except form_of become concept graph edges."""
        for s, rel, e in triples:
            if rel == 'form_of':
                continue  # already handled in form groups
            cid_s = self._ensure_word_has_concept(s)
            cid_e = self._ensure_word_has_concept(e)
            if cid_s != cid_e:
                pair = (cid_s, cid_e) if cid_s < cid_e else (cid_e, cid_s)
                if rel not in self.relations[pair]:
                    self.relations[pair].append(rel)

        # Store relations per concept
        for (ci, cj), rels in self.relations.items():
            for rel in rels:
                self.concepts[ci]['relations'][rel].append(cj)
                self.concepts[cj]['relations'][rel].append(ci)

    def _build_hierarchy(self, triples):
        """Build is_a hierarchy: concept → parent concept."""
        for s, rel, e in triples:
            if rel == 'is_a':
                cid_s = self.word_to_cid.get(s)
                cid_e = self.word_to_cid.get(e)
                if cid_s is not None and cid_e is not None and cid_s != cid_e:
                    self.parents[cid_s] = cid_e
                    if cid_e not in self.children:
                        self.children[cid_e] = []
                    if cid_s not in self.children[cid_e]:
                        self.children[cid_e].append(cid_s)

    def build_meta_concepts(self):
        """L2 clustering: Louvain on concept relation graph."""
        try:
            import networkx as nx
            from community import community_louvain
        except ImportError:
            print("  networkx/python-louvain not available, skipping meta-concepts")
            return

        G = nx.Graph()
        for cid in self.concepts:
            G.add_node(cid)
        for (ci, cj), _ in self.relations.items():
            G.add_edge(ci, cj)

        if G.number_of_edges() == 0:
            print("  No concept relations, L2 skipped")
            return

        partition = community_louvain.best_partition(G)
        self.meta_labels = partition
        self.cid_to_metas = defaultdict(list)
        for cid, mid in partition.items():
            self.cid_to_metas[mid].append(cid)

        print(f"  L2: {len(self.cid_to_metas)} meta-concepts from {G.number_of_nodes()} concepts")

    def compute_vectors(self, word_vectors=None, dim=256):
        """Compute concept centroid vectors.
        If word_vectors provided: avg of word vectors in concept.
        Otherwise: random initialization (for bootstrapping)."""
        rng = np.random.RandomState(42)

        for cid, concept in self.concepts.items():
            words = self.cid_to_words[cid]
            if word_vectors:
                # Average available word vectors
                vecs = [word_vectors[w] for w in words if w in word_vectors]
                if vecs:
                    concept['vector'] = np.mean(vecs, axis=0)
                    norm = np.linalg.norm(concept['vector'])
                    if norm > 0:
                        concept['vector'] /= norm
                else:
                    concept['vector'] = rng.randn(dim).astype(np.float32)
                    concept['vector'] /= np.linalg.norm(concept['vector'])
            else:
                concept['vector'] = rng.randn(dim).astype(np.float32)
                concept['vector'] /= np.linalg.norm(concept['vector'])

    def save(self, path=None):
        """Save skeleton to disk."""
        path = path or SKELETON_PATH
        data = {
            'concepts': {str(cid): c for cid, c in self.concepts.items()},
            'word_to_cid': self.word_to_cid,
            'cid_to_words': {str(cid): ws for cid, ws in self.cid_to_words.items()},
            'relations': {f'{ci},{cj}': rels for (ci, cj), rels in self.relations.items()},
            'meta_labels': {str(cid): mid for cid, mid in self.meta_labels.items()},
            'cid_to_metas': {str(mid): cids for mid, cids in self.cid_to_metas.items()},
            'parents': {str(c): p for c, p in self.parents.items()},
            'children': {str(c): ch for c, ch in self.children.items()},
            'n_concepts': self.n_concepts,
        }
        # Convert numpy arrays to lists for JSON
        for cid_str in data['concepts']:
            vec = data['concepts'][cid_str]['vector']
            if vec is not None:
                data['concepts'][cid_str]['vector'] = vec.tolist()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"  Saved skeleton to {path}")

    def load(self, path=None):
        """Load skeleton from disk."""
        path = path or SKELETON_PATH
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.concepts = {int(cid): c for cid, c in data['concepts'].items()}
        self.word_to_cid = data['word_to_cid']
        self.cid_to_words = {int(cid): ws for cid, ws in data['cid_to_words'].items()}
        self.relations = {}
        for key, rels in data['relations'].items():
            ci, cj = key.split(',')
            self.relations[(int(ci), int(cj))] = rels
        self.meta_labels = {int(cid): mid for cid, mid in data['meta_labels'].items()}
        self.cid_to_metas = {int(mid): cids for mid, cids in data['cid_to_metas'].items()}
        self.parents = {int(c): p for c, p in data['parents'].items()}
        self.children = {int(c): ch for c, ch in data['children'].items()}
        self.n_concepts = data['n_concepts']

        # Restore vectors from lists
        for cid in self.concepts:
            vec = self.concepts[cid]['vector']
            if vec is not None:
                self.concepts[cid]['vector'] = np.array(vec, dtype=np.float32)
        return self

    def concept_of(self, word):
        """Get concept ID for a word. Returns None if unknown."""
        return self.word_to_cid.get(word.lower())

    def anchor_of(self, word):
        """Get anchor word for a word's concept."""
        cid = self.concept_of(word)
        if cid is not None:
            return self.concepts[cid]['anchor']
        return None

    def satellites_of(self, word):
        """Get all satellite words for a word's concept."""
        cid = self.concept_of(word)
        if cid is not None:
            return self.concepts[cid]['satellites']
        return []

    def related_concepts(self, cid, rel_type=None):
        """Get concepts related to given concept."""
        if cid not in self.concepts:
            return []
        result = set()
        for (ci, cj), rels in self.relations.items():
            if ci == cj:
                continue
            if ci == cid:
                if rel_type is None or rel_type in rels:
                    result.add(cj)
            elif cj == cid:
                if rel_type is None or rel_type in rels:
                    result.add(ci)
        return list(result)

    def get_connections(self, cid, max_depth=2):
        """Get all concepts within max_depth steps from cid in the relation graph."""
        connected = {cid}
        frontier = {cid}
        for _ in range(max_depth):
            new_frontier = set()
            for c in frontier:
                for rel_cid in self.related_concepts(c):
                    if rel_cid not in connected:
                        new_frontier.add(rel_cid)
            frontier = new_frontier
            connected.update(frontier)
        return connected - {cid}

    def top_concepts(self, k=20):
        """Get the largest concepts (by number of words)."""
        sorted_cids = sorted(self.concepts.keys(),
                            key=lambda c: self.concepts[c]['size'], reverse=True)
        return [(c, self.concepts[c]['anchor'], self.concepts[c]['size'])
                for c in sorted_cids[:k]]

    def word_similarity(self, w1, w2):
        """Concept-level similarity between two words.
        Same concept = 1.0, related concepts = 0.5, else 0.0."""
        c1 = self.concept_of(w1)
        c2 = self.concept_of(w2)
        if c1 is None or c2 is None:
            return 0.0
        if c1 == c2:
            return 1.0
        if c2 in self.related_concepts(c1):
            return 0.5
        return 0.0

    def sentence_concepts(self, words):
        """Map a list of words (tokens) to their concept sequence.
        Returns: [(word, cid, is_anchor), ...]
        Unknown words get cid=None."""
        result = []
        for w in words:
            wl = w.strip('.,!?;:()[]\'\"«»').lower()
            cid = self.concept_of(wl)
            is_anchor = False
            if cid is not None:
                is_anchor = (self.concepts[cid]['anchor'] == wl)
            result.append((w, cid, is_anchor))
        return result


if __name__ == '__main__':
    sk = ConceptSkeleton()
    sk.build()
    print(f"\nTop 20 concepts:")
    for cid, anchor, size in sk.top_concepts(20):
        print(f"  [{cid:4d}] {anchor:15s} ({size:4d} words)")
    print(f"\nTotal concepts: {sk.n_concepts}")
    if sk.relations:
        print(f"Concept relations: {len(sk.relations)}")
    if sk.parents:
        print(f"Hierarchy nodes: {len(sk.parents)}")

    # Test some queries
    for test_word in ['собака', 'собаки', 'собакой', 'армия', 'война']:
        cid = sk.concept_of(test_word)
        if cid is not None:
            c = sk.concepts[cid]
            print(f"  {test_word} → concept [{cid}] anchor='{c['anchor']}', "
                  f"satellites={c['satellites'][:5]}")
            rels = sk.related_concepts(cid)
            if rels:
                rel_anchors = [sk.concepts[r]['anchor'] for r in rels[:5]]
                print(f"    related: {rel_anchors}")

    # Build meta-concepts
    sk.build_meta_concepts()
    if sk.cid_to_metas:
        print(f"\nMeta-concepts: {len(sk.cid_to_metas)}")
        for mid, cids in sorted(sk.cid_to_metas.items())[:5]:
            anchors = [sk.concepts[c]['anchor'] for c in cids[:8]]
            print(f"  M{mid}: {anchors}")
