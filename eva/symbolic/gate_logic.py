"""
GateLogic — многоуровневая система бинарных gates.
Уровни: meta → concept → word → BPE (+ sentence_type, paragraph)

Каждый gate = факт существования перехода (0/1). Без частот.
Gates только добавляются, никогда не удаляются.

valid_mask(context) = INTERSECTION всех gates на всех уровнях.
"""
import sys, json, numpy as np
from collections import defaultdict

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')


class GateLevel:
    """Один иерархический уровень: from_id → set[to_id]."""

    def __init__(self, name, n_from, n_to):
        self.name = name
        self.n_from = n_from
        self.n_to = n_to
        self.gates = {}       # from_id -> set[to_id]
        self._frozen = False

    def observe(self, from_id, to_id):
        """Добавить gate: переход from→to возможен."""
        if self._frozen:
            return
        if from_id not in self.gates:
            self.gates[from_id] = set()
        self.gates[from_id].add(to_id)

    def observe_sequence(self, ids):
        for i in range(len(ids) - 1):
            self.observe(ids[i], ids[i + 1])

    def valid_to(self, from_id):
        """Вернуть set разрешённых to_id. None = нет ограничений."""
        return self.gates.get(from_id)

    def has_gate(self, from_id, to_id):
        return from_id in self.gates and to_id in self.gates[from_id]

    def freeze(self):
        self._frozen = True

    def size(self):
        return sum(len(v) for v in self.gates.values())

    def save(self, path):
        data = {'name': self.name, 'n_from': self.n_from, 'n_to': self.n_to,
                'gates': {str(k): list(v) for k, v in self.gates.items()}}
        json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False)

    def load(self, path):
        data = json.load(open(path, 'r', encoding='utf-8'))
        self.name = data['name']
        self.n_from = data['n_from']
        self.n_to = data['n_to']
        self.gates = {int(k): set(v) for k, v in data['gates'].items()}
        self._frozen = True


class GateLogic:
    """
    Коллекция GateLevel + precomputed expansions в V-маски.
    
    Уровни:
      meta(12) → concept(48) → word(2442) → BPE(4101)
      s_type(5) → sentence type transition
      para(topic_set) → paragraph topic membership
    """

    def __init__(self, hv=None, ag=None, config=None):
        if config is None:
            from eva.symbolic.auto_config import AutoConfig
            config = AutoConfig()
        self.config = config
        self.hv = hv
        self.ag = ag
        self.V = config.vocab_size
        self.l1_offset = config.n_clusters
        self.n_clusters = config.n_clusters
        self.n_metas = config.n_metas

        self.levels = {
            'meta':    GateLevel('meta',    self.n_metas,    self.n_metas),
            'concept': GateLevel('concept', self.n_clusters, self.n_clusters),
            'word':    GateLevel('word',    self.V, self.V),
            'bpe':     GateLevel('bpe',     self.V, self.V),
        }
        self.s_type_level = GateLevel('s_type', config.gate_s_type_count, config.gate_s_type_count)

        # Para topic: set of concept IDs for current paragraph
        self.para_concepts = set()

        # Precomputed expansions: level_id → V-length bool mask
        # Filled by _build_expansions()
        self._expansions = {}
        self._expansions_built = False

    # --- Hierarchy helpers ---

    def meta_of(self, concept_cluster):
        if self.ag is not None:
            cid = self.l1_offset + concept_cluster
            mid = self.ag.cid_to_mid.get(cid)
            if mid is not None:
                return mid - self.l1_offset  # mid includes L1_OFFSET
            # Fallback: check tid_to_cid for a token in this concept
            for tid in self.ag.cid_to_tids.get(cid, []):
                tc = self.ag.tid_to_cid.get(tid)
                if tc is not None:
                    mid = self.ag.cid_to_mid.get(tc[0])
                    if mid is not None:
                        return mid - self.l1_offset
        return 0

    def concepts_in_meta(self, meta_id):
        if self.ag is not None:
            mid = self.l1_offset + meta_id
            cids = self.ag.mid_to_cids.get(mid, [])
            return [c - self.l1_offset for c in cids]
        return list(range(self.n_clusters))

    def tokens_of_concept(self, concept_cluster):
        """Вернуть список BPE token IDs для концепта."""
        if self.ag is not None:
            cid = self.l1_offset + concept_cluster
            # cid_to_tids: L1_OFFSET+cluster → list[tid]
            tids = self.ag.cid_to_tids.get(cid, [])
            if not tids and hasattr(self.ag, 'get_members'):
                tids = self.ag.get_members(cid)
            if isinstance(tids, list):
                return tids
            # Some AGs have numpy array
            return list(tids) if hasattr(tids, '__iter__') else []
        return []

    # --- Building expansions ---

    def _build_expansions(self):
        """Предвычислить level_id → V-length mask."""
        self._expansions = {}

        # meta → V tokens
        meta_masks = {}
        for m in range(self.n_metas):
            mask = np.zeros(self.V, dtype=bool)
            for c in self.concepts_in_meta(m):
                for tid in self.tokens_of_concept(c):
                    mask[tid] = True
            meta_masks[m] = mask
        self._expansions['meta'] = meta_masks

        # concept → V tokens
        conc_masks = {}
        for c in range(self.n_clusters):
            mask = np.zeros(self.V, dtype=bool)
            for tid in self.tokens_of_concept(c):
                mask[tid] = True
            conc_masks[c] = mask
        self._expansions['concept'] = conc_masks

        # word → V tokens (identity: one type-2 token)
        word_masks = {}
        for tid in range(self.V):
            mask = np.zeros(self.V, dtype=bool)
            mask[tid] = True
            word_masks[tid] = mask
        self._expansions['word'] = word_masks

        # bpe → V tokens (identity)
        bpe_masks = {}
        for tid in range(self.V):
            mask = np.zeros(self.V, dtype=bool)
            mask[tid] = True
            bpe_masks[tid] = mask
        self._expansions['bpe'] = bpe_masks

        self._expansions_built = True

    def _level_mask(self, level_name, from_id):
        """V-length mask for a single gate at a single level."""
        if not self._expansions_built:
            self._build_expansions()
        masks = self._expansions.get(level_name)
        if masks is None:
            return np.ones(self.V, dtype=bool)
        return masks.get(from_id, np.zeros(self.V, dtype=bool))

    # --- Observe ---

    def observe(self, text_hierarchy):
        """
        Текстовый иерархический парсер → gates на всех уровнях.
        """
        if not self._expansions_built:
            self._build_expansions()

        if not text_hierarchy.sentences and hasattr(text_hierarchy, 'parse'):
            text_hierarchy.parse()

        for sent in text_hierarchy.sentences:
            # BPE level: все пары соседних токенов
            toks = sent.tokens
            self.levels['bpe'].observe_sequence(toks)

            # Word level: только type-2 → type-2
            type2s = [t for t in toks if t < 4096 and self.hv.token_type[t] == 2]
            self.levels['word'].observe_sequence(type2s)

            # Concept level
            concepts = []
            for t in type2s:
                cid = self.ag.get_concept(t)
                if cid is not None:
                    concepts.append(cid - self.l1_offset)
            self.levels['concept'].observe_sequence(concepts)

            # Meta level
            metas = [self.meta_of(c) for c in concepts]
            self.levels['meta'].observe_sequence(metas)

        # Sentence type transitions
        s_types = [s.s_type for s in text_hierarchy.sentences if s.s_type]
        type_ids = [self.config.s_type_map.get(t, 0) for t in s_types]
        self.s_type_level.observe_sequence(type_ids)

        print(f"GateLogic observed {len(text_hierarchy.sentences)} sentences:")
        for name, level in self.levels.items():
            print(f"  {name:8s}: {level.size()} gates")
        print(f"  s_type : {self.s_type_level.size()} gates")

    def observe_online(self, ctx, chosen_token, content_token=-1):
        """
        Online learning: наблюдать один шаг генерации и добавить gates.
        Вызывать ПОСЛЕ выбора токена.
        """
        pos_in_word = ctx.get('pos_in_word', 0)
        prev_token = ctx.get('prev_token_id', -1)

        if pos_in_word == 0 and chosen_token != 3:
            self.levels['bpe'].observe(prev_token, chosen_token)
            tt = self.hv.token_type if self.hv else []
            if chosen_token < len(tt) and tt[chosen_token] == 2 and prev_token < len(tt) and tt[prev_token] == 2:
                self.levels['word'].observe(prev_token, chosen_token)
                cid_prev = self.ag.get_concept(prev_token) if self.ag else None
                cid_cur = self.ag.get_concept(chosen_token) if self.ag else None
                if cid_prev is not None and cid_cur is not None:
                    c_prev = cid_prev - self.l1_offset
                    c_cur = cid_cur - self.l1_offset
                    if 0 <= c_prev < self.n_clusters and 0 <= c_cur < self.n_clusters:
                        self.levels['concept'].observe(c_prev, c_cur)
                        m_prev = self.meta_of(c_prev)
                        m_cur = self.meta_of(c_cur)
                        self.levels['meta'].observe(m_prev, m_cur)

    def observe_sentence(self, tokens, prev_s_type=None, cur_s_type=None):
        """
        Наблюдать одно сгенерированное предложение → добавить gates на всех уровнях.
        Вызывать ПОСЛЕ завершения предложения (self-play).
        """
        # BPE level: все пары соседних токенов
        self.levels['bpe'].observe_sequence(tokens)

        # Word level: только type-2 → type-2
        type2s = [t for t in tokens if t < 4096 and self.hv.token_type[t] == 2]
        self.levels['word'].observe_sequence(type2s)

        # Concept level
        concepts = []
        for t in type2s:
            cid = self.ag.get_concept(t)
            if cid is not None:
                concepts.append(cid - self.l1_offset)
        self.levels['concept'].observe_sequence(concepts)

        # Meta level
        metas = [self.meta_of(c) for c in concepts]
        self.levels['meta'].observe_sequence(metas)

        # Sentence type transition
        if prev_s_type is not None and cur_s_type is not None:
            pid = self.config.s_type_map.get(prev_s_type, 0)
            cid = self.config.s_type_map.get(cur_s_type, 0)
            self.s_type_level.observe(pid, cid)

        # Mark expansions stale
        self._expansions_built = False

    # --- Inference ---

    def valid_mask(self, context, V=None):
        """
        INTERSECTION всех gates на всех уровнях.

        context dict:
          prev_token     — последний BPE токен
          prev_type2     — последний type-2
          prev_concept   — cluster idx последнего type-2
          prev_meta      — meta idx последнего type-2
          pos_in_word    — позиция в слове
          prev_s_type    — тип предыдущего предложения
          para_topic     — set концептов текущего абзаца
        """
        if V is None:
            V = self.V
        masks = []

        # BPE gate (применяется всегда)
        prev_token = context.get('prev_token')
        if prev_token is not None:
            valid = self.levels['bpe'].valid_to(prev_token)
            if valid is not None:
                m = np.zeros(V, dtype=bool)
                for tid in valid:
                    if tid < V:
                        m[tid] = True
                masks.append(m)

        # Word gate (только на piw=0, тип-к тип-2)
        if context.get('pos_in_word') == 0:
            prev_type2 = context.get('prev_type2')
            if prev_type2 is not None:
                valid = self.levels['word'].valid_to(prev_type2)
                if valid is not None:
                    m = np.zeros(V, dtype=bool)
                    for tid in valid:
                        if tid < V:
                            m[tid] = True
                    masks.append(m)

            # Concept gate
            prev_concept = context.get('prev_concept')
            if prev_concept is not None:
                valid = self.levels['concept'].valid_to(prev_concept)
                if valid is not None:
                    m = np.zeros(V, dtype=bool)
                    for c in valid:
                        for tid in self.tokens_of_concept(c):
                            m[tid] = True
                    masks.append(m)

            # Meta gate
            prev_meta = context.get('prev_meta')
            if prev_meta is not None:
                valid = self.levels['meta'].valid_to(prev_meta)
                if valid is not None:
                    m = np.zeros(V, dtype=bool)
                    for me in valid:
                        for c in self.concepts_in_meta(me):
                            for tid in self.tokens_of_concept(c):
                                m[tid] = True
                    masks.append(m)

        # Sentence type gate (на первом слове после EOS)
        prev_s_type = context.get('prev_s_type')
        if prev_s_type is not None:
            valid = self.s_type_level.valid_to(prev_s_type)
            if valid is not None:
                context['_valid_s_types'] = valid

        # Paragraph gate
        para = context.get('para_topic')
        if para:
            m = np.zeros(V, dtype=bool)
            for c in para:
                for tid in self.tokens_of_concept(c):
                    m[tid] = True
            masks.append(m)

        if not masks:
            return np.ones(V, dtype=bool)

        result = masks[0].copy()
        for m in masks[1:]:
            result &= m
        return result

    def save(self, dir_path):
        import os, json
        os.makedirs(dir_path, exist_ok=True)
        for name, level in self.levels.items():
            level.save(os.path.join(dir_path, f'{name}.json'))
        self.s_type_level.save(os.path.join(dir_path, 's_type.json'))
        
        # Write annotation JSON with labels
        ag = self.ag
        hv = self.hv
        annot = {
            'levels': {},
            'concept_labels': {},
            'meta_labels': {},
            'token_labels': {},
        }
        if ag:
            for cid in sorted(ag.cid_to_tids.keys()):
                c = cid - ag.L1_OFFSET
                name = ag.cid_label.get(cid, f'C{c}')
                mid = ag.cid_to_mid.get(cid)
                mname = ag.mid_label.get(mid, '?') if mid else '?'
                annot['concept_labels'][c] = {'name': name, 'meta': mname}
            for mid in sorted(ag.mid_to_cids.keys()):
                m = mid - ag.L2_OFFSET
                annot['meta_labels'][m] = ag.mid_label.get(mid, f'M{m}')
        if hv:
            for level_name, level in self.levels.items():
                labels = {}
                for from_id in sorted(level.gates.keys()):
                    if level_name == 'meta':
                        labels[from_id] = annot.get('meta_labels', {}).get(from_id, str(from_id))
                    elif level_name == 'concept':
                        labels[from_id] = annot.get('concept_labels', {}).get(from_id, str(from_id))
                    elif level_name == 'word':
                        tid_text = hv.decode([from_id]).strip() if hv else str(from_id)
                        labels[from_id] = tid_text
                    elif level_name == 'bpe':
                        tid_text = hv.decode([from_id]).strip() if hv else str(from_id)
                        labels[from_id] = tid_text
                annot['levels'][level_name] = {
                    'n_from': level.n_from,
                    'n_to': level.n_to,
                    'gate_count': level.size(),
                    'from_labels': labels,
                }
        # s_type labels
        s_label = {}
        for from_id in sorted(self.s_type_level.gates.keys()):
            name = self.config.s_type_map_rev.get(from_id, str(from_id))
            s_label[from_id] = name
        annot['levels']['s_type'] = {
            'n_from': self.s_type_level.n_from,
            'n_to': self.s_type_level.n_to,
            'gate_count': self.s_type_level.size(),
            'from_labels': s_label,
        }
        with open(os.path.join(dir_path, 'annotation.json'), 'w', encoding='utf-8') as f:
            json.dump(annot, f, ensure_ascii=False, indent=2)

    def load(self, dir_path):
        import os
        for name in self.levels:
            path = os.path.join(dir_path, f'{name}.json')
            if os.path.exists(path):
                self.levels[name].load(path)
        s_path = os.path.join(dir_path, 's_type.json')
        if os.path.exists(s_path):
            self.s_type_level.load(s_path)
        self._build_expansions()


if __name__ == '__main__':
    from eva.symbolic.text_hierarchy import TextHierarchy
    from eva.symbolic.bpe_tokenizer import HierarchicalVocab
    from eva.symbolic.association_graph import AssociationGraph
    from eva.symbolic.heads import HeadsEnsemble

    hv = HierarchicalVocab()
    heads = HeadsEnsemble('real_data/v8/heads_meta.pkl', 'real_data/v8')
    ag = AssociationGraph(n_clusters=48, n_metas=12)
    ag.build(heads.log_prob_csr, hv.token_type, decode_fn=hv.decode)

    th = TextHierarchy(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru.txt', hv)
    sents = th.parse()
    th.analyze_concepts(ag)

    gl = GateLogic(hv=hv, ag=ag)
    gl.observe(th)
    gl.save(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\gates')

    # Test valid_mask
    ctx = {'prev_token': 475, 'pos_in_word': 0, 'prev_type2': 475,
           'prev_concept': 1, 'prev_meta': 1}
    mask = gl.valid_mask(ctx)
    print(f"\nTest mask for prev_token=475 (сказал): {int(mask.sum())} valid tokens")
