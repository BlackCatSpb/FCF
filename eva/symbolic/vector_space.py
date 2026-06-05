import sys, math, numpy as np
from collections import defaultdict

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.generation_loop import select_token
from eva.symbolic.structural_rules import StructuralRules
from eva.symbolic.gate_logic import GateLogic
from eva.symbolic.pattern_learner import PatternLearner




class VectorSpace:
    """
    Непрерывное пространство concept vectors.
    Токен → SVD-вектор (32-dim). Близость векторов = семантическая близость.
    Никаких K-means. Порог активации регулирует ширину "образа".
    """
    
    def __init__(self, assoc_graph, hv):
        self.ag = assoc_graph
        self.hv = hv
        self.dim = None
        
        # Build vector lookup: type-2 token → SVD vector
        self.token_vectors = {}  # tid -> (32,) ndarray
        self.token_inv = {}      # tid -> bool (is in SVD space)
        
        if self.ag.starter_embeddings is not None:
            self.dim = self.ag.starter_embeddings.shape[1]
            for i, tid in enumerate(self.ag.starter_list):
                self.token_vectors[tid] = self.ag.starter_embeddings[i]
                self.token_inv[tid] = True
        else:
            print("WARNING: no SVD embeddings in AssociationGraph")
    
    def has_vector(self, tid):
        """Проверяет, есть ли у токена SVD-вектор."""
        return tid in self.token_vectors
    
    def get_vector(self, tid):
        """SVD-вектор токена (32-dim)."""
        return self.token_vectors.get(tid)
    
    def normalize(self, v):
        """L2-normalize vector."""
        n = np.linalg.norm(v)
        if n > 1e-10:
            return v / n
        return v
    
    def similarity(self, tid_a, tid_b):
        """Косинусная близость между двумя токенами."""
        va = self.token_vectors.get(tid_a)
        vb = self.token_vectors.get(tid_b)
        if va is None or vb is None:
            return 0.0
        return float(np.dot(self.normalize(va), self.normalize(vb)))
    
    def topk_similar(self, tid, k=20, exclude_set=None):
        """
        Топ-k токенов, ближайших к tid в SVD-пространстве.
        Возвращает [(tid, cosine_sim)].
        """
        v = self.token_vectors.get(tid)
        if v is None:
            return []
        vn = self.normalize(v)
        
        sims = []
        for other_tid, other_v in self.token_vectors.items():
            if other_tid == tid:
                continue
            if exclude_set and other_tid in exclude_set:
                continue
            sim = float(np.dot(vn, self.normalize(other_v)))
            sims.append((other_tid, sim))
        
        sims.sort(key=lambda x: -x[1])
        return sims[:k]
    
    def topk_to_vector(self, vector, k=20, exclude_set=None):
        """
        Топ-k токенов, ближайших к произвольному вектору.
        vector: (32,) ndarray
        Returns: [(tid, cosine_sim)]
        """
        vn = self.normalize(vector)
        sims = []
        for tid, tv in self.token_vectors.items():
            if exclude_set and tid in exclude_set:
                continue
            sim = float(np.dot(vn, self.normalize(tv)))
            sims.append((tid, sim))
        sims.sort(key=lambda x: -x[1])
        return sims[:k]
    
    def word_vector(self, word):
        """Найти SVD-вектор для слова."""
        bpe = self.hv.encode(word)
        if not bpe:
            return None
        first = bpe[0]
        first_d = self.hv.decode([first]).strip()
        for t in range(4096):
            if t < len(self.hv.token_type) and self.hv.token_type[t] == 2:
                if self.hv.decode([t]).strip() == first_d:
                    return self.get_vector(t)
        return None
    
    # ---- Composition ----
    def compose(self, vectors, weights=None):
        """
        Композиция: взвешенная интерполяция векторов.
        """
        if not vectors:
            return None
        if weights is None:
            weights = [1.0 / len(vectors)] * len(vectors)
        w = np.array(weights, dtype=np.float32)
        w = w / w.sum()
        result = np.zeros_like(vectors[0])
        for v, weight in zip(vectors, w):
            result += v * weight
        return result


class VectorGenerator:
    """
    Генерация через векторное пространство.
    
    Принцип: на piw=0 семантический переход через SVD-близость,
    на piw>=1 — head-продолжение слова (как обычно).
    """
    
    def __init__(self, heads_obj, assoc_graph, hv, rule_extractor=None):
        self.heads = heads_obj
        self.ag = assoc_graph
        self.hv = hv
        self.vs = VectorSpace(assoc_graph, hv)
        
        # Rules
        self.rules = []
        if rule_extractor is not None:
            self.rules = getattr(rule_extractor, 'affixation_rules', []) + \
                         getattr(rule_extractor, 'syntax_rules', [])
        
        self.V = 4101
        self.tt = hv.token_type
        
        # Blocked tokens
        self.REPLACEMENT_TOKENS = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
            120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
            130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
            140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
            150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
            160, 228, 229, 230, 231, 232, 233, 234, 235, 236,
            237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
            247, 248, 249, 250, 251, 252, 253, 254, 255, 256,
            257, 258, 259, 260, 261, 262, 272, 274, 277, 298,
            299, 334, 1498}
        self.IGNORED = {0, 1, 2, 4, 5}
        self.BANNED = {3}
        
        # Pattern Learner (самоорганизация шаблонов)
        self._pattern_learner = None
        self._pattern_learner_trained = False
        # Semantic coherence weight
        self.sem_weight = 5.0
        self.sem_boost_count = 5  # only boost top-5 SVD-nearest tokens
        
        # Binary structural matrix: heads → 1/0 (возможно/невозможно)
        self._build_structural_matrix()
        
        # StructuralRules for multi-level constraints
        self.structural_rules = None
        
        # GateLogic: multi-level binary gates
        self.gates = None
        
        # Cross-sentence context tracking
        self._prev_sentence_last_concept = -1   # cluster idx (0-47)
        self._prev_sentence_type = 'start'
        
        # Precomputed: concept -> token_ids for fast lookup
        self._concept_to_tids = {}  # cluster_idx -> set[tid]
        self._build_concept_to_tids()
        
        # Last content word tracking (for semantic context at word boundaries)
        self._last_content_tid = -1
        
        # ---- Word accumulation (for full-word semantic anchoring) ----
        self._current_word_tokens = []  # tokens of current in-progress word
        self._full_word_anchor = None   # (type2_tid, word_text) last complete word
        
        # ---- ConceptNet-refined vectors ----
        self._refined_vectors_path = None
        
        # ---- Word completion prefix map (type-2→type-3 semantic continuation) ----
        self._prefix_type3_set = {} # prefix_tid -> set[type3_tid]
        self._prefix_type2 = set()  # type-2 tokens that are prefixes (have type-3 continuations)
        self._build_word_prefix_map()

        # ---- Word completion state ----
        self._word_prefix = -1        # type-2 prefix of current in-progress word
        self._current_word_type3s = set()  # type-3 tokens already emitted for current word
    
    def _build_structural_matrix(self):
        """Строит структурную матрицу: возможные transitions между type-2 токенами.
        Heads = не частоты, а факт существования перехода в данных.
        Каждый переход, который хоть раз встретился — структурно возможен.
        """
        self.structural = {}  # tid_in -> set of tid_out that can follow
        
        csr = self.heads.log_prob_csr
        n_structural = 0
        for src in range(self.V):
            if src >= csr.shape[0]:
                continue
            row = csr[src].tocoo()
            targets = set()
            for col in row.col:
                if col < self.V:
                    targets.add(int(col))
            if targets:
                self.structural[src] = targets
                n_structural += len(targets)
        
        print("Structural matrix: %d transitions, %.1f avg per token" % (
            n_structural, n_structural / max(len(self.structural), 1)))
        
        # Detect prefix type-2 tokens: those that have type-3 continuations
        self._prefix_type2 = set()
        for tid2 in range(self.V):
            if tid2 >= len(self.tt) or self.tt[tid2] != 2:
                continue
            follow = self.structural.get(tid2, set())
            for t in follow:
                if t < len(self.tt) and self.tt[t] == 3:
                    self._prefix_type2.add(tid2)
                    break
        print(f"  Prefix type-2: {len(self._prefix_type2)} / {sum(1 for t in range(self.V) if t < len(self.tt) and self.tt[t] == 2)}")
        
        # EOS fix: add EOS as possible transition from every source
        # so sentences can end after any word (structural decision, not frequency-based)
        for src in range(self.V):
            if src in self.structural:
                self.structural[src].add(3)
            else:
                self.structural[src] = {3}
        
        # EOS fix: add transitions FROM EOS to all type-2 sentence starters
        # so new sentences can begin after EOS
        eos_next = set()
        for tid in range(self.V):
            if tid < len(self.tt) and self.tt[tid] == 2:
                eos_next.add(tid)
        self.structural[3] = eos_next
        print("  EOS patched: can end any word, can start new sentences")
    
    def _build_concept_to_tids(self):
        """Precompute cluster_idx -> set[tid] for fast cross-sentence lookup."""
        self._concept_to_tids = {}
        for tid in range(self.V):
            if tid >= len(self.tt) or self.tt[tid] != 2:
                continue
            cid = self.ag.get_concept(tid)
            if cid is not None:
                cluster = cid - self.ag.L1_OFFSET
                if 0 <= cluster < self.ag.n_clusters:
                    self._concept_to_tids.setdefault(cluster, set()).add(tid)
    
    def _build_word_prefix_map(self):
        """
        Build prefix data for semantic word continuation at piw>=1.
        Uses structural matrix to identify which type-3 tokens follow each prefix.
        """
        self._prefix_type3_set = {}  # prefix_tid -> set[type3_tid]
        for tid in range(self.V):
            if tid >= len(self.tt) or not self._is_content_token(tid):
                continue
            follow = self.structural.get(tid, set())
            # Only type-3 that decode to pure letters (no punctuation, numbers)
            type3 = set()
            for t in follow:
                if t < len(self.tt) and self.tt[t] == 3:
                    txt = self.hv.decode([t]).strip()
                    if txt and txt[0].isalpha():
                        type3.add(t)
            if type3:
                self._prefix_type3_set[tid] = type3
        
        n_prefixes = len(self._prefix_type3_set)
        n_paths = sum(len(v) for v in self._prefix_type3_set.values())
        print(f"  Prefix type-3 map: {n_prefixes} prefixes, {n_paths} type-3 continuations")
    
    def _prefix_boost_scores(self, ctx, valid_mask, content_token=-1):
        """
        Boost prefix type-2 tokens at piw=0 based on SVD similarity between
        the prefix and the last content word. This lets the model generate
        morphologically complex words via prefix+continuation instead of
        only atomic type-2 tokens.
        """
        scores = np.zeros(self.V, dtype=np.float32)
        pos_in_word = ctx.get('pos_in_word', -1)
        if pos_in_word != 0 or content_token < 0:
            return scores
        if not self.vs.has_vector(content_token):
            return scores

        # Check if the PREVIOUS generated token is a prefix
        # If so, also boost its type-3 continuations as alternatives to new words
        prev_tid = ctx.get('token_id', -1)
        if prev_tid >= 0 and prev_tid in self._prefix_type3_set:
            # We're at piw=0 AFTER generating a prefix.
            # The prefix _is_ the current token. Boost type-3 continuations
            # so that completing the word is competitive with starting a new one.
            prefix_tid = prev_tid
            if self.vs.has_vector(prefix_tid):
                sim = self.vs.similarity(prefix_tid, content_token)
                if sim > 0:
                    for tid3 in self._prefix_type3_set[prefix_tid]:
                        if valid_mask[tid3]:
                            scores[tid3] = sim * 8.0

        for prefix_tid in self._prefix_type3_set:
            if not valid_mask[prefix_tid]:
                continue
            if self.vs.has_vector(prefix_tid):
                sim = self.vs.similarity(prefix_tid, content_token)
                if sim > 0:
                    scores[prefix_tid] = sim * 4.0

        return scores

    def _continuation_semantic_scores(self, ctx, content_token, valid_mask):
        """
        Semantic scores for type-3 continuations (piw>=1).
        At piw>=1, the prefix token has already been selected.
        Score each valid type-3 continuation of this prefix by SVD similarity
        between the prefix and the last content word.
        """
        scores = np.zeros(self.V, dtype=np.float32)
        pos_in_word = ctx.get('pos_in_word', -1)
        if pos_in_word < 1 or content_token < 0:
            return scores
        if not self._current_word_tokens:
            return scores
        
        prefix_tid = self._current_word_tokens[0]
        if not self._is_content_token(prefix_tid):
            return scores
        if not self.vs.has_vector(prefix_tid) or not self.vs.has_vector(content_token):
            return scores
        
        sim = self.vs.similarity(prefix_tid, content_token)
        if sim <= 0:
            return scores
        
        valid_type3 = self._prefix_type3_set.get(prefix_tid, set())
        for tid3 in valid_type3:
            if valid_mask[tid3]:
                scores[tid3] = sim * 1.0
        
        return scores
    
    def load_structural_rules(self, path):
        """Load StructuralRules from JSON."""
        if not path:
            return
        self.structural_rules = StructuralRules()
        self.structural_rules.load(path)

    def load_refined_vectors(self, path):
        """Load ConceptNet-refined SVD vectors. Overrides per-token vectors in vs."""
        import pickle, os
        if not path or not os.path.exists(path):
            print(f'WARNING: refined vectors not found at {path}')
            return
        with open(path, 'rb') as f:
            refined = pickle.load(f)
        n_overridden = 0
        for tid, vec in refined.items():
            if tid in self.vs.token_vectors:
                self.vs.token_vectors[tid] = vec.astype(np.float32)
                n_overridden += 1
        self._refined_vectors_path = path
        print(f'Refined vectors loaded: {n_overridden}/{len(refined)} overridden')

    def _concepts_for_prev(self, prev_concept_cluster):
        """Get valid next concept clusters for cross-sentence transition."""
        if self.structural_rules is None or prev_concept_cluster < 0:
            return None
        valid_next = self.structural_rules.get_valid_next_concepts(prev_concept_cluster)
        if not valid_next:
            return None
        return valid_next
    
    def load_gates(self, gates_dir):
        """Load GateLogic from saved gates directory."""
        self.gates = GateLogic(hv=self.hv, ag=self.ag)
        self.gates.load(gates_dir)
        print(f"  Gates loaded: {sum(self.gates.levels[l].size() for l in self.gates.levels)}")
    
    def _gate_context(self, ctx, prev_token, content_token):
        """Построить multi-level context для GateLogic.valid_mask()."""
        context = {
            'prev_token': prev_token,
            'pos_in_word': ctx.get('pos_in_word', -1),
        }
        if content_token >= 0 and self._is_content_token(content_token):
            context['prev_type2'] = content_token
            cid = self.ag.get_concept(content_token)
            if cid is not None:
                cluster = cid - self.ag.L1_OFFSET
                if 0 <= cluster < self.ag.n_clusters:
                    context['prev_concept'] = cluster
                    if self.gates is not None:
                        context['prev_meta'] = self.gates.meta_of(cluster)
        if prev_token == 3 and self._prev_sentence_type != 'start':
            s_type_map = {'statement': 0, 'question': 1, 'exclamation': 2,
                          'dialogue': 3, 'french': 4}
            context['prev_s_type'] = s_type_map.get(self._prev_sentence_type, 0)
            if self._sentences_in_paragraph > 0 and self._paragraph_topic:
                context['para_topic'] = self._paragraph_topic
        return context
    
    def _gate_scores(self, ctx, gate_ctx, valid_mask_arr):
        """
        Gate-based structural scores: INTERSECTION всех gates на всех уровнях.
        Заменяет _structural_scores когда gates загружены.
        """
        scores = np.full(self.V, -np.inf, dtype=np.float32)
        gate_mask = self.gates.valid_mask(gate_ctx)
        combined = gate_mask & valid_mask_arr
        for tid in range(self.V):
            if combined[tid]:
                scores[tid] = 1.0
        return scores
    
    def _valid_mask(self, ctx):
        """Стены: правила."""
        mask = np.ones(self.V, dtype=bool)
        
        for t in self.IGNORED | self.BANNED | self.REPLACEMENT_TOKENS:
            if t < self.V:
                mask[t] = False
        for t in range(4096, self.V):
            mask[t] = False
        
        pos_in_word = ctx.get('pos_in_word', -1)
        word_num = ctx.get('word_num', -1)
        flags = ctx.get('flags', 0)
        is_word_end = (flags >> 1) & 1
        
        # piw=0: только WORD_STARTER
        if pos_in_word == 0:
            for tid in range(self.V):
                if tid < len(self.tt) and self.tt[tid] != 2:
                    mask[tid] = False
            if word_num < 3:
                mask[3] = False
            
            # ПРАВИЛО: на wn=0 только content words (многобуквенные)
            # Первое слово задаёт semantic anchor, однобуквенные не годятся
            if word_num == 0:
                for tid in range(self.V):
                    if tid < len(self.tt) and self.tt[tid] == 2:
                        text = self.hv.decode([tid]).strip()
                        if len(text) <= 1:
                            mask[tid] = False
            
            # ПРАВИЛО: на wn >= 2 блокировать function words
            if word_num >= 2:
                function_words = {'в', 'с', 'к', 'у', 'о', 'и', 'а', '–', '—', '1', '0'}
                for tid in range(self.V):
                    if tid < len(self.tt) and self.tt[tid] == 2:
                        text = self.hv.decode([tid]).strip().lower()
                        if text in function_words:
                            mask[tid] = False
            
            # Блокировать латинские буквы (не русский контент)
            for tid in range(self.V):
                if tid < len(self.tt) and self.tt[tid] == 2:
                    text = self.hv.decode([tid]).strip()
                    if text and text[0].isascii() and text[0].isalpha():
                        mask[tid] = False
            
            # Блокировать короткие type-2 префиксы (длина < 4 = псевдослова)
            for tid in self._prefix_type2:
                if tid < self.V:
                    text = self.hv.decode([tid]).strip()
                    if len(text) < 4:
                        mask[tid] = False
            
            # Блокировать любые type-2 короче 3 букв (обрубки вроде "пр", "См", "пл")
            for tid in range(self.V):
                if tid < len(self.tt) and self.tt[tid] == 2:
                    text = self.hv.decode([tid]).strip()
                    if len(text) < 3:
                        mask[tid] = False
        
        # piw≥1: WORD_CONT — allow type-2 (new word), type-3 (structural cont), EOS
        if pos_in_word >= 1:
            for tid in range(self.V):
                if tid < len(self.tt):
                    if self.tt[tid] == 0:
                        mask[tid] = False
            mask[3] = True if word_num >= 2 else False

            # Если у текущего префикса есть непотраченные type-3 продолжения —
            # блокируем ВСЕ type-2. Слово должно быть завершено через type-3.
            word_prefix = ctx.get('word_prefix', -1)
            used_type3s = ctx.get('used_type3s', set())
            if word_prefix >= 0 and word_prefix in self._prefix_type3_set:
                remaining = self._prefix_type3_set[word_prefix] - used_type3s
                if remaining:
                    for tid in range(self.V):
                        if tid < len(self.tt) and self.tt[tid] == 2:
                            mask[tid] = False

        # На piw=0 после префикса: разрешить только type-3 продолжения,
        # блокировать все type-2 (слово должно быть завершено через type-3)
        if pos_in_word == 0:
            prev_tid = ctx.get('token_id', -1)
            if prev_tid >= 0 and prev_tid in self._prefix_type3_set:
                for tid in range(self.V):
                    if tid < len(self.tt) and self.tt[tid] == 2:
                        mask[tid] = False
                for tid3 in self._prefix_type3_set[prev_tid]:
                    if tid3 < self.V:
                        mask[tid3] = True
        
        # EOS разрешён в конце слова
        if is_word_end and word_num >= 2:
            mask[3] = True
        
        # Антиповтор: блокировать слово, уже появившееся в последних 15 токенах
        if pos_in_word == 0:
            recent = ctx.get('context_tokens', [])
            seen_text = set()
            seen_concept = {}  # cid -> count
            for t in recent:
                if t < len(self.tt) and self.tt[t] == 2:
                    text = self.hv.decode([t]).strip().lower()
                    if self._is_content_token(t) and len(text) > 2:
                        seen_text.add(text)
                    cid = self.ag.get_concept(t)
                    if cid is not None:
                        seen_concept[cid] = seen_concept.get(cid, 0) + 1
            for t in range(self.V):
                if t >= len(self.tt) or self.tt[t] != 2:
                    continue
                # Текстовая репетиция: ни одного повтора
                text = self.hv.decode([t]).strip().lower()
                if text in seen_text:
                    mask[t] = False
                # Концептуальная репетиция: не повторять тот же концепт 3+ раз
                cid = self.ag.get_concept(t)
                if cid is not None and seen_concept.get(cid, 0) >= 2:
                    mask[t] = False
        
        return mask
    
    def _compute_full_word_anchor(self, word_tokens):
        """
        Из токенов законченного слова → type-2 токен полного слова для SVD-якоря.
        Сначала полное слово; если нет — longest prefix type-2.
        """
        if not word_tokens:
            return -1
        try:
            word_text = self.hv.tokenizer.decode(word_tokens).strip()
            if not word_text:
                return -1
            
            # Try full word first
            key = ' ' + word_text
            encoded = self.hv.encode(key)
            for t in encoded:
                if t < 4096 and self.tt[t] == 2:
                    decoded = self.hv.decode([t]).strip()
                    if decoded == word_text:
                        return t  # exact match!
                    # Full word split into preexisting type-2 + continuation
                    if decoded.startswith(word_text[0]) and len(decoded) > 1:
                        return t  # best available multi-char prefix
            
            # Fallback: shorter prefixes
            for p_len in range(min(len(word_text)-1, 10), 0, -1):
                if p_len < 2:
                    break
                prefix = word_text[:p_len]
                encoded = self.hv.encode(' ' + prefix)
                for t in encoded:
                    if t < 4096 and self.tt[t] == 2:
                        decoded = self.hv.decode([t]).strip()
                        if decoded.startswith(word_text[0]) and len(decoded) > 1:
                            return t
            
            return -1
        except Exception:
            return -1
    
    def _is_content_token(self, tid):
        """Проверяет, что токен — content word (многобуквенный type-2)."""
        if tid >= 4096 or tid < 0:
            return False
        if tid >= len(self.tt) or self.tt[tid] != 2:
            return False
        text = self.hv.decode([tid]).strip()
        if not text:
            return False
        # Однобуквенные (предлоги/союзы) — не content
        if len(text) <= 1:
            return False
        # Латинские буквы — не русский контент
        if text[0].isascii() and text[0].isalpha():
            return False
        return True

    # ─── Example Path (образ-скелет) ──────────────────────────────────

    def _extract_example_path(self, text):
        hv = self.hv
        ag = self.ag
        tokens = hv.encode(' ' + text)
        concept_sequence = []
        token_sequence = []
        word_count = 0
        prev_cluster = None
        for t in tokens:
            if t >= 4096:
                continue
            if self.tt[t] != 2:
                continue
            decoded = hv.decode([t]).strip()
            # Фильтруем мусор: пустые строки, латиница, одиночные символы
            if not decoded:
                continue
            if decoded[0].isascii() and decoded[0].isalpha():
                continue
            if len(decoded) <= 1:
                continue
            c = ag.get_concept(t)
            if c is not None:
                cluster = c - ag.L1_OFFSET
                if 0 <= cluster < ag.n_clusters:
                    if prev_cluster is None or cluster != prev_cluster:
                        concept_sequence.append(cluster)
                        token_sequence.append(t)
                        word_count += 1
                        prev_cluster = cluster
        if not concept_sequence:
            return None
        concept_sequence.append(-1)  # EOS
        token_sequence.append(-1)    # EOS
        # Определяем тип предложения
        s_type = 'statement'
        first_word_text = text.strip().split()[0].lower().strip('—–-«».,!?;:()"')
        if text.strip().endswith('?'):
            s_type = 'question'
        elif text.strip().endswith('!'):
            s_type = 'exclamation'
        elif text.strip().startswith('—') or first_word_text in ('—', '-'):
            s_type = 'dialogue'
        else:
            # Если есть PatternLearner со статистикой — используем его
            pl = getattr(self, '_pattern_learner', None)
            if pl and (pl.token_s_type_dist or pl.concept_s_type_dist) and token_sequence:
                first_tid = token_sequence[0]
                first_c = concept_sequence[0] if concept_sequence else -1
                s_type = pl.detect_s_type(first_concept=first_c, first_tid=first_tid)
        return {
            's_type': s_type,
            'concept_sequence': concept_sequence,
            'token_sequence': token_sequence,
            'word_count': word_count,
            'text': text,
        }

    def _example_path_scores(self, ctx, valid, forced_path):
        scores = np.zeros(self.V, dtype=np.float32)
        pos_in_word = ctx.get('pos_in_word', -1)
        if pos_in_word != 0:
            return scores
        word_num = ctx.get('word_num', -1)
        if word_num < 0:
            return scores
        seq = forced_path.get('concept_sequence', [])
        tokens_seq = forced_path.get('token_sequence')
        if tokens_seq is None:
            tokens_seq = []
        if word_num >= len(seq):
            return scores
        expected_cluster = seq[word_num]
        expected_tid = tokens_seq[word_num] if word_num < len(tokens_seq) else -1
        
        # В конце пути — форсируем EOS
        if expected_cluster == -1:
            if 3 < self.V and valid[3]:
                scores[3] = 10.0
            return scores
        
        # Буст конкретного токена из примера
        if expected_tid >= 0 and expected_tid < self.V and valid[expected_tid]:
            scores[expected_tid] = 12.0
        
        # Буст остальных токенов из концепта
        if 0 <= expected_cluster < self.ag.n_clusters:
            cid_full = self.ag.L1_OFFSET + expected_cluster
            members = self.ag.cid_to_tids.get(cid_full, [])
            for tid in members:
                if tid < self.V and valid[tid] and scores[tid] == 0.0:
                    scores[tid] = 6.0
        return scores
        word_num = ctx.get('word_num', -1)
        if word_num < 0:
            return scores
        seq = forced_path.get('concept_sequence', [])
        tokens_seq = forced_path.get('token_sequence')
        if tokens_seq is None:
            tokens_seq = []
        if word_num >= len(seq):
            return scores
        expected_cluster = seq[word_num]
        expected_tid = tokens_seq[word_num] if word_num < len(tokens_seq) else -1
        
        # Буст конкретного токена из примера
        if expected_tid >= 0 and expected_tid < self.V and valid[expected_tid]:
            scores[expected_tid] = 12.0
        
        # Буст остальных токенов из концепта
        if 0 <= expected_cluster < self.ag.n_clusters:
            cid_full = self.ag.L1_OFFSET + expected_cluster
            members = self.ag.cid_to_tids.get(cid_full, [])
            for tid in members:
                if tid < self.V and valid[tid] and scores[tid] == 0.0:
                    scores[tid] = 6.0
        return scores

    def _word_importance(self, tid):
        if tid >= len(self.tt) or self.tt[tid] != 2:
            return 0.0
        text = self.hv.decode([tid]).strip()
        if not text or len(text) <= 2:
            return 0.0
        if text[0].isascii() and text[0].isalpha():
            return 0.0
        l = len(text)
        if l >= 6:
            return 1.0
        if l >= 4:
            cid = self.ag.get_concept(tid)
            if cid is not None:
                cluster = cid - self.ag.L1_OFFSET
                members = self.ag.get_members(cluster)
                if len(members) >= 10:
                    return 1.0
            return 0.7
        return 0.3

    def _select_spine(self, ctx, prev_sent_concept=-1, seed_word=None):
        spine = []
        hv = self.hv
        ag = self.ag
        cur_cid = None
        if seed_word:
            seed_tokens = hv.encode(' ' + seed_word)
            for t in seed_tokens:
                if t < 4096 and self.tt[t] == 2:
                    c = ag.get_concept(t)
                    if c is not None:
                        cur_cid = c - ag.L1_OFFSET
                        break
        if cur_cid is None and prev_sent_concept >= 0:
            cur_cid = prev_sent_concept
        if cur_cid is None:
            return []
        roles = ['subject', 'verb', 'object']
        for pos in range(3):
            if cur_cid is None or cur_cid >= ag.n_clusters:
                break
            cid_full = ag.L1_OFFSET + cur_cid
            outgoing = ag.transition_ci.get(cid_full, [])
            if not outgoing:
                break
            best_cj = None
            # Try all outgoing transitions to find one with usable members
            outgoing_sorted = sorted(outgoing, key=lambda x: -x[1])
            for cj_id, lp in outgoing_sorted:
                if cj_id < ag.L1_OFFSET:
                    continue
                cj = cj_id - ag.L1_OFFSET
                if not (0 <= cj < ag.n_clusters) or cj == cur_cid:
                    continue
                members = ag.cid_to_tids.get(cj_id, [])
                if not members:
                    continue
                for tid in members:
                    if not self._is_content_token(tid) or tid >= 4096:
                        continue
                    if self._word_importance(tid) >= 0.7 and len(hv.decode([tid]).strip()) >= 3:
                        best_cj = cj
                        break
                if best_cj is not None:
                    break
            if best_cj is None:
                break
            members = ag.cid_to_tids.get(ag.L1_OFFSET + best_cj, [])
            if not members:
                break
            best_tid = None
            best_score = -1.0
            for tid in members:
                if not self._is_content_token(tid) or tid >= 4096:
                    continue
                imp = self._word_importance(tid)
                if imp < 0.7:
                    continue
                txt = hv.decode([tid]).strip()
                if len(txt) < 3:
                    continue
                score = imp * (1.0 if best_tid is None else 0.9)
                if score > best_score:
                    best_score = score
                    best_tid = tid
            if best_tid is not None:
                spine.append((best_tid, best_cj, roles[pos]))
                cur_cid = best_cj
            else:
                break
        return spine

    def _spine_scores(self, ctx, valid_mask, spine):
        scores = np.zeros(self.V, dtype=np.float32)
        pos_in_word = ctx.get('pos_in_word', -1)
        if pos_in_word != 0:
            return scores
        word_num = ctx.get('word_num', -1)
        if word_num < 0:
            return scores
        for sp_idx, (tid, cid, role) in enumerate(spine):
            if sp_idx <= word_num <= sp_idx + 1:
                if tid < self.V and valid_mask[tid]:
                    scores[tid] = 3.0
        return scores

    def _structural_scores(self, ctx, prev_token, valid_mask, prev_sentence_concept=-1,
                           paragraph_topic=None):
        """
        Структурные scores (БЕЗ частотного bias).
        На piw=0: все transitions из structural matrix равны (1.0).
          Если это первое слово после EOS, дополнительно фильтруем
          по cross-sentence concept rules + paragraph topic.
        На piw>=1: все type-3 токены равны.
        """
        scores = np.full(self.V, -np.inf, dtype=np.float32)
        pos_in_word = ctx.get('pos_in_word', -1)
        
        if pos_in_word == 0:
            flags = ctx.get('flags', 0)
            is_word_end = (flags >> 1) & 1
            word_num = ctx.get('word_num', -1)
            
            # Structural: any observed transition = equally valid
            allowed = self.structural.get(prev_token, set())
            for tid in allowed:
                if tid < self.V and valid_mask[tid]:
                    scores[tid] = 1.0
            
            # EOS: mark structurally valid (score = structural base 1.0)
            # Actual EOS selection is handled in generate_step via separate probability
            if valid_mask[3]:
                scores[3] = 1.0
            
            # Cross-sentence concept constraint: at sentence start
            if prev_sentence_concept >= 0 and prev_token == 3:
                valid_next = self._concepts_for_prev(prev_sentence_concept)
                if valid_next is not None and len(valid_next) < self.ag.n_clusters - 2:
                    for tid in range(self.V):
                        if not np.isfinite(scores[tid]):
                            continue
                        if tid >= len(self.tt) or self.tt[tid] != 2:
                            continue
                        cid = self.ag.get_concept(tid)
                        if cid is not None:
                            cluster = cid - self.ag.L1_OFFSET
                            if cluster not in valid_next:
                                scores[tid] = -np.inf
            
            # Paragraph topic constraint: at sentence start within a paragraph
            # First word must belong to paragraph's topic concept set
            if paragraph_topic and len(paragraph_topic) > 0 and prev_token == 3:
                for tid in range(self.V):
                    if not np.isfinite(scores[tid]):
                        continue
                    if tid >= len(self.tt) or self.tt[tid] != 2:
                        continue
                    cid = self.ag.get_concept(tid)
                    if cid is not None:
                        cluster = cid - self.ag.L1_OFFSET
                        if cluster not in paragraph_topic:
                            scores[tid] = -np.inf
        else:
            # Continuation: only structurally valid type-3 tokens
            allowed = self.structural.get(prev_token, set())
            for tid in allowed:
                if tid < self.V and valid_mask[tid]:
                    scores[tid] = 1.0
            # EOS also valid at word boundary
            if valid_mask[3] and ctx.get('flags', 0) >> 1 & 1:
                scores[3] = 1.0
        
        return scores
    
    def _semantic_scores(self, ctx, prev_token, content_token, valid_mask):
        """
        Семантический выбор через SVD-пространство.
        
        Ключевой принцип: семантика определяется ПОСЛЕДНИМ CONTENT WORD
        (многобуквенным type-2). Однобуквенные предлоги/союзы не несут
        семантического сигнала.
        
        content_token: последний многобуквенный type-2 в контексте.
        """
        scores = np.full(self.V, 0.0, dtype=np.float32)
        pos_in_word = ctx.get('pos_in_word', -1)
        word_num = ctx.get('word_num', -1)
        
        if pos_in_word != 0:
            return scores
        
        # Определяем semantic anchor: последний content word
        anchor = content_token if content_token >= 0 else prev_token
        if not self._is_content_token(anchor):
            if self._is_content_token(prev_token):
                anchor = prev_token
            else:
                return scores  # нет semantic anchor
        
        if not self.vs.has_vector(anchor):
            return scores
        
        # Semantic boost: топ-50 SVD-ближайших, boosting только content-токены
        similar = self.vs.topk_similar(anchor, k=50)
        content_boosted = 0
        for tid, sim in similar:
            if content_boosted >= self.sem_boost_count:
                break
            if tid >= self.V or not valid_mask[tid]:
                continue
            if sim <= 0:
                continue
            if self._is_content_token(tid):
                scores[tid] += sim * self.sem_weight
                content_boosted += 1
        
        return scores
    
    def _find_speech_anchor(self, s_type):
        """Choose response verb anchor for sentence-type semantic boost."""
        if s_type in ('dialogue', 'question'):
            return 475  # сказал — universal speech verb
        return -1
    
    def _sentence_type_boost(self, ctx, valid_mask, prev_sentence_type=None,
                             words_in_sentence=0):
        """
        Sentence-type semantic boost.
        После dialogue/question: добавляем SVD-близость к глаголу речи "сказал"
        к scores первых 3 слов предложения. Это не структурное правило,
        а семантический bias: даже если content_token — не глагол, response-глаголы получают boost.
        """
        scores = np.zeros(self.V, dtype=np.float32)
        if prev_sentence_type not in ('dialogue', 'question'):
            return scores
        if ctx.get('pos_in_word', -1) != 0:
            return scores
        if words_in_sentence >= 3:
            return scores
        
        anchor = self._find_speech_anchor(prev_sentence_type)
        if anchor < 0 or not self.vs.has_vector(anchor):
            return scores
        
        similar = self.vs.topk_similar(anchor, k=20)
        for tid, sim in similar:
            if tid < self.V and valid_mask[tid]:
                if self._is_content_token(tid) and sim > 0:
                    # Weight decays with word position: word 1 gets full, word 2 gets half
                    weight = 2.0 * (1.0 - words_in_sentence * 0.3)
                    scores[tid] = sim * max(weight, 0.5)
        return scores
    
    def _concept_pmi_scores(self, ctx, valid_mask, content_token=-1,
                            prev_sent_concept=-1):
        """
        PMI-based concept transition scores.
        Uses AssociationGraph.transition_ci: P(concept_j | concept_i) from corpus.
        At word boundaries, boost tokens in top-K most likely next concepts
        based on the LAST content token's concept.
        """
        scores = np.zeros(self.V, dtype=np.float32)
        pos_in_word = ctx.get('pos_in_word', -1)
        if pos_in_word != 0:
            return scores

        # Use content_token's concept for within-sentence transitions
        # Fall back to prev_sent_concept for cross-sentence transitions
        last_concept_id = -1
        if content_token >= 0 and self._is_content_token(content_token):
            last_concept_id = self.ag.get_concept(content_token)
        if last_concept_id is None or last_concept_id < 0:
            if prev_sent_concept >= 0 and prev_sent_concept < self.ag.n_clusters:
                last_concept_id = self.ag.L1_OFFSET + prev_sent_concept
        if last_concept_id is None or last_concept_id < 0:
            return scores

        outgoing = self.ag.transition_ci.get(last_concept_id, [])
        if not outgoing:
            return scores

        top_k = min(5, len(outgoing))
        top_transitions = sorted(outgoing, key=lambda x: -x[1])[:top_k]
        for cj_id, log_prob in top_transitions:
            members = self.ag.cid_to_tids.get(cj_id, [])
            boost = float(np.exp(log_prob))
            if boost < 0.01:
                continue
            for tid in members:
                if tid < self.V and valid_mask[tid]:
                    scores[tid] += boost * 3.0

        return scores
    
    def generate_step(self, ctx, prev_token, content_token=-1, temperature=0.5,
                      prev_sentence_concept=-1, prev_sentence_type=None,
                      paragraph_topic=None, spine=None, forced_path=None):
        """
        Один шаг генерации.
        prev_token: последний токен (может быть type-2 или type-3)
        content_token: последний многобуквенный type-2 (для семантики)
        prev_sentence_concept: cluster idx последнего концепта предыдущего предложения
        prev_sentence_type: тип предыдущего предложения (для sentence-type контекста)
        paragraph_topic: set of concept cluster IDs for current paragraph
        spine: [(tid, concept_id, role), ...] — стержневые слова предложения
        forced_path: dict from _extract_example_path — концепт-скелет для подражания
        """
        # 1. Стены
        valid = self._valid_mask(ctx)
        if not valid.any():
            return 3, {"reason": "all blocked"}
        
        # 2. Структурные scores: Gates (если загружены) или structural matrix
        if self.gates is not None:
            gate_ctx = self._gate_context(ctx, prev_token, content_token)
            structural = self._gate_scores(ctx, gate_ctx, valid)
        else:
            structural = self._structural_scores(ctx, prev_token, valid, prev_sentence_concept,
                                                 paragraph_topic)
        
        # 3. Семантические scores (SVD-близость к последнему content word)
        semantic = self._semantic_scores(ctx, prev_token, content_token, valid)
        
        # 4. Concept scores (PMI-based concept transition)
        concept_pmi = self._concept_pmi_scores(ctx, valid, content_token,
                                               prev_sentence_concept)
        
        # 5. Sentence-type boost: после диалога/вопроса boost глаголов речи
        s_type_boost = self._sentence_type_boost(
            ctx, valid, prev_sentence_type,
            words_in_sentence=getattr(self, '_words_in_current_sentence', 0)
        )
        
        # 6. Prefix boost: на piw=0 бустить префиксы, чьи продолжения семантически подходят
        prefix_boost = self._prefix_boost_scores(ctx, valid, content_token)
        
        # 7. Continuation semantic scores: на piw>=1 бустить type-3 по SVD законченного слова
        cont_sem = self._continuation_semantic_scores(ctx, content_token, valid)
        
        # 8. Spine boost: стержневые слова предложения
        spine_boost = self._spine_scores(ctx, valid, spine or [])
        
        # 9. Example path boost: концепт-скелет из примера
        example_boost = self._example_path_scores(ctx, valid, forced_path) if forced_path else np.zeros(self.V, dtype=np.float32)
        
        # 10. Комбинируем: structural + semantic + concept + s_type + prefix + cont_sem + spine + example
        scores = structural.copy()
        for tid in range(self.V):
            if not valid[tid]:
                continue
            base = 0.0
            if np.isfinite(scores[tid]):
                base = scores[tid]
            if semantic[tid] != 0.0:
                base += semantic[tid]
            if concept_pmi[tid] != 0.0:
                base += concept_pmi[tid]
            if s_type_boost[tid] != 0.0:
                base += s_type_boost[tid]
            if prefix_boost[tid] != 0.0:
                base += prefix_boost[tid]
            if cont_sem[tid] != 0.0:
                base += cont_sem[tid]
            if spine_boost[tid] != 0.0:
                base += spine_boost[tid]
            if example_boost[tid] != 0.0:
                base += example_boost[tid]
            scores[tid] = base
        
        # 11. Стена
        scores[~valid] = -np.inf
        
        # 12. EOS decision: at word boundaries, probabilistically end sentence
        pos_in_word = ctx.get('pos_in_word', -1)
        flags = ctx.get('flags', 0)
        is_word_end = (flags >> 1) & 1
        word_num = ctx.get('word_num', -1)
        
        if pos_in_word == 0 and is_word_end and word_num >= 4:
            import random as _random
            # Slower EOS curve: 1% × word_num, max 0.25
            p_eos = min(0.25, 0.01 * (word_num - 3))
            if _random.random() < p_eos:
                return 3, {
                    "valid_tokens": int(valid.sum()),
                    "chosen": 3,
                    "chosen_text": "<<$>>",
                    "reasons": ["EOS(%.2f)" % p_eos]
                }
        
        # 13. Fallback если всё -inf
        if not np.any(np.isfinite(scores)):
            scores = semantic.copy() + concept_pmi.copy() + s_type_boost.copy()
            scores[~valid] = -np.inf
        
        # 14. Выбор
        next_tok = select_token(scores, temperature=temperature)
        
        explanation = {
            "valid_tokens": int(valid.sum()),
            "chosen": int(next_tok),
            "chosen_text": self.hv.decode([next_tok]).strip(),
            "reasons": []
        }
        
        # Что повлияло?
        if self._is_content_token(next_tok):
            explanation["reasons"].append("content")
        if concept_pmi[next_tok] != 0.0:
            explanation["reasons"].append("concept_pmi")
        if content_token >= 0 and self.vs.has_vector(content_token) and self.vs.has_vector(next_tok):
            sim = self.vs.similarity(content_token, next_tok)
            if sim > 0.2:
                explanation["reasons"].append("semantic(%.2f)" % sim)
        
        # Report sentence-type boost if it affected choice
        if s_type_boost[next_tok] != 0.0 and prev_sentence_type in ('dialogue', 'question'):
            explanation["reasons"].append("s_type_%s" % prev_sentence_type[:4])
        
        # Report example path boost
        if forced_path and example_boost[next_tok] != 0.0:
            word_num = ctx.get('word_num', -1)
            seq = forced_path.get('concept_sequence', [])
            if 0 <= word_num < len(seq):
                expected_cluster = seq[word_num]
                explanation["reasons"].append("example_C%d" % expected_cluster)
        
        # Report cross-sentence constraint if active
        if prev_sentence_concept >= 0 and prev_token == 3:
            explanation["prev_sent_concept"] = int(prev_sentence_concept)
            explanation["reasons"].append("cross_sent_C%d" % prev_sentence_concept)
            total_struct = int(np.sum(np.isfinite(structural)))
            explanation["cross_sent_allowed"] = total_struct
        
        return next_tok, explanation
    
    def generate(self, max_tokens=40, seed_word=None, target_composition=None, temperature=0.5,
                 example=None, auto_pattern=False, text_hierarchy=None):
        """
        Генерация: structural + semantic, без frequency bias.
        
        seed_word: начальное слово (опционально)
        target_composition: [(word, weight), ...] — целевая композиция (опционально)
        example: строка-пример для подражания (опционально)
        """
        tokens = [2]  # BOS
        explanations = []
        content_token = -1  # последний content word (для семантического контекста)
        self._current_word_tokens = []  # reset word accumulation
        self._full_word_anchor = None
        self._active_spine = []  # sentence spine for current sentence
        
        # Example path: извлечь концепт-скелет из примера
        self._example_path = None
        if example:
            self._example_path = self._extract_example_path(example)
            if self._example_path:
                print("Example path: s_type=%s concepts=%s" % (
                    self._example_path['s_type'],
                    self._example_path['concept_sequence']))
                self._prev_sentence_type = self._example_path['s_type']
        
        # Auto pattern: самоорганизация шаблонов из корпуса
        self._active_pattern_seq = None
        if auto_pattern and self._example_path is None:
            if not self._pattern_learner_trained and text_hierarchy is not None:
                self._pattern_learner = PatternLearner(self.hv, self.ag, self.gates)
                self._pattern_learner.learn(text_hierarchy)
                self._pattern_learner_trained = True
            
            if self._pattern_learner and self._pattern_learner.patterns:
                matched = self._pattern_learner.match(seed_word=seed_word)
                if matched:
                    self._active_pattern_seq = matched
                    self._example_path = self._pattern_learner.to_forced_path(matched)
                    desc = self._pattern_learner.describe(matched)
                    if desc:
                        print("Pattern match: concepts=%s freq=%d meta=%s" % (
                            desc['concepts'], desc['freq'], desc['meta']))
                    self._prev_sentence_type = self._example_path['s_type']
        
        # Sentence tracking for cross-sentence constraints
        self._prev_sentence_last_concept = -1
        self._prev_sentence_type = 'start'
        self._sentence_start_pos = len(tokens)  # where current sentence started
        self._generated_sentence_count = 0
        self._words_in_current_sentence = 0
        
        # Paragraph tracking (абзац = 3-5 предложений на одну тему)
        self._paragraph_topic = set()   # set of concept cluster IDs for current paragraph
        self._sentences_in_paragraph = 0
        import random as _rnd
        self._max_paragraph_sentences = _rnd.randint(3, 5)
        
        # Seed word: encode " word" to get proper type-2 token
        if seed_word:
            # Infer sentence type from seed word via token/concept statistics
            if self._pattern_learner and (self._pattern_learner.token_s_type_dist or self._pattern_learner.concept_s_type_dist):
                seed_tokens_for_type = self.hv.encode(' ' + seed_word)
                for t in seed_tokens_for_type:
                    if t < 4096 and self.tt[t] == 2:
                        c = self.ag.get_concept(t)
                        cluster = (c - self.ag.L1_OFFSET) if c is not None else -1
                        s_t = self._pattern_learner.detect_s_type(first_concept=cluster if cluster >= 0 else None, first_tid=t)
                        self._prev_sentence_type = s_t
                        break
            
            seed_tokens = self.hv.encode(' ' + seed_word)
            seed_type2 = None
            for t in seed_tokens:
                if t < 4096 and self.tt[t] == 2:
                    seed_type2 = t
                    break
            if seed_type2 is not None:
                tokens.append(seed_type2)
                if self._is_content_token(seed_type2):
                    content_token = seed_type2
                    self._current_word_tokens = [seed_type2]
                    self._word_prefix = seed_type2
                    self._current_word_type3s = set()

        
        # Initial BOS: BOS is token 2, which is not in the structural matrix
        # First token is always from heads (no structural context yet)
        if len(tokens) == 1 and not seed_word:
            meta = self.hv.metadata_from_ids(tokens)
            ctx = dict(meta[-1])
            ctx['context_tokens'] = []
            valid = self._valid_mask(ctx)
            
            # Если есть example_path — форсируем первый концепт из пути,
            # разрешая ожидаемый токен даже если _valid_mask его блокирует
            if self._example_path:
                seq = self._example_path.get('concept_sequence', [])
                tok_seq = self._example_path.get('token_sequence', [])
                if seq and seq[0] >= 0:
                    expected_cluster = seq[0]
                    expected_tok = tok_seq[0] if tok_seq else -1
                    if expected_tok >= 0 and expected_tok < self.V:
                        valid[expected_tok] = True  # обход _valid_mask для токена из примера
            
            # Use heads for very first token
            try:
                hs = self.heads.individual_scores(ctx)
                w = np.array([1.0, 1.0, 3.0, 0.5, 0.2, 0.5], dtype=np.float32)
                scores = np.dot(w, hs)
            except:
                scores = np.full(self.V, -np.inf, dtype=np.float32)
            
            # Если есть example_path — форсируем первый концепт из пути
            if self._example_path:
                seq = self._example_path.get('concept_sequence', [])
                tok_seq = self._example_path.get('token_sequence', [])
                if seq and seq[0] >= 0:
                    expected_cluster = seq[0]
                    expected_tok = tok_seq[0] if tok_seq else -1
                    if 0 <= expected_cluster < self.ag.n_clusters:
                        cid_full = self.ag.L1_OFFSET + expected_cluster
                        members = self.ag.cid_to_tids.get(cid_full, [])
                        for tid in range(self.V):
                            if valid[tid]:
                                if tid == expected_tok:
                                    scores[tid] = 20.0
                                elif tid in members:
                                    scores[tid] = 10.0
                                else:
                                    scores[tid] = -np.inf
            
            # Suppress all single-letter tokens at first position
            for t in range(self.V):
                if t < len(self.tt) and self.tt[t] == 2:
                    text = self.hv.decode([t]).strip()
                    if len(text) <= 1:
                        scores[t] = -np.inf
            scores[~valid] = -np.inf
            next_tok = select_token(scores, temperature=min(temperature, 0.5))
            tokens.append(next_tok)
            chosen_text = self.hv.decode([next_tok]).strip()
            if self._is_content_token(next_tok):
                content_token = next_tok
                # Compute full-word anchor for first word
                full_enc = self.hv.encode(' ' + chosen_text)
                for ft in full_enc:
                    if ft < 4096 and self.tt[ft] == 2 and self._is_content_token(ft):
                        if self.vs.has_vector(ft):
                            content_token = ft
                            break
            exp = {"valid_tokens": int(valid.sum()), "chosen": int(next_tok),
                   "chosen_text": chosen_text, "reasons": ["first"]}
            explanations.append(exp)
        
        # Main generation loop
        while len(tokens) < max_tokens:
            meta = self.hv.metadata_from_ids(tokens)
            ctx = dict(meta[-1])
            ctx['context_tokens'] = tokens[-15:] if len(tokens) > 5 else tokens
            ctx['word_prefix'] = self._word_prefix
            ctx['used_type3s'] = self._current_word_type3s.copy()
            
            prev = tokens[-1] if len(tokens) > 1 else 2
            
            # Cross-sentence context: after EOS, pass previous sentence's last concept
            if prev == 3:
                psc = self._prev_sentence_last_concept
            else:
                psc = -1
            
            # Generate
            # Paragraph topic constraint: apply at sentence start (prev == 3)
            para_topic = self._paragraph_topic if (
                prev == 3 and self._sentences_in_paragraph > 0
            ) else None
            
            # Sentence spine: pre-select 3 core words at sentence start
            if prev == 3 or len(tokens) <= 2:
                spine = self._select_spine(ctx, prev_sent_concept=psc, seed_word=seed_word if len(tokens) <= 2 else None)
                self._active_spine = spine
            else:
                spine = self._active_spine
            
            next_tok, exp = self.generate_step(
                ctx, prev, content_token, temperature,
                prev_sentence_concept=psc,
                prev_sentence_type=self._prev_sentence_type if prev == 3 else None,
                paragraph_topic=para_topic,
                spine=spine,
                forced_path=self._example_path
            )
            
            # Word accumulation: track tokens to decode full word later
            pos_in_word = ctx.get('pos_in_word', 0)
            cur_token_type = self.tt[next_tok] if next_tok < len(self.tt) else 0
            
            if cur_token_type == 2:
                # New word starting — previous word is complete
                self._word_prefix = next_tok
                self._current_word_type3s = set()
                if self._current_word_tokens:
                    anchor = self._compute_full_word_anchor(self._current_word_tokens)
                    if anchor >= 0 and self._is_content_token(anchor):
                        content_token = anchor  # full-word SVD anchor!
                    self._current_word_tokens = [next_tok]
                else:
                    self._current_word_tokens = [next_tok]
                self._words_in_current_sentence += 1
            elif cur_token_type == 3:
                self._current_word_tokens.append(next_tok)
                self._current_word_type3s.add(next_tok)
            else:
                # Punctuation/special — word is done
                self._word_prefix = -1
                self._current_word_type3s = set()
                if self._current_word_tokens:
                    anchor = self._compute_full_word_anchor(self._current_word_tokens)
                    if anchor >= 0 and self._is_content_token(anchor):
                        content_token = anchor
                    self._current_word_tokens = []
                self._current_word_tokens.append(next_tok)  # include punct in next word
            

            
            # Target composition guidance
            if target_composition and ctx.get('word_num', 0) >= 2:
                # Compute target vector
                vectors = []
                weights = []
                for word, w in target_composition:
                    v = self.vs.word_vector(word)
                    if v is not None:
                        vectors.append(v)
                        weights.append(w)
                if vectors and len(vectors) == len(target_composition):
                    tv = self.vs.compose(vectors, weights)
                    if tv is not None:
                        valid = self._valid_mask(ctx) if hasattr(self, '_valid_mask') else np.ones(self.V, dtype=bool)
                        near = self.vs.topk_to_vector(tv, k=10)
                        scores2 = np.full(self.V, -np.inf, dtype=np.float32)
                        for tid, sim in near:
                            if tid < self.V and valid[tid] and sim > 0:
                                scores2[tid] = sim * 5.0
                        scores2[~valid] = -np.inf
                        if np.any(np.isfinite(scores2)):
                            next_tok = select_token(scores2, temperature=temperature)
                            exp["reasons"].append("composition")
                            exp["chosen"] = int(next_tok)
                            exp["chosen_text"] = self.hv.decode([next_tok]).strip()
                            if self._is_content_token(next_tok):
                                full_enc = self.hv.encode(' ' + exp['chosen_text'])
                                for ft in full_enc:
                                    if ft < 4096 and self.tt[ft] == 2 and self._is_content_token(ft):
                                        content_token = ft
                                        break
                                content_token = next_tok
            
            tokens.append(next_tok)
            explanations.append(exp)
            
            if next_tok == 3:
                # Word tracking reset at sentence boundary
                self._word_prefix = -1
                self._current_word_type3s = set()
                # Clear example path for next sentence
                self._example_path = None
                
                # Отчёт успеха для авто-шаблона
                if self._active_pattern_seq is not None and self._pattern_learner is not None:
                    type2_tokens = [t for t in tokens[self._sentence_start_pos:] if t < 4096 and self.tt[t] == 2]
                    path = self._pattern_learner.to_forced_path(self._active_pattern_seq)
                    expected = path.get('concept_sequence', []) if path else []
                    matches = 0
                    total = min(len(type2_tokens), len(expected) - 1)
                    for i in range(total):
                        if i < len(expected) - 1:
                            c = self.ag.get_concept(type2_tokens[i])
                            if c is not None and (c - self.ag.L1_OFFSET) == expected[i]:
                                matches += 1
                    match_ratio = matches / max(1, total)
                    self._pattern_learner.report_outcome(
                        self._active_pattern_seq,
                        success=True,
                        word_count=len(type2_tokens),
                        match_ratio=match_ratio
                    )
                    self._active_pattern_seq = None
                
                # Sentence complete: save last concept for cross-sentence constraint
                if content_token >= 0:
                    cid = self.ag.get_concept(content_token)
                    if cid is not None:
                        self._prev_sentence_last_concept = cid - self.ag.L1_OFFSET
                    else:
                        self._prev_sentence_last_concept = -1
                else:
                    self._prev_sentence_last_concept = -1
                
                # Detect sentence type from punctuation + concept statistics
                sent_tokens = tokens[self._sentence_start_pos:]
                sent_text = self.hv.decode(sent_tokens).strip()
                prev_type_before = self._prev_sentence_type
                
                if '?' in sent_text:
                    self._prev_sentence_type = 'question'
                elif '!' in sent_text:
                    self._prev_sentence_type = 'exclamation'
                elif any(m in sent_text for m in ['—', '–', '«', '"']):
                    self._prev_sentence_type = 'dialogue'
                elif sent_tokens and self._pattern_learner and (self._pattern_learner.token_s_type_dist or self._pattern_learner.concept_s_type_dist):
                    first_tid = sent_tokens[0]
                    if first_tid < 4096 and self.tt[first_tid] == 2:
                        c = self.ag.get_concept(first_tid)
                        cluster = (c - self.ag.L1_OFFSET) if c is not None else -1
                        self._prev_sentence_type = self._pattern_learner.detect_s_type(
                            first_concept=cluster if cluster >= 0 else None,
                            first_tid=first_tid)
                else:
                    self._prev_sentence_type = 'statement'
                
                # Self-play: observe this sentence into gates
                if self.gates is not None:
                    self.gates.observe_sentence(
                        sent_tokens,
                        prev_s_type=prev_type_before,
                        cur_s_type=self._prev_sentence_type
                    )
                
                self._generated_sentence_count += 1
                self._sentence_start_pos = len(tokens)
                self._words_in_current_sentence = 0
                
                # Paragraph tracking: build concept set from this sentence's content words
                sent_concepts = set()
                for tid in sent_tokens:
                    if tid < len(self.tt) and self.tt[tid] == 2:
                        cid = self.ag.get_concept(tid)
                        if cid is not None:
                            sent_concepts.add(cid - self.ag.L1_OFFSET)
                
                if self._sentences_in_paragraph == 0:
                    # First sentence of paragraph: set topic
                    self._paragraph_topic = sent_concepts
                    self._sentences_in_paragraph = 1
                else:
                    overlap = sent_concepts & self._paragraph_topic
                    if (overlap or len(self._paragraph_topic) < 3):
                        # Topic overlap or broad topic: continue paragraph
                        self._paragraph_topic |= sent_concepts  # expand topic
                        self._sentences_in_paragraph += 1
                    else:
                        # No overlap: paragraph wraps naturally
                        pass
                
                # Check if paragraph should end
                if self._sentences_in_paragraph >= self._max_paragraph_sentences:
                    self._paragraph_topic = sent_concepts
                    self._sentences_in_paragraph = 0
                    import random as _rnd
                    self._max_paragraph_sentences = _rnd.randint(3, 5)
        
        # Отчёт: результат авто-шаблона
        if self._active_pattern_seq is not None and self._pattern_learner is not None:
            reached_eos = next_tok == 3 if len(tokens) > 1 else False
            type2_tokens = [t for t in tokens if t < 4096 and self.tt[t] == 2]
            path = self._pattern_learner.to_forced_path(self._active_pattern_seq)
            expected = path.get('concept_sequence', []) if path else []
            matches = 0
            total = min(len(type2_tokens), len(expected) - 1)
            for i in range(total):
                if i < len(expected) - 1:
                    c = self.ag.get_concept(type2_tokens[i])
                    if c is not None and (c - self.ag.L1_OFFSET) == expected[i]:
                        matches += 1
            match_ratio = matches / max(1, total)
            self._pattern_learner.report_outcome(
                self._active_pattern_seq,
                success=reached_eos,
                word_count=len(type2_tokens),
                match_ratio=match_ratio
            )
        
        return {
            'tokens': tokens,
            'text': self.hv.decode(tokens),
            'explanations': explanations,
        }
    
    def print_trace(self, result):
        print("\n=== ВЕКТОРНАЯ ГЕНЕРАЦИЯ ===")
        print("Text: %s\n" % result['text'])
        for i, exp in enumerate(result.get('explanations', [])):
            reasons = ', '.join(exp.get('reasons', [])) or 'heads'
            print("  [%2d] '%s' V=%d %s" % (i, exp['chosen_text'], 
                  exp['valid_tokens'], reasons))
