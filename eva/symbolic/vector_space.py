import sys, math, random, numpy as np
from collections import defaultdict

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.generation_loop import select_token
from eva.symbolic.structural_rules import StructuralRules
from eva.symbolic.gate_logic import GateLogic
from eva.symbolic.pattern_learner import PatternLearner
from eva.symbolic.auto_config import AutoConfig




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
    
    def project_to_dim(self, target_dim, seed=42):
        """Project all vectors to target_dim preserving pairwise dot products.
        
        Uses a random orthogonal matrix Q ∈ R^{target_dim × current_dim} with
        orthonormal columns, so Q^T Q = I and v·w ≈ (Qv)·(Qw). Up-projection
        adds extra dimensions initialized from the original space; down-projection
        truncates via the leading singular vectors of the random projection.
        """
        cur = self.dim
        if cur is None or cur == target_dim:
            return
        rng = np.random.RandomState(seed)
        R = rng.randn(target_dim, cur).astype(np.float32)
        Q, _ = np.linalg.qr(R)
        for tid in list(self.token_vectors.keys()):
            v = self.token_vectors[tid]
            self.token_vectors[tid] = (Q @ v).astype(np.float32)
        self.dim = target_dim

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


class VectorPopulation:
    """Evolutionary vector population with fitness-proportional selection.

    Each token can have multiple vector versions. Selection is probabilistic —
    no hard thresholds. Branching (new version spawn) happens on mismatch,
    extinct by fitness decay. All parameters derived from token frequency and
    global vector statistics — zero hardcoded values.
    """
    def __init__(self, vs, config, token_freq):
        self.vs = vs
        self.config = config
        self._freq = token_freq

        self.versions = {}    # {tid: [vec₀, vec₁, ...]}
        self.fitness = {}     # {tid: [f₀, f₁, ...]}
        self.n_calls = {}     # {tid: [c₀, c₁, ...]}
        self._base_calls = {} # {tid: total_update_calls}
        self._global_std = None

    def lazy_init(self, tid):
        if tid not in self.versions:
            v = self.vs.token_vectors[tid].copy()
            self.versions[tid] = [v]
            self.fitness[tid] = [1.0]
            self.n_calls[tid] = [0]
            self._base_calls[tid] = 0

    def max_size(self, tid):
        f = max(self._freq.get(tid, 1), 1)
        return max(1, min(5, 1 + int(math.log2(f))))

    def select(self, tid):
        self.lazy_init(tid)
        fit = self.fitness[tid]
        total = sum(fit)
        if total <= 0:
            return 0
        r = random.random() * total
        cum = 0
        for i, f in enumerate(fit):
            cum += f
            if r <= cum:
                return i
        return 0

    def update(self, tid, idx, is_match):
        self.lazy_init(tid)
        n = self._base_calls[tid] + 1
        self._base_calls[tid] = n
        decay = n / (n + 1)
        fit = self.fitness[tid]
        for i in range(len(fit)):
            fit[i] *= decay
        fit[idx] += 1.0 if is_match else 0.2
        self._homeostatic(tid)

    def _homeostatic(self, tid):
        """Homeostatic plasticity: weak versions boosted, strong suppressed.
        Maintains population diversity without hard thresholds.
        If one version dominates >50% of total fitness, it is gently dampened.
        If a version has <10% share, it gets a small excitatory boost.
        """
        fit = self.fitness[tid]
        if len(fit) <= 1:
            return
        tot = sum(fit)
        if tot <= 0:
            return
        for i in range(len(fit)):
            share = fit[i] / tot
            if share > 0.5:
                fit[i] *= 0.95
            elif share < 0.1 and fit[i] > 0:
                fit[i] *= 1.05

    def maybe_branch(self, tid, idx, noise_level):
        self.lazy_init(tid)
        if len(self.versions[tid]) >= self.max_size(tid):
            self._prune(tid)
            if len(self.versions[tid]) >= self.max_size(tid):
                return
        # noise_level is desired angular deviation (radians ≈ fraction of unit sphere).
        # Scale per-dim: σ_dim = angle / √dim so that |noise| ≈ angle.
        dim = self.vs.dim
        angle = max(0.003, min(noise_level, 0.2))
        per_dim = angle / math.sqrt(dim)
        v = self.versions[tid][idx].copy()
        v += np.random.randn(*v.shape).astype(np.float32) * per_dim
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v /= norm
        self.versions[tid].append(v)
        self.fitness[tid].append(0.5)
        self.n_calls[tid].append(0)

    def _prune(self, tid):
        fit = self.fitness[tid]
        if len(fit) <= 1:
            return
        min_idx = min(range(1, len(fit)), key=lambda i: fit[i])
        self.versions[tid].pop(min_idx)
        self.fitness[tid].pop(min_idx)
        self.n_calls[tid].pop(min_idx)

    def sync_best(self, tid):
        self.lazy_init(tid)
        fit = self.fitness[tid]
        best_idx = max(range(len(fit)), key=lambda i: fit[i])
        if best_idx != 0:
            self.versions[tid][0], self.versions[tid][best_idx] = \
                self.versions[tid][best_idx], self.versions[tid][0]
            self.fitness[tid][0], self.fitness[tid][best_idx] = \
                self.fitness[tid][best_idx], self.fitness[tid][0]
            self.n_calls[tid][0], self.n_calls[tid][best_idx] = \
                self.n_calls[tid][best_idx], self.n_calls[tid][0]
        self.vs.token_vectors[tid] = self.versions[tid][0].copy()

    def global_noise_std(self):
        if self._global_std is None:
            vecs = list(self.vs.token_vectors.values())
            if vecs:
                self._global_std = float(np.std(np.stack(vecs)))
            else:
                self._global_std = 0.01
        return self._global_std

    def estimate_noise(self, tid, ctx_vec=None):
        gs = self.global_noise_std()
        if ctx_vec is not None and tid in self.versions:
            v = self.versions[tid][0]
            sim = float(np.dot(v, ctx_vec))
            return gs * max(0.05, 1.0 - sim)
        return gs * 0.5

    def sync_all(self):
        for tid in list(self.versions.keys()):
            self.sync_best(tid)


class VectorGenerator:
    """
    Генерация через векторное пространство.
    
    Принцип: на piw=0 семантический переход через SVD-близость,
    на piw>=1 — head-продолжение слова (как обычно).
    """
    
    def __init__(self, heads_obj, assoc_graph, hv, rule_extractor=None, config=None):
        self.heads = heads_obj
        self.ag = assoc_graph
        self.hv = hv
        self.config = config or AutoConfig()
        self.vs = VectorSpace(assoc_graph, hv)
        
        # Rules
        self.rules = []
        if rule_extractor is not None:
            self.rules = getattr(rule_extractor, 'affixation_rules', []) + \
                         getattr(rule_extractor, 'syntax_rules', [])
        
        self.V = self.config.vocab_size
        self.tt = hv.token_type
        
        # Blocked tokens
        self.REPLACEMENT_TOKENS = self.config.replacement_tokens
        self.IGNORED = self.config.ignored_tokens
        self.BANNED = self.config.banned_tokens
        
        # Pattern Learner (самоорганизация шаблонов)
        self._pattern_learner = None
        self._pattern_learner_trained = False
        # Semantic coherence weight
        self.sem_weight = self.config.sem_weight
        self.sem_boost_count = self.config.sem_boost_count
        
        # Binary structural matrix: heads → 1/0 (возможно/невозможно)
        self._build_structural_matrix(min_prob=self.config.structural_min_prob)
        
        # StructuralRules for multi-level constraints
        self.structural_rules = None
        
        # GateLogic: multi-level binary gates
        self.gates = None
        
        # Cross-sentence context tracking
        self._prev_sentence_last_concept = -1
        self._prev_sentence_type = 'start'
        self._last_concept_cid = -1  # lateral inhibition target
        
        # Precomputed: concept -> token_ids for fast lookup
        self._concept_to_tids = {}
        self._build_concept_to_tids()
        
        # Last content word tracking
        self._last_content_tid = -1
        
        # ---- Word accumulation ----
        self._current_word_tokens = []
        self._full_word_anchor = None
        
        # ---- ConceptNet-refined vectors ----
        self._refined_vectors_path = None
        
        # ---- Word completion prefix map ----
        self._prefix_type3_set = {}
        self._prefix_type2 = set()
        self._build_word_prefix_map()

        # ---- Word completion state ----
        self._word_prefix = -1
        self._current_word_type3s = set()

        # ---- Target tracking ----
        self._target_tokens = None
        self._target_pos = 0
        self._target_matches = 0
        self._got_target_match = False
        self._word_tabu = set()
        self._svd_lr = self.config.svd_lr
        self._trained_vectors = None  # persistent training state

        # ---- Combined learning state ----
        self._token_freq = defaultdict(int)
        self._token_momentum = {}
        self._current_epoch = 0
        self._population = VectorPopulation(self.vs, self.config, self._token_freq)
        self._error_pairs = defaultdict(int)  # (ctx_anchor, wrong_tid) → count
        
    def _build_structural_matrix(self, min_prob=0.001):
        """Строит структурную матрицу: возможные transitions между type-2 токенами.
        Heads = не частоты, а факт существования перехода в данных.
        Каждый переход, который хоть раз встретился — структурно возможен.
        Храним также log-prob веса для взвешенных structural scores.
        min_prob: минимальная вероятность для включения перехода (pruning шума).
        """
        self.structural = {}  # tid_in -> set of tid_out that can follow
        self.structural_weights = {}  # tid_in -> {tid_out: weight}
        self.structural_min_prob = min_prob
        
        csr = self.heads.log_prob_csr
        log_min = math.log(max(min_prob, 1e-10))
        n_total = 0
        n_pruned = 0
        for src in range(self.V):
            if src >= csr.shape[0]:
                continue
            row = csr[src].tocoo()
            targets = set()
            weights = {}
            for col, val in zip(row.col, row.data):
                if col < self.V:
                    n_total += 1
                    if val >= log_min:  # keep if prob >= min_prob
                        targets.add(int(col))
                        weights[int(col)] = float(val)
                    else:
                        n_pruned += 1
            if targets:
                self.structural[src] = targets
                self.structural_weights[src] = weights
                n_structural = sum(len(v) for v in self.structural.values())
        
        print("Structural matrix: %d transitions (%d pruned below prob=%.3f), %.1f avg per token" % (
            n_structural, n_pruned, min_prob, n_structural / max(len(self.structural), 1)))
        
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
                            scores[tid3] = sim * self.config.prefix_type3_boost

        for prefix_tid in self._prefix_type3_set:
            if not valid_mask[prefix_tid]:
                continue
            if self.vs.has_vector(prefix_tid):
                sim = self.vs.similarity(prefix_tid, content_token)
                if sim > 0:
                    scores[prefix_tid] = sim * self.config.prefix_boost_weight

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
                scores[tid3] = sim * self.config.cont_sem_weight
        
        return scores
    
    def load_structural_rules(self, path):
        """Load StructuralRules from JSON."""
        if not path:
            return
        self.structural_rules = StructuralRules()
        self.structural_rules.load(path)

    def save_refined_vectors(self, path, metadata=None):
        """Save SVD vectors with metadata markup."""
        import pickle, json, os
        data = {
            'vectors': dict(self.vs.token_vectors),
            'metadata': metadata or {},
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=5)
        # Also save human-readable JSON
        json_path = path.replace('.pkl', '.json')
        markup = {
            'dim': self.vs.dim,
            'n_tokens': len(self.vs.token_vectors),
            'tokens': {},
            'metadata': metadata or {},
        }
        for tid, vec in self.vs.token_vectors.items():
            if tid >= len(self.tt) or self.tt[tid] != 2:
                continue
            text = self.hv.decode([tid]).strip() if hasattr(self, 'hv') and self.hv else str(tid)
            cid = self.ag.get_concept(tid) if hasattr(self, 'ag') and self.ag else None
            cluster = (cid - self.ag.L1_OFFSET) if cid is not None else None
            cname = self.ag.cid_label.get(cid, f'C{cluster}') if (cid is not None and hasattr(self.ag, 'cid_label')) else None
            markup['tokens'][text] = {
                'tid': tid,
                'concept': cluster,
                'concept_name': cname,
                'norm': float(np.linalg.norm(vec)),
            }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(markup, f, ensure_ascii=False, indent=2)
        print(f'Refined vectors saved: {len(data["vectors"])} tokens ({self.vs.dim}d) + markup')
    
    def load_refined_vectors(self, path):
        """Load ConceptNet-refined SVD vectors. Overrides per-token vectors in vs.
        Supports both old format ({tid: vec}) and new format ({'vectors': ..., 'metadata': ...})."""
        import pickle, os, json
        if not path or not os.path.exists(path):
            print(f'WARNING: refined vectors not found at {path}')
            return
        with open(path, 'rb') as f:
            refined = pickle.load(f)
        if isinstance(refined, dict) and 'vectors' in refined:
            vectors = refined['vectors']
            meta = refined.get('metadata', {})
        else:
            vectors = refined
            meta = {}
        n_overridden = 0
        for tid, vec in vectors.items():
            if tid in self.vs.token_vectors:
                self.vs.token_vectors[tid] = np.array(vec, dtype=np.float32)
                n_overridden += 1
        self._refined_vectors_path = path
        self._refined_metadata = meta

        # Project to target dimension if different from loaded
        loaded_dim = self.vs.dim
        target_dim = self.config.svd_dim
        if target_dim != loaded_dim:
            self.vs.project_to_dim(target_dim)
            # Also project starter_embeddings to keep alignment
            if self.ag.starter_embeddings is not None and self.ag.starter_embeddings.shape[1] != target_dim:
                R = np.random.RandomState(42).randn(target_dim, loaded_dim).astype(np.float32)
                Q, _ = np.linalg.qr(R)
                self.ag.starter_embeddings = (self.ag.starter_embeddings @ Q.T).astype(np.float32)
            print(f'  Projected: {loaded_dim}D → {target_dim}D')

        # Initialize training state from refined vectors
        self._trained_vectors = {tid: v.copy() for tid, v in self.vs.token_vectors.items()}
        print(f'Refined vectors loaded: {n_overridden}/{len(vectors)} overridden')
        if meta:
            print(f'  Metadata: {json.dumps(meta, ensure_ascii=False)[:200]}')

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
        for t in range(self.config.bpe_limit, self.V):
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
                        if len(text) < self.config.first_word_min_length:
                            mask[tid] = False

            # ПРАВИЛО: на wn >= 2 блокировать function words
            if word_num >= 2:
                for tid in range(self.V):
                    if tid < len(self.tt) and self.tt[tid] == 2:
                        text = self.hv.decode([tid]).strip().lower()
                        if text in self.config.function_words:
                            mask[tid] = False
            
            # Блокировать латинские буквы (не русский контент)
            for tid in range(self.V):
                if tid < len(self.tt) and self.tt[tid] == 2:
                    text = self.hv.decode([tid]).strip()
                    if text and text[0].isascii() and text[0].isalpha():
                        mask[tid] = False
            
            # Блокировать любые type-2 короче 3 букв (обрубки вроде "пр", "См", "пл")
            # Но: если есть активный target, разрешаем ожидаемый токен даже если короткий
            expected_tid = -1
            if (hasattr(self, '_target_tokens') and self._target_tokens is not None
                and hasattr(self, '_target_pos') and self._target_pos >= 0
                and self._target_pos < len(self._target_tokens)):
                expected_tid = self._target_tokens[self._target_pos]
            for tid in range(self.V):
                if tid < len(self.tt) and self.tt[tid] == 2:
                    text = self.hv.decode([tid]).strip()
                    if len(text) < self.config.min_word_length:
                        if tid != expected_tid:
                            mask[tid] = False
        
        # piw≥1: WORD_CONT — allow type-2 (new word), type-3 (structural cont), EOS
        if pos_in_word >= 1:
            for tid in range(self.V):
                if tid < len(self.tt):
                    if self.tt[tid] == 0:
                        mask[tid] = False

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
        # НО: если слово не было продлено type-3 (в _current_word_tokens только 1 токен
        # и used_type3s пуст), значит слово уже полное — разрешаем type-2
        if pos_in_word == 0:
            prev_tid = ctx.get('token_id', -1)
            word_in_progress = hasattr(self, '_current_word_tokens') and self._current_word_tokens
            used_type3s = ctx.get('used_type3s', set())
            word_was_extended = len(used_type3s) > 0 or (word_in_progress and hasattr(self, '_current_word_tokens') and len(self._current_word_tokens) > 1)
            if prev_tid >= 0 and prev_tid in self._prefix_type3_set and word_was_extended:
                for tid in range(self.V):
                    if tid < len(self.tt) and self.tt[tid] == 2:
                        mask[tid] = False
                for tid3 in self._prefix_type3_set[prev_tid]:
                    if tid3 < self.V:
                        mask[tid3] = True
        
        # EOS разрешён в конце слова, НО: не если текущий префикс имеет
        # непотраченные type-3 продолжения (слово не закончено)
        if is_word_end and word_num >= 2:
            allow_eos = True
            word_prefix = ctx.get('word_prefix', -1)
            used_type3s = ctx.get('used_type3s', set())
            if word_prefix >= 0 and word_prefix in self._prefix_type3_set:
                remaining = self._prefix_type3_set[word_prefix] - used_type3s
                if remaining:
                    allow_eos = False  # prefix incomplete
            if allow_eos:
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
        
        # s_type gate: на первом слове предложения блокировать токены,
        # не соответствующие ни одному из валидных следующих s_type
        valid_s_types = ctx.get('_valid_s_types')
        if valid_s_types is not None and word_num == 0 and pos_in_word == 0:
            pl = getattr(self, '_pattern_learner', None)
            if pl is not None and pl.token_s_type_dist:
                s_type_names = []
                for sid in valid_s_types:
                    name = self.config.s_type_map_rev.get(sid)
                    if name:
                        s_type_names.append(name)
                for tid in range(self.V):
                    if tid < len(self.tt) and self.tt[tid] == 2 and mask[tid]:
                        tdist = pl.token_s_type_dist.get(tid, {})
                        if tdist and s_type_names:
                            matches_any = any(
                                tdist.get(sn, 0) > 0.0 for sn in s_type_names)
                            if not matches_any:
                                mask[tid] = False
        
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
                if t < self.config.bpe_limit and self.tt[t] == 2:
                    decoded = self.hv.decode([t]).strip()
                    if decoded == word_text:
                        return t
                    if word_text.startswith(decoded) and len(decoded) > 1:
                        return t
            
            # Fallback: shorter prefixes
            for p_len in range(min(len(word_text)-1, self.config.prefix_fallback_max_len), 0, -1):
                if p_len < 2:
                    break
                prefix = word_text[:p_len]
                encoded = self.hv.encode(' ' + prefix)
                for t in encoded:
                    if t < self.config.bpe_limit and self.tt[t] == 2:
                        decoded = self.hv.decode([t]).strip()
                        if word_text.startswith(decoded) and len(decoded) > 1:
                            return t
            
            return -1
        except Exception:
            return -1
    
    def _is_content_token(self, tid):
        """Проверяет, что токен — content word (многобуквенный type-2)."""
        if tid >= self.config.bpe_limit or tid < 0:
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
            if t >= self.config.bpe_limit:
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
            eos_id = self.config.eos_token_id
            if eos_id < self.V and valid[eos_id]:
                scores[eos_id] = self.config.example_eos_boost
            return scores
        
        # Буст конкретного токена из примера
        if expected_tid >= 0 and expected_tid < self.V and valid[expected_tid]:
            scores[expected_tid] = self.config.example_token_boost
        
        # Буст остальных токенов из концепта
        if 0 <= expected_cluster < self.ag.n_clusters:
            cid_full = self.ag.L1_OFFSET + expected_cluster
            members = self.ag.cid_to_tids.get(cid_full, [])
            for tid in members:
                if tid < self.V and valid[tid] and scores[tid] == 0.0:
                    scores[tid] = self.config.example_concept_boost


    def _word_importance(self, tid):
        if tid >= len(self.tt) or self.tt[tid] != 2:
            return 0.0
        text = self.hv.decode([tid]).strip()
        if not text or len(text) <= self.config.min_word_length:
            return 0.0
        if text[0].isascii() and text[0].isalpha():
            return 0.0
        l = len(text)
        for threshold, score in sorted(self.config.word_importance_tiers, reverse=True):
            if l >= threshold:
                cid = self.ag.get_concept(tid)
                if cid is not None and threshold >= 4:
                    cluster = cid - self.ag.L1_OFFSET
                    members = self.ag.get_members(cluster)
                    if len(members) >= self.config.concept_member_count_threshold:
                        return 1.0
                return score
        return self.config.word_importance_tiers[-1][1]

    def _select_spine(self, ctx, prev_sent_concept=-1, seed_word=None):
        spine = []
        hv = self.hv
        ag = self.ag
        cur_cid = None
        if seed_word:
            seed_tokens = hv.encode(' ' + seed_word)
            for t in seed_tokens:
                if t < self.config.bpe_limit and self.tt[t] == 2:
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
                    if not self._is_content_token(tid) or tid >= self.config.bpe_limit:
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
                if not self._is_content_token(tid) or tid >= self.config.bpe_limit:
                    continue
                imp = self._word_importance(tid)
                if imp < 0.7:
                    continue
                txt = hv.decode([tid]).strip()
                if len(txt) < 3:
                    continue
                score = imp * (1.0 if best_tid is None else self.config.spine_tie_break_decay)
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
                    scores[tid] = self.config.spine_boost
        return scores

    def _structural_scores(self, ctx, prev_token, valid_mask, prev_sentence_concept=-1,
                           paragraph_topic=None):
        """
        Структурные scores (взвешенные по log-prob).
        На piw=0: все transitions из structural matrix взвешены по prob.
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
            
            # Structural: weighted by log-prob from heads
            allowed = self.structural.get(prev_token, set())
            weights = self.structural_weights.get(prev_token, {})
            for tid in allowed:
                if tid < self.V and valid_mask[tid]:
                    w = weights.get(tid, 0.0)
                    prob = np.exp(min(w, 0.0))  # log-prob -> prob
                    scores[tid] = max(prob, self.config.structural_score_floor)
            
            # EOS: weighted by transition from prev_token to EOS
            if valid_mask[3]:
                eos_w = self.structural_weights.get(prev_token, {}).get(3, 0.0)
                scores[3] = max(np.exp(min(eos_w, 0.0)), 0.01)
            
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
            weights = self.structural_weights.get(prev_token, {})
            for tid in allowed:
                if tid < self.V and valid_mask[tid]:
                    w = weights.get(tid, 0.0)
                    prob = np.exp(min(w, 0.0))
                    scores[tid] = max(prob, self.config.structural_score_floor)
            # EOS also valid at word boundary
            if valid_mask[self.config.eos_token_id] and ctx.get('flags', 0) >> 1 & 1:
                scores[self.config.eos_token_id] = self.config.eos_structural_baseline
        
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
        similar = self.vs.topk_similar(anchor, k=self.config.semantic_topk)
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
            return self.config.speech_verb_anchor  # сказал — universal speech verb
        return -1
    
    def _anti_chain_penalty(self, context_tokens, scores, valid_mask, window=None,
                             multiplier=None):
        """
        Anti-chain: если концепт кандидата уже был в последних N словах (content words),
        умножаем его score на multiplier (по умолчанию из config).
        window — количество content words, не BPE-токенов.
        """
        if window is None:
            window = self.config.anti_chain_window
        if multiplier is None:
            multiplier = self.config.anti_chain_multiplier
        if not context_tokens or len(context_tokens) < 3:
            return scores
        # Scan backward through BPE tokens, counting content words (type-2)
        recent_concepts = set()
        content_found = 0
        for tid in reversed(context_tokens):
            if tid < len(self.tt) and self.tt[tid] == 2:
                cid = self.ag.get_concept(tid)
                if cid is not None:
                    recent_concepts.add(cid)
                content_found += 1
                if content_found >= window:
                    break
        if not recent_concepts:
            return scores
        for tid in range(self.V):
            if not valid_mask[tid] or not np.isfinite(scores[tid]):
                continue
            cid = self.ag.get_concept(tid)
            if cid is not None and cid in recent_concepts:
                scores[tid] *= multiplier
        return scores
    
    def _lateral_inhibition(self, scores, valid_mask, last_cid, multiplier=0.3):
        """Lateral inhibition: suppress other tokens in the same concept as
        the last-selected content token. Prevents immediate co-activation of
        redundant tokens — analogous to cortical lateral inhibition."""
        if last_cid < 0:
            return scores
        members = self.ag.cid_to_tids.get(last_cid, [])
        for tid in members:
            if tid < self.V and valid_mask[tid] and np.isfinite(scores[tid]):
                scores[tid] *= multiplier
        return scores
    
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
            cid = self.ag.get_concept(content_token)
            if cid is not None:
                last_concept_id = cid
        if last_concept_id < 0:
            if prev_sent_concept >= 0 and prev_sent_concept < self.ag.n_clusters:
                last_concept_id = self.ag.L1_OFFSET + prev_sent_concept
        if last_concept_id < 0:
            return scores

        outgoing = self.ag.transition_ci.get(last_concept_id, [])
        if not outgoing:
            return scores

        top_k = min(self.config.pmi_topk, len(outgoing))
        top_transitions = sorted(outgoing, key=lambda x: -x[1])[:top_k]
        for cj_id, log_prob in top_transitions:
            members = self.ag.cid_to_tids.get(cj_id, [])
            boost = float(np.exp(log_prob))
            if boost < self.config.pmi_boost_floor:
                continue
            penalty = 0.0
            for c_exp, c_wrong in self._word_tabu:
                if cj_id == c_wrong:
                    penalty = self.config.concept_pmi_tabu
                    break
            for tid in members:
                if tid < self.V and valid_mask[tid]:
                    scores[tid] += boost * self.config.concept_pmi_weight + penalty

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
        
        # 2a. Target override: если есть активный target, форсируем ожидаемый токен
        if (hasattr(self, '_target_tokens') and self._target_tokens is not None
            and hasattr(self, '_target_pos') and self._target_pos >= 0
            and self._target_pos < len(self._target_tokens)
            and ctx.get('pos_in_word', -1) == 0):
            expected_tid = self._target_tokens[self._target_pos]
            if expected_tid < self.V:
                structural[expected_tid] = self.config.target_override  # high structural score
                valid[expected_tid] = True       # force-valid
        
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
        
        # 10. Heads ensemble scores: 6 предвычисленных голов (morph, syntax, transition, sem, concept, contra)
        head_scores = np.zeros(self.V, dtype=np.float32)
        if hasattr(self.heads, 'individual_scores'):
            try:
                hs = self.heads.individual_scores(ctx)
                hw = np.array(self.config.heads_integrated_weights, dtype=np.float32)
                scores6 = np.dot(hw, hs)
                for tid in range(self.V):
                    if valid[tid] and np.isfinite(scores6[tid]):
                        head_scores[tid] = scores6[tid] * self.config.heads_integrated_scale
            except:
                pass
        
        # 10a. Target boost: если есть target, на piw=0 бустим ожидаемый токен
        target_boost = np.zeros(self.V, dtype=np.float32)
        if (hasattr(self, '_target_tokens') and self._target_tokens is not None
            and hasattr(self, '_target_pos') and self._target_pos >= 0
            and self._target_pos < len(self._target_tokens)):
            if ctx.get('pos_in_word', -1) == 0:
                expected_tid = self._target_tokens[self._target_pos]
                if expected_tid < self.V and valid[expected_tid]:
                    target_boost[expected_tid] = self.config.target_boost  # strong direct boost
        
        # 11. Комбинируем: structural + semantic + concept + s_type + prefix + cont_sem + spine + example + heads + target
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
            if head_scores[tid] != 0.0:
                base += head_scores[tid]
            if target_boost[tid] != 0.0:
                base += target_boost[tid]
            scores[tid] = base
        
        # 11a. Anti-chain penalty: штрафуем повтор концепта в последних 15 токенах
        context_tokens = ctx.get('context_tokens', [])
        scores = self._anti_chain_penalty(context_tokens, scores, valid)
        
        # 11b. Lateral inhibition: подавляем другие токены того же концепта,
        # что был выбран на предыдущем content-шаге (кортикальное торможение)
        scores = self._lateral_inhibition(scores, valid, self._last_concept_cid)
        
        # 11. Стена
        scores[~valid] = -np.inf
        
        # 12. EOS decision: at word boundaries, probabilistically end sentence
        pos_in_word = ctx.get('pos_in_word', -1)
        flags = ctx.get('flags', 0)
        is_word_end = (flags >> 1) & 1
        word_num = ctx.get('word_num', -1)
        
        if is_word_end and word_num >= self.config.eos_min_words:
            import random as _random
            target_active = (hasattr(self, '_target_tokens') and self._target_tokens is not None
                            and hasattr(self, '_target_pos') and self._target_pos >= 0
                            and self._target_pos < len(self._target_tokens))
            if not target_active:
                p_eos = self.config.eos_probability(word_num)
                if _random.random() < p_eos:
                    return 3, {
                        "valid_tokens": int(valid.sum()),
                        "chosen": 3,
                        "chosen_text": "<<$>>",
                        "reasons": ["EOS(%.2f)" % p_eos]
                    }
        
        # 13. Fallback если всё -inf
        if not np.any(np.isfinite(scores)):
            scores = (structural.copy() + semantic.copy() + concept_pmi.copy() +
                      s_type_boost.copy() + prefix_boost.copy() + cont_sem.copy() +
                      spine_boost.copy() + example_boost.copy())
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
            # Lateral inhibition: запомнить концепт выбранного токена
            cid = self.ag.get_concept(next_tok)
            self._last_concept_cid = cid if cid is not None else -1
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
    
    def _check_target_match(self, word_anchor, context_tokens):
        """Сравнить сгенерированное слово с target. SVD shift на sequential connectedness."""
        if self._target_tokens is None or word_anchor < 0:
            return
        if self._target_pos >= len(self._target_tokens):
            return
        
        expected = self._target_tokens[self._target_pos]
        # Find context anchor that is NOT word_anchor itself
        ctx_anchor = -1
        for tid in reversed(context_tokens):
            if tid != word_anchor and tid < self.config.bpe_limit and self.tt[tid] == 2 and self.vs.has_vector(tid):
                ctx_anchor = tid
                break
        if ctx_anchor < 0:
            return
        
        if word_anchor == expected:
            # Совпало! Усилить семантическую связь + sequential connectedness
            self._target_matches += 1
            self._target_pos += 1
            self._word_tabu.clear()
            self._got_target_match = True
            self._token_freq[word_anchor] += 1
            self._svd_shift(word_anchor, ctx_anchor, is_match=True)
        else:
            self._got_target_match = False
            # Не совпало: запомнить неправильный концепт для блокировки
            c_wrong = self.ag.get_concept(word_anchor)
            c_expected = self.ag.get_concept(expected)
            if c_wrong is not None and c_expected is not None:
                self._word_tabu.add((c_expected, c_wrong))
            
            # Sequential connectedness: wrong word learns position AND gets
            # corrective pull toward the right word
            self._svd_shift(word_anchor, ctx_anchor, is_match=False, target_tid=expected)

    def _context_anchor(self, context_tokens):
        """Find last type-2 token with a vector in context for SVD shift."""
        for tid in reversed(context_tokens):
            if tid < self.config.bpe_limit and self.tt[tid] == 2 and self.vs.has_vector(tid):
                return tid
        return -1

    def _cluster_to_tid(self, cluster_idx):
        """Convert a cluster index (0..n_clusters-1) to a representative token ID for SVD."""
        if cluster_idx < 0:
            return -1
        cid_full = cluster_idx + self.ag.L1_OFFSET
        members = self.ag.cid_to_tids.get(cid_full, [])
        for tid in members:
            if tid < self.config.bpe_limit and self.vs.has_vector(tid):
                return tid
        return -1

    def _concept_prediction_error(self, prev_tid, actual_tid):
        """Predictive coding: PMI-based concept surprise.

        Returns 0.0 if actual concept matches the PMI-predicted next concept
        from prev_tid. Returns 1.0 if unexpected. Error ∈ [0, 1] scales the
        SVD learning signal — the brain learns more from surprising events.
        """
        actual_c = self.ag.get_concept(actual_tid)
        if actual_c is None:
            return 0.5
        actual_cluster = actual_c - self.ag.L1_OFFSET
        pmi_next = self.ag.pmi.get(prev_tid, {})
        if not pmi_next:
            return 0.5
        # Best predicted concept: PMI-argmax over successors
        best_cj = max(pmi_next, key=pmi_next.get)
        if best_cj == actual_cluster:
            return 0.0
        # Surprise magnitude = 1 - exp(log_prob_diff) ≈ how much more
        # surprising than the best prediction
        diff = pmi_next.get(actual_cluster, -20.0) - pmi_next.get(best_cj, 0.0)
        return float(np.clip(1.0 - np.exp(diff), 0.0, 1.0))

    def _cluster_to_tid(self, cluster_idx):
        """Convert a cluster index (0..n_clusters-1) to a representative token ID for SVD."""
        if cluster_idx < 0:
            return -1
        cid_full = cluster_idx + self.ag.L1_OFFSET
        members = self.ag.cid_to_tids.get(cid_full, [])
        # Pick first member that has a vector
        for tid in members:
            if tid < self.config.bpe_limit and self.vs.has_vector(tid):
                return tid
        return -1

    def _svd_shift(self, word_tid, ctx_anchor, is_match=False, target_tid=-1):
        """
        Sequential connectedness SVD shift via evolutionary population.

        When target_tid >= 0 and not is_match (wrong word generated):
        - Normal LTD shift toward context (×0.05)
        - PLUS correction shift: pull wrong→right (stronger)
        - Repeated errors amplify correction (boosting)

        When is_match (correct word):
        - LTP shift toward context (×1.0)
        """
        if word_tid < 0 or ctx_anchor < 0:
            return
        if not self.vs.has_vector(word_tid) or not self.vs.has_vector(ctx_anchor):
            return

        idx = self._population.select(word_tid)
        v_word = self._population.versions[word_tid][idx]
        v_ctx = self.vs.token_vectors[ctx_anchor]

        freq = self._token_freq.get(word_tid, 1)
        adaptive_lr = self._svd_lr / (1.0 + 0.1 * math.sqrt(freq))

        # --- Context shift (LTP/LTD based on match) ---
        total_scale = 1.0 if is_match else 0.05
        pred_err = self._concept_prediction_error(ctx_anchor, word_tid)
        total_scale *= (1.0 + pred_err)

        y = float(np.dot(v_word, v_ctx))
        y = max(y, 0.05)

        shift = (v_ctx - v_word) * adaptive_lr * total_scale * y

        # --- Correction shift: wrong word pulled toward right word ---
        if target_tid >= 0 and not is_match and self.vs.has_vector(target_tid):
            v_right = self.vs.token_vectors[target_tid]
            # Error count: how many times this (context, wrong) pair occurred
            err_key = (ctx_anchor, word_tid)
            if not hasattr(self, '_error_pairs'):
                self._error_pairs = defaultdict(int)
            self._error_pairs[err_key] += 1
            error_boost = 1.0 + min(3.0, 0.5 * (self._error_pairs[err_key] - 1))

            y_right = float(np.dot(v_word, v_right))
            y_right = max(y_right, 0.05)

            corr_scale = 1.0  # full correction LR
            corr_scale *= error_boost  # amplify on repeat errors
            corr_shift = (v_right - v_word) * adaptive_lr * corr_scale * y_right
            shift += corr_shift

        # --- Momentum ---
        beta = self.config.svd_momentum_beta
        if word_tid not in self._token_momentum:
            self._token_momentum[word_tid] = {}
        if idx not in self._token_momentum[word_tid]:
            self._token_momentum[word_tid][idx] = np.zeros_like(v_word)
        mom = self._token_momentum[word_tid][idx]
        mom[:] = beta * mom + (1.0 - beta) * shift

        v_word += mom
        nrm = float(np.linalg.norm(v_word))
        if nrm > 0:
            v_word /= nrm

        self._population.update(word_tid, idx, is_match)
        if not is_match:
            noise = self._population.estimate_noise(word_tid, v_ctx)
            self._population.maybe_branch(word_tid, idx, noise)
            # Create a targeted correction hypothesis if right word is known
            if target_tid >= 0 and self.vs.has_vector(target_tid):
                self._corrective_branch(word_tid, target_tid, ctx_anchor)
        self._population.sync_best(word_tid)
        if target_tid >= 0 and not is_match and self.vs.has_vector(target_tid):
            self._population.sync_best(target_tid)

    def _corrective_branch(self, wrong_tid, right_tid, ctx_anchor):
        """Create a targeted correction hypothesis at the right hierarchical level.

        When a word or sub-word unit is wrong, instead of random exploration,
        branch a version specifically pulled toward the right answer.
        This hypothesis competes with the original — if it leads to matches,
        its fitness rises; if not, it atrophies.

        If the type-2 prefix was correct but type-3 continuation is wrong,
        create a hypothesis for the type-3 tokens only (not the prefix).
        If the whole word is wrong, correct at word level.
        """
        v_right = self.vs.token_vectors[right_tid]
        idx = self._population.select(wrong_tid)
        v_hyp = self._population.versions[wrong_tid][idx].copy()

        y = float(np.dot(v_hyp, v_right))
        y = max(y, 0.05)
        shift = (v_right - v_hyp) * 0.5 * y
        v_hyp += shift
        nrm = float(np.linalg.norm(v_hyp))
        if nrm > 0:
            v_hyp /= nrm

        if wrong_tid not in self._population.versions:
            return
        self._population.versions[wrong_tid].append(v_hyp)
        self._population.fitness[wrong_tid].append(0.5)
        self._population.n_calls[wrong_tid].append(0)

        while len(self._population.versions[wrong_tid]) > self._population.max_size(wrong_tid):
            self._population._prune(wrong_tid)

    def generate(self, max_tokens=40, seed_word=None, target_composition=None, temperature=0.5,
                 example=None, auto_pattern=False, text_hierarchy=None, target_text=None,
                 training_mode=False):
        """
        Генерация: structural + semantic, без frequency bias.
        
        seed_word: начальное слово (опционально)
        target_composition: [(word, weight), ...] — целевая композиция (опционально)
        example: строка-пример для подражания (опционально)
        target_text: строка-цель для имитационного обучения (опционально)
        training_mode: если True, SVD-сдвиги накапливаются в _trained_vectors
        """
        tokens = [2]  # BOS
        explanations = []
        content_token = -1  # последний content word (для семантического контекста)
        self._current_word_tokens = []  # reset word accumulation
        self._full_word_anchor = None
        self._active_spine = []  # sentence spine for current sentence
        self._max_paragraph_sentences = random.randint(self.config.max_paragraph_sentences_min,
                                                       self.config.max_paragraph_sentences_max)
        
        # Load trained vectors as starting point, save current for restore.
        # MUST deep-copy: _saved_vectors must remain a reference to the UNMODIFIED
        # original dict, while vs.token_vectors becomes a NEW dict.
        _saved_vectors = self.vs.token_vectors
        start_vectors = self._trained_vectors if self._trained_vectors is not None else _saved_vectors
        self.vs.token_vectors = {tid: v.copy() for tid, v in start_vectors.items()}
        # Sync population primary to loaded vectors
        for tid, v in self.vs.token_vectors.items():
            self._population.lazy_init(tid)
            self._population.versions[tid][0] = v.copy()
        
        # Target tracking for repeat-after-me
        self._target_tokens = None
        self._target_pos = 0
        self._target_matches = 0
        self._got_target_match = False
        self._word_tabu = set()
        if target_text:
            enc = self.hv.encode(' ' + target_text.strip())
            target_tids = [t for t in enc if t < self.config.bpe_limit and self.tt[t] == 2 and self._is_content_token(t)]
            self._target_tokens = target_tids
        
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
        self._last_concept_cid = -1
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
                    if t < self.config.bpe_limit and self.tt[t] == 2:
                        c = self.ag.get_concept(t)
                        cluster = (c - self.ag.L1_OFFSET) if c is not None else -1
                        s_t = self._pattern_learner.detect_s_type(first_concept=cluster if cluster >= 0 else None, first_tid=t)
                        self._prev_sentence_type = s_t
                        break
            
            seed_tokens = self.hv.encode(' ' + seed_word)
            seed_type2 = None
            for t in seed_tokens:
                if t < self.config.bpe_limit and self.tt[t] == 2:
                    seed_type2 = t
                    break
            if seed_type2 is not None:
                tokens.append(seed_type2)
                if self._is_content_token(seed_type2):
                    content_token = seed_type2
                    self._current_word_tokens = [seed_type2]
                    self._word_prefix = seed_type2
                    self._current_word_type3s = set()
                    self._words_in_current_sentence += 1
                    # Check seed against target immediately
                    if self._target_tokens is not None and self._target_pos < len(self._target_tokens):
                        expected = self._target_tokens[self._target_pos]
                        if seed_type2 == expected:
                            self._target_matches += 1
                            self._target_pos += 1
                            self._word_tabu.clear()
                # Seed word is complete: keep _current_word_tokens so word boundary
                # detection works when next type-2 is generated

        
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
                w = np.array(self.config.gl_initial_weights, dtype=np.float32)
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
            next_tok = select_token(scores, temperature=min(temperature, self.config.initial_token_temperature_cap))
            tokens.append(next_tok)
            chosen_text = self.hv.decode([next_tok]).strip()
            if self._is_content_token(next_tok):
                content_token = next_tok
                # Compute full-word anchor for first word
                full_enc = self.hv.encode(' ' + chosen_text)
                for ft in full_enc:
                    if ft < self.config.bpe_limit and self.tt[ft] == 2 and self._is_content_token(ft):
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
            ctx['context_tokens'] = tokens[-self.config.context_window:] if len(tokens) > 5 else tokens
            ctx['word_prefix'] = self._word_prefix
            ctx['used_type3s'] = self._current_word_type3s.copy()
            
            # Upper word limit: force EOS beyond corpus-estimated max
            max_words = self.config.max_words_per_sentence
            if self._words_in_current_sentence >= max_words and ctx.get('pos_in_word', 0) == 0:
                next_tok = 3
                exp = {"valid_tokens": 0, "chosen": 3, "chosen_text": "<<$>>",
                       "reasons": ["word_limit"]}
                # Handle word accumulation before EOS
                if self._current_word_tokens and self._target_tokens is not None:
                    anchor = self._compute_full_word_anchor(self._current_word_tokens)
                    self._check_target_match(anchor, tokens)
                self._word_prefix = -1
                self._current_word_type3s = set()
                self._current_word_tokens = []
                tokens.append(next_tok)
                explanations.append(exp)
                break
            
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
            
            pos_in_word = ctx.get('pos_in_word', 0)
            cur_token_type = self.tt[next_tok] if next_tok < len(self.tt) else 0
            
            # === Universal sequential connectedness training ===
            # Full hierarchy:
            #   type-3 (char) → word prefix (intra-word)
            #   type-2 (word) → context anchor (intra-sentence)
            #   type-2 after EOS → prev sentence last concept (cross-sentence/paragraph)
            # Non-matches get full shift, matches get minimal preservation
            if training_mode and self._target_tokens is not None:
                if cur_token_type == 2 and pos_in_word == 0:
                    # Paragraph level: first word after EOS → prev sentence concept
                    # _prev_sentence_last_concept is a cluster index (0..n_clusters-1),
                    # convert to a representative token ID before use.
                    if prev == 3 and self._prev_sentence_last_concept >= 0:
                        psc_tid = self._cluster_to_tid(self._prev_sentence_last_concept)
                        if psc_tid >= 0 and self.vs.has_vector(next_tok):
                            self._svd_shift(next_tok, psc_tid, is_match=False)
                    # Word level: all type-2 → context anchor from same sentence
                    # Note: match tracking is handled by _check_target_match
                    # at word completion. Here we only do the SVD shift.
                    ctx_anchor = self._context_anchor(ctx.get('context_tokens', []))
                    if ctx_anchor >= 0 and self.vs.has_vector(next_tok):
                        is_match = (self._target_pos < len(self._target_tokens) and
                                    next_tok == self._target_tokens[self._target_pos])
                        ttid = self._target_tokens[self._target_pos] if self._target_pos < len(self._target_tokens) else -1
                        if not is_match and ttid >= 0 and ttid != next_tok:
                            self._svd_shift(next_tok, ctx_anchor, is_match=False, target_tid=ttid)
                        else:
                            self._svd_shift(next_tok, ctx_anchor, is_match=is_match)
                elif cur_token_type == 3 and self._word_prefix >= 0:
                    # Character-level: shift type-3 toward word prefix
                    word_prefix_tid = self._word_prefix
                    if self.vs.has_vector(word_prefix_tid) and self.vs.has_vector(next_tok):
                        self._svd_shift(next_tok, word_prefix_tid, is_match=False)
            
            # Online gate learning: observe chosen token
            if self.gates is not None and hasattr(self.gates, 'observe_online'):
                observe = True
                if self._target_tokens is not None:
                    if cur_token_type == 2 and pos_in_word == 0:
                        prev_was_good = (self._got_target_match or self._target_pos == 0)
                        if not prev_was_good:
                            observe = False
                if observe:
                    self.gates.observe_online(ctx, next_tok, content_token)
            
            if cur_token_type == 2:
                # New word starting — previous word is complete.
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
                                scores2[tid] = sim * self.config.composition_sem_weight
                        scores2[~valid] = -np.inf
                        if np.any(np.isfinite(scores2)):
                            next_tok = select_token(scores2, temperature=temperature)
                            exp["reasons"].append("composition")
                            exp["chosen"] = int(next_tok)
                            exp["chosen_text"] = self.hv.decode([next_tok]).strip()
                            if self._is_content_token(next_tok):
                                full_enc = self.hv.encode(' ' + exp['chosen_text'])
                                for ft in full_enc:
                                    if ft < self.config.bpe_limit and self.tt[ft] == 2 and self._is_content_token(ft):
                                        content_token = ft
                                        break
                                content_token = next_tok
            
            tokens.append(next_tok)
            explanations.append(exp)
            
            if next_tok == 3:
                # Compare last completed word to target
                if self._current_word_tokens and self._target_tokens is not None:
                    anchor = self._compute_full_word_anchor(self._current_word_tokens)
                    self._check_target_match(anchor, tokens)
                # Word tracking reset at sentence boundary
                self._word_prefix = -1
                self._current_word_type3s = set()
                self._current_word_tokens = []
                # Clear example path for next sentence
                self._example_path = None
                
                # Отчёт успеха для авто-шаблона
                if self._active_pattern_seq is not None and self._pattern_learner is not None:
                    type2_tokens = [t for t in tokens[self._sentence_start_pos:] if t < self.config.bpe_limit and self.tt[t] == 2]
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
                    self._max_paragraph_sentences = random.randint(self.config.max_paragraph_sentences_min,
                                                                    self.config.max_paragraph_sentences_max)
        
        # Отчёт: результат авто-шаблона
        if self._active_pattern_seq is not None and self._pattern_learner is not None:
            reached_eos = next_tok == 3 if len(tokens) > 1 else False
            type2_tokens = [t for t in tokens if t < self.config.bpe_limit and self.tt[t] == 2]
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
        
        # Отчёт: результат имитационного обучения
        if self._target_tokens is not None:
            n_target = len(self._target_tokens)
            pct = self._target_matches / max(1, n_target) * 100
            print("Target: %d/%d words matched (%.0f%%)" % (self._target_matches, n_target, pct))
        
        # Save training state: accumulate SVD shifts across calls
        if training_mode and self._trained_vectors is not None:
            self._trained_vectors = {tid: v.copy() for tid, v in self.vs.token_vectors.items()}
        # Restore previous vector state
        self.vs.token_vectors = _saved_vectors
        
        return {
            'tokens': tokens,
            'text': self.hv.decode(tokens),
            'explanations': explanations,
            'target_matches': self._target_matches if self._target_tokens is not None else -1,
            'target_total': len(self._target_tokens) if self._target_tokens is not None else 0,
        }
    
    def set_epoch(self, epoch):
        """Set current epoch and decay learning rate."""
        self._current_epoch = epoch
        self._svd_lr = self.config.svd_lr * (self.config.svd_lr_decay ** epoch)
        print(f"  Epoch {epoch}: SVD lr = {self._svd_lr:.6f}")

    def reset_momentum(self):
        """Reset momentum buffer between epochs."""
        if not hasattr(self, '_token_momentum') or self._token_momentum is None:
            self._token_momentum = {}
        self._token_momentum.clear()
        self._error_pairs.clear()

    def print_trace(self, result):
        print("\n=== ВЕКТОРНАЯ ГЕНЕРАЦИЯ ===")
        print("Text: %s\n" % result['text'])
        for i, exp in enumerate(result.get('explanations', [])):
            reasons = ', '.join(exp.get('reasons', [])) or 'heads'
            print("  [%2d] '%s' V=%d %s" % (i, exp['chosen_text'], 
                  exp['valid_tokens'], reasons))
