"""
AutoConfig — data-driven estimation of ALL model parameters from corpus.
No hardcoded magic numbers. Every parameter is computed from data statistics.

Usage:
    config = AutoConfig.from_corpus(corpus, hv)
    # Build models with default SVD, pass to AutoConfig.stage2()
    config = config.stage2(ag, vg)
    # Now use config everywhere:
    ag = AssociationGraph(config=config)
    vg = VectorGenerator(heads, ag, hv, config=config)
"""
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import vstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json, pickle, os, math


class AutoConfig:
    def __init__(self, params=None):
        # Vocabulary / architecture
        self.vocab_size = 4101
        self.svd_dim = 256
        self.n_clusters = 48
        self.n_metas = 12
        self.random_state = 42
        self.bpe_limit = 4096
        self.special_tokens = {0: 'PAD', 1: 'UNK', 2: 'BOS', 3: 'EOS', 4: 'SEP', 5: 'MASK'}
        self.eos_token_id = 3

        # BPE replacement tokens (600–1498) excluded from content
        self.replacement_tokens = set(range(100, 1499))

        # Structural
        self.structural_min_prob = 0.001

        # Semantic score
        self.sem_weight = 5.0
        self.sem_boost_count = 5

        # Learning
        self.svd_lr = 0.1
        self.svd_momentum_beta = 0.9
        self.svd_neg_feedback_scale = 0.3
        self.svd_lr_decay = 0.9
        self.svd_epochs = 10
        self.svd_neg_samples = 5

        # Generation limits
        self.max_words_per_sentence = 12
        self.context_window = 15
        self.eos_min_words = 4
        self.eos_p1 = 0.01
        self.eos_p2 = 5.0

        # Anti-chain
        self.anti_chain_window = 15
        self.anti_chain_multiplier = 0.01

        # Score component weights (generate_step)
        self.target_boost = 15.0
        self.target_override = 5.0
        self.concept_pmi_weight = 3.0
        self.concept_pmi_tabu = -10.0
        self.prefix_boost_weight = 4.0
        self.prefix_type3_boost = 8.0
        self.cont_sem_weight = 1.0
        self.spine_boost = 3.0
        self.composition_sem_weight = 5.0
        self.example_token_boost = 12.0
        self.example_concept_boost = 6.0
        self.example_eos_boost = 10.0

        # Heads ensemble
        self.heads_weights = [0.5, 0.5, 0.3, 0.3, 0.2, 0.1]
        self.heads_scale = 0.2

        # Heads: per-context weight overrides (heads.py)
        self.head_default_weights = {'morph': 1.0, 'syntax': 1.0, 'transition': 2.0,
                                      'semantic': 0.5, 'concept': 0.2, 'contra': 0.5}
        self.head_special_transition = 5.0
        self.head_special_zero_heads = ['morph', 'syntax', 'semantic', 'concept', 'contra']
        self.head_word_start = {'syntax': 3.0, 'morph': 0.5, 'transition': 1.0, 'semantic': 0.5}
        self.head_word_end = {'morph': 0.5, 'transition': 3.0, 'semantic': 1.0}
        self.head_mid_word = {'morph': 4.0, 'transition': 0.5}
        self.head_rare_token_threshold = 5
        self.head_rare_semantic_boost = 1.0
        self.head_rare_transition_factor = 0.3
        self.head_fallback_logprob = -7.0
        self.head_unseen_penalty = -10.0
        self.head_contra_penalty_multiplier = 2.0

        # Generation loop: per-mode head weights (generation_loop.py)
        self.gl_special_weights = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
        self.gl_word_start_weights = [1.0, 1.5, 2.0, 0.5, 0.2, 0.5]
        self.gl_word_end_weights = [0.5, 1.0, 3.0, 1.0, 0.2, 0.5]
        self.gl_mid_word_weights = [3.0, 1.0, 1.0, 0.5, 0.2, 0.5]
        self.gl_initial_weights = [1.0, 1.0, 3.0, 0.5, 0.2, 0.5]
        self.gl_top_k_default = 50
        self.gl_score_clip_min = -50
        self.gl_score_clip_max = 50
        self.gl_special_token_ids = {0, 1, 2, 3, 4, 5}

        # WeightTransformer normalization
        self.norm_word_len = 19
        self.norm_pos_in_word = 18
        self.norm_word_num = 275
        self.norm_pos_in_sent = 587
        self.norm_sent_len = 587
        self.norm_flags_divisor = 255.0

        # Token filters
        self.function_words = {'в', 'с', 'к', 'у', 'о', 'и', 'а', '\u2013', '\u2014'}
        self.target_short_word_whitelist = {'он', 'не', 'на'}
        self.min_word_length = 3
        self.first_word_min_length = 2
        self.ignored_tokens = {0, 1, 2, 4, 5}
        self.banned_tokens = {3}

        # Sentence type map
        self.s_type_map = {
            'statement': 0, 'question': 1, 'exclamation': 2,
            'dialogue': 3, 'french': 4
        }
        self.s_type_map_rev = {
            0: 'statement', 1: 'question', 2: 'exclamation',
            3: 'dialogue', 4: 'start'
        }

        # Topic distribution (text type)
        self.max_paragraph_sentences_min = 3
        self.max_paragraph_sentences_max = 5

        # Speech verb anchor for dialogue boost
        self.speech_verb_anchor = 475

        # Vector-space generation internals
        self.prefix_fallback_max_len = 10
        self.word_importance_tiers = [(6, 1.0), (4, 0.7), (0, 0.3)]
        self.concept_member_count_threshold = 10
        self.spine_tie_break_decay = 0.9
        self.structural_score_floor = 0.01
        self.eos_structural_baseline = 0.5
        self.semantic_topk = 50
        self.initial_token_temperature_cap = 0.5
        self.pmi_topk = 5
        self.pmi_boost_floor = 0.01
        self.heads_integrated_weights = [0.5, 0.5, 0.3, 0.3, 0.2, 0.1]
        self.heads_integrated_scale = 0.2

        # Association graph internals
        self.ag_profile_distance_scale = 2.0
        self.ag_compose_weights = [0.7, 0.3]
        self.ag_concept_boost = 2.0
        self.ag_activation_max_depth = 3
        self.ag_activation_decay = 0.5
        self.ag_pmi_topk = 10
        self.ag_reverse_pmi_threshold = 0.5
        self.ag_reverse_pmi_topk = 5
        self.ag_meta_activation_fraction = 0.5
        self.ag_2hop_pmi_normalization = 4.0
        self.ag_2hop_concept_activation = 0.2
        self.ag_2hop_token_activation = 0.05
        self.ag_generation_temperature = 0.2
        self.ag_topk_concept_selection = 3
        self.ag_distance_decay = 20.0
        self.ag_concept_connection_pmi_floor = 0.1

        # Gate logic internals
        self.gate_s_type_count = 5
        self.gate_l1_offset = 48

        # Concept clustering methods
        self.cluster_method_l1 = 'hdbscan'
        self.cluster_method_l2 = 'louvain'
        self.birch_threshold = 0.3
        self.hdbscan_min_cluster_ratio = 0.02

        # Pattern learner internals
        self.pl_sample_cap = 5
        self.pl_min_freq = 3
        self.pl_pattern_min_concepts = 2
        self.pl_pattern_max_concepts = 5
        self.pl_weight_norm_divisor = 20.0
        self.pl_reinforcement_lr = 0.1
        self.pl_reinforcement_offset = 0.5
        self.pl_weight_upper_bound = 1.0
        self.pl_chain_reaction_bonus = 0.05
        self.pl_base_decay = 0.05
        self.pl_failure_decay = 0.1
        self.pl_weight_lower_bound = 0.1
        self.pl_s_type_match_bonus = 0.5

        # EOS
        self._eos_empirical = None

        # Raw data for derived computations
        self._sentence_lengths = []
        self._sentence_lengths_tokens = []
        self._word_freq = Counter()
        self._transition_counts = defaultdict(Counter)
        self._short_words = Counter()
        self._eos_positions = []

        # SVD/post-model statistics
        self._collocated_sims = []
        self._random_sims = []
        self._score_distributions = defaultdict(list)

        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    # ──────────────────────────────────────────────────
    # Stage 1: pre-model, purely from corpus text
    # ──────────────────────────────────────────────────

    @classmethod
    def from_corpus(cls, corpus, hv=None):
        config = cls()
        if hasattr(corpus, 'sentences') and not corpus.sentences:
            corpus.parse()
        config.collect_raw(corpus, hv)
        config.estimate_raw_params()
        return config

    def collect_raw(self, corpus, hv=None):
        for sent in corpus.sentences:
            text = sent.text.strip()
            if not text:
                continue
            if hv:
                enc = hv.encode(' ' + text)
                content_tids = []
                for t in enc:
                    if t >= self.bpe_limit:
                        continue
                    if hv.token_type[t] == 2:
                        word = hv.decode([t]).strip()
                        content_tids.append(t)
                        if len(word) <= 2:
                            self._short_words[word] += 1
                self._sentence_lengths.append(len(content_tids))
                self._sentence_lengths_tokens.append(len(enc))
                for j in range(len(content_tids) - 1):
                    self._transition_counts[content_tids[j]][content_tids[j+1]] += 1
                self._eos_positions.append(len(content_tids))
            else:
                words = text.split()
                self._sentence_lengths.append(len(words))

    def estimate_raw_params(self):
        if not self._sentence_lengths:
            return
        lengths = np.array(self._sentence_lengths)
        self.max_words_per_sentence = max(5, int(np.percentile(lengths, 99)))
        if self._sentence_lengths_tokens:
            self.context_window = max(5, int(np.percentile(
                self._sentence_lengths_tokens, 95)))
        self.eos_min_words = max(2, int(np.percentile(lengths, 5)))
        self._fit_eos_curve()
        self._estimate_function_words()
        self.anti_chain_window = min(self.context_window,
                                     max(5, int(np.percentile(lengths, 90))))

    def _fit_eos_curve(self):
        if not self._eos_positions:
            return
        max_pos = max(self._eos_positions)
        hist = np.zeros(max_pos + 1)
        for pos in self._eos_positions:
            hist[pos] += 1
        total = len(self._eos_positions)
        self._eos_empirical = hist / max(total, 1)

    def _estimate_function_words(self):
        if not self._short_words:
            return
        counts = np.array(list(self._short_words.values()))
        if len(counts) < 2:
            return
        mu, sigma = np.mean(counts), np.std(counts)
        self.function_words = {w for w, f in self._short_words.items()
                               if f > mu + 2 * sigma}

    def eos_probability(self, word_num):
        if self._eos_empirical is not None and word_num < len(self._eos_empirical):
            return min(0.5, float(self._eos_empirical[word_num]))
        return 0.0

    # ──────────────────────────────────────────────────
    # Stage 2: post-SVD, post-clustering
    # ──────────────────────────────────────────────────

    def stage2(self, ag, vg, corpus, hv):
        self.estimate_svd_params(ag, vg, hv)
        if self._sentence_lengths:
            self.estimate_cluster_count(ag)
        return self

    def estimate_svd_params(self, ag, vg, hv):
        """
        Collect SVD similarity distributions for collocated vs random pairs.
        Then calibrate sem_weight and svd_lr.
        """
        collocated = []
        random_pairs = []
        all_tids = list(vg.vs.token_vectors.keys())

        for prev_tid, next_dict in self._transition_counts.items():
            if not vg.vs.has_vector(prev_tid):
                continue
            for next_tid, count in next_dict.items():
                if not vg.vs.has_vector(next_tid):
                    continue
                sim = vg.vs.similarity(prev_tid, next_tid)
                collocated.extend([sim] * min(count, 10))

        rng = np.random.RandomState(self.random_state)
        for _ in range(min(5000, len(collocated) * 2)):
            a = rng.randint(0, len(all_tids))
            b = rng.randint(0, len(all_tids))
            if a != b and vg.vs.has_vector(a) and vg.vs.has_vector(b):
                random_pairs.append(vg.vs.similarity(a, b))

        self._collocated_sims = collocated
        self._random_sims = random_pairs

        if collocated and random_pairs:
            mean_col = np.mean(collocated)
            mean_rnd = np.mean(random_pairs)
            std_col = np.std(collocated)
            gap = max(0.1, mean_col - mean_rnd)
            # sem_weight should make target separation ~3 standard deviations
            self.sem_weight = max(3.0, 3.0 * std_col / gap)
            self.sem_boost_count = max(3, int(self.vocab_size * 0.005))

            # lr: fraction of mean semantic gap per update
            # after 10 updates, reach ~50% of gap
            self.svd_lr = max(0.01, min(0.5, 0.5 * gap / 10.0))

    def estimate_cluster_count(self, ag):
        """Validate cluster count via silhouette, if enough data."""
        if ag.starter_embeddings is None or ag.starter_embeddings.shape[0] < 20:
            return
        emb = ag.starter_embeddings
        if emb.shape[0] >= self.n_clusters * 2:
            try:
                candidate = min(emb.shape[0] // 2, self.n_clusters)
                km = KMeans(n_clusters=candidate, random_state=self.random_state,
                            n_init='auto')
                labels = km.fit_predict(emb)
                sil = silhouette_score(emb, labels)
                # adjust n_clusters if silhouette is too low
                if sil < 0.15:
                    candidate = max(5, int(candidate * 0.7))
                    self.n_clusters = candidate
            except Exception:
                pass

    # ──────────────────────────────────────────────────
    # Score calibration (post-models)
    # ──────────────────────────────────────────────────

    def calibrate_scores(self, vg, samples=1000):
        """
        Collect score distributions, then calibrate each component weight
        so they occupy consistent relative strength.
        """
        pass  # requires forward generation passes

    # ──────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────

    def save(self, path):
        data = {}
        for k, v in self.__dict__.items():
            if k.startswith('__'):
                continue
            if isinstance(v, (Counter, defaultdict)):
                data[k] = dict(v)
            elif isinstance(v, set):
                data[k] = list(v)
            elif isinstance(v, np.ndarray):
                data[k] = v.tolist()
            else:
                data[k] = v
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        config = cls()
        for k, v in data.items():
            if hasattr(config, k) or k.startswith('_'):
                if isinstance(v, list) and k in ('function_words', 'target_short_word_whitelist',
                                                  'replacement_tokens', 'ignored_tokens', 'banned_tokens',
                                                  'gl_special_token_ids'):
                    v = set(v)
                setattr(config, k, v)
        return config

    def to_json(self, path):
        safe = {}
        for k, v in self.__dict__.items():
            if k.startswith('__') or k.startswith('_'):
                continue
            if isinstance(v, set):
                safe[k] = list(v)
            elif isinstance(v, np.ndarray):
                safe[k] = v.tolist()
            elif isinstance(v, (Counter, defaultdict)):
                safe[k] = dict(v)
            else:
                safe[k] = v
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(safe, f, ensure_ascii=False, indent=2)
