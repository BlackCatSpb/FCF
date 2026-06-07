"""
HeadsEnsemble — 6 data-driven heads (morph, syntax, transition, semantic, concept, contra).

Все операции — numpy arrays, без Python loops per-token.
WeightTransformer учит взвешивать heads.
"""
import math, pickle, os
from typing import Optional
import numpy as np


class HeadsEnsemble:
    """Все 6 heads. Векторизовано: score_all(V) = O(V) numpy."""

    def __init__(self, meta_path: str, csr_path: Optional[str] = None,
                 default_weights: Optional[dict] = None,
                 config: Optional['AutoConfig'] = None):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        self.V = meta.get('V', getattr(config, 'vocab_size', 4101))
        if config is None:
            from eva.symbolic.auto_config import AutoConfig
            config = AutoConfig()
        self.config = config

        # Precomputed log-prob arrays from rebuild script
        self.morph_logprob = meta.get('morph_logprob', {})
        self.syntax_logprob = meta.get('syntax_logprob', {})

        # Transition: load CSR log-prob matrix
        if csr_path is None:
            csr_path = os.path.join(os.path.dirname(meta_path), 'hierarchical')
        log_prob_csr_file = os.path.join(csr_path, 'log_prob_csr.npz')
        if os.path.exists(log_prob_csr_file):
            from scipy.sparse import load_npz
            self.log_prob_csr = load_npz(log_prob_csr_file)
        else:
            self.log_prob_csr = None

        # Semantic similarity — sparse, only allocate if data exists
        trans_sim = meta.get('trans_sim_sparse', {})
        if trans_sim:
            self.semantic_sim = np.zeros((self.V, self.V), dtype=np.float32)
            for tid, neighbors in trans_sim.items():
                for neighbor_id, sim in neighbors:
                    self.semantic_sim[tid, neighbor_id] = sim
        else:
            self.semantic_sim = None

        # Contradiction penalty — sparse
        contra = meta.get('contra_pairs', [])
        if contra:
            self.contra_penalty = np.zeros((self.V, self.V), dtype=np.float32)
            for ta, tb, s in contra:
                self.contra_penalty[ta, tb] = float(s)
                self.contra_penalty[tb, ta] = float(s)
        else:
            self.contra_penalty = None

        # Concept scores
        cs = meta.get('concept_scores', None)
        if cs is not None:
            self.concept_scores = np.asarray(cs, dtype=np.float32)
        else:
            self.concept_scores = np.ones(self.V, dtype=np.float32) * 0.5

        # Token counts for rare-token detection
        self.token_counts = np.asarray(meta.get('token_counts', np.ones(self.V)), dtype=np.int32)

        self.default_weights = default_weights or dict(self.config.head_default_weights)

    def compute_weights(self, context: dict) -> dict:
        w = dict(self.default_weights)
        flags = context.get('flags', 0)
        pos_in_word = context.get('pos_in_word', -1)
        word_len = context.get('word_len', 0)
        prev = context.get('prev_token_id', None)

        is_word_start = (flags >> 0) & 1
        is_word_end = (flags >> 1) & 1
        is_special = (flags >> 5) & 1

        if is_special:
            w['transition'] = self.config.head_special_transition
            for k in self.config.head_special_zero_heads:
                w[k] = 0.0
        elif is_word_start:
            w.update(self.config.head_word_start)
        elif is_word_end:
            w.update(self.config.head_word_end)
        elif pos_in_word > 0 and word_len > 2:
            frac = pos_in_word / max(word_len, 1)
            if 0.2 < frac < 0.8:
                w.update(self.config.head_mid_word)

        if prev is not None and prev < self.V:
            if int(self.token_counts[prev]) < self.config.head_rare_token_threshold:
                w['semantic'] += self.config.head_rare_semantic_boost
                w['transition'] *= self.config.head_rare_transition_factor

        return w

    def individual_scores(self, context: dict) -> np.ndarray:
        """Return (6, V) array: each row is one head's score vector."""
        out = np.zeros((6, self.V), dtype=np.float32)
        wl = context.get('word_len', 0)
        piw = context.get('pos_in_word', -1)
        wn = context.get('word_num', -1)
        prev = context.get('prev_token_id', None)
        ctx_toks = context.get('context_tokens', [])

        # 0: morph
        if piw in self.morph_logprob:
            out[0] = self.morph_logprob[piw]
        else:
            out[0] = np.full(self.V, self.config.head_fallback_logprob, dtype=np.float32)

        # 1: syntax
        if wn in self.syntax_logprob:
            out[1] = self.syntax_logprob[wn]
        else:
            out[1] = np.full(self.V, self.config.head_fallback_logprob, dtype=np.float32)

        # 2: transition
        if prev is not None and prev < self.V and self.log_prob_csr is not None:
            row = self.log_prob_csr[prev].tocoo()
            for col_idx, val in zip(row.col, row.data):
                out[2, col_idx] = val
        out[2][out[2] == 0] = self.config.head_unseen_penalty  # unseen transitions get low score

        # 3: semantic
        if ctx_toks and self.semantic_sim is not None:
            for ct in ctx_toks[-3:]:
                if ct < self.V:
                    out[3] += self.semantic_sim[ct]

        # 4: concept
        out[4] = self.concept_scores

        # 5: contra (penalty)
        if ctx_toks and self.contra_penalty is not None:
            for ct in ctx_toks[-3:]:
                if ct < self.V:
                    out[5] = np.maximum(out[5], self.contra_penalty[ct])

        return out

    def score_all(self, context: dict) -> np.ndarray:
        weights = self.compute_weights(context)
        scores = np.zeros(self.V, dtype=np.float32)

        w = weights.get('morph', 0.0)
        if w != 0.0:
            piw = context.get('pos_in_word', -1)
            if piw in self.morph_logprob:
                scores += w * self.morph_logprob[piw]

        w = weights.get('syntax', 0.0)
        if w != 0.0:
            wn = context.get('word_num', -1)
            if wn in self.syntax_logprob:
                scores += w * self.syntax_logprob[wn]

        w = weights.get('transition', 0.0)
        if w != 0.0 and self.log_prob_csr is not None:
            prev = context.get('prev_token_id', None)
            if prev is not None and prev < self.V:
                row = self.log_prob_csr[prev].tocoo()
                for col_idx, val in zip(row.col, row.data):
                    scores[col_idx] += w * val

        w = weights.get('semantic', 0.0)
        if w != 0.0 and self.semantic_sim is not None:
            ctx = context.get('context_tokens', [])
            if ctx:
                sem = np.zeros(self.V, dtype=np.float32)
                for ct in ctx[-3:]:
                    if ct < self.V:
                        sem += self.semantic_sim[ct]
                scores += w * sem

        w = weights.get('concept', 0.0)
        if w != 0.0:
            scores += w * self.concept_scores

        w = weights.get('contra', 0.0)
        if w != 0.0 and self.contra_penalty is not None:
            ctx = context.get('context_tokens', [])
            if ctx:
                penalty = np.zeros(self.V, dtype=np.float32)
                for ct in ctx[-3:]:
                    if ct < self.V:
                        penalty = np.maximum(penalty, self.contra_penalty[ct])
                scores -= w * penalty * self.config.head_contra_penalty_multiplier

        return scores

    def best_token(self, context: dict) -> int:
        return int(np.argmax(self.score_all(context)))

    def top_k(self, context: dict, k: int = 5) -> list:
        scores = self.score_all(context)
        top_idx = np.argsort(-scores)[:k]
        return [(int(tid), float(scores[tid])) for tid in top_idx]

    def score_token(self, token_id: int, context: dict) -> float:
        return float(self.score_all(context)[token_id])

    def token_text(self, tid: int) -> str:
        """Return text for a token ID. Requires BPEVocab loaded."""
        if not hasattr(self, '_vocab'):
            try:
                from eva.symbolic.bpe_tokenizer import BPEVocab
                self._vocab = BPEVocab()
            except ImportError:
                return str(tid)
        return self._vocab.decode([tid]) if hasattr(self._vocab, 'decode') else str(tid)
