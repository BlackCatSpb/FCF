"""
GenerationLoop — autoregressive generation with 6 heads + HierarchicalVocab.

Architecture: HierarchicalVocab → token IDs → metadata → 6 head scores → mask → select.
No CoordinatePacker, no boundary tokens (WORD_OPEN/CLOSE).
Word boundaries are implicit in token types (WORD_STARTER = has leading space).
"""
import sys, math, random
from typing import Optional
import numpy as np

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
from eva.symbolic.bpe_tokenizer import HierarchicalVocab


def apply_masks(scores: np.ndarray, tokens: list, config) -> np.ndarray:
    """Zero out invalid token scores. Returns modified scores."""
    masked = scores.copy()
    V = config.vocab_size

    # Always block: PAD, UNK, BOS, SEP, MASK (indices 4096-4100 are unused boundary slots)
    for t in config.gl_special_token_ids:
        masked[t] = -np.inf
    for t in range(config.bpe_limit, V):
        masked[t] = -np.inf

    # Block replacement tokens (decodes to "�")
    for t in config.replacement_tokens:
        masked[t] = -np.inf

    # After BOS: only WORD_STARTER tokens allowed
    if len(tokens) == 1 and hasattr(config, 'token_type_arr'):
        for t in range(V):
            if t < len(config.token_type_arr) and config.token_type_arr[t] != 2:
                masked[t] = -np.inf

    # EOS only after enough words generated
    if len(tokens) < config.eos_min_words:
        masked[config.eos_token_id] = -np.inf

    return masked


def select_token(scores: np.ndarray, temperature: float = 0.0,
                 top_k: int = 50, config=None) -> int:
    """Argmax (temp=0) or temperature-sampled with top-k filtering."""
    V = config.vocab_size if config else 4101
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

    if top_k > 0 and top_k < V:
        cutoff = np.partition(scores, -top_k)[-top_k]  # k-th largest
        blocked = scores < cutoff
        scores = scores.copy()
        scores[blocked] = -np.inf

    if temperature <= 0:
        return int(np.argmax(scores))

    clip_min = config.gl_score_clip_min if config else -50
    clip_max = config.gl_score_clip_max if config else 50
    scores = scores - scores.max()
    probs = np.exp(np.clip(scores / temperature, clip_min, clip_max))
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        return int(np.argmax(scores))
    probs /= total
    return int(np.random.choice(V, p=probs))


class GenerationLoop:
    """Generation loop — HierarchicalVocab → heads → mask → select."""

    def __init__(self, heads_obj, transformer=None,
                 weight_fn=None, max_tokens: int = 200,
                 device=None, config=None):
        from eva.symbolic.auto_config import AutoConfig
        self.config = config or AutoConfig()
        self.heads = heads_obj
        self.transformer = transformer
        self.vocab = HierarchicalVocab()
        self.max_tokens = max_tokens
        self.device = device or 'cpu'
        self.V = self.config.vocab_size
        self.EOS = self.config.eos_token_id
        self.BOS = 2

    def generate(self, temperature: float = 0.0, seed=None,
                 return_compact: bool = False) -> list:
        """Generate a sentence. Returns token IDs or dict with compact track."""
        if seed is not None:
            np.random.seed(seed)

        tokens = [self.BOS]
        if return_compact:
            compact_frames = []

        while len(tokens) < self.max_tokens:
            meta = self.vocab.metadata_from_ids(tokens)
            ctx = dict(meta[-1])
            ctx['context_tokens'] = tokens[-4:-1] if len(tokens) > 1 else tokens
            word_num = ctx['word_num']

            weights = self._compute_weights(ctx)

            scores = np.zeros(self.V, dtype=np.float32)
            try:
                head_scores = self.heads.individual_scores(ctx)
                scores = np.dot(weights, head_scores)
            except Exception:
                scores = np.ones(self.V, dtype=np.float32) * -1e9
                scores[self.EOS] = 0

            scores = apply_masks(scores, tokens, self.config)
            next_tok = select_token(scores, temperature, self.config.gl_top_k_default, self.config)

            if next_tok == self.EOS:
                tokens.append(self.EOS)
                break

            tokens.append(next_tok)

            if return_compact:
                compact_frames.append(next_tok)

        result = {'tokens': tokens}
        if return_compact:
            result['compact'] = np.array(compact_frames, dtype=np.uint16)
        return result if return_compact else tokens

    def _compute_weights(self, ctx: dict) -> np.ndarray:
        """Compute 6-d weight vector from context."""
        w = np.array(self.config.gl_initial_weights, dtype=np.float32)
        flags = ctx.get('flags', 0)
        pos_in_word = ctx.get('pos_in_word', -1)
        word_len = ctx.get('word_len', 0)

        is_word_start = (flags >> 0) & 1
        is_word_end = (flags >> 1) & 1
        is_special = (flags >> 5) & 1

        if is_special:
            w = np.array(self.config.gl_special_weights, dtype=np.float32)
        elif is_word_start:
            w = np.array(self.config.gl_word_start_weights, dtype=np.float32)
        elif is_word_end:
            w = np.array(self.config.gl_word_end_weights, dtype=np.float32)
        elif pos_in_word > 0:
            w = np.array(self.config.gl_mid_word_weights, dtype=np.float32)

        if self.transformer is not None:
            try:
                tw = self._transformer_weights(ctx)
                w = tw
            except Exception:
                pass

        return w

    def _transformer_weights(self, ctx: dict) -> np.ndarray:
        """WeightTransformer: context → 6-d weight vector."""
        import torch
        d = self.device
        NORM = self.config
        prev = torch.tensor([ctx['prev_token_id']], dtype=torch.long, device=d)
        wl = torch.tensor([ctx['word_len'] / max(NORM.norm_word_len, 1)], dtype=torch.float32, device=d)
        piw = torch.tensor([ctx['pos_in_word'] / max(NORM.norm_pos_in_word, 1)], dtype=torch.float32, device=d)
        wn = torch.tensor([ctx['word_num'] / max(NORM.norm_word_num, 1)], dtype=torch.float32, device=d)
        pis = torch.tensor([ctx['pos_in_sent'] / max(NORM.norm_pos_in_sent, 1)], dtype=torch.float32, device=d)
        sl = torch.tensor([ctx['sent_len'] / max(NORM.norm_sent_len, 1)], dtype=torch.float32, device=d)
        fl = torch.tensor([ctx['flags'] / NORM.norm_flags_divisor], dtype=torch.float32, device=d)
        with torch.no_grad():
            w = self.transformer(prev, wl, piw, wn, pis, sl, fl)
        return w.squeeze(0).cpu().numpy().astype(np.float32)

    def decode_tokens(self, tokens: list) -> str:
        """Decode token IDs to readable string."""
        return self.vocab.decode(tokens)
