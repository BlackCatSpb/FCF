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

V = 4101  # heads produce scores for all 4101 (0-4095 BPE + 5 boundary slots)
BOS, EOS = 2, 3
PAD, UNK, SEP, MASK = 0, 1, 4, 5

# 136 BPE tokens that decode to replacement character U+FFFD
REPLACEMENT_TOKENS = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
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


def apply_masks(scores: np.ndarray, tokens: list,
                word_num: int, token_type_arr=None) -> np.ndarray:
    """Zero out invalid token scores. Returns modified scores."""
    masked = scores.copy()

    # Always block: PAD, UNK, BOS, SEP, MASK (indices 4096-4100 are unused boundary slots)
    for t in (PAD, UNK, BOS, SEP, MASK):
        masked[t] = -np.inf
    for t in range(4096, V):
        masked[t] = -np.inf

    # Block replacement tokens (decodes to "�")
    for t in REPLACEMENT_TOKENS:
        masked[t] = -np.inf

    # After BOS: only WORD_STARTER tokens allowed
    if len(tokens) == 1 and token_type_arr is not None:
        for t in range(V):
            if t < len(token_type_arr) and token_type_arr[t] != 2:
                masked[t] = -np.inf

    # EOS only after enough words generated
    if word_num < 3:
        masked[EOS] = -np.inf

    return masked


def select_token(scores: np.ndarray, temperature: float = 0.0,
                 top_k: int = 50) -> int:
    """Argmax (temp=0) or temperature-sampled with top-k filtering."""
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

    if top_k > 0 and top_k < V:
        cutoff = np.partition(scores, -top_k)[-top_k]  # k-th largest
        blocked = scores < cutoff
        scores = scores.copy()
        scores[blocked] = -np.inf

    if temperature <= 0:
        return int(np.argmax(scores))

    scores = scores - scores.max()
    probs = np.exp(np.clip(scores / temperature, -50, 50))
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        return int(np.argmax(scores))
    probs /= total
    return int(np.random.choice(V, p=probs))


class GenerationLoop:
    """Generation loop — HierarchicalVocab → heads → mask → select."""

    def __init__(self, heads_obj, transformer=None,
                 weight_fn=None, max_tokens: int = 200,
                 device=None):
        self.heads = heads_obj
        self.transformer = transformer
        self.vocab = HierarchicalVocab()
        self.max_tokens = max_tokens
        self.device = device or 'cpu'

    def generate(self, temperature: float = 0.0, seed=None,
                 return_compact: bool = False) -> list:
        """Generate a sentence. Returns token IDs or dict with compact track."""
        if seed is not None:
            np.random.seed(seed)

        tokens = [BOS]
        if return_compact:
            compact_frames = []

        while len(tokens) < self.max_tokens:
            # Compute metadata for current sequence
            meta = self.vocab.metadata_from_ids(tokens)
            ctx = dict(meta[-1])  # last step = where next token will go
            ctx['context_tokens'] = tokens[-4:-1] if len(tokens) > 1 else tokens
            word_num = ctx['word_num']

            weights = self._compute_weights(ctx)

            # Head scores
            scores = np.zeros(V, dtype=np.float32)
            try:
                head_scores = self.heads.individual_scores(ctx)
                scores = np.dot(weights, head_scores)
            except Exception:
                scores = np.ones(V, dtype=np.float32) * -1e9
                scores[EOS] = 0  # fallback: just end

            # Apply masks
            scores = apply_masks(scores, tokens, word_num, self.vocab.token_type)

            # Select next token
            next_tok = select_token(scores, temperature)

            if next_tok == EOS:
                tokens.append(EOS)
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
        w = np.array([1.0, 1.0, 3.0, 0.5, 0.2, 0.5], dtype=np.float32)
        flags = ctx.get('flags', 0)
        pos_in_word = ctx.get('pos_in_word', -1)
        word_len = ctx.get('word_len', 0)

        is_word_start = (flags >> 0) & 1
        is_word_end = (flags >> 1) & 1
        is_special = (flags >> 5) & 1

        if is_special:
            w = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif is_word_start:
            w = np.array([1.0, 1.5, 2.0, 0.5, 0.2, 0.5], dtype=np.float32)
        elif is_word_end:
            w = np.array([0.5, 1.0, 3.0, 1.0, 0.2, 0.5], dtype=np.float32)
        elif piw > 0:  # mid-word position → morph matters
            w = np.array([3.0, 1.0, 1.0, 0.5, 0.2, 0.5], dtype=np.float32)

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
        NORM = {'word_len': 19, 'pos_in_word': 18, 'word_num': 275,
                'pos_in_sent': 587, 'sent_len': 587}
        prev = torch.tensor([ctx['prev_token_id']], dtype=torch.long, device=d)
        wl = torch.tensor([ctx['word_len'] / max(NORM['word_len'], 1)], dtype=torch.float32, device=d)
        piw = torch.tensor([ctx['pos_in_word'] / max(NORM['pos_in_word'], 1)], dtype=torch.float32, device=d)
        wn = torch.tensor([ctx['word_num'] / max(NORM['word_num'], 1)], dtype=torch.float32, device=d)
        pis = torch.tensor([ctx['pos_in_sent'] / max(NORM['pos_in_sent'], 1)], dtype=torch.float32, device=d)
        sl = torch.tensor([ctx['sent_len'] / max(NORM['sent_len'], 1)], dtype=torch.float32, device=d)
        fl = torch.tensor([ctx['flags'] / 255.0], dtype=torch.float32, device=d)
        with torch.no_grad():
            w = self.transformer(prev, wl, piw, wn, pis, sl, fl)
        return w.squeeze(0).cpu().numpy().astype(np.float32)

    def decode_tokens(self, tokens: list) -> str:
        """Decode token IDs to readable string."""
        return self.vocab.decode(tokens)
