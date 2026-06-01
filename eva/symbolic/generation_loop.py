"""
GenerationLoop — autoregressive generation with 6 heads + optional WeightTransformer.

Architecture: context builder → 6 head scores → weights → weighted sum → mask → select.
"""
import sys, math, random
from typing import Optional, Callable
import numpy as np
import torch

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
from coordinate_packer import CoordinatePacker

V = 4101
SPECIAL = set(range(0, 5)) | {156, 157, 158, 159, 160}
SENT_OPEN, SENT_CLOSE = 0, 159
WORD_OPEN, WORD_CLOSE = 157, 158
NORM = {'word_len': 19, 'pos_in_word': 18, 'word_num': 275, 'pos_in_sent': 587, 'sent_len': 587}


def build_context(tokens: list, pos_in_word: int, word_len: int,
                  word_num: int, pos_in_sent: int, sent_len: int,
                  flags: int) -> dict:
    """Build context dict for head scoring."""
    prev = tokens[-1] if tokens else SENT_OPEN
    ctx_toks = tokens[max(0, len(tokens) - 3):]
    return {
        'token_id': prev,
        'pos_in_word': pos_in_word,
        'word_len': word_len,
        'word_num': word_num,
        'pos_in_sent': pos_in_sent,
        'sent_len': sent_len,
        'prev_token_id': prev,
        'flags': flags,
        'context_tokens': ctx_toks,
    }


def default_rule_weights(context: dict) -> np.ndarray:
    """Rule-based weight vector (6,). Matches HeadsEnsemble.compute_weights."""
    w = np.array([1.0, 1.0, 2.0, 0.5, 0.2, 0.5], dtype=np.float32)
    flags = context.get('flags', 0)
    pos_in_word = context.get('pos_in_word', -1)
    word_len = context.get('word_len', 0)
    prev = context.get('prev_token_id', None)

    is_word_start = (flags >> 0) & 1
    is_word_end = (flags >> 1) & 1
    is_special = (flags >> 5) & 1

    if is_special:
        w = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float32)
    elif is_word_start:
        w = np.array([0.5, 3.0, 1.0, 0.5, 0.2, 0.5], dtype=np.float32)
    elif is_word_end:
        w = np.array([0.5, 1.0, 3.0, 1.0, 0.2, 0.5], dtype=np.float32)
    elif pos_in_word > 0 and word_len > 2:
        frac = pos_in_word / max(word_len, 1)
        if 0.2 < frac < 0.8:
            w = np.array([4.0, 1.0, 0.5, 0.5, 0.2, 0.5], dtype=np.float32)

    if prev is not None and prev < V:
        # Rare prev token → trust semantic more
        pass  # simplified

    return w


def compute_weighted_scores(heads_obj, context: dict,
                            weights: np.ndarray) -> np.ndarray:
    """Compute weighted sum of 6 head score vectors."""
    head_scores = heads_obj.individual_scores(context)  # (6, V)
    return np.dot(weights, head_scores)  # (V,)


def apply_masks(scores: np.ndarray, tokens: list, pos_in_word: int,
                word_len: int, is_special_context: bool) -> np.ndarray:
    """Zero out invalid token scores. Returns modified scores."""
    masked = scores.copy()

    # ─── Special context (between WORD_CLOSE and next WORD_OPEN) ───
    if is_special_context:
        for s in SPECIAL:
            if s not in (WORD_OPEN, SENT_CLOSE):
                masked[s] = -np.inf
        return masked

    # ─── Content context ───
    # Block all special tokens
    for s in SPECIAL:
        masked[s] = -np.inf

    # Allow WORD_CLOSE only after 2+ content tokens in word
    if pos_in_word >= 2:
        # Smooth ramp: low at pos=2 (~0.5), high at pos=6 (~8.0)
        bonus = 8.0 / (1.0 + math.exp(-(pos_in_word - 5)))
        masked[WORD_CLOSE] = max(0.5, bonus)

    # No WORD_CLOSE right after WORD_OPEN
    if len(tokens) >= 1 and tokens[-1] == WORD_OPEN:
        pass  # already blocked by loop above

    return masked


def after_selection_update(tokens: list, next_tok: int, pos_in_word: int,
                           word_len: int) -> tuple:
    """Update state after token selection. Returns (pos_in_word, word_len, flags)."""
    flags = 0
    if next_tok == WORD_CLOSE:
        flags |= 2  # word_end
        pos_in_word = 0
    elif next_tok == WORD_OPEN:
        flags |= 1  # word_start
        flags |= (1 << 5)  # is_special
        pos_in_word = 0
    elif next_tok == SENT_CLOSE:
        pass
    else:
        pos_in_word += 1
        if pos_in_word > word_len:
            word_len = pos_in_word
    return pos_in_word, word_len, flags


def select_token(scores: np.ndarray, temperature: float = 0.0) -> int:
    """Argmax (temp=0) or temperature-sampled token selection."""
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
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
    """Full generation loop — context → score → mask → select → repeat."""

    def __init__(self, heads_obj, transformer: Optional[Callable] = None,
                 weight_fn: Optional[Callable] = None, max_tokens: int = 200,
                 device: Optional[torch.device] = None):
        self.heads = heads_obj
        self.transformer = transformer
        self.weight_fn = weight_fn or default_rule_weights
        self.packer = CoordinatePacker()
        self.max_tokens = max_tokens
        self.device = device or torch.device('cpu')

    def generate(self, temperature: float = 0.0, seed: Optional[int] = None,
                 return_coords: bool = False) -> list:
        """Generate a sentence. Returns list of token IDs or trajectory dict."""
        if seed is not None:
            np.random.seed(seed)

        tokens = [SENT_OPEN]
        coords = []
        pos_in_word = -1
        word_len = 1
        word_num = -1
        pos_in_sent = 0
        sent_len = self.max_tokens

        while len(tokens) < self.max_tokens:
            prev = tokens[-1]
            flags = 0

            # Determine next token and compute its position
            pack_piw = pos_in_word  # position for THIS token (before update)
            pack_wl = word_len
            pack_wn = word_num
            pack_flags = flags

            if prev == SENT_OPEN:
                next_tok = WORD_OPEN
                word_num += 1
                word_len = 5
                pos_in_word = 0
                flags = (1 << 5)
                weights = np.array([0, 0, 5.0, 0, 0, 0], dtype=np.float32)
            elif prev == WORD_OPEN:
                ctx = build_context(tokens, 0, word_len, word_num,
                                    pos_in_sent, sent_len, 0)
                weights = self.weight_fn(ctx)
                if self.transformer is not None:
                    weights = self._transformer_weights(ctx, weights)
                scores = compute_weighted_scores(self.heads, ctx, weights)
                scores = apply_masks(scores, tokens, 0, word_len,
                                    is_special_context=False)
                next_tok = select_token(scores, temperature)
                pos_in_word, word_len, flags = after_selection_update(
                    tokens, next_tok, 0, word_len)
                pack_piw = 0
                pack_wl = word_len
            elif prev == WORD_CLOSE:
                ctx = build_context(tokens, 0, word_len, word_num,
                                    pos_in_sent, sent_len, (1 << 5))
                weights = self.weight_fn(ctx)
                if self.transformer is not None:
                    weights = self._transformer_weights(ctx, weights)
                masked = np.full(V, -np.inf, dtype=np.float32)
                masked[WORD_OPEN] = 0
                if word_num >= 3:
                    masked[SENT_CLOSE] = -2.0
                next_tok = select_token(masked, temperature)
                if next_tok == SENT_CLOSE:
                    tokens.append(SENT_CLOSE)
                    break
                word_num += 1
                word_len = 5
                pos_in_word = 0
                flags = (1 << 5)
                weights = np.array([0, 0, 5.0, 0, 0, 0], dtype=np.float32)
                pack_piw = 0
                pack_wl = 5
                pack_wn = word_num
            elif prev == SENT_CLOSE:
                break
            else:
                ctx = build_context(tokens, pos_in_word, word_len, word_num,
                                    pos_in_sent, sent_len, 0)
                weights = self.weight_fn(ctx)
                if self.transformer is not None:
                    weights = self._transformer_weights(ctx, weights)
                scores = compute_weighted_scores(self.heads, ctx, weights)
                scores = apply_masks(scores, tokens, pos_in_word, word_len,
                                    is_special_context=False)
                next_tok = select_token(scores, temperature)
                pack_piw = pos_in_word
                pack_wl = word_len
                pos_in_word, word_len, flags = after_selection_update(
                    tokens, next_tok, pos_in_word, word_len)

            tokens.append(next_tok)
            pos_in_sent += 1

            if return_coords:
                h = self.packer.pack_token(
                    token_id=next_tok,
                    pos_in_word=pack_piw,
                    word_len=pack_wl,
                    word_num=pack_wn,
                    pos_in_sent=pos_in_sent,
                    sent_len=sent_len,
                    flags=pack_flags,
                )
                from eva.symbolic.reserved_dims import fill_reserved
                h = fill_reserved(h, weights, np.zeros(6), next_tok, self.heads)
                coords.append(h)

        if return_coords:
            return {'tokens': tokens, 'coords': np.stack(coords) if coords else np.zeros((0, 384))}
        return tokens

    def _transformer_weights(self, ctx: dict, fallback: np.ndarray) -> np.ndarray:
        try:
            d = self.device
            prev = torch.tensor([ctx['prev_token_id']], dtype=torch.long, device=d)
            wl = torch.tensor([ctx['word_len'] / NORM['word_len']], dtype=torch.float32, device=d)
            piw = torch.tensor([ctx['pos_in_word'] / max(NORM['pos_in_word'], 1)], dtype=torch.float32, device=d)
            wn = torch.tensor([ctx['word_num'] / max(NORM['word_num'], 1)], dtype=torch.float32, device=d)
            pis = torch.tensor([ctx['pos_in_sent'] / max(NORM['pos_in_sent'], 1)], dtype=torch.float32, device=d)
            sl = torch.tensor([ctx['sent_len'] / max(NORM['sent_len'], 1)], dtype=torch.float32, device=d)
            fl = torch.tensor([ctx['flags'] / 255.0], dtype=torch.float32, device=d)

            with torch.no_grad():
                w = self.transformer(prev, wl, piw, wn, pis, sl, fl)
            return w.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception:
            return fallback

    def decode_tokens(self, tokens: list) -> str:
        """Decode token IDs to readable string."""
        try:
            from eva.symbolic.bpe_tokenizer import BPEVocab
            vocab = BPEVocab()
            return vocab.decode(tokens)
        except ImportError:
            return ' '.join(str(t) for t in tokens)
