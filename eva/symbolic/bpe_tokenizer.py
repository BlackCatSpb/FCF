"""BPE tokenizer for EVA — trained on full_corpus_ru.txt, vocab_size=4096."""

import os, json, pickle
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from typing import List, Optional


BPE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'bpe_tokenizer.json')
CORPUS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'full_corpus_ru.txt')

# ─── Token type constants ───
SPECIAL_T = 0       # special tokens (0-5): PAD, UNK, BOS, EOS, SEP, MASK
BYTE_T = 1          # single-byte tokens (6-155): punctuation, digits, symbols
WORD_STARTER_T = 2  # BPE tokens with leading space: start a new word
WORD_CONT_T = 3     # BPE tokens without leading space: continue current word


class HierarchicalVocab:
    """
    Vocabulary where each token ID encodes its hierarchical role.
    Word boundaries are IMPLICIT in token types — no WORD_OPEN/WORD_CLOSE needed.

    Token layout:
      0-5:   SPECIAL_T — special tokens (PAD, UNK, BOS, EOS, SEP, MASK)
      6-155: BYTE_T — single printable characters (punctuation, digits)
      156+:   WORD_STARTER_T or WORD_CONT_T — BPE merge tokens
              classified by decoded content (leading space = starter)

    Metadata (pos_in_word, word_num, flags) is derived from the ID sequence.
    """

    def __init__(self, path: Optional[str] = None):
        path = path or BPE_PATH
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self._build_type_table()

    def _build_type_table(self):
        """Classify every token ID by its decoded content."""
        self.token_type = np.zeros(self.vocab_size, dtype=np.uint8)
        for i in range(self.vocab_size):
            if i < 6:
                self.token_type[i] = SPECIAL_T
            elif i < 156:
                self.token_type[i] = BYTE_T
            else:
                decoded = self.tokenizer.decode([i])
                if decoded.startswith(' '):
                    self.token_type[i] = WORD_STARTER_T
                else:
                    self.token_type[i] = WORD_CONT_T
        # Replacement tokens (decode to \ufffd) — treated as continuers
        for i in range(6, self.vocab_size):
            if '\ufffd' in self.tokenizer.decode([i]):
                self.token_type[i] = WORD_CONT_T

    def type_name(self, token_id):
        return ['SPECIAL', 'BYTE', 'WORD_STARTER', 'WORD_CONT'][int(self.token_type[token_id])]

    def is_byte(self, token_id):
        return self.token_type[token_id] == BYTE_T

    def is_special(self, token_id):
        return self.token_type[token_id] == SPECIAL_T

    def encode(self, text: str) -> List[int]:
        """Encode text with BPE, NO boundary markers, NO BOS/EOS."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode IDs to text."""
        if skip_special:
            ids = [i for i in ids if i >= 6 or i in (2, 3)]
        return self.tokenizer.decode(ids)

    def metadata_from_ids(self, ids: List[int]) -> List[dict]:
        """
        Compute (pos_in_word, word_len, word_num, pos_in_sent, sent_len, flags)
        for each position. No boundary tokens needed — types encode hierarchy.
        """
        L = len(ids)
        piw = [0] * L
        wl = [5] * L
        wn = [0] * L
        fl = [0] * L

        cur_word = -1
        cur_pos = 0
        prev_type = SPECIAL_T
        word_start_idx = {}  # word_number -> first step index

        # Pass 1: word number and position in word
        for t, tid in enumerate(ids):
            tt = int(self.token_type[tid])
            if tt == SPECIAL_T:
                piw[t] = 0
                wn[t] = -1  # not part of any word
                fl[t] = 1 << 5  # is_special
                if tid == 2:
                    fl[t] |= 1 << 2  # sent_start
                elif tid == 3:
                    fl[t] |= 1 << 3  # sent_end
                prev_type = SPECIAL_T
            elif tt == WORD_STARTER_T:
                cur_word += 1
                cur_pos = 0
                piw[t] = 0
                wn[t] = cur_word
                word_start_idx[cur_word] = t
                fl[t] = 1 << 0  # word_start
                if cur_word == 0:
                    fl[t] |= 1 << 2  # sent_start
                prev_type = WORD_STARTER_T
            elif prev_type == SPECIAL_T:
                # Token right after SPECIAL (e.g. BOS) but NOT a WORD_STARTER
                # Treat as a standalone "word" of its own (punctuation, continuation)
                cur_word += 1
                cur_pos = 0
                piw[t] = 0
                wn[t] = cur_word
                word_start_idx[cur_word] = t
                fl[t] = 1 << 0  # word_start
                prev_type = WORD_CONT_T
            else:
                cur_pos += 1
                piw[t] = cur_pos
                wn[t] = cur_word
                prev_type = WORD_CONT_T

        # Pass 2: word lengths and word_end flags
        for t in range(L):
            w = wn[t]
            if w >= 0:
                start = word_start_idx.get(w, t)
                wl[t] = t - start + 1
                if t == L - 1 or wn[t + 1] != w:
                    fl[t] |= 1 << 1  # word_end

        return [
            {'pos_in_word': piw[t], 'word_len': wl[t], 'word_num': max(wn[t], 0),
             'pos_in_sent': t, 'sent_len': L, 'flags': fl[t],
             'token_id': ids[t], 'prev_token_id': ids[t - 1] if t > 0 else ids[t]}
            for t in range(L)
        ]


def train_bpe(vocab_size=4096, corpus_path: Optional[str] = None, save_path: Optional[str] = None):
    corpus_path = corpus_path or CORPUS_PATH
    save_path = save_path or BPE_PATH

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"],
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train([corpus_path], trainer)
    tokenizer.save(save_path)
    print(f'BPE tokenizer saved to {save_path} (vocab_size={tokenizer.get_vocab_size()})')
    return tokenizer


class BPEVocab:
    """BPE vocab wrapper, compatible with CharacterVocab interface."""

    def __init__(self, path: Optional[str] = None):
        path = path or BPE_PATH
        if not os.path.exists(path):
            print(f'Tokenizer not found at {path}, training...')
            self.tokenizer = train_bpe(save_path=path)
        else:
            self.tokenizer = Tokenizer.from_file(path)

        self.vocab_size = self.tokenizer.get_vocab_size()
        self.PAD_IDX = self.tokenizer.token_to_id('<PAD>')
        self.UNK_IDX = self.tokenizer.token_to_id('<UNK>')
        self.BOS_IDX = self.tokenizer.token_to_id('<BOS>')
        self.EOS_IDX = self.tokenizer.token_to_id('<EOS>')

        # Boundary tokens — not in BPE vocab, handled separately
        self.GAP_FILLER_IDX = self.vocab_size
        self.WORD_OPEN_IDX = self.vocab_size + 1
        self.WORD_CLOSE_IDX = self.vocab_size + 2
        self.SENT_OPEN_IDX = self.vocab_size + 3
        self.SENT_CLOSE_IDX = self.vocab_size + 4

        self.vocab_size_with_boundaries = self.vocab_size + 5
        self.max_char_idx = self.vocab_size_with_boundaries - 1

    def encode(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        return [self.BOS_IDX] + ids + [self.EOS_IDX]

    def encode_with_boundaries(self, text: str) -> List[int]:
        import re
        sentences = re.split(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z])', text)
        result = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            result.append(self.SENT_OPEN_IDX)
            # Split into words; each word is BPE-encoded individually
            for word in sent.split():
                clean = word.strip('.,;:!?()[]{}«»—–-…\"\'')
                if clean:
                    result.append(self.WORD_OPEN_IDX)
                    result.extend(self.tokenizer.encode(clean).ids)
                    result.append(self.WORD_CLOSE_IDX)
                for ch in word:
                    if ch in '.,;:!?()[]{}«»—–-…\"\'':
                        result.append(self.tokenizer.encode(ch).ids[0])
            result.append(self.SENT_CLOSE_IDX)
        return result

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        result = []
        for i in ids:
            if i == self.WORD_OPEN_IDX:
                if result and not result[-1].isspace():
                    result.append(' ')
                continue
            if i == self.WORD_CLOSE_IDX:
                continue
            if skip_special and i in {self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX,
                                       self.GAP_FILLER_IDX, self.SENT_OPEN_IDX, self.SENT_CLOSE_IDX}:
                continue
            if i < self.vocab_size:
                tok = self.tokenizer.decode([i])
                # Insert space before uppercase (word start in BPE)
                if (tok and tok[0].isupper() and result
                    and not result[-1].isspace() and result[-1] not in ' («'):
                    result.append(' ')
                result.append(tok)
        return ''.join(result)

    def __len__(self) -> int:
        return self.vocab_size_with_boundaries

    def save(self, path: str):
        self.tokenizer.save(path)

    @staticmethod
    def load(path: str):
        return BPEVocab(path)
