"""BPE tokenizer for EVA — trained on full_corpus_ru.txt, vocab_size=4096."""

import os, json, pickle
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from typing import List, Optional


BPE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'bpe_tokenizer.json')
CORPUS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'full_corpus_ru.txt')


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
