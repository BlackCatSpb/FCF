"""
CharacterVocab — символьный словарь для EVA.

~400 символов: кириллица, латиница, цифры, пунктуация, спецсимволы.
Никаких BPE-токенов — только атомарные символы.
"""
from typing import List, Dict, Optional, Tuple
import numpy as np

RUSSIAN_CHARS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
LATIN_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
PUNCTUATION = ".,;:!?()[]{}'\"«»—–-… \t\n\r"
SPECIAL = "<PAD><UNK><BOS><EOS><SEP><MASK>"  # Спецтокены через комбинацию символов

ALL_CHARS = sorted(set(RUSSIAN_CHARS + LATIN_CHARS + DIGITS + PUNCTUATION))


class CharacterVocab:
    """Символьный словарь: каждый символ — отдельный токен."""

    def __init__(self):
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.BOS_IDX = 2
        self.EOS_IDX = 3

        self._char_to_idx: Dict[str, int] = {
            "<PAD>": self.PAD_IDX,
            "<UNK>": self.UNK_IDX,
            "<BOS>": self.BOS_IDX,
            "<EOS>": self.EOS_IDX,
        }
        self._idx_to_char: Dict[int, str] = {v: k for k, v in self._char_to_idx.items()}

        idx = 4
        for ch in ALL_CHARS:
            self._char_to_idx[ch] = idx
            self._idx_to_char[idx] = ch
            idx += 1

        self.vocab_size = len(self._char_to_idx)

    def encode(self, text: str) -> List[int]:
        """Кодирует текст в последовательность символьных индексов."""
        return [self.BOS_IDX] + [self._char_to_idx.get(ch, self.UNK_IDX) for ch in text] + [self.EOS_IDX]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Декодирует индексы обратно в строку."""
        chars = []
        for i in ids:
            if skip_special and i in (self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX):
                continue
            chars.append(self._idx_to_char.get(i, "?"))
        return "".join(chars)

    def encode_batch(self, texts: List[str], max_len: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Кодирует батч текстов с паддингом.
        Returns: (input_ids [B, L], attention_mask [B, L])
        """
        B = len(texts)
        input_ids = np.full((B, max_len), self.PAD_IDX, dtype=np.int64)
        attention_mask = np.zeros((B, max_len), dtype=np.float32)

        for i, text in enumerate(texts):
            ids = self.encode(text)[:max_len]
            input_ids[i, :len(ids)] = ids
            attention_mask[i, :len(ids)] = 1.0

        return input_ids, attention_mask

    def char_to_idx(self, ch: str) -> int:
        return self._char_to_idx.get(ch, self.UNK_IDX)

    def idx_to_char(self, idx: int) -> str:
        return self._idx_to_char.get(idx, "?")

    def itos(self, ids: List[int]) -> str:
        return self.decode(ids)

    def stoi(self, text: str) -> List[int]:
        return self.encode(text)

    def __len__(self) -> int:
        return self.vocab_size
