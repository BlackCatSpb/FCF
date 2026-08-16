"""Трёхуровневый морфемный токенизатор: символы → морфемы → слова.

Иерархическое «сито»:
  L1 (слово)    — только слова, собранные из словарных морфем (валидные сборки);
  L2 (морфема)  — приставки/корни/суффиксы/окончания (сегменты, разделённые маркером \\u037E);
  L3 (символ)   — полный fallback: любые байты/символы, <unk> не возникает.

Интерфейс повторяет eva/symbolic/sp_compat.SPCompatTokenizer:
  encode(text) -> list[int]; decode(ids) -> str; id_to_piece(i) -> str;
  vocab_size; pad_id/bos_id/eos_id/unk_id.

Совместим с sp_compat.load_piece_model: формат "morpheme-v1" распознаётся по полю
data["format"] и возвращается экземпляр MorphemeTokenizer.

id-пространство: 0-3 спецтокены, затем символы, затем морфемы, затем слова.
"""

import json
import re

SEP = '\u037E'
SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]


class MorphemeTokenizer:
    def __init__(self, data):
        self._specials = dict(data['specials'])
        self._char = data['char']
        self._morph = data['morph']
        self._word = data['word']
        self._chains = data.get('chains', {})
        self._vocab_size = int(data['vocab_size'])
        self._id_to_piece = [None] * self._vocab_size
        for s, i in self._specials.items():
            self._id_to_piece[i] = s
        for ch, i in self._char.items():
            self._id_to_piece[i] = ch
        for m, i in self._morph.items():
            self._id_to_piece[i] = m.replace(SEP, '')
        for w, i in self._word.items():
            self._id_to_piece[i] = w
        self._char_rev = {i: ch for ch, i in self._char.items()}
        self._morph_rev = {i: m for m, i in self._morph.items()}
        self._word_rev = {i: w for w, i in self._word.items()}
        self._morph_max_len = max((len(m) for m in self._morph), default=0)
        self._morph_by_first = {}
        for m in sorted(self._morph, key=len, reverse=True):
            self._morph_by_first.setdefault(m[0], []).append(m)
        self._cache = {}

    # ── интерфейс ───────────────────────────────────────────────

    def vocab_size(self):
        return self._vocab_size

    def pad_id(self):
        return self._specials["<|pad|>"]

    def bos_id(self):
        return self._specials["<|bos|>"]

    def eos_id(self):
        return self._specials["<|eos|>"]

    def unk_id(self):
        return self._specials["<|unk|>"]

    def id_to_piece(self, i):
        return self._id_to_piece[i]

    def IdToPiece(self, i):
        return self._id_to_piece[i]

    def piece_size(self):
        return self._vocab_size

    def encode(self, text):
        """str -> list[int]. Входной маркер \\u037E игнорируется (это метка разбора)."""
        if SEP in text:
            text = text.replace(SEP, '')
        out = []
        for tok in re.findall(r'\w+|[^\w\s]|\s', text):
            if tok.isspace():
                cid = self._char.get(tok)
                if cid is not None:
                    out.append(cid)
                continue
            out.extend(self._encode_word(tok))
        return out

    def encode_pieces(self, text):
        return [self._id_to_piece[i] for i in self.encode(text)]

    def decode(self, ids):
        return ''.join(self._id_to_piece[i] for i in ids)

    # ── уровни ──────────────────────────────────────────────────

    def _encode_word(self, w):
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        ids = self._encode_word_uncached(w)
        if len(self._cache) < 200000:
            self._cache[w] = ids
        return ids

    def _encode_word_uncached(self, w):
        ids = self._encode_word_plain(w)
        if ids is not None:
            return ids
        if w[0].isupper() and len(w) > 1:
            lower = self._encode_word_plain(w.lower())
            if lower and all(x != self.unk_id() for x in lower):
                cid = self._char.get(w[0])
                if cid is not None:
                    return [cid] + lower
        return [self._char.get(ch, self.unk_id()) for ch in w]

    def _encode_word_plain(self, w):
        """Разбор без ветки регистра (не рекурсивный)."""
        wid = self._word.get(w)
        if wid is not None:
            return [wid]
        segs = self._chains.get(w)
        if segs is not None:
            m = [self._morph.get(s) for s in segs]
            if all(x is not None for x in m):
                return list(m)
        segs = self._greedy_split(w)
        if segs is not None:
            return list(segs)
        return None

    def _greedy_split(self, w):
        """Жадный разбор слева-направо по словарю морфем (самый длинный матч).

        Возвращает id-список или None, если слово не собирается целиком.
        Индексация по первому символу: кандидатов на позицию ~десятки, не десятки тысяч.
        """
        if not w or not self._morph_by_first:
            return None
        res = []
        i = 0
        n = len(w)
        while i < n:
            matched = None
            cands = self._morph_by_first.get(w[i])
            if cands is not None:
                limit = min(self._morph_max_len, n - i)
                for m in cands:
                    if len(m) > limit:
                        continue
                    if w.startswith(m, i):
                        matched = m
                        break
            if matched is None:
                return None
            res.append(self._morph[matched])
            i += len(matched)
        return res

    # ── сериализация ────────────────────────────────────────────

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            return cls(json.load(f))


def looks_like_morpheme(data):
    return isinstance(data, dict) and data.get('format') == 'morpheme-v1'