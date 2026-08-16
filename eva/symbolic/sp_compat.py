"""SPCompat — тонкий адаптер: HF ByteLevel BPE (WideBind 65K) под API SentencePiece.

FCF ожидает от токенизатора интерфейс SentencePieceProcessor:
  encode(text, add_bos=False, add_eos=False, out_type=None) -> list[int] | list[str]
  decode(ids) -> str
  vocab_size() -> int
  IdToPiece(i) / id_to_piece(i) -> str   (▁ = маркер пробела, как в SP)
  PieceToId(p) / piece_to_id(p) -> int | None
  EncodeAsPieces(text) / encode_as_pieces(text) -> list[str]

WideBind-токенизатор — ByteLevel BPE: токены хранятся в байтовом (Latin-1)
представлении, пробелы кодируются 'Ġ' (U+0120). Ключевые отличия и решения:
  * IdToPiece реализуется через decode([cid]) — даёт читаемый текст;
    токены с ведущим пробелом возвращаются как '▁' + текст (аналог SP).
  * PieceToId — через encode(word): ровно один токен -> его id, иначе None
    (token_to_id по читаемому слову у ByteLevel не работает).
  * encode/decode полностью эквивалентны HF (спецтокены 0-3 = pad/bos/eos/unk).
"""
from __future__ import annotations

import os
import json


def _bpe_like(tok, idx: int) -> str:
    """id_to_token может хранить токен в байтовом виде — это нормально."""
    try:
        return tok.id_to_token(idx)
    except Exception:
        return None


class SPCompatTokenizer:
    """SentencePiece-совместимая обёртка над HF Tokenizer (ByteLevel BPE)."""

    def __init__(self, tok, pad_id: int = 0, bos_id: int = 1,
                 eos_id: int = 2, unk_id: int = 3):
        self._tok = tok
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        # get_vocab_size() стоит ~14ms на вызов (Rust binding) — кэшируем
        self._vocab_size = self._tok.get_vocab_size()

    def vocab_size(self) -> int:
        return self._vocab_size

    # ── Кодирование ─────────────────────────────────────────────

    def encode(self, text, add_bos: bool = False, add_eos: bool = False,
               out_type=None):
        """Токенизация текста. add_bos/add_eos игнорируются (BOS/EOS в FCF
        добавляются на уровне генератора константами _BOS_ID/_EOS_ID)."""
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if out_type == str or out_type == 'str':
            return [self.id_to_piece(i) for i in ids]
        return ids

    def decode(self, ids):
        """Декодирование в текст (спецтокены пропускаются)."""
        if isinstance(ids, (int,)):
            ids = [ids]
        return self._tok.decode(list(ids), skip_special_tokens=True)

    # ── Словарь ─────────────────────────────────────────────────

    def IdToPiece(self, i: int) -> str:
        """Текст токена. Ведущий пробел ByteLevel -> '▁' (SP-нотация)."""
        if i < 0 or i >= self.vocab_size():
            return ''
        text = self._tok.decode([i], skip_special_tokens=True)
        if text.startswith(' '):
            return '\u2581' + text[1:]
        return text

    def PieceToId(self, p) -> int | None:
        """Обратный lookup. Через encode: единственный токен -> id."""
        if p is None:
            return None
        q = p[1:] if p.startswith('\u2581') else p
        # Спецтокены
        for name, idx in (('<|pad|>', self.pad_id), ('<|bos|>', self.bos_id),
                          ('<|eos|>', self.eos_id), ('<|unk|>', self.unk_id),
                          ('<pad>', self.pad_id), ('<bos>', self.bos_id),
                          ('<eos>', self.eos_id), ('<unk>', self.unk_id)):
            if q == name:
                return idx
        q = ' ' + q if p.startswith('\u2581') else q
        ids = self._tok.encode(q, add_special_tokens=False).ids
        if len(ids) == 1:
            return ids[0]
        # Прямой поиск по тексту токена (редкие цельные слова)
        for i in range(self.vocab_size()):
            if self._tok.id_to_token(i) == q:
                return i
        return None

    def EncodeAsPieces(self, text) -> list:
        return [self.id_to_piece(i) for i in self.encode(text)]

    # ── Snake-case алиасы (используются в concept_space) ────────

    def id_to_piece(self, i: int) -> str:
        return self.IdToPiece(i)

    def piece_to_id(self, p) -> int | None:
        return self.PieceToId(p)

    def encode_as_pieces(self, text) -> list:
        return self.EncodeAsPieces(text)

    def piece_size(self) -> int:
        return self.vocab_size()

    # ── Совместимость со старым __main__-кодом ──────────────────

    def __len__(self) -> int:
        return self.vocab_size()

    def __repr__(self) -> str:
        return (f'<SPCompatTokenizer vocab={self.vocab_size()} '
                f'backend={type(self._tok).__name__}>')


def load_piece_model(path) -> SPCompatTokenizer:
    """Загрузить токенизатор по расширению.

    *.model            -> sentencepiece (возвращает SP-процессор)
    *tokenizer.json*   -> HF ByteLevel BPE (оборачивает в SPCompatTokenizer)
    *morpheme*/.json с format=morpheme-v1 -> MorphemeTokenizer (символы→морфемы→слова)
    """
    base = os.path.basename(str(path)).lower()
    if 'tokenizer.json' in base or base.endswith('.json'):
        with open(str(path), encoding='utf-8') as f:
            data = json.load(f)
        if data.get('format') == 'morpheme-v1':
            from eva.symbolic.morpheme_tokenizer import MorphemeTokenizer
            return MorphemeTokenizer(data)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(path))
        return SPCompatTokenizer(tok)
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=str(path))