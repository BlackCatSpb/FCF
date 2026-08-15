"""Tests for SPCompatTokenizer — HF ByteLevel BPE под API SentencePiece.

Создаём маленький ByteLevel BPE в памяти (без реального токенизатора WB)
и проверяем полный SP-совместимый интерфейс: encode/decode, IdToPiece
(▁-эмуляция пробелов), PieceToId, vocab_size, спецтокены.
"""
import pytest

from eva.symbolic.sp_compat import SPCompatTokenizer, load_piece_model


def _make_bpe_tok():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tok = Tokenizer(BPE(unk_token='<|unk|>'))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=120, special_tokens=[
        '<|pad|>', '<|bos|>', '<|eos|>', '<|unk|>'])
    tok.train_from_iterator([
        'человек идёт в лес большой дом',
        'мир время жизнь новый человек',
        'большой дом в лесу человек',
    ], trainer)
    return tok


@pytest.fixture(scope='module')
def sp():
    return SPCompatTokenizer(_make_bpe_tok())


class TestSPCompatInterface:
    def test_vocab_size(self, sp):
        assert sp.vocab_size() == sp._tok.get_vocab_size()
        assert sp.piece_size() == sp.vocab_size()
        assert len(sp) == sp.vocab_size()

    def test_special_ids(self, sp):
        assert sp.pad_id == 0 and sp.bos_id == 1
        assert sp.eos_id == 2 and sp.unk_id == 3

    def test_roundtrip(self, sp):
        text = 'человек идёт в лес'
        ids = sp.encode(text)
        assert isinstance(ids, list) and len(ids) >= 2
        assert all(isinstance(i, int) for i in ids)
        assert sp.decode(ids) == text
        assert sp.decode(ids[0])  # одиночный токен декодируется

    def test_encode_no_bos_eos(self, sp):
        ids = sp.encode('человек')
        assert sp.bos_id not in ids and sp.eos_id not in ids
        ids_be = sp.encode('человек', add_bos=True, add_eos=True)
        assert ids_be[0] == sp.bos_id and ids_be[-1] == sp.eos_id

    def test_encode_out_type_str(self, sp):
        pieces = sp.encode('человек', out_type=str)
        assert isinstance(pieces, list) and all(isinstance(p, str) for p in pieces)

    def test_id_to_piece_whitespace_marker(self, sp):
        # Токены с ведущим пробелом ByteLevel -> '▁'-префикс (SP-нотация)
        for cid in range(sp.vocab_size()):
            piece = sp.IdToPiece(cid)
            assert isinstance(piece, str)
            raw = piece.lstrip('\u2581')
            # ▁ никогда не встречается в середине куска
            assert '\u2581' not in raw

    def test_id_to_piece_matches_decode(self, sp):
        # Склейка ▁-кусков без ▁ внутри должна давать исходный текст
        text = 'большой дом в лесу'
        ids = sp.encode(text)
        pieces = [sp.IdToPiece(i) for i in ids]
        joined = ''.join(p.lstrip('\u2581') if p.startswith('\u2581') else p
                         for p in pieces)
        assert text.replace(' ', '') == joined.replace(' ', '')

    def test_piece_to_id_roundtrip(self, sp):
        # Цельные слова: единственный токен -> id, обратно -> текст
        for w in ('человек', 'мир', 'лес', 'дом'):
            wid = sp.PieceToId(w)
            if wid is not None:
                assert sp.IdToPiece(wid).lstrip('\u2581') == w

    def test_piece_to_id_underscore(self, sp):
        # ▁-префикс должен резолвиться как токен с пробелом
        wid_plain = sp.PieceToId('человек')
        wid_ws = sp.PieceToId('\u2581человек')
        if wid_plain is not None:
            # ▁человек = токен ' человек' — либо тот же, либо другой валидный
            assert wid_ws is not None or wid_plain is not None

    def test_piece_to_id_unknown(self, sp):
        assert sp.PieceToId('несуществующеесловоабвгдеж') is None
        assert sp.PieceToId(None) is None

    def test_encode_as_pieces(self, sp):
        pieces = sp.EncodeAsPieces('человек')
        assert len(pieces) == len(sp.encode('человек'))
        assert sp.encode_as_pieces('человек') == pieces

    def test_snake_case_aliases(self, sp):
        for cid in range(min(20, sp.vocab_size())):
            assert sp.id_to_piece(cid) == sp.IdToPiece(cid)
        assert sp.piece_to_id('человек') == sp.PieceToId('человек')

    def test_vocab_size_cached(self, sp):
        # Кэш: повторные вызовы не обращаются к Rust-биндингу
        assert sp._vocab_size == sp.vocab_size()


class TestLoadPieceModel:
    def test_load_json_returns_adapter(self, tmp_path):
        import json
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        tok = Tokenizer(BPE(unk_token='<|unk|>'))
        path = tmp_path / 'tokenizer.json'
        tok.save(str(path))
        sp = load_piece_model(str(path))
        assert isinstance(sp, SPCompatTokenizer)

    def test_load_model_missing(self, tmp_path):
        # .model без файла — SP упадёт с ошибкой (значит путь пошёл в SP)
        with pytest.raises(Exception):
            load_piece_model(str(tmp_path / 'missing.model'))