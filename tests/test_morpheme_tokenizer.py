"""Tests for MorphemeTokenizer — трёхуровневое сито: символы → морфемы → слова.

Словарь строится в памяти (мини-версия morpheme-65k) и проверяется полный
SP-совместимый интерфейс: encode/decode, уровни, маркер \\u037E, регистр.
"""
import pytest

from eva.symbolic.morpheme_tokenizer import (
    MorphemeTokenizer, looks_like_morpheme, SEP, SPECIALS)


def _make_data():
    # id: 0-3 спец, 4+ символы, затем морфемы, затем слова
    char = {' ': 4, 'а': 5, 'б': 6, 'в': 7, 'г': 8, 'е': 9, 'и': 10,
            'й': 11, 'к': 12, 'л': 13, 'м': 14, 'н': 15, 'о': 16, 'п': 17,
            'р': 18, 'с': 19, 'т': 20, 'у': 21, 'х': 22, 'ь': 23, 'я': 24,
            'А': 25, 'Б': 26, 'Г': 27, 'Ч': 49, '.': 28, ',': 29, '1': 30, '2': 31}
    morph = {'пере': 32, 'ход': 33, 'чел': 34, 'овек': 35, 'идет': 36,
             'лес': 37, 'больш': 38, 'ой': 39, 'дом': 40, 'крейсерск': 41,
             'ую': 42, 'проб': 43, 'еж': 44, 'али': 45}
    word = {'человек': 46, 'через': 47, 'дорогу': 48}
    return {
        'format': 'morpheme-v1',
        'specials': {s: i for i, s in enumerate(SPECIALS)},
        'char': char, 'morph': morph, 'word': word,
        'chains': {'переход': ['пере', 'ход'], 'человек': ['чел', 'овек'],
                   'большой': ['больш', 'ой']},
        'vocab_size': 50,
    }


@pytest.fixture(scope='module')
def mt():
    return MorphemeTokenizer(_make_data())


class TestMorphemeInterface:
    def test_vocab(self, mt):
        assert mt.vocab_size() == 50
        assert mt.piece_size() == 50
        assert mt.pad_id() == 0
        assert mt.bos_id() == 1
        assert mt.eos_id() == 2
        assert mt.unk_id() == 3

    def test_word_level(self, mt):
        assert mt.encode('человек') == [46]

    def test_chain_level(self, mt):
        assert mt.encode('переход') == [32, 33]

    def test_morph_level(self, mt):
        assert mt.encode('переход') == [32, 33]

    def test_greedy_split(self, mt):
        assert mt.encode('большой') == [38, 39]

    def test_char_fallback(self, mt):
        ids = mt.encode('пй')
        assert all(i >= 4 for i in ids)
        assert ids == [17, 11]

    def test_marker_stripped(self, mt):
        assert mt.encode('пере' + SEP + 'ход') == [32, 33]

    def test_upper_word(self, mt):
        assert mt.encode('Человек') == [49, 46]

    def test_punct_split(self, mt):
        ids = mt.encode('переход.')
        assert ids == [32, 33, 28]

    def test_unknown_char_unk(self, mt):
        ids = mt.encode('щ')
        assert ids == [3]

    def test_roundtrip(self, mt):
        text = 'переход через дорогу большой дом'
        assert mt.decode(mt.encode(text)) == text

    def test_encode_pieces(self, mt):
        pieces = mt.encode_pieces('переход')
        assert pieces == ['пере', 'ход']

    def test_id_to_piece(self, mt):
        assert mt.IdToPiece(46) == 'человек'
        assert mt.id_to_piece(32) == 'пере'


class TestDetection:
    def test_looks_like_morpheme(self):
        assert looks_like_morpheme({'format': 'morpheme-v1'})
        assert not looks_like_morpheme({'format': 'bpe'})
        assert not looks_like_morpheme('x')


class TestLoadPieceModel:
    def test_load_via_sp_compat(self, tmp_path, mt):
        import json
        p = tmp_path / 'tok.json'
        p.write_text(json.dumps(_make_data()), encoding='utf-8')
        from eva.symbolic.sp_compat import load_piece_model
        tok = load_piece_model(str(p))
        assert isinstance(tok, MorphemeTokenizer)
        assert tok.encode('переход') == [32, 33]