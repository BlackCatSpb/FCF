from collections import Counter, defaultdict
import json, os, sys, time
import numpy as np
from eva.symbolic.fractal_encoding import path as zck_path
from eva.symbolic.fcf_config import EnvironmentResolver

try:
    from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab as NatMorph, Doc
except ImportError:
    Segmenter = NewsEmbedding = NewsMorphTagger = NatMorph = Doc = None


_ENV = EnvironmentResolver()


class MorphVocab:
    """Morphological vocabulary with custom Zeckendorf paths per concept.

    CID range: 0..N_cids-1 (flat, matches SP vocab).
    Each concept can have a custom Zeckendorf path stored in _path_override.

    Path types:
        service word (ADP, CCONJ, etc.): path = ZCK(cid)           [standard]
        content word (known lemma+form):  path = ZCK(lemma_rank)[:12]
                                            + ZCK(form_rank)[:4]
        BPE fallback:                     path = ZCK(cid)           [standard]
    """

    def __init__(self, sp_model_path=None, vocab_size=146000):
        if sp_model_path is None:
            sp_model_path = _ENV.bpe_model_path
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.vocab_size = min(vocab_size, self._sp.vocab_size())

        # word -> CID (direct SP ID or morph-overridden ID)
        self.word_cache = {}
        # CID -> word (reverse index for O(1) decode)
        self._cid_to_word = {}
        # CID -> custom Zeckendorf path tuple (service/fallback CIDs not stored here)
        self._path_override = {}

        # Service POS tags
        self._service_pos = {'ADP', 'CCONJ', 'SCONJ', 'PART', 'PRON', 'PUNCT'}
        self._service_words = set()

        # Lemma/form frequency rankings for path construction
        self._lemma_rank = {}    # lemma -> rank (0..N)
        self._form_rank = {}     # grammeme string -> rank (0..N)
        self._word_info = {}     # word -> (lemma, grammeme, is_service)

    def zeckendorf_path(self, cid):
        """Custom Zeckendorf path for concept cid."""
        if cid in self._path_override:
            return self._path_override[cid]
        return zck_path(cid)

    # ── Build from corpus ──────────────────────────────────────

    @classmethod
    def build(cls, corpus_path=None,
              sp_model_path=None):
        """Build morphological vocabulary: parse corpus, assign custom paths."""
        if corpus_path is None:
            corpus_path = _ENV.raw_corpus_path
        if sp_model_path is None:
            sp_model_path = _ENV.bpe_model_path
        self = cls(sp_model_path=sp_model_path)

        if Segmenter is None:
            raise ImportError("natasha not installed. Run: pip install natasha")

        segmenter = Segmenter()
        emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(emb)
        nat_morph = NatMorph()

        print("  Parsing corpus with Natasha...")
        t0 = time.perf_counter()
        lemma_counts = Counter()
        form_counts = Counter()
        word_info = {}  # word.lower() -> (lemma, grammeme, is_service)
        service_words = set()

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                doc = Doc(line)
                doc.segment(segmenter)
                doc.tag_morph(morph_tagger)
                for token in doc.tokens:
                    token.lemmatize(nat_morph)
                    word = token.text.lower()
                    lemma = token.lemma.lower() if token.lemma else word
                    pos = token.pos or 'X'

                    gram_parts = [pos]
                    feats = token.feats or {}
                    for k in ['Number', 'Person', 'Gender', 'Case', 'Tense',
                              'VerbForm', 'Mood', 'Voice', 'Aspect',
                              'Animacy', 'Degree', 'Variant', 'Definite']:
                        v = feats.get(k)
                        if v:
                            gram_parts.append(f'{k}={v}')
                    gram = '|'.join(gram_parts)

                    is_service = pos in self._service_pos
                    if is_service:
                        service_words.add(word)

                    lemma_counts[lemma] += 1
                    form_counts[gram] += 1
                    word_info[word] = (lemma, gram, is_service)

                if lineno % 10000 == 0 and lineno > 0:
                    print(f"    processed {lineno} lines...", end='\r')

        t1 = time.perf_counter()
        print(f"    done: {len(word_info)} unique words, "
              f"{len(lemma_counts)} lemmas, {len(form_counts)} forms in {t1-t0:.1f}s")

        # Rank lemmas and forms by frequency
        for rank, (lemma, _) in enumerate(lemma_counts.most_common()):
            self._lemma_rank[lemma] = rank
        for rank, (gram, _) in enumerate(form_counts.most_common()):
            self._form_rank[gram] = rank

        self._service_words = service_words
        self._word_info = word_info

        # Build word_cache: word -> CID (direct SP ID)
        all_sp_tokens = {}
        for i in range(self.vocab_size):
            piece = self._sp.IdToPiece(i)
            token = piece.lstrip('▁').lower()
            all_sp_tokens[token] = i
            if not token:
                all_sp_tokens[piece.lower()] = i

        # Match morph words to their SP IDs
        n_content = 0
        n_service = 0
        for word, (lemma, gram, is_service) in word_info.items():
            if word in all_sp_tokens:
                cid = all_sp_tokens[word]
                self.word_cache[word] = cid
                self._cid_to_word[cid] = word
                if is_service:
                    n_service += 1
                else:
                    n_content += 1

        # Build path overrides for content words (shared lemma -> same path prefix)
        for word in self.word_cache:
            if word not in word_info:
                continue
            lemma, gram, is_service = word_info[word]
            if is_service:
                continue  # service words keep standard path

            cid = self.word_cache[word]
            lemma_rank = self._lemma_rank.get(lemma)
            form_rank = self._form_rank.get(gram)
            if lemma_rank is None:
                continue

            # path = ZCK(lemma_rank)[:12] + ZCK(form_rank)[:4]
            lp = zck_path(lemma_rank)[:12]
            fp = zck_path(form_rank)[:4] if form_rank is not None else (0,) * 4
            self._path_override[cid] = lp + fp

        print(f"  Cached: {len(self.word_cache)} words"
              f" ({n_service} service, {n_content} content)")
        print(f"  Path overrides: {len(self._path_override)} content concepts")
        return self

    # ── Serialization ──────────────────────────────────────────

    def get_path_overrides(self):
        return self._path_override

    def save(self, path):
        data = {
            'word_cache': self.word_cache,
            'path_override': {str(k): list(v) for k, v in self._path_override.items()},
            'lemma_rank': self._lemma_rank,
            'form_rank': self._form_rank,
            'service_words': list(self._service_words),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  Saved MorphVocab to {path}")

    @classmethod
    def load(cls, path, sp_model_path=None):
        if sp_model_path is None:
            sp_model_path = _ENV.bpe_model_path
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self = cls(sp_model_path=sp_model_path)
        # Filter word_cache to CIDs within current vocab (may differ from build-time)
        vs = self.vocab_size
        self.word_cache = {w: c for w, c in data['word_cache'].items() if c < vs}
        self._path_override = {int(k): tuple(v) for k, v in data['path_override'].items()
                               if int(k) < vs}
        self._lemma_rank = data['lemma_rank']
        self._form_rank = data['form_rank']
        self._service_words = set(data['service_words'])
        return self

    def decode_cid(self, cid):
        """Convert CID back to (word, is_morph) for generation output."""
        if cid >= self.vocab_size:
            return None, False
        if cid in self._cid_to_word:
            return self._cid_to_word[cid], True
        piece = self._sp.IdToPiece(cid)
        return piece.lstrip('▁'), False


def build_morph_vocab(corpus_path=None,
                      sp_model_path=None,
                      output=None):
    if corpus_path is None:
        corpus_path = _ENV.raw_corpus_path
    if sp_model_path is None:
        sp_model_path = _ENV.bpe_model_path
    if output is None:
        output = _ENV.morph_vocab_path
    mv = MorphVocab.build(corpus_path, sp_model_path=sp_model_path)
    mv.save(output)
    return mv


if __name__ == '__main__':
    import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    build_morph_vocab()
