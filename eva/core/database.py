"""
database.py — hierarchical storage manager ("Хранилище").

Wraps the compact hierarchical storage and heads DB into a single API.
Manages: sentences, transitions, morph/syntax distributions, head metadata.
"""
import os, json, pickle, time
import numpy as np
from scipy.sparse import load_npz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
V5_DIR = os.path.join(ROOT, 'real_data', 'v5')
HIER_DIR = os.path.join(V5_DIR, 'hierarchical')
WIKI_DIR = os.path.join(V5_DIR, 'wikipedia')
META_PATH = os.path.join(V5_DIR, 'heads_meta.pkl')


class Database:
    """Single entry point for all stored data."""

    def __init__(self):
        self.load_time = 0.0
        self.stats = {}

    def load(self):
        t0 = time.time()
        # Meta
        with open(META_PATH, 'rb') as f:
            self.meta = pickle.load(f)
        # Sentences — War and Peace
        sent_data = np.load(os.path.join(HIER_DIR, 'sentences.npz'))
        self.sent_tokens = sent_data['tokens']
        self.sent_lens = sent_data['token_lens']
        self.sent_word_counts = sent_data['word_counts']
        self.sent_word_spans = sent_data['word_spans']
        # Sentences — Wikipedia RU (if exists)
        self.wiki_sent_tokens = None
        self.wiki_sent_lens = None
        self.wiki_sent_word_counts = None
        self.wiki_sent_word_spans = None
        wiki_sent_path = os.path.join(WIKI_DIR, 'sentences.npz')
        if os.path.exists(wiki_sent_path):
            wsd = np.load(wiki_sent_path)
            self.wiki_sent_tokens = wsd['tokens']
            self.wiki_sent_lens = wsd['token_lens']
            self.wiki_sent_word_counts = wsd['word_counts']
            self.wiki_sent_word_spans = wsd['word_spans']
        # Transitions
        self.trans_csr = load_npz(os.path.join(HIER_DIR, 'transitions_csr.npz'))
        self.log_prob_csr = load_npz(os.path.join(HIER_DIR, 'log_prob_csr.npz'))
        # Token counts
        self.token_counts = np.load(os.path.join(HIER_DIR, 'token_counts.npz'))['counts']
        # Morph/syntax cache (loaded on demand)
        self._morph_cache = None
        self._syntax_cache = None
        # Metadata
        self.V = int(self.meta.get('V', 4101))
        self.load_time = time.time() - t0
        self.stats = self.meta.get('stats', {})
        self.stats['V'] = self.V

    @property
    def morph_cache(self):
        if self._morph_cache is None:
            self._morph_cache = np.load(
                os.path.join(HIER_DIR, 'morph_cache.npz'), allow_pickle=True)
        return self._morph_cache

    @property
    def syntax_cache(self):
        if self._syntax_cache is None:
            self._syntax_cache = np.load(
                os.path.join(HIER_DIR, 'syntax_cache.npz'), allow_pickle=True)
        return self._syntax_cache

    def get_sentence(self, idx):
        """Reconstruct a WP sentence by index."""
        ptr = int(np.sum(self.sent_lens[:idx])) if idx > 0 else 0
        L = int(self.sent_lens[idx])
        nw = int(self.sent_word_counts[idx])
        tokens = list(self.sent_tokens[ptr:ptr + L])
        wptr = int(np.sum(self.sent_word_counts[:idx])) * 2 if idx > 0 else 0
        spans = []
        for j in range(nw):
            s = int(self.sent_word_spans[wptr + 2 * j])
            e = int(self.sent_word_spans[wptr + 2 * j + 1])
            spans.append((s, e))
        return {'tokens': tokens, 'word_spans': spans, 'length': L}

    def get_wiki_sentence(self, idx):
        """Reconstruct a Wikipedia sentence by index."""
        if self.wiki_sent_tokens is None:
            return None
        ptr = int(np.sum(self.wiki_sent_lens[:idx])) if idx > 0 else 0
        L = int(self.wiki_sent_lens[idx])
        nw = int(self.wiki_sent_word_counts[idx])
        tokens = list(self.wiki_sent_tokens[ptr:ptr + L])
        wptr = int(np.sum(self.wiki_sent_word_counts[:idx])) * 2 if idx > 0 else 0
        spans = []
        for j in range(nw):
            s = int(self.wiki_sent_word_spans[wptr + 2 * j])
            e = int(self.wiki_sent_word_spans[wptr + 2 * j + 1])
            spans.append((s, e))
        return {'tokens': tokens, 'word_spans': spans, 'length': L}

    @property
    def n_sentences(self):
        return len(self.sent_lens)

    @property
    def n_wiki_sentences(self):
        return len(self.wiki_sent_lens) if self.wiki_sent_lens is not None else 0

    @property
    def total_sentences(self):
        return self.n_sentences + self.n_wiki_sentences

    @property
    def n_tokens(self):
        return int(np.sum(self.sent_lens))

    @property
    def n_words(self):
        return int(np.sum(self.sent_word_counts))

    def get_transition_count(self, src, dst):
        """Get raw transition count between two tokens."""
        if src < self.V and dst < self.V:
            return int(self.trans_csr[src, dst])
        return 0

    def summary(self):
        stats = self.meta.get('stats', {})
        return {
            'sentences': stats.get('n_sentences', self.total_sentences),
            'tokens': stats.get('n_tokens', self.n_tokens),
            'words': stats.get('n_words', self.n_words),
            'vocab': self.V,
            'transitions': self.trans_csr.nnz,
            'contra_pairs': stats.get('n_contra_pairs', len(self.meta.get('contra_pairs', []))),
            'ngrams': stats.get('n_ngrams', len(self.meta.get('ngram_sparse', []))),
            'load_time': f'{self.load_time:.2f}s',
            'disk': self._disk_usage(),
        }

    def _disk_usage(self):
        total = 0
        for root, dirs, files in os.walk(V5_DIR):
            for f in files:
                fp = os.path.join(root, f)
                total += os.path.getsize(fp)
        return f'{total / 1024 / 1024:.1f} MB'

    def save_meta(self):
        """Save updated meta back to disk."""
        with open(META_PATH, 'wb') as f:
            pickle.dump(self.meta, f, protocol=5)
