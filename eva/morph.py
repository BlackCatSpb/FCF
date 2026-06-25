"""
Unified Russian morphological decomposition for FCF.

Provides:
  - pymorphy3-based decomposition (primary, lazy singleton)
  - rule-based fallback (prefix+stem+ending tables)
  - `decompose_word(word)` → dict or None  (primary interface)
  - `annotate_word(word)` → word with \u037E morpheme separators
  - `annotate_corpus_line(line)` → line with all Cyrillic words annotated
  - `validate_alignment(words, sp, e5_model)` → cos-sim alignment check
  - `DecompositionMethod` enum

All methods accept `method` kwarg: 'pymorphy3', 'rule', or 'auto' (pymorphy3→rule).

Usage:
    from eva.morph import decompose_word, annotate_word
    parts = decompose_word('приносили')
    # {'PREFIX': 'при', 'ROOT': 'нос', 'ENDING': 'или'}

    annotated = annotate_word('приносили')
    # 'при;нос;или'
"""

import os
import re
import sys
import math
import time
import typing as t
from collections import Counter
from pathlib import Path

import numpy as np

# ── Constants ──────────────────────────────────────────────────────

SEP = '\u037E'  # Greek question mark — morpheme boundary marker

# Russian consonants for ending boundary detection
CONSONANTS = frozenset('бвгджзйклмнпрстфхцчшщ')

# Common Russian prefixes (sorted by length descending for greedy match)
PREFIXES = [
    'вз', 'воз', 'вос', 'вы', 'до', 'за', 'из', 'ис',
    'на', 'над', 'наи', 'не', 'недо', 'низ', 'нис',
    'о', 'об', 'обез', 'обес', 'пере', 'по', 'под',
    'подо', 'пра', 'пред', 'пре', 'при', 'про',
    'раз', 'рас', 'со', 'с', 'у', 'без', 'бес',
    'вне', 'внутри', 'меж', 'между',
    'после', 'сверх', 'через',
    'анти', 'архи', 'гипер', 'де', 'дис', 'ин',
    'контр', 'суб', 'супер', 'ультра', 'экс',
]

# Common Russian endings (approximate)
ENDINGS = [
    'а', 'ы', 'е', 'у', 'ой', 'ую', 'ою',
    'ей', 'ий', 'ие', 'ия', 'ию', 'ием', 'иях',
    'ами', 'ях', 'ах', 'ов', 'ев', 'ём', 'ем',
    'ам', 'ом', 'ею', 'о', 'ых', 'им', 'ими',
    'ешь', 'ет', 'ем', 'ете', 'ут', 'ют', 'ат', 'ят',
    'ал', 'ла', 'ло', 'ли', 'ть', 'ти', 'чь',
    'л', 'на', 'ся', 'сь', 'ого', 'его', 'ому', 'ему',
    'ым', 'им', 'ыми', 'ими', 'ых', 'их',
]

# Regex for splitting text into Cyrillic words and non-Cyrillic segments
RE_CYRILLIC_WORD = re.compile(r'[а-яёА-ЯЁ]+|[^а-яёА-ЯЁ]+')
RE_EXTRA_SPACES = re.compile(r' {2,}')
RE_NONPRINT = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

# ── Helpers ────────────────────────────────────────────────────────


def _has_cyrillic(text: str, min_ratio: float = 0.3) -> bool:
    """Check if text contains a sufficient ratio of Cyrillic characters."""
    cyr = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    if not cyr:
        return False
    total = sum(1 for c in text if c.isalpha())
    return total == 0 or (cyr / total) >= min_ratio


def _clean_line(text: str) -> str:
    """Strip, remove non-printable chars, collapse spaces."""
    s = text.strip()
    s = RE_NONPRINT.sub('', s)
    s = RE_EXTRA_SPACES.sub(' ', s)
    return s.strip('—–-− \t')


# ── Lazy pymorphy3 singleton ──────────────────────────────────────

_PYMORPHY_ANALYZER = None


def _get_pymorphy():
    global _PYMORPHY_ANALYZER
    if _PYMORPHY_ANALYZER is None:
        import pymorphy3
        _PYMORPHY_ANALYZER = pymorphy3.MorphAnalyzer()
    return _PYMORPHY_ANALYZER


# ── Rule-based decomposition ────────────────────────────────────────


def _rule_decompose(word: str) -> t.Optional[t.Dict[str, str]]:
    """Rule-based Russian morpheme decomposition: prefix+stem+ending.

    Returns dict of {role: string} or None if confidence < threshold.
    """
    w = word.lower().strip()
    if len(w) < 3:
        return None

    result: t.Dict[str, str] = {}
    rest = w

    # 1. Split prefix (greedy, longest match)
    for p in sorted(PREFIXES, key=len, reverse=True):
        if not rest.startswith(p):
            continue
        min_stem = 3 if len(p) == 1 else 2
        if len(rest) <= len(p) + min_stem:
            continue
        # Check that prefix is followed by a consonant (morphotactic constraint)
        nxt = rest[len(p)]
        if nxt in 'аеёиоуыэюя':
            continue
        result['PREFIX'] = p
        rest = rest[len(p):]
        break

    # 2. Split ending (greedy, longest match, preceded by consonant)
    for e in sorted(ENDINGS, key=len, reverse=True):
        if len(rest) > len(e) + 1 and rest.endswith(e):
            pre = rest[-(len(e) + 1)]
            if pre in CONSONANTS:
                result['ENDING'] = e
                rest = rest[:-len(e)]
                break

    # 3. Remainder is root
    if rest:
        result['ROOT'] = rest

    # Confidence check
    n_parts = len(result)
    if n_parts < 2:
        return None
    if len(result.get('ROOT', '')) < 2:
        return None

    return result


# ── pymorphy3 decomposition ────────────────────────────────────────


def _pymorphy_decompose(word: str) -> t.Optional[t.Dict[str, str]]:
    """Fallback decomposition using pymorphy3 morphological analysis.

    Returns dict of {role: string} or None if parse confidence too low.
    """
    morph = _get_pymorphy()
    parsed = morph.parse(word)
    if not parsed or parsed[0].score < 0.3:
        return None

    p = parsed[0]
    nf = p.normal_form.lower()
    w = word.lower().strip()

    if w == nf or len(w) < 4:
        return {'ROOT': nf}

    # Strip known prefix from word (same as rule-based)
    rest = w
    pfx = ''
    for pfx_candidate in sorted(PREFIXES, key=len, reverse=True):
        if not rest.startswith(pfx_candidate):
            continue
        min_stem = 3 if len(pfx_candidate) == 1 else 2
        if len(rest) <= len(pfx_candidate) + min_stem:
            continue
        # If normal_form also starts with this prefix, it's part of the root
        if nf.startswith(pfx_candidate) and len(nf) > len(pfx_candidate) + min_stem:
            continue
        pfx = pfx_candidate
        rest = rest[len(pfx):]
        break

    # Strip same prefix from normal_form
    nf_stem = nf
    if pfx:
        for pfx_candidate in sorted(PREFIXES, key=len, reverse=True):
            if nf_stem.startswith(pfx_candidate):
                nf_stem = nf_stem[len(pfx_candidate):]
                break

    result: t.Dict[str, str] = {}
    if pfx:
        result['PREFIX'] = pfx

    # Align rest with nf_stem via longest common prefix
    i = 0
    while i < min(len(rest), len(nf_stem)) and rest[i] == nf_stem[i]:
        i += 1

    if i >= 2:
        result['ROOT'] = rest[:i]
        if i < len(rest):
            result['ENDING'] = rest[i:]
    elif len(rest) <= len(nf_stem) + 3:
        # Word form similar length to lemma — consonant boundary split
        split_pos = max(2, len(rest) - 2)
        while split_pos < len(rest) and rest[split_pos] not in CONSONANTS:
            split_pos += 1
        if split_pos < len(rest):
            result['ROOT'] = rest[:split_pos]
            result['ENDING'] = rest[split_pos:]
        else:
            result['ROOT'] = rest
    else:
        result['ROOT'] = nf_stem if len(nf_stem) >= 2 else rest[:3]
        suffix = rest[len(result['ROOT']):] if rest.startswith(result['ROOT']) else rest[i:]
        if suffix:
            result['ENDING'] = suffix

    return result


# ── Public API ──────────────────────────────────────────────────────


def decompose_word(word: str, method: str = 'auto') -> t.Optional[t.Dict[str, str]]:
    """Decompose a Russian word into morphemes.

    Args:
        word: Russian word to decompose (case-insensitive).
        method: 'pymorphy3' — use pymorphy3 only
                'rule' — use rule-based only
                'auto' — pymorphy3 first, rule-based fallback

    Returns:
        dict of {role: string} (keys: PREFIX, ROOT, ENDING) or None.
    """
    w = word.lower().strip()
    if len(w) < 3 or not _has_cyrillic(w):
        return None
    if all(c in '.,!?;:()[]{}«»—–-…\'\"1234567890 ' for c in w):
        return None

    if method in ('auto', 'pymorphy3'):
        try:
            result = _pymorphy_decompose(word)
            if result is not None:
                return result
        except Exception:
            pass
        if method == 'pymorphy3':
            return None

    return _rule_decompose(word)


def annotate_word(word: str, method: str = 'auto') -> str:
    """Annotate a single word with morpheme separators.

    Returns word with SEP (\u037E) between morphemes, or original word
    if decomposition fails.
    """
    parts = decompose_word(word, method=method)
    if parts is None:
        return word
    ordered = []
    for role in ('PREFIX', 'ROOT', 'ENDING'):
        if role in parts:
            ordered.append(parts[role])
    return SEP.join(ordered)


def annotate_corpus_line(line: str, method: str = 'auto') -> str:
    """Annotate all Cyrillic words in a line with morpheme separators.

    Non-Cyrillic segments (punctuation, numbers, spaces) are preserved.
    """
    segments = RE_CYRILLIC_WORD.findall(line)
    result = []
    for seg in segments:
        if re.match(r'^[а-яёА-ЯЁ]+$', seg):
            result.append(annotate_word(seg, method=method))
        else:
            result.append(seg)
    return ''.join(result)


# ── Corpus annotation (streaming) ─────────────────────────────────


def annotate_corpus_file(
    input_path: str,
    output_path: str,
    method: str = 'auto',
    max_lines: int = 0,
    progress_interval: int = 100000,
) -> t.Dict[str, int]:
    """Stream-annotate a corpus file with morpheme separators.

    Reads line by line, annotates each, writes to output.
    Memory-efficient: O(max_line_length), not O(corpus).

    Returns stats dict.
    """
    stats: t.Dict[str, int] = Counter()
    t0 = time.time()
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if max_lines and i >= max_lines:
                break
            line = _clean_line(line)
            if not line:
                continue
            annotated = annotate_corpus_line(line, method=method)
            fout.write(annotated + '\n')
            stats['lines'] += 1
            if (i + 1) % progress_interval == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1:>9}] {rate:.0f} L/s", flush=True)
    return dict(stats)


# ── Alignment validation ───────────────────────────────────────────


def validate_alignment(
    words: t.List[str],
    sp,
    e5_model=None,
    sample_size: int = 500,
    device: str = 'cpu',
) -> t.Dict[str, float]:
    """Validate alignment between BPE tokenization and morph decomposition.

    For each word, computes:
    1. e5 embedding of the full word
    2. e5 embedding of the VSA bundle of BPE subword tokens
    3. e5 embedding of the VSA bundle of morphologically decomposed parts
    4. Cosine similarity between (2) and (1), (3) and (1)

    Returns dict with mean/std cos-sim for both methods.

    Args:
        words: list of Russian words to test
        sp: SentencePieceProcessor (loaded)
        e5_model: SentenceTransformer model for embedding (loaded or None)
        sample_size: max words to test
    """
    if e5_model is None:
        from sentence_transformers import SentenceTransformer
        e5_model = SentenceTransformer('intfloat/multilingual-e5-base', device=device)

    rng = np.random.RandomState(42)
    if len(words) > sample_size:
        words = rng.choice(sorted(set(words)), sample_size, replace=False).tolist()

    bpe_sims = []
    morph_sims = []

    for word in words:
        if not _has_cyrillic(word) or len(word) < 3:
            continue
        try:
            target = e5_model.encode(word, normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            continue

        # BPE subword bundle
        ids = sp.encode(word)
        pieces = [sp.IdToPiece(i).replace('\u2581', '').strip() for i in ids]
        pieces = [p for p in pieces if p and len(p) >= 2]
        if pieces:
            try:
                bpe_embs = e5_model.encode(pieces, normalize_embeddings=True, show_progress_bar=False)
                bpe_bundle = np.mean(bpe_embs, axis=0).astype(np.float32)
                bpe_bundle /= max(np.linalg.norm(bpe_bundle), 1e-10)
                bpe_sims.append(float(np.dot(bpe_bundle, target)))
            except Exception:
                pass

        # Morph decomposition bundle
        parts = decompose_word(word)
        if parts and len(parts) >= 2:
            m_texts = [m for _, m in parts.items()]
            try:
                m_embs = e5_model.encode(m_texts, normalize_embeddings=True, show_progress_bar=False)
                m_bundle = np.mean(m_embs, axis=0).astype(np.float32)
                m_bundle /= max(np.linalg.norm(m_bundle), 1e-10)
                morph_sims.append(float(np.dot(m_bundle, target)))
            except Exception:
                pass

    result = {}
    if bpe_sims:
        arr = np.array(bpe_sims)
        result['bpe_mean'] = float(arr.mean())
        result['bpe_std'] = float(arr.std())
    if morph_sims:
        arr = np.array(morph_sims)
        result['morph_mean'] = float(arr.mean())
        result['morph_std'] = float(arr.std())

    return result


# ── Vocabulary coverage ────────────────────────────────────────────


def vocab_coverage(words: t.List[str], sp) -> t.Dict[str, float]:
    """Compute BPE vocabulary coverage statistics.

    Returns dict with coverage at various token counts (1, 2, 3, 5, 10 tokens).
    """
    lengths = [len(sp.encode(w)) for w in words]
    arr = np.array(lengths)
    total = len(arr)
    return {
        'unique_words': total,
        '1_token': float((arr == 1).sum() / total),
        '2_tokens': float((arr <= 2).sum() / total),
        '3_tokens': float((arr <= 3).sum() / total),
        '5_tokens': float((arr <= 5).sum() / total),
        '10_tokens': float((arr <= 10).sum() / total),
        'mean_tokens': float(arr.mean()),
        'median_tokens': float(np.median(arr)),
        'max_tokens': int(arr.max()),
    }
