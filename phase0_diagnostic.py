"""Phase 0: Morphological decomposition diagnostics."""

import sys, os, json, random
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE = os.path.dirname(os.path.abspath(__file__))

def stem_suffix_ending_russian(word):
    """Simple rule-based Russian decomposition: try to split stem + ending."""
    endings = ['а', 'ы', 'е', 'у', 'ой', 'ой', 'ую', 'ою',
               'ей', 'ий', 'ие', 'ия', 'ию', 'ием', 'иях',
               'ами', 'ях', 'ах', 'ов', 'ев', 'ём', 'ем',
               'ам', 'ом', 'ой', 'ею', 'о', 'ых', 'им', 'ими',
               'ешь', 'ет', 'ем', 'ете', 'ут', 'ют', 'ат', 'ят',
               'ал', 'ла', 'ло', 'ли', 'ть', 'ти', 'чь',
               'л', 'на', 'ся', 'сь']
    # Try to split: find longest ending match
    for end in sorted(endings, key=len, reverse=True):
        if len(word) > len(end) + 1 and word.endswith(end):
            stem = word[:-len(end)]
            return stem, end
    return word, ''


def decompose_known_prefixed(word, root_candidates):
    """Try to split off known Russian prefixes."""
    prefixes = ['вз', 'воз', 'вос', 'вы', 'до', 'за', 'из', 'ис',
                'на', 'над', 'наи', 'не', 'недо', 'низ', 'нис',
                'о', 'об', 'обез', 'обес', 'пере', 'по', 'под',
                'подо', 'пра', 'пред', 'пре', 'при', 'про',
                'раз', 'рас', 'со', 'с', 'у', 'без', 'бес',
                'вне', 'внутри', 'меж', 'между', 'около',
                'после', 'сверх', 'через', 'чрез',
                'анти', 'архи', 'гипер', 'де', 'дис', 'ин',
                'контр', 'суб', 'супер', 'ультра',
                'экс']
    for pfx in sorted(prefixes, key=len, reverse=True):
        if word.startswith(pfx) and len(word) > len(pfx) + 2:
            rest = word[len(pfx):]
            # Check if rest could be a known root (starts with consonant in most cases)
            if rest[0] in 'аеёиоуыэюя':
                continue  # unlikely root start after prefix
            return pfx, rest
    return '', word


def main():
    # Load sp model for CID lookup
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(BASE, 'real_data', 'bpe_ru_146k.model'))

    # Load morph_vocab.json for lemma info
    morph_path = os.path.join(BASE, 'real_data', 'morph_vocab.json')
    if os.path.exists(morph_path):
        with open(morph_path, encoding='utf-8') as f:
            mv = json.load(f)
        word_cache = mv.get('word_cache', {})
        lemma_rank = mv.get('_lemma_rank', mv.get('lemma_rank', {}))
        print(f"Loaded morph_vocab: {len(word_cache)} words, {len(lemma_rank)} lemmas")
    else:
        print("No morph_vocab.json found, building from SP tokens...")
        word_cache = {}
        lemma_rank = {}
        for i in range(sp.vocab_size()):
            piece = sp.IdToPiece(i)
            word = piece.replace('\u2581', '').lower().strip()
            if word and not all(c in '.,!?;:()[]{}«»—–-…\'\""' for c in word):
                word_cache[word] = i

    # Sample 2000 words
    rng = random.Random(42)
    items = list(word_cache.items())
    rng.shuffle(items)
    sample = items[:2000]

    # Try natasha for morphological analysis
    try:
        from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab as NatMorph, Doc
        _segmenter = Segmenter()
        _emb = NewsEmbedding()
        _tagger = NewsMorphTagger(_emb)
        _nat_morph = NatMorph()
        has_natasha = True
        print("natasha: available")
    except ImportError:
        has_natasha = False
        print("natasha: NOT available, using rule-based")

    results = []
    root_candidates = set()
    for word, cid in sample:
        word_lower = word.replace('\u2581', '').lower().strip()
        if not word_lower or len(word_lower) < 3:
            continue
        entry = {'word': word_lower, 'cid': cid}

        # Decompose: use rule-based for stem/ending/prefix
        pfx, rest = decompose_known_prefixed(word_lower, root_candidates)
        if pfx:
            entry['prefix'] = pfx
            entry['root_stem'] = rest
            stem, ending = stem_suffix_ending_russian(rest)
        else:
            stem, ending = stem_suffix_ending_russian(word_lower)
        entry['stem'] = stem
        entry['ending'] = ending

        if has_natasha and len(word_lower) >= 3:
            try:
                doc = Doc(word_lower)
                doc.segment(_segmenter)
                doc.tag_morph(_tagger)
                for token in doc.tokens:
                    token.lemmatize(_nat_morph)
                    entry['lemma'] = token.lemma.lower()
                    entry['pos'] = token.pos
                    if token.feats:
                        entry['grammemes'] = str(token.feats)
                    break
            except Exception:
                pass
        if 'lemma' not in entry:
            entry['lemma'] = word_lower

        results.append(entry)
        if len(entry.get('stem', '')) > 2:
            root_candidates.add(entry['stem'])
        if 'root_stem' in entry and len(entry['root_stem']) > 2:
            root_candidates.add(entry['root_stem'])

    # Summary stats
    n = len(results)
    n_prefixed = sum(1 for r in results if r.get('prefix'))
    n_ending = sum(1 for r in results if r.get('ending'))
    n_with_lemma = sum(1 for r in results if r.get('lemma') and r['lemma'] != r['word'])
    stem_len = [len(r.get('stem', r.get('root_stem', ''))) for r in results]
    ending_len = [len(r.get('ending', '')) for r in results]

    print(f"\n=== Phase 0: Morphological Diagnostics ===")
    if has_natasha:
        print(f"  Words with lemma from natasha: {n_with_lemma}/{n} ({100*n_with_lemma//n}%)")
    print(f"  Words with prefix detected: {n_prefixed} ({100*n_prefixed//n}%)")
    print(f"  Words with ending split: {n_ending} ({100*n_ending//n}%)")
    print(f"  Unique root candidates: {len(root_candidates)}")
    print(f"  Avg stem length: {sum(stem_len)/max(len(stem_len),1):.1f}")
    print(f"  Avg ending length: {sum(ending_len)/max(len(ending_len),1):.1f}")
    print(f"  MorphemeField estimated size: ~{len(root_candidates)} unique morphemes")

    # Show examples
    print("\n  Decomposition examples:")
    for r in results[:15]:
        parts = []
        if r.get('prefix'):
            parts.append(f"pref={r['prefix']}")
        if r.get('stem'):
            parts.append(f"stem={r['stem']}")
        elif r.get('root_stem'):
            parts.append(f"root={r['root_stem']}")
        if r.get('ending'):
            parts.append(f"end={r['ending']}")
        if r.get('lemma') and r['lemma'] != r.get('word', ''):
            parts.append(f"lemma={r['lemma']}")
        if r.get('pos'):
            parts.append(f"POS={r['pos']}")
        conf_str = ""
        print(f"    {r['word']:20s} -> {' | '.join(parts)} {conf_str}")

    # Fallback recommendation
    print(f"\n  Recommendation for Phase 1-2:")
    if has_natasha and n_with_lemma / max(n, 1) > 0.5:
        print(f"    USE natasha for lemma+POS, rule-based for stem/ending/prefix")
    else:
        print(f"    natasha available but only {n_with_lemma}/{n} have lemma. Use rule-based.")
    print(f"    Rule-based stem+ending works for {n_ending} of {n} words")

if __name__ == '__main__':
    main()
