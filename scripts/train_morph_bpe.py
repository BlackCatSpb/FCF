"""
Train a morpheme-aware SentencePiece BPE model for FCF.

Level 1: BPE trained on corpus pre-segmented at morpheme boundaries.
         The separator token \u037E (GREEK QUESTION MARK) marks morpheme
         junctions, biasing BPE toward learning morpheme-sized tokens.

Level 2: Fallback analyzer — runtime decomposition for words not in
         SentencePiece vocabulary, using e5 embeddings and VSA bundle.

Usage:
    # Step 1: train morph-aware BPE
    python scripts/train_morph_bpe.py --corpus real_data/full_corpus_ru.txt \
        --output real_data/bpe_morph --vocab-size 256000 --device cpu

    # Step 2: seed concept vectors with e5 + morph decomposition
    python scripts/seed_embeddings.py --cs real_data/concept_space.json \
        --sp real_data/bpe_morph.model --all --device cpu
"""

import argparse
import os
import sys
import time
import math
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# -- Morpheme boundary separator token --
# Using a rare Unicode char that won't appear in normal text
SEP = '\u037E'  # Greek question mark

# Russian consonants for ending detection
CONSONANTS = set('\u0431\u0432\u0433\u0434\u0436\u0437\u0439\u043a\u043b'
                 '\u043c\u043d\u043f\u0440\u0441\u0442\u0444\u0445\u0446'
                 '\u0447\u0448\u0449')


def decompose_word(word):
    """Rule-based Russian morpheme decomposition: prefix+root+ending.
    Returns list of parts or None."""
    w = word.lower().strip()
    if len(w) < 3:
        return None

    prefixes = ['вз','воз','вос','вы','до','за','из','ис','на','над','наи',
                'не','недо','низ','нис','о','об','обез','обес','пере','по',
                'под','подо','пра','пред','пре','при','про','раз','рас',
                'со','с','у','без','бес','вне','внутри','меж','между',
                'после','сверх','через','анти','архи','гипер','де','дис',
                'ин','контр','суб','супер','ультра','экс']
    endings = ['а','ы','е','у','ой','ую','ою','ей','ий','ие','ия','ию',
               'ием','иях','ами','ях','ах','ов','ев','ём','ем','ам','ом',
               'ею','о','ых','им','ими','ешь','ет','ем','ете','ут','ют',
               'ат','ят','ал','ла','ло','ли','ть','ти','чь','л','на','ся',
               'сь','ого','его','ому','ему','ым','им','ыми','ими','ых','их']

    parts = []
    rest = w

    # Prefix
    for p in sorted(prefixes, key=len, reverse=True):
        if rest.startswith(p) and len(rest) > len(p) + 2:
            parts.append(('PREFIX', p))
            rest = rest[len(p):]
            break

    # Ending
    for e in sorted(endings, key=len, reverse=True):
        if len(rest) > len(e) + 1 and rest.endswith(e) and \
                rest[-(len(e) + 1)] in CONSONANTS:
            parts.append(('ENDING', e))
            rest = rest[:-len(e)]
            break

    if rest:
        parts.append(('ROOT', rest))

    if len(parts) >= 2:
        return parts
    return None


def augment_corpus(words, max_words=0):
    """Add SEP between morpheme parts for BPE training.
    Returns (augmented_text, stats_dict)."""
    stats = Counter()
    augmented = []
    total = len(words)
    limit = max_words if max_words > 0 else total

    for i, w in enumerate(words):
        if i >= limit:
            break
        parts = decompose_word(w)
        if parts:
            # Insert SEP between morpheme parts: "при+нос+и+ть"
            morph_text = SEP.join(p for _, p in parts)
            augmented.append(morph_text)
            stats['decomposed'] += 1
        else:
            augmented.append(w)
            stats['kept_whole'] += 1

        if (i + 1) % 50000 == 0:
            pct = 100 * (i + 1) / limit
            print(f"  [{pct:5.1f}%] {i+1}/{limit} words", flush=True)

    return '\n'.join(augmented), stats


def main():
    parser = argparse.ArgumentParser(description='Train morph-aware BPE for FCF')
    parser.add_argument('--corpus', required=True, help='Path to Russian corpus')
    parser.add_argument('--output', required=True, help='Output prefix for BPE model')
    parser.add_argument('--vocab-size', type=int, default=256000, help='BPE vocabulary size')
    parser.add_argument('--max-words', type=int, default=0, help='Max words to process (0=all)')
    parser.add_argument('--device', default='cpu', help='Torch device for e5 (cpu/cuda)')
    args = parser.parse_args()

    # -- 1. Load corpus --
    print("Loading corpus...", flush=True)
    with open(args.corpus, 'r', encoding='utf-8') as f:
        text = f.read()
    words = sorted(set(text.split()))
    print(f"  Unique words: {len(words)}")
    print(f"  Total chars: {len(text):,}")

    # -- 2. Decompose and augment --
    print(f"\nDecomposing words (max={args.max_words or 'all'})...", flush=True)
    t0 = time.time()
    aug_text, stats = augment_corpus(words, args.max_words)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Decomposed: {stats['decomposed']}")
    print(f"  Kept whole: {stats['kept_whole']}")

    # -- 3. Validate decomposition quality with e5 --
    print(f"\nValidating decomposition quality with e5...", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('intfloat/multilingual-e5-base', device=args.device)

        sample_words = [w for w in words if len(decompose_word(w) or []) >= 2][:500]
        if sample_words:
            word_embs = model.encode(sample_words, normalize_embeddings=True,
                                     show_progress_bar=False, batch_size=512)
            sims = []
            for word, target in zip(sample_words, word_embs):
                parts = decompose_word(word)
                if parts:
                    m_texts = [p for _, p in parts]
                    m_embs = model.encode(m_texts, normalize_embeddings=True,
                                          show_progress_bar=False)
                    bundle = np.mean(m_embs, axis=0)
                    bundle /= np.linalg.norm(bundle)
                    sims.append(float(np.dot(bundle, target)))

            arr = np.array(sims)
            print(f"  Samples: {len(sample_words)}")
            print(f"  Bundle cos: mean={arr.mean():.4f} std={arr.std():.4f}")
            print(f"  Ceiling (self-cos): 1.0000")
            # Baseline: random
            rng = np.random.RandomState(42)
            rnd_sims = []
            for target in word_embs:
                rnd = rng.randn(768).astype(np.float32)
                rnd /= np.linalg.norm(rnd)
                rnd_sims.append(float(np.dot(rnd, target)))
            print(f"  Random baseline: mean={np.mean(rnd_sims):.4f} std={np.std(rnd_sims):.4f}")
    except Exception as e:
        print(f"  Validation skipped: {e}")

    # -- 4. Train SentencePiece on augmented text --
    print(f"\nTraining SentencePiece (vocab_size={args.vocab_size})...", flush=True)
    aug_path = args.output + '_augmented.txt'
    with open(aug_path, 'w', encoding='utf-8') as f:
        f.write(aug_text)

    import sentencepiece as spm
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=aug_path,
        model_prefix=args.output,
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        pad_id=0, unk_id=0, bos_id=1, eos_id=2,
        pad_piece='<pad>', unk_piece='<unk>', bos_piece='<s>', eos_piece='</s>',
        train_extremely_large_corpus=True,
        shuffle_input_sentence=False,
        split_by_unicode_script=True,
        byte_fallback=True,
        num_threads=4,
    )
    print(f"  Trained in {time.time()-t0:.1f}s")
    print(f"  Model: {args.output}.model")
    print(f"  Vocab: {args.output}.vocab")

    # -- 5. Load and test --
    sp = spm.SentencePieceProcessor()
    sp.load(args.output + '.model')
    print(f"\n  Vocab size: {sp.vocab_size()}")

    # Test on sample words
    test_words = ['приносили', 'доходный', 'загородный', 'перестройка',
                  'безвозмездный', 'водопроводчик', 'антиконституционный']
    print(f"\n  Sample encodings:")
    for w in test_words:
        ids = sp.encode(w)
        pieces = [sp.IdToPiece(i) for i in ids]
        print(f"    {w:25s} -> {' '.join(pieces)}")

    # -- 6. Coverage stats --
    n_covered = sum(1 for w in words if sp.encode(w) and len(sp.encode(w)) <= 5)
    print(f"\n  Coverage (<=5 BPE tokens): {n_covered}/{len(words)} ({100*n_covered/len(words):.1f}%)")
    print(f"\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
