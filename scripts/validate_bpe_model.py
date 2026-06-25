"""
Validate a trained BPE model: coverage, sample encodings, e5 alignment.

Usage:
    python scripts/validate_bpe_model.py real_data/bpe_morph.model
    python scripts/validate_bpe_model.py real_data/bpe_morph.model --sample-words 500 --e5
"""

import argparse
import os
import sys
import random
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eva.morph import vocab_coverage, _has_cyrillic, validate_alignment, SEP


def main():
    parser = argparse.ArgumentParser(description='Validate BPE model')
    parser.add_argument('model', help='Path to .model file')
    parser.add_argument('--corpus', default='real_data/full_corpus_ru_morph.txt',
                        help='Corpus for word extraction')
    parser.add_argument('--sample-words', type=int, default=500,
                        help='Words for coverage test')
    parser.add_argument('--e5', action='store_true',
                        help='Run e5 alignment validation (loads ~1 GB model)')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(args.model)
    print(f"Vocab size: {sp.vocab_size()}")
    size_mb = os.path.getsize(args.model) / 1e6
    print(f"Model:      {args.model} ({size_mb:.1f} MB)")

    SP_WS = '\u2581'

    # Sample encodings
    test_words = ['приносили', 'доходный', 'загородный', 'перестройка',
                  'безвозмездный', 'водопроводчик', 'антиконституционный',
                  'природа', 'выходил', 'подводный', 'приносить']

    print("\n--- Sample encodings ---")
    for w in test_words:
        ids = sp.encode(w)
        pieces_str = ' '.join(sp.IdToPiece(i).replace(SP_WS, '_') for i in ids)
        print(f"  {w:25s} -> {pieces_str}  [{len(ids)} tok]")

    # Load unique words from corpus
    print(f"\nLoading words from corpus...")
    t0 = time.time()
    words = set()
    with open(args.corpus, 'r', encoding='utf-8') as f:
        for line in f:
            for w in line.strip().split():
                wc = w.replace(SEP, '').strip().lower()
                if len(wc) >= 3 and _has_cyrillic(wc):
                    words.add(wc)
            if len(words) >= args.sample_words * 4:
                break
    words = sorted(words)[:args.sample_words]
    print(f"  {len(words)} unique words in {time.time()-t0:.1f}s")

    # Coverage
    cov = vocab_coverage(words, sp)
    print("\n--- Coverage ---")
    for k in ['unique_words', 'mean_tokens', 'median_tokens', 'max_tokens']:
        v = cov.get(k, 0)
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.2f}")
        else:
            print(f"  {k:20s}: {v}")
    for k in ['1_token', '2_tokens', '3_tokens', '5_tokens', '10_tokens']:
        v = cov.get(k, 0.0) * 100
        print(f"  {k:20s}: {v:.1f}%")

    # e5 alignment
    if args.e5:
        print("\n--- e5 Alignment Validation ---")
        t0 = time.time()
        align = validate_alignment(
            words, sp,
            sample_size=min(500, len(words)),
            device=args.device,
        )
        print(f"  Computed in {time.time()-t0:.1f}s")
        if 'morph_mean' in align:
            print(f"  Morph bundle cos:  mean={align['morph_mean']:.4f} std={align['morph_std']:.4f}")
        if 'bpe_mean' in align:
            print(f"  BPE bundle cos:    mean={align['bpe_mean']:.4f} std={align['bpe_std']:.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
