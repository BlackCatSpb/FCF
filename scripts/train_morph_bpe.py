"""
Train a morpheme-aware SentencePiece BPE model for FCF.

Two modes:
  1. Streaming mode (default):  read corpus line-by-line, annotate morphemes
     on-the-fly via eva.morph, write augmented text, train BPE.
  2. Pretokenized mode:         corpus already contains \u037E morpheme
     separators (produced by prepare_wiki_corpus.py --morph-annotate).
     Just train BPE directly.

Memory-efficient: O(max_line_length), never loads entire corpus.

Usage:
    # Streaming mode (annotate on-the-fly)
    python scripts/train_morph_bpe.py --corpus real_data/full_corpus_ru_clean.txt \
        --output real_data/bpe_morph --vocab-size 256000

    # Pretokenized mode (corpus already has \u037E)
    python scripts/train_morph_bpe.py --corpus real_data/full_corpus_ru_morph.txt \
        --output real_data/bpe_morph --vocab-size 256000 --pretokenized

    # Full pipeline with validation
    python scripts/train_morph_bpe.py --corpus real_data/full_corpus_ru_morph.txt \
        --output real_data/bpe_morph --vocab-size 256000 --pretokenized \
        --validate --sample-words 1000

    # Quick test run
    python scripts/train_morph_bpe.py --corpus real_data/sample.txt \
        --output real_data/bpe_morph_test --vocab-size 16000 --max-lines 10000
"""

import argparse
import os
import sys
import time
import math
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eva.morph import SEP, annotate_corpus_line, _has_cyrillic, vocab_coverage


# ── Streaming corpus annotation ───────────────────────────────────


def annotate_corpus_stream(
    input_path: str,
    output_path: str,
    max_lines: int = 0,
    progress_interval: int = 100000,
) -> dict:
    """Read corpus line-by-line, annotate morphemes, write augmented.

    Returns stats dict.
    """
    stats = Counter()
    t0 = time.time()
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            annotated = annotate_corpus_line(line)
            fout.write(annotated + '\n')
            stats['lines'] += 1
            if (i + 1) % progress_interval == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1:>9}] {rate:.0f} L/s", flush=True)
    elapsed = time.time() - t0
    stats['elapsed_s'] = elapsed
    stats['lines_per_sec'] = stats['lines'] / elapsed if elapsed > 0 else 0
    return dict(stats)


# ── BPE training ──────────────────────────────────────────────────


def train_bpe(
    input_path: str,
    output_prefix: str,
    vocab_size: int,
    num_threads: int = 4,
) -> dict:
    """Train SentencePiece BPE model on the augmented corpus.

    Returns model metadata dict.
    """
    import sentencepiece as spm

    t0 = time.time()
    print(f"  Input:  {input_path}", flush=True)
    print(f"  Output: {output_prefix}.model", flush=True)
    print(f"  Vocab:  {vocab_size}", flush=True)

    # pad_id=-1: no padding piece (prevents conflict with unk_id=0)
    # hard_vocab_limit=False: allows slightly smaller final vocab when
    #   byte-fallback tokens consume some slots (common with byte_fallback=True)
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        pad_id=-1,
        unk_id=0, bos_id=1, eos_id=2,
        unk_piece='<unk>', bos_piece='<s>', eos_piece='</s>',
        train_extremely_large_corpus=True,
        shuffle_input_sentence=False,
        split_by_unicode_script=True,
        byte_fallback=True,
        hard_vocab_limit=False,
        num_threads=num_threads,
    )
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s", flush=True)
    return {'elapsed_s': elapsed, 'vocab_size': vocab_size}


# ── Validation ─────────────────────────────────────────────────────


def run_validation(
    sp,
    corpus_path: str,
    sample_words: int = 500,
    validate_e5: bool = True,
    device: str = 'cpu',
) -> dict:
    """Validate the trained BPE model: coverage, sample encoding, e5 alignment."""
    from eva.morph import validate_alignment

    results = {}

    # Load unique words from corpus (sample only)
    print("  Loading unique words from corpus...", flush=True)
    words = set()
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            for w in line.strip().split():
                w_clean = w.replace(SEP, '').strip().lower()
                if len(w_clean) >= 3 and _has_cyrillic(w_clean):
                    words.add(w_clean)
                    if len(words) >= sample_words * 3:
                        break
            if len(words) >= sample_words * 3:
                break
    words = sorted(words)[:sample_words]
    print(f"  Unique words: {len(words)}", flush=True)

    # Coverage
    cover = vocab_coverage(words, sp)
    results['coverage'] = cover
    print(f"  1-token:  {cover['1_token']*100:.1f}%", flush=True)
    print(f"  2-tokens: {cover['2_tokens']*100:.1f}%", flush=True)
    print(f"  5-tokens: {cover['5_tokens']*100:.1f}%", flush=True)
    print(f"  Mean tokens/word: {cover['mean_tokens']:.2f}", flush=True)
    print(f"  Median tokens/word: {cover['median_tokens']:.2f}", flush=True)

    # Sample encodings (replace SentencePiece whitespace marker for terminal)
    SP_WS = '\u2581'
    test_words = ['приносили', 'доходный', 'загородный', 'перестройка',
                  'безвозмездный', 'водопроводчик', 'антиконституционный',
                  'природа', 'выходил', 'подводный']
    print(f"\n  Sample encodings:", flush=True)
    for w in test_words:
        ids = sp.encode(w)
        pieces = [sp.IdToPiece(i).replace(SP_WS, '_') for i in ids]
        pieces_str = ' '.join(pieces)
        print(f"    {w:30s} -> {pieces_str:60s}  [{len(ids)} tok]", flush=True)

    # e5 alignment validation
    if validate_e5:
        print(f"\n  Validating e5 alignment ({sample_words} words)...", flush=True)
        try:
            align = validate_alignment(
                words[:sample_words], sp,
                device=device, sample_size=min(500, sample_words),
            )
            results['alignment'] = align
            if 'morph_mean' in align:
                print(f"  Morph bundle cos:  mean={align['morph_mean']:.4f} std={align['morph_std']:.4f}",
                      flush=True)
            if 'bpe_mean' in align:
                print(f"  BPE bundle cos:    mean={align['bpe_mean']:.4f} std={align['bpe_std']:.4f}",
                      flush=True)
        except Exception as e:
            print(f"  e5 validation skipped: {e}", flush=True)

    return results


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Train morph-aware BPE for FCF')
    parser.add_argument('--corpus', required=True,
                        help='Path to Russian corpus')
    parser.add_argument('--output', required=True,
                        help='Output prefix for BPE model')
    parser.add_argument('--vocab-size', type=int, default=256000,
                        help='BPE vocabulary size (default: 256000)')
    parser.add_argument('--pretokenized', action='store_true',
                        help='Corpus already has \\u037E morph separators (skip annotation)')
    parser.add_argument('--max-lines', type=int, default=0,
                        help='Max lines to process (0=all)')
    parser.add_argument('--validate', action='store_true',
                        help='Run post-training validation (coverage + e5)')
    parser.add_argument('--sample-words', type=int, default=500,
                        help='Sample words for validation')
    parser.add_argument('--no-e5', action='store_true',
                        help='Skip e5 validation (saves ~1 GB RAM)')
    parser.add_argument('--device', default='cpu',
                        help='Torch device for e5 (cpu/cuda)')
    parser.add_argument('--threads', type=int, default=4,
                        help='SentencePiece training threads')
    parser.add_argument('--keep-augmented', action='store_true',
                        help='Keep the augmented text file after training')
    args = parser.parse_args()

    t_start = time.time()
    aug_path = args.output + '_augmented.txt'

    # ── 1. Prepare augmented corpus ──────────────────────────────
    if args.pretokenized:
        print("[Mode] Pretokenized — using corpus as-is (already has morpheme separators)",
              flush=True)
        aug_path = args.corpus
        stats = {'lines': 0, 'method': 'pretokenized'}
        if args.max_lines:
            # Create subset for quick testing
            subset_path = args.output + '_subset.txt'
            print(f"  Subsetting to {args.max_lines} lines -> {subset_path}")
            with open(args.corpus, 'r', encoding='utf-8') as fin, \
                 open(subset_path, 'w', encoding='utf-8') as fout:
                for i, line in enumerate(fin):
                    if i >= args.max_lines:
                        break
                    fout.write(line)
            aug_path = subset_path
            stats = {'lines': args.max_lines, 'method': 'pretokenized_subset'}
        print(f"  Corpus: {aug_path}", flush=True)
    else:
        print("[Mode] Streaming — annotating morphemes on-the-fly", flush=True)
        if not os.path.exists(args.corpus):
            print(f"ERROR: Corpus not found: {args.corpus}", file=sys.stderr)
            return 1
        stats = annotate_corpus_stream(
            args.corpus, aug_path,
            max_lines=args.max_lines,
        )
        stats['method'] = 'streaming'
        print(f"  Annotated {stats['lines']} lines in {stats['elapsed_s']:.1f}s "
              f"({stats['lines_per_sec']:.0f} L/s)",
              flush=True)

    # ── 2. Train BPE ─────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"Training SentencePiece (vocab_size={args.vocab_size})...", flush=True)
    bpe_stats = train_bpe(
        aug_path, args.output,
        vocab_size=args.vocab_size,
        num_threads=args.threads,
    )

    # ── 3. Load and validate ─────────────────────────────────────
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(args.output + '.model')
    print(f"\n  Vocab size: {sp.vocab_size()}", flush=True)
    print(f"  Model: {args.output}.model", flush=True)
    print(f"  Vocab: {args.output}.vocab", flush=True)

    if args.validate:
        print(f"\n{'='*60}", flush=True)
        print("Validation...", flush=True)
        val_results = run_validation(
            sp, args.corpus,
            sample_words=args.sample_words,
            validate_e5=not args.no_e5,
            device=args.device,
        )
    else:
        val_results = {}

    # ── 4. Cleanup ───────────────────────────────────────────────
    if not args.keep_augmented and not args.pretokenized:
        if os.path.exists(aug_path):
            os.remove(aug_path)
            print(f"\n  Removed augmented temp file: {aug_path}", flush=True)

    # ── 5. Summary report ────────────────────────────────────────
    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"MORPH BPE TRAINING COMPLETE", flush=True)
    print(f"  Total time:  {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)", flush=True)
    print(f"  Model:       {args.output}.model ({os.path.getsize(args.output+'.model')/1e6:.1f} MB)",
          flush=True)
    print(f"  Vocab size:  {sp.vocab_size()}", flush=True)
    print(f"  Corpus:      {args.corpus}", flush=True)
    if val_results:
        cov = val_results.get('coverage', {})
        if cov:
            print(f"  Coverage:    1-tok={cov['1_token']*100:.1f}%, "
                  f"5-tok={cov['5_tokens']*100:.1f}%, "
                  f"mean={cov['mean_tokens']:.2f} tok/word", flush=True)
    print(f"\nTo use with FCF:")
    print(f"  python train_full.py --fresh --learned-fields --morph-bpe {args.output}.model "
          f"--seed-e5 --e5-morph-bundle", flush=True)
    print(f"{'='*60}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
