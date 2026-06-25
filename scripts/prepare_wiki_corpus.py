"""
Download, clean, and prepare Wikipedia-RuDataset for FCF training.

Outputs:
  real_data/full_corpus_ru_clean.txt    — cleaned text (one sentence per line)
  real_data/full_corpus_ru_morph.txt    — with morpheme separators (for BPE training)
  real_data/val_corpus.txt              — validation split
  real_data/wiki_prep_report.txt        — preparation report

Usage:
    # Download + prepare
    python scripts/prepare_wiki_corpus.py --download

    # With morph annotations (for morph-aware BPE training)
    python scripts/prepare_wiki_corpus.py --download --morph-annotate

    # Use existing parquet files
    python scripts/prepare_wiki_corpus.py --wiki-parquet path/to/wikipedia.parquet

    # Full pipeline
    python scripts/prepare_wiki_corpus.py --download --out-dir real_data \
        --max-lines 100000 --val-split 0.001 --min-chars 20 --max-chars 20000
"""

import argparse
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eva.morph import annotate_corpus_line, _clean_line, _has_cyrillic


def load_wikipedia_parquet(parquet_path: str, max_lines: int = 0):
    """Load text from Wikipedia-RuDataset Parquet. Auto-detect text columns."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in ['system', 'user', 'assistant']:
            if col in df.columns and isinstance(row[col], str) and len(row[col]) > 5:
                parts.append(row[col].strip())
        text = ' '.join(parts) if parts else ''
        for col in ['text', 'content', 'article', 'sentence', 'page_text', 'body']:
            if col in df.columns and isinstance(row[col], str) and len(row[col]) > 5:
                text = row[col].strip()
                break
        if text:
            texts.append(text)
        if max_lines and len(texts) >= max_lines:
            break
    return texts


def download_wikipedia(dest_dir: str):
    """Download Wikipedia-RuDataset from Hugging Face to Parquet."""
    from datasets import load_dataset
    print("  Downloading aiplatforms/wikipedia-RuDataset from HuggingFace...")
    t0 = time.time()
    ds = load_dataset("aiplatforms/wikipedia-RuDataset", split="train")
    print(f"  Downloaded {len(ds)} rows in {time.time() - t0:.1f}s")
    parquet_dir = Path(dest_dir) / "wiki_download"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = str(parquet_dir / "wikipedia_rudataset.parquet")
    ds.to_parquet(parquet_path)
    print(f"  Saved to {parquet_path}")
    return parquet_path


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (Russian-aware)."""
    if len(text) < 10:
        return [text] if text.strip() else []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) >= 10]


def main():
    parser = argparse.ArgumentParser(description='Prepare Wikipedia corpus for FCF training')
    parser.add_argument('--download', action='store_true',
                        help='Download datasets from HuggingFace')
    parser.add_argument('--wiki-parquet', default=None,
                        help='Path to existing Wikipedia Parquet file')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: real_data)')
    parser.add_argument('--max-lines', type=int, default=0,
                        help='Max lines to process (0 = all)')
    parser.add_argument('--val-split', type=float, default=0.001,
                        help='Fraction for validation set')
    parser.add_argument('--min-chars', type=int, default=15,
                        help='Minimum characters per line')
    parser.add_argument('--max-chars', type=int, default=20000,
                        help='Maximum characters per line')
    parser.add_argument('--morph-annotate', action='store_true',
                        help='Also produce morph-annotated corpus for BPE training')
    args = parser.parse_args()

    base_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / 'real_data'
    base_dir.mkdir(parents=True, exist_ok=True)

    clean_path = base_dir / 'full_corpus_ru_clean.txt'
    morph_path = base_dir / 'full_corpus_ru_morph.txt'
    val_path = base_dir / 'val_corpus.txt'
    report_path = base_dir / 'wiki_prep_report.txt'
    report_lines = []

    all_texts = []

    # 1. Load Wikipedia
    wiki_path = args.wiki_parquet
    if wiki_path is None and args.download:
        wiki_path = download_wikipedia(str(base_dir))
    if wiki_path and os.path.exists(wiki_path):
        print(f"Loading Wikipedia from {wiki_path}...")
        t0 = time.time()
        texts = load_wikipedia_parquet(wiki_path, args.max_lines)
        print(f"  {len(texts)} documents in {time.time() - t0:.1f}s")
        all_texts.extend(texts)
        report_lines.append(f"Wikipedia: {len(texts)} documents")
    else:
        print("No Wikipedia data. Use --download or --wiki-parquet.")

    if not all_texts:
        print("No data loaded. Exiting.")
        return 1

    print(f"\nTotal documents: {len(all_texts)}")

    # 2. Split into sentences
    print("Splitting into sentences...")
    t0 = time.time()
    sentences = []
    for doc in all_texts:
        if isinstance(doc, str):
            sentences.extend(split_sentences(doc))
        elif isinstance(doc, dict):
            for v in doc.values():
                if isinstance(v, str) and len(v) > 10:
                    sentences.extend(split_sentences(v))
    print(f"  {len(sentences)} sentences in {time.time() - t0:.1f}s")
    report_lines.append(f"Sentences: {len(sentences)}")

    # 3. Clean and filter
    print("Cleaning and filtering...")
    t0 = time.time()
    stats = Counter()
    kept = []

    for s in sentences:
        s = _clean_line(s)
        if not s:
            stats['empty'] += 1
            continue
        if len(s) < args.min_chars:
            stats['too_short'] += 1
            continue
        if len(s) > args.max_chars:
            stats['too_long'] += 1
            continue
        if not _has_cyrillic(s):
            stats['no_cyrillic'] += 1
            continue
        kept.append(s)

    print(f"  Kept: {len(kept)} | Filtered: {sum(stats.values())} in {time.time() - t0:.1f}s")
    report_lines.append(f"After cleaning: {len(kept)} ({sum(stats.values())} filtered)")

    # 4. Shuffle and split train/val
    import random
    rng = random.Random(42)
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_split))
    val_lines = kept[:n_val]
    train_lines = kept[n_val:]
    print(f"  Train: {len(train_lines)} | Val: {len(val_lines)}")
    report_lines.append(f"Train: {len(train_lines)}")
    report_lines.append(f"Val: {len(val_lines)}")

    # 5. Write clean corpus (no morph annotations)
    print(f"Writing clean corpus to {clean_path}...")
    t0 = time.time()
    with open(clean_path, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')
    print(f"  Done in {time.time() - t0:.1f}s")

    # 6. Optionally write morph-annotated corpus
    if args.morph_annotate:
        print(f"Writing morph-annotated corpus to {morph_path}...")
        t0 = time.time()
        with open(morph_path, 'w', encoding='utf-8') as f:
            for line in train_lines:
                f.write(annotate_corpus_line(line) + '\n')
        print(f"  Done in {time.time() - t0:.1f}s")

    # 7. Write validation corpus
    print(f"Writing validation corpus to {val_path}...")
    with open(val_path, 'w', encoding='utf-8') as f:
        for line in val_lines:
            f.write(line + '\n')

    # 8. Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("WIKIPEDIA CORPUS PREPARATION REPORT\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Args: {vars(args)}\n\n")
        for line in report_lines:
            f.write(f"  {line}\n")
        f.write("\nFilter stats:\n")
        for k, v in sorted(stats.items(), key=lambda x: -x[1]):
            f.write(f"  {k}: {v}\n")

    # File sizes
    clean_size = os.path.getsize(clean_path) / (1024 * 1024)
    print(f"\nDone! Report: {report_path}")
    print(f"  {clean_path}  ({clean_size:.1f} MB, {len(train_lines)} lines)")
    print(f"  {val_path}  ({os.path.getsize(val_path) / (1024 * 1024):.1f} MB, {len(val_lines)} lines)")
    if args.morph_annotate:
        print(f"  {morph_path}  ({os.path.getsize(morph_path) / (1024 * 1024):.1f} MB)")
    print(f"\nTo train:")
    print(f"  python train_full.py --fresh --learned-fields")
    print(f"\nTo train morph-aware BPE:")
    print(f"  python scripts/train_morph_bpe.py --corpus {morph_path} --output {base_dir}/bpe_morph --vocab-size 256000")
    print(f"  python train_full.py --fresh --learned-fields --morph-bpe {base_dir}/bpe_morph.model")
    print(f"\nTo seed rare concepts with e5:")
    print(f"  python train_full.py --fresh --learned-fields --seed-e5 --e5-morph-bundle")
    return 0


if __name__ == '__main__':
    sys.exit(main())
