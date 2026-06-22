"""Prune qwen_knowledge.npz: deduplicate + filter by count >= min_count.

Usage:
  python prune_qwen_knowledge.py
  python prune_qwen_knowledge.py --min-count 3 --input real_data/qwen_knowledge.npz --output real_data/qwen_knowledge.npz
"""

import os, sys
import numpy as np


def prune(input_path, output_path, min_count=3):
    print(f"Loading: {input_path}")
    data = np.load(input_path)
    rows = data['rows']; cols = data['cols']; vals = data['vals']; counts = data['counts']
    raw_n = len(rows)
    print(f"  Raw entries: {raw_n:,}")

    # Aggregate duplicate keys
    seen = {}  # key64 → (sum_cos_weighted, sum_counts)
    for i in range(raw_n):
        a, b = int(rows[i]), int(cols[i])
        key = (a << 32) | b
        c = int(counts[i])
        if key not in seen:
            seen[key] = [float(vals[i]) * c, c]
        else:
            seen[key][0] += float(vals[i]) * c
            seen[key][1] += c
    unique_n = len(seen)
    print(f"  Unique pairs: {unique_n:,}")

    # Filter by count
    pruned = {k: v for k, v in seen.items() if v[1] >= min_count}
    pruned_n = len(pruned)
    print(f"  After filter (count>={min_count}): {pruned_n:,} ({pruned_n/max(unique_n,1)*100:.1f}%)")

    # Build arrays
    out_rows = np.empty(pruned_n, dtype=np.uint32)
    out_cols = np.empty(pruned_n, dtype=np.uint32)
    out_vals = np.empty(pruned_n, dtype=np.float32)
    out_counts = np.empty(pruned_n, dtype=np.uint32)
    for i, (key, (sum_w, cnt)) in enumerate(pruned.items()):
        out_rows[i] = key >> 32
        out_cols[i] = key & 0xFFFFFFFF
        out_vals[i] = np.float32(sum_w / cnt)
        out_counts[i] = min(cnt, 4294967295)

    np.savez_compressed(output_path,
                        rows=out_rows, cols=out_cols,
                        vals=out_vals, counts=out_counts)
    print(f"Saved: {output_path} ({pruned_n:,} pairs)")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    default_input = os.path.join(base, 'real_data', 'qwen_knowledge.npz')
    import argparse
    p = argparse.ArgumentParser(description="Prune qwen_knowledge.npz")
    p.add_argument("--input", default=default_input)
    p.add_argument("--output", default=default_input)
    p.add_argument("--min-count", type=int, default=3)
    args = p.parse_args()
    prune(args.input, args.output, args.min_count)
