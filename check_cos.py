"""Check pairwise cosine stats and nearest-neighbor sanity."""

import sys, os, numpy as np

def check(prefix):
    f = np.load(prefix + ".codes.npz")
    codes, cids, basis = f["codes"], f["cids"], f["basis"]

    # Random sample for pairwise stats
    n = min(5000, len(cids))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(cids), n, replace=False)
    vecs = codes[idx] @ basis
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    vecs /= norms

    # Pairwise cos: 5000 x 5000 matrix
    dots = vecs @ vecs.T
    triu = np.triu_indices(n, k=1)
    all_cos = dots[triu]
    mu = float(np.mean(all_cos))
    std = float(np.std(all_cos))
    max_cos = float(np.max(all_cos))
    min_cos = float(np.min(all_cos))
    frac_gt_05 = float(np.mean(all_cos > 0.5)) * 100

    # Top-1 NN for each sampled point
    top1_sims = []
    for i in range(min(1000, n)):
        row = dots[i]
        row[i] = -2  # exclude self
        top1 = float(np.max(row))
        top1_sims.append(top1)
    top1_mu = float(np.mean(top1_sims))
    top1_std = float(np.std(top1_sims))

    name = os.path.basename(prefix)
    print(f"{name}: N={n}")
    print(f"  pairwise cos:  mean={mu:.4f}  std={std:.4f}  max={max_cos:.4f}  min={min_cos:.4f}")
    print(f"  fraction >0.5: {frac_gt_05:.1f}%")
    print(f"  top-1 NN sim:  mean={top1_mu:.4f}  std={top1_std:.4f}")
    print(f"  expected max NN for random {n}x768: ~{np.sqrt(2*np.log(n)/768):.3f}")
    if mu > 0.02:
        print(f"  WARNING: cos_mean={mu:.4f} > 0.02 -> partial collapse!")
    if mu > 0.08:
        print(f"  COLLAPSE: cos_mean={mu:.4f} exceeds collapse guard threshold!")
    if top1_mu > 0.3:
        print(f"  NOTE: top-1 NN sim={top1_mu:.3f} > 0.3 -- vectors are very similar")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        check(p)
