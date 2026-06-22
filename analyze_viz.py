"""Analyze PCA cluster structure of a checkpoint."""

import sys, json, os, numpy as np
from sklearn.decomposition import PCA
from collections import Counter

def analyze(prefix):
    npz_file = prefix + ".codes.npz"
    json_file = prefix + ".json"
    f = np.load(npz_file)
    codes, cids, basis = f["codes"], f["cids"], f["basis"]

    with open(json_file) as fp:
        meta = json.load(fp)

    # subsample 10000
    n = min(10000, len(cids))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(cids), n, replace=False)
    vecs = codes[idx] @ basis
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    vecs /= norms

    pca = PCA(2, random_state=0).fit(vecs)
    xy = pca.transform(vecs)

    print(f"=== {os.path.basename(prefix)} ===")
    print(f"  Tokens sampled: {n} / {len(cids)}")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f}  PC2: {pca.explained_variance_ratio_[1]:.4f}")
    print(f"  sum first 2: {sum(pca.explained_variance_ratio_[:2]):.4f}")

    # KMeans-like: find dense clusters by scanning PC1 slices
    x = xy[:, 0]
    y = xy[:, 1]
    x_bins = 20
    y_bins = 20
    x_edges = np.linspace(x.min(), x.max(), x_bins + 1)
    y_edges = np.linspace(y.min(), y.max(), y_bins + 1)
    density_grid = np.zeros((y_bins, x_bins))
    cid_grid = [[[] for _ in range(x_bins)] for _ in range(y_bins)]
    for i in range(n):
        xi = np.searchsorted(x_edges[1:], x[i], side='left')
        yi = np.searchsorted(y_edges[1:], y[i], side='left')
        xi = min(xi, x_bins - 1)
        yi = min(yi, y_bins - 1)
        density_grid[yi, xi] += 1
        cid_grid[yi][xi].append(int(cids[idx[i]]))

    # Find top-5 densest cells
    flat_idx = np.argsort(-density_grid.ravel())[:5]
    print(f"\n  Top-5 densest 2D cells (% of plotted points):")
    for fi in flat_idx:
        yi_b, xi_b = divmod(fi, x_bins)
        count = int(density_grid[yi_b, xi_b])
        pct = count / n * 100
        cids_in_cell = cid_grid[yi_b][xi_b]
        print(f"    cell [{yi_b},{xi_b}]: {count} pts ({pct:.1f}%) — e.g. CID {cids_in_cell[:3]}")

    # PCA loadings: top-10 contributing original features
    loadings_pc1 = pca.components_[0]
    top_feats = np.argsort(-np.abs(loadings_pc1))[:10]
    # Project back: these are latent dims that most influence PC1
    basis_load = basis[:, top_feats]  # (latent_dim, 10) → avg vector direction
    print(f"\n  Top-10 latent dims for PC1: {top_feats.tolist()}")
    print(f"  Mean PC1 coord: {x.mean():.4f} ± {x.std():.4f}")
    print(f"  Mean PC2 coord: {y.mean():.4f} ± {y.std():.4f}")
    print(f"  x range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")

    # Check if PC1 captures frequency vs semantics
    print(f"\n  Interpretation:")
    if pca.explained_variance_ratio_[0] > 0.05:
        print(f"    PC1={pca.explained_variance_ratio_[0]:.1%} -> strong semantic/variance axis forming")
        print(f"    Uniform sphere hypothesis REJECTED - structure emerging")
    else:
        print(f"    PC1={pca.explained_variance_ratio_[0]:.1%} -> near-uniform sphere")
        print(f"    No significant structure yet")

if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else r"real_data\concept_space_70k"
    analyze(prefix)
