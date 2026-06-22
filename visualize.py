"""Standalone checkpoint visualizer — reads .codes.npz, PCA + scatter."""

import argparse
import json
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_checkpoint(path_prefix):
    """Load concept_space_{path_prefix}.codes.npz and matching JSON."""
    base = os.path.splitext(path_prefix)[0]  # strip any .json/.npz
    npz_file = base + ".codes.npz"
    json_file = base + ".json"

    npz = np.load(npz_file)
    codes = npz["codes"]          # (N, latent_dim)
    cids = npz["cids"]            # (N,)
    basis = npz["basis"]          # (latent_dim, dim)

    with open(json_file) as f:
        meta = json.load(f)
    vocab_size = meta.get("vocab_size", len(cids))

    # Project to vector space: (N, latent_dim) @ (latent_dim, dim) → (N, dim)
    vecs = codes @ basis
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    vecs /= norms  # unit sphere

    return vecs, cids, vocab_size


def main():
    ap = argparse.ArgumentParser(description="FCF Checkpoint Visualizer")
    ap.add_argument("checkpoint", nargs="?",
                    default=r"real_data\concept_space_40k",
                    help="Path prefix (without .codes.npz/.json)")
    ap.add_argument("--n-tokens", type=int, default=5000,
                    help="Number of tokens to plot (default 5000)")
    ap.add_argument("--labels", type=int, default=30,
                    help="Number of label annotations (default 30)")
    ap.add_argument("--sp-model",
                    default=r"real_data\bpe_ru_146k.model",
                    help="SentencePiece model for token labels")
    ap.add_argument("--html", action="store_true",
                    help="Also generate interactive HTML via plotly")
    ap.add_argument("--output", "-o", default=None,
                    help="Save figure to file (shows window if omitted)")
    args = ap.parse_args()

    print(f"Loading {args.checkpoint} …")
    vecs, cids, vocab_size = load_checkpoint(args.checkpoint)
    n = min(args.n_tokens, len(vecs))
    print(f"  {n} / {len(vecs)} vectors, {vocab_size} vocab")

    # Sample evenly
    rng = np.random.RandomState(42)
    idx = rng.choice(len(vecs), n, replace=False)
    vecs_sample = vecs[idx]
    cids_sample = cids[idx]

    print("PCA 768D -> 2D ..")
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(vecs_sample)
    var = pca.explained_variance_ratio_
    print(f"  Explained variance: PC1={var[0]:.3f} PC2={var[1]:.3f}")

    # Color by distance from origin (or field density)
    colors = np.linalg.norm(xy, axis=1)

    # Token labels
    if args.sp_model and args.labels > 0:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=args.sp_model)
        # Pick tokens at extremes of PC1/PC2
        order_pc1 = np.argsort(xy[:, 0])
        order_pc2 = np.argsort(xy[:, 1])
        label_idx = np.unique(np.concatenate([
            order_pc1[:args.labels // 2],
            order_pc1[-args.labels // 2:],
            order_pc2[:args.labels // 2],
            order_pc2[-args.labels // 2:],
        ]))
    else:
        sp = None
        label_idx = []

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(14, 10))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap="viridis",
                    s=4, alpha=0.6, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("Distance from origin (2D)")

    for i in label_idx:
        cid = int(cids_sample[i])
        if sp:
            try:
                label = sp.IdToPiece(cid)
            except Exception:
                label = f"[{cid}]"
        else:
            label = str(cid)
        label = label.replace("▁", "").strip()
        if len(label) > 20:
            label = label[:18] + "…"
        ax.annotate(label, xy[i], fontsize=5, alpha=0.8,
                    arrowprops=dict(arrowstyle="->", lw=0.3, alpha=0.4))

    ax.set_title(f"FCF — {n} tokens (PCA {var[0]:.1%} / {var[1]:.1%})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.15)

    if args.html:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            df_labels = {i: (
                sp.IdToPiece(int(cids_sample[i])).replace("▁", "").strip()
                if sp else str(int(cids_sample[i])))
                for i in label_idx}

            fig_ply = go.Figure(data=go.Scattergl(
                x=xy[:, 0], y=xy[:, 1],
                mode="markers+text" if len(label_idx) > 0 else "markers",
                marker=dict(size=2, color=colors, colorscale="Viridis",
                            showscale=True),
                text=[df_labels.get(i, "") for i in range(len(xy))],
                textposition="top center",
                hovertext=[f"CID {int(cids_sample[i])}"
                           for i in range(len(xy))],
            ))
            fig_ply.update_layout(title=f"FCF — {n} tokens (interactive)",
                                  width=1200, height=800)
            html_path = os.path.splitext(args.checkpoint)[0] + "_viz.html"
            fig_ply.write_html(html_path)
            print(f"  HTML saved: {html_path}")
        except ImportError:
            print("  plotly not installed, skipping HTML")

    if args.output:
        plt.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"  Saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
