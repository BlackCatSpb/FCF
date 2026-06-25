"""Check PC1 after removing freq=0 outliers."""

import sys, os, json, numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

def analyze(prefix):
    npz_file = prefix + ".codes.npz"
    json_file = prefix + ".json"
    sp_model = os.path.join(os.path.dirname(prefix), "bpe_ru_146k.model")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=sp_model)

    f = np.load(npz_file)
    codes, cids, basis = f["codes"], f["cids"], f["basis"]
    with open(json_file) as fp:
        meta = json.load(fp)
    usage = meta.get("concept_usage", {})

    rng = np.random.RandomState(42)
    n = min(15000, len(cids))
    samp_idx = rng.choice(len(cids), n, replace=False)

    # Vectors
    vecs = codes[samp_idx] @ basis
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    vecs /= norms
    samp_cids = cids[samp_idx]

    # Metadata per sampled token
    freqs = np.array([usage.get(str(int(c)), usage.get(int(c), 0)) for c in samp_cids], dtype=float)
    is_subword = np.array([not sp.IdToPiece(int(c)).startswith("\u2581") for c in samp_cids], dtype=float)

    seen = freqs > 0
    unseen = freqs == 0
    name = os.path.basename(prefix)
    print(f"=== {name} ===")
    print(f"  Sampled: {n} total, {int(seen.sum())} seen, {int(unseen.sum())} freq=0")

    for label, mask in [("ALL", slice(None)), ("SEEN", seen), ("UNSEEN", unseen)]:
        v = vecs[mask] if isinstance(mask, np.ndarray) else vecs
        sc = samp_cids[mask] if isinstance(mask, np.ndarray) else samp_cids
        if len(v) < 50:
            continue
        pca = PCA(2, random_state=0).fit(v)
        xy = pca.transform(v)
        fs = freqs[mask] if isinstance(mask, np.ndarray) else freqs
        sw = is_subword[mask] if isinstance(mask, np.ndarray) else is_subword
        r1, _ = pearsonr(xy[:, 0], fs)
        r2, _ = pearsonr(xy[:, 0], sw)
        print(f"\n  [{label}] PC1={pca.explained_variance_ratio_[0]:.4f} PC2={pca.explained_variance_ratio_[1]:.4f}  N={len(v)}")
        print(f"    r(PC1,freq)={r1:.4f}  r(PC1,subword)={r2:.4f}")

        order = np.argsort(xy[:, 0])
        print(f"    Extreme PC-:")
        for i in order[:8]:
            cid = int(sc[i])
            t = sp.IdToPiece(cid).replace("\u2581", " ").strip()
            print(f"      {t:25s} PC1={xy[i,0]:+.3f} freq={int(fs[i])}")
        print(f"    Extreme PC+:")
        for i in order[-8:]:
            cid = int(sc[i])
            t = sp.IdToPiece(cid).replace("\u2581", " ").strip()
            print(f"      {t:25s} PC1={xy[i,0]:+.3f} freq={int(fs[i])}")

if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else r"real_data\concept_space_75k"
    analyze(prefix)
