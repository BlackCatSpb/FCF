"""Nearest neighbors: seen-only vs all vs random baseline."""

import sys, os, json, numpy as np

def check(prefix):
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

    # All vectors
    vecs = codes @ basis
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    vecs /= norms

    # Seen token mask (usage > 0)
    seen_mask = np.array([usage.get(str(int(c)), usage.get(int(c), 0)) > 0 for c in cids])
    seen_cids = cids[seen_mask]
    seen_vecs = vecs[seen_mask]
    n_seen = len(seen_cids)
    print(f"=== {os.path.basename(prefix)} ===")
    print(f"  Total CIDs: {len(cids)}  Seen: {n_seen}  Unseen: {len(cids) - n_seen}")

    # For each SEEN token, find NN among ALL tokens vs among SEEN-only
    rng = np.random.RandomState(42)
    sample = rng.choice(n_seen, min(200, n_seen), replace=False)

    from sklearn.neighbors import NearestNeighbors

    # NN among all (full brute)
    nn_all = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    nn_all.fit(vecs)
    top1_all = []
    for i in sample:
        q = seen_vecs[i:i+1]
        dists, nidxs = nn_all.kneighbors(q)
        # first is self, skip it
        for d, ni in zip(dists[0][1:], nidxs[0][1:]):
            sim = 1 - d
            top1_all.append(sim)
            break

    # NN among seen only
    nn_seen = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    nn_seen.fit(seen_vecs)
    top1_seen = []
    for i in sample:
        q = seen_vecs[i:i+1]
        dists, nidxs = nn_seen.kneighbors(q)
        for d, ni in zip(dists[0][1:], nidxs[0][1:]):
            sim = 1 - d
            top1_seen.append(sim)
            break

    print(f"  Top-1 NN (all CIDs):    mean={np.mean(top1_all):.4f}  std={np.std(top1_all):.4f}")
    print(f"  Top-1 NN (seen only):   mean={np.mean(top1_seen):.4f}  std={np.std(top1_seen):.4f}")
    random_top1 = np.sqrt(2 * np.log(len(seen_cids)) / codes.shape[1])
    print(f"  Random baseline (seen): ~{random_top1:.3f}")

    # Show some example NN pairs among seen tokens
    print("\n  Sample NN pairs (seen-only):")
    for i in sample[:15]:
        q = seen_vecs[i:i+1]
        dists, nidxs = nn_seen.kneighbors(q)
        q_cid = int(seen_cids[i])
        try:
            q_name = sp.IdToPiece(q_cid).replace("\u2581", " ").strip()
        except:
            q_name = f"[{q_cid}]"
        line = f"    {q_name:20s} ->"
        for d, ni in zip(dists[0][1:], nidxs[0][1:]):
            ni_cid = int(seen_cids[ni])
            try:
                ni_name = sp.IdToPiece(ni_cid).replace("\u2581", " ").strip()
            except:
                ni_name = f"[{ni_cid}]"
            sim = 1 - d
            line += f"  {ni_name:15s}({sim:.3f})"
        print(line)

    # Check whether seen tokens have structure: PCA on seen-only
    from sklearn.decomposition import PCA
    pca = PCA(2, random_state=0).fit(seen_vecs)
    print(f"\n  PCA on seen-only: PC1={pca.explained_variance_ratio_[0]:.4f} PC2={pca.explained_variance_ratio_[1]:.4f}")

if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else r"real_data\concept_space_75k"
    check(prefix)
