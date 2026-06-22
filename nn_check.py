"""Check nearest neighbors in 768D space for semantic coherence."""

import sys, json, os, numpy as np

def nn(prefix):
    npz_file = prefix + ".codes.npz"
    base = os.path.dirname(prefix)
    sp_model = os.path.join(base, "bpe_ru_146k.model") if base else "real_data/bpe_ru_146k.model"
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=sp_model)

    f = np.load(npz_file)
    codes, cids, basis = f["codes"], f["cids"], f["basis"]
    vecs = codes @ basis
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    vecs /= norms

    # Build index
    from sklearn.neighbors import NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute")
    nn_model.fit(vecs)

    seeds = ["любовь", "человек", "город", "собака", "большой", "хороший", "вода", "говорить"]
    print(f"=== {os.path.basename(prefix)} ===")
    for word in seeds:
        wid = sp.PieceToId("\u2581" + word)
        if wid < 0 or wid >= len(cids):
            # try without space prefix
            wid = sp.PieceToId(word)
        if wid < 0 or wid >= len(cids):
            print(f"\n  [{word}] NOT FOUND")
            continue
        idx = np.where(cids == wid)[0]
        if len(idx) == 0:
            print(f"\n  [{word}] CID {wid} not in checkpoint")
            continue
        idx = idx[0]
        dists, nidxs = nn_model.kneighbors(vecs[idx:idx+1])
        print(f"\n  [{word}] (CID {wid}):")
        for i, (nidx, d) in enumerate(zip(nidxs[0], dists[0])):
            if i == 0:
                continue  # self
            nc = int(cids[nidx])
            try:
                nt = sp.IdToPiece(nc).replace("\u2581", " ").strip()
            except:
                nt = f"[CID{nc}]"
            sim = 1 - d
            print(f"    {i:2d}. {nt:20s} (CID {nc:6d}) sim={sim:.4f}")

if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else r"real_data\concept_space_75k"
    nn(prefix)
