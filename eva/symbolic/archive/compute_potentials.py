"""
EVA v5 — Чтение потенциальных связей из полных треков.
Вычисляем: транзишены, дельты, контексты, плотности.
Никакого обучения — только статистика по реальным данным.
"""
import sys, os, pickle, time
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
from scipy import sparse as sp
from coordinate_packer import CoordinatePacker
from eva.symbolic.trajectory_store import TrajectoryStore
from eva.symbolic.bpe_tokenizer import BPEVocab

print("="*60)
print("ЧТЕНИЕ ПОТЕНЦИАЛЬНЫХ СВЯЗЕЙ")
print("="*60)

packer = CoordinatePacker()
cv = BPEVocab()
V = packer.V  # 4101

# ─── 1. Load trajectories ───
print("\n[1] Loading trajectories...")
store = TrajectoryStore()
store.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\trajectory_store_v5.pkl')
print(f"  {store.total_stored} trajectories")

# ─── 2. Compute transition statistics ───
print("\n[2] Computing transition matrix + delta vectors...")
t0 = time.time()

# Transition count: dense 4101x4101 = 67 MB (ok)
trans_count = np.zeros((V, V), dtype=np.int32)
# Delta vectors: store sparsely, only observed transitions
delta_dict = {}  # (src, dst) -> [384] float32, accumulated
delta_cnt = {}   # (src, dst) -> int, count
h_by_token = {tid: [] for tid in range(V)}

total_transitions = 0
for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    traj = htraj.symbol_trajectory  # [L, 384]
    tokens = htraj.ids
    
    L = len(tokens)
    for t in range(L - 1):
        src = tokens[t]
        dst = tokens[t + 1]
        delta = traj[t + 1] - traj[t]
        
        trans_count[src, dst] += 1
        key = (src, dst)
        if key in delta_dict:
            delta_dict[key] += delta.astype(np.float32)
            delta_cnt[key] += 1
        else:
            delta_dict[key] = delta.astype(np.float32).copy()
            delta_cnt[key] = 1
        total_transitions += 1
    
    # Collect h[t] per token (sample first 5000 for memory)
    for t in range(L):
        tid = tokens[t]
        if len(h_by_token.get(tid, [])) < 100:
            h_by_token.setdefault(tid, []).append(traj[t])

    if (idx + 1) % 5000 == 0:
        print(f"  {idx+1}/{store.total_stored} processed ({total_transitions:,} transitions)")

print(f"  Done: {total_transitions:,} transitions in {time.time()-t0:.0f}s")
print(f"  Unique transitions: {(trans_count > 0).sum()}")
print(f"  Density: {(trans_count > 0).sum() / (V*V) * 100:.2f}% of {V}x{V} matrix filled")

# ─── 3. Transition probabilities + delta means ───
print("\n[3] Computing transition probabilities...")

# P(token_j | token_i) = count(i→j) / count(i→*)
row_sums = trans_count.sum(axis=1, keepdims=True)
row_sums = np.maximum(row_sums, 1)  # avoid div by 0
trans_prob = trans_count.astype(np.float64) / row_sums

# Average delta per transition (sparse dict -> dense only for nonzero)
delta_mean_sparse = {}
mask_src_dst = list(delta_dict.keys())
print(f"  Computing {len(mask_src_dst)} delta means...")
for key in mask_src_dst:
    delta_mean_sparse[key] = delta_dict[key] / delta_cnt[key]

# Free raw accumulators
del delta_dict
del delta_cnt

# Build delta_norm array for statistics
delta_norms = np.array([np.linalg.norm(v) for v in delta_mean_sparse.values()])

# ─── 4. Token context vectors ───
print("\n[4] Computing token context signatures...")
context_vectors = np.zeros((V, packer.DIM), dtype=np.float32)
context_counts = np.zeros(V, dtype=np.int32)
for tid, vecs in h_by_token.items():
    if vecs:
        context_vectors[tid] = np.mean(vecs, axis=0)
        context_counts[tid] = len(vecs)

# Which tokens have enough data?
tokens_with_data = context_counts > 0
print(f"  Tokens with context data: {tokens_with_data.sum()}/{V}")

# ─── 5. Analysis of transition patterns ───
print("\n[5] Analysis:")
print(f"  Transitions: {total_transitions:,}")
print(f"  Unique transitions: {(trans_count > 0).sum():,}")
print(f"  Avg contexts/token: {context_counts[tokens_with_data].mean():.1f}")

# Top transitions
top_n = 20
flat_idx = np.argsort(-trans_count, axis=None)
top_src = flat_idx[:top_n] // V
top_dst = flat_idx[:top_n] % V
print(f"\n  Top-{top_n} transitions:")
for i in range(top_n):
    s, d = top_src[i], top_dst[i]
    c = trans_count[s, d]
    s_text = cv.decode([s], skip_special=False)[:10]
    d_text = cv.decode([d], skip_special=False)[:10]
    d_norm = np.linalg.norm(delta_mean_sparse.get((s, d), np.zeros(packer.DIM)))
    print(f"    '{s_text}' ({s:4d}) -> '{d_text}' ({d:4d}) : {c:5d} times, "
          f"P={trans_prob[s,d]:.3f}, delta_norm={d_norm:.3f}")

# Entropy per source token
entropy = np.zeros(V)
for src in range(V):
    p = trans_prob[src]
    p = p[p > 0]
    if len(p) > 0:
        entropy[src] = -np.sum(p * np.log2(p))

max_ent = np.argsort(-entropy)[:10]
print(f"\n  Highest entropy tokens (most unpredictable):")
for tid in max_ent:
    text = cv.decode([tid], skip_special=False)[:10]
    n_dest = (trans_count[tid] > 0).sum()
    print(f"    '{text}' ({tid:4d}) : ent={entropy[tid]:.2f}, {n_dest} destinations")

# ─── 6. Delta vector statistics ───
print(f"\n  Delta norms:")
print(f"    mean={delta_norms.mean():.3f} std={delta_norms.std():.3f}")
print(f"    min={delta_norms.min():.3f} max={delta_norms.max():.3f}")

# ─── 7. Save compact database ───
print("\n[6] Saving potential database...")
save_dir = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5'

# Sparse save: only non-zero rows and columns
rows, cols = np.where(trans_count > 0)
n_nonzero = len(rows)

pot_db = {
    'transition_count': trans_count,
    'transition_prob': trans_prob,
    'delta_mean_sparse': delta_mean_sparse,
    'context_vectors': context_vectors,
    'context_counts': context_counts,
    'token_entropy': entropy,
    'n_transitions': total_transitions,
    'n_unique': n_nonzero,
    'density': n_nonzero / (V * V),
    'stats': {
        'n_trajectories': store.total_stored,
        'n_tokens': sum(len(htraj.ids) for htraj in store.hierarchical),
        'n_transitions_total': total_transitions,
        'n_unique_transitions': n_nonzero,
        'delta_norm_mean': float(delta_norms.mean()),
        'delta_norm_std': float(delta_norms.std()),
        'avg_contexts_per_token': float(context_counts[tokens_with_data].mean()),
        'n_delta_vectors': len(delta_mean_sparse),
    }
}

with open(os.path.join(save_dir, 'potential_db.pkl'), 'wb') as f:
    pickle.dump(pot_db, f, protocol=5)

print(f"  Saved: potential_db.pkl")
print(f"  Size estimation:")
for key, val in pot_db.items():
    if isinstance(val, np.ndarray):
        mb = val.nbytes / 1024 / 1024
        print(f"    {key}: {val.shape} = {mb:.1f} MB")
    elif isinstance(val, dict):
        print(f"    {key}: {len(val)} entries")

print(f"\n{'='*60}")
print(f"DATABASE READY")
print(f"{'='*60}")
print(f"  Transition count:  {n_nonzero:,} unique pairs")
print(f"  Matrix density:    {n_nonzero / (V*V) * 100:.4f}%")
print(f"  Delta norm range:  [{delta_norms.min():.3f}, {delta_norms.max():.3f}]")
print(f"  Context coverage:  {tokens_with_data.sum()}/{V} tokens")
print(f"{'='*60}")
