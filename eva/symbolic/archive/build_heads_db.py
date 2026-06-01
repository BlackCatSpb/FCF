"""
build_heads_db.py — вычислить и сохранить все статистики для v5 Heads.

Сохраняет в real_data/v5/heads_db.pkl:
  - morph_dist:        word_len → pos_in_word → {token_id: count}
  - syntax_dist:       word_num → {token_id: count} (word-start token per position)
  - sent_start_dist:   sent_len → {token_id: count} (sentence-start token)
  - trans_prob:        [V, V] — матрица вероятностей переходов (log)
  - trans_sim_sparse:  token_id → [(neighbor_id, similarity)] (transition-pattern similarity, top-20)
  - contra_pairs:      (token_a, token_b, similarity) where P=0 but expected
  - concept_regions:   attractor field gap scores per token
  - token_counts:      total occurrences of each token
  - word_type_counts:  first-token-of-word count per token_id
"""
import sys, os, pickle, math, time
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
from collections import defaultdict
from coordinate_packer import CoordinatePacker
from eva.symbolic.trajectory_store import TrajectoryStore
from eva.symbolic.bpe_tokenizer import BPEVocab

packer = CoordinatePacker()
cv = BPEVocab()
V = packer.V

SAVE_DIR = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5'
os.makedirs(SAVE_DIR, exist_ok=True)

print("="*60)
print("BUILD HEADS DATABASE")
print("="*60)

# ─── 1. Load existing data ───
print("\n[1] Loading trajectories + potential_db...")
store = TrajectoryStore()
store.load(os.path.join(SAVE_DIR, 'trajectory_store_v5.pkl'))
with open(os.path.join(SAVE_DIR, 'potential_db.pkl'), 'rb') as f:
    pot_db = pickle.load(f)

trans_count = pot_db['transition_count']     # [V, V]
trans_prob = pot_db['transition_prob']       # [V, V]
context_vectors = pot_db['context_vectors']  # [V, 384]
context_counts = pot_db['context_counts']    # [V]

# ─── 2. MORPH distribution ───
print("\n[2] Computing morphology distribution (word_len → pos → token)...")
morph = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
n_words = 0

for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    traj = htraj.symbol_trajectory
    tokens = htraj.ids
    
    in_word = False
    word_toks = []
    for t, tid in enumerate(tokens):
        info = packer.unpack_token(traj[t])
        flags = info['flags']
        is_start = (flags >> packer.F_WORD_START) & 1
        is_end = (flags >> packer.F_WORD_END) & 1
        is_special = (flags >> packer.F_SPECIAL) & 1
        
        if is_start and not is_special:
            in_word = True
            word_toks = [tid]
        elif in_word:
            if not is_special:
                word_toks.append(tid)
            if is_end:
                wl = len(word_toks)
                for pi, wt in enumerate(word_toks):
                    morph[wl][pi][wt] += 1
                in_word = False
                n_words += 1
                word_toks = []

# Convert to dict-of-dicts for JSON-safe saving
morph_dict = {}
for wl in morph:
    morph_dict[wl] = {}
    for pos in morph[wl]:
        morph_dict[wl][pos] = dict(morph[wl][pos])

# ─── 3. SYNTAX distribution ───
print("\n[3] Computing syntax distribution (word_num → word_start_token)...")
syntax = defaultdict(lambda: defaultdict(int))

for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    traj = htraj.symbol_trajectory
    tokens = htraj.ids
    
    in_word = False
    wn = -1
    for t, tid in enumerate(tokens):
        info = packer.unpack_token(traj[t])
        flags = info['flags']
        is_start = (flags >> packer.F_WORD_START) & 1
        is_end = (flags >> packer.F_WORD_END) & 1
        is_special = (flags >> packer.F_SPECIAL) & 1
        
        if is_start and not is_special:
            wn += 1
            syntax[wn][tid] += 1  # first token of this word
            in_word = True
        elif is_end:
            in_word = False

syntax_dict = {wn: dict(d) for wn, d in syntax.items()}

# ─── 4. SENTENCE-START distribution ───
print("\n[4] Computing sentence-start distribution...")
sent_start = defaultdict(int)
sent_len_dist = defaultdict(lambda: defaultdict(int))

for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    traj = htraj.symbol_trajectory
    tokens = htraj.ids
    L = len(tokens)
    
    for t, tid in enumerate(tokens):
        info = packer.unpack_token(traj[t])
        flags = info['flags']
        is_start = (flags >> packer.F_SENT_START) & 1
        is_end = (flags >> packer.F_SENT_END) & 1
        
        # First non-special token after SENT_OPEN
        if is_start and (flags >> packer.F_SPECIAL) & 1 == 0:
            sent_start[tid] += 1
        
        if is_end:
            sent_len = info.get('sent_len', 0)
            if sent_len:
                sent_len_dist[sent_len][tid] += 1

# ─── 5. Transition-pattern similarity ───
print("\n[5] Computing transition-pattern similarity...")
row_sum = trans_count.sum(axis=1, keepdims=True).astype(np.float64)
col_sum = trans_count.sum(axis=0, keepdims=True).astype(np.float64)

# Only for tokens with min 10 total transitions AND non-zero row+col norm
total_trans = trans_count.sum(axis=1) + trans_count.sum(axis=0)
valid_mask = total_trans > 10

row_norm = np.zeros((V, V), dtype=np.float32)
col_norm = np.zeros((V, V), dtype=np.float32)
for tid in range(V):
    if valid_mask[tid]:
        rs = float(row_sum[tid, 0])   # row_sum = [V, 1]
        cs = float(col_sum[0, tid])   # col_sum = [1, V]
        row_norm[tid] = (trans_count[tid].astype(np.float64) / max(rs, 1.0)).astype(np.float32)
        col_norm[tid] = (trans_count[:, tid].astype(np.float64) / max(cs, 1.0)).astype(np.float32)
valid_tokens = np.where(valid_mask)[0]
n_valid = len(valid_tokens)

print(f"  Computing for {n_valid} tokens...")

row_valid = row_norm[valid_tokens]
col_valid = col_norm[valid_tokens]
eps = 1e-10
row_valid_n = row_valid / (np.linalg.norm(row_valid, axis=1, keepdims=True) + eps)
col_valid_n = col_valid / (np.linalg.norm(col_valid, axis=1, keepdims=True) + eps)

row_sim = row_valid_n @ row_valid_n.T
col_sim = col_valid_n @ col_valid_n.T
trans_sim = 0.5 * row_sim + 0.5 * col_sim

# For each token, store top-20 most similar by transition patterns
trans_sim_sparse = {}
for i, tid in enumerate(valid_tokens):
    sims = trans_sim[i]
    top_k = np.argsort(-sims)[1:21]  # exclude self
    trans_sim_sparse[tid] = [(int(valid_tokens[j]), float(sims[j])) for j in top_k]

# ─── 6. CONTRADICTION candidates ───
print("\n[6] Computing contradiction candidates...")
contra_pairs = []

# Contradiction = high trans_sim but P(src→dst) = 0 and P(dst→src) = 0
# Only BPE subword tokens (ID > 160) — punctuation/char contradictions are expected
special_tokens = set(range(0, 5)) | {156, 157, 158, 159, 160}  # BOS,UNK,PAD,EOS,SEP,PUNCT,WORD*,SENT*
for i, ta in enumerate(valid_tokens):
    if ta in special_tokens:
        continue
    for j, tb in enumerate(valid_tokens):
        if j <= i:
            continue
        if tb in special_tokens:
            continue
        if trans_count[ta, tb] == 0 and trans_count[tb, ta] == 0:
            s = trans_sim[i, j]
            if s > 0.85:  # stricter threshold
                contra_pairs.append((int(ta), int(tb), float(s)))

print(f"  Found {len(contra_pairs)} contradiction pairs (P=0, trans_sim>0.8)")

# ─── 7. CONCEPT regions: attractor potential per token ───
print("\n[7] Computing concept regions (attractor field)...")

# Use context_vectors as token position proxies
# For each token, compute its attractor field potential
# A "concept" = region of high potential surrounded by low potential
# We compute: concept_score[tid] = 1/(1+potential[tid])
# Higher score = sparser region = potential concept

# Load attractor field
import torch
af_path = os.path.join(SAVE_DIR, 'attractor_field_v5.pt')
if os.path.exists(af_path):
    try:
        af_data = torch.load(af_path, map_location='cpu', weights_only=False)
        if isinstance(af_data, dict):
            # Extract centers and counts
            centers = af_data.get('centers', None)
            counts = af_data.get('counts', None)
            if centers is None:
                # Try nested structure
                for key in ['attractors.centers', 'centers', 'attractor_centers']:
                    if key in af_data:
                        centers = af_data[key]
                        break
            if isinstance(centers, torch.Tensor) and len(centers) > 0:
                centers_np = centers.numpy() if hasattr(centers, 'numpy') else np.array(centers)
                print(f"  Attractor centers: {centers_np.shape}")
                
                # Compute potential at each token's context vector
                token_pos = context_vectors  # [V, 384]
                
                # Sample centers if too many
                n_centers = min(10000, len(centers_np))
                center_sample = centers_np[:n_centers]
                
                # Batch compute distances
                from scipy.spatial.distance import cdist
                token_pos_eff = context_vectors[:2000]  # first 2000 tokens
                
                # Compute adaptive sigma: median distance between token positions
                # In ±1 metadata space (97 active dims), typical distance ≈ sqrt(97*4) = 19.7
                # Use 3 distances to estimate
                sample_dists = cdist(token_pos_eff[:100], center_sample[:100], metric='euclidean')
                adaptive_sigma = float(np.median(sample_dists)) / 3.0  # so ~95% within 3 sigma
                print(f"  Adaptive sigma = {adaptive_sigma:.2f} (median dist = {float(np.median(sample_dists)):.2f})")
                
                dists = cdist(token_pos_eff, center_sample, metric='euclidean')
                potentials = np.exp(-dists ** 2 / (2 * adaptive_sigma ** 2)).sum(axis=1)
                
                # Normalize: concept_score = 1/(1+potential/total_centers)
                # potential ≈ total_centers in dense region, ≈ 0 in sparse region
                normalized = potentials / len(center_sample)
                concept_scores = 1.0 - normalized  # 0 = dense, 1 = sparse (concept)
                concept_scores = np.clip(concept_scores, 0.0, 1.0)
                print(f"  Concept scores (first 10, normalized): {concept_scores[:10]}")
                print(f"  Range: [{concept_scores.min():.3f}, {concept_scores.max():.3f}]")
            else:
                print(f"  No valid centers found in attractor field")
                concept_scores = None
        else:
            print(f"  Attractor field type: {type(af_data)}")
            concept_scores = None
    except Exception as e:
        print(f"  Error: {e}")
        concept_scores = None
else:
    print(f"  No attractor field at {af_path}")
    concept_scores = None

# ─── 8. Token frequency ───
print("\n[8] Computing token frequencies...")
token_counts = defaultdict(int)
for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    tokens = htraj.ids
    for tid in tokens:
        token_counts[tid] += 1

# ─── 9. Save ───
print("\n[9] Saving heads database...")
heads_db = {
    'morph_dist': morph_dict,
    'syntax_dist': syntax_dict,
    'sent_start_dist': dict(sent_start),
    'sent_len_dist': {sl: dict(d) for sl, d in sent_len_dist.items()},
    'trans_prob': trans_prob.astype(np.float32),
    'trans_count': trans_count.astype(np.int32),
    'trans_sim_sparse': trans_sim_sparse,
    'contra_pairs': contra_pairs,
    'token_counts': dict(token_counts),
    'context_vectors': context_vectors,
    'V': V,
    'stats': {
        'n_trajectories': store.total_stored,
        'n_words': n_words,
        'morph_len_range': [int(k) for k in sorted(morph_dict.keys())],
        'syntax_max_word': max(syntax_dict.keys()),
        'n_valid_tokens': n_valid,
        'n_contra_pairs': len(contra_pairs),
    }
}

if concept_scores is not None:
    heads_db['concept_scores'] = concept_scores

save_path = os.path.join(SAVE_DIR, 'heads_db.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(heads_db, f, protocol=5)

print(f"\n  Saved: {save_path}")
print(f"  Stats:")
for k, v in heads_db['stats'].items():
    print(f"    {k}: {v}")

print("\n" + "="*60)
print("HEADS DATABASE READY")
print("="*60)
