"""
rebuild_heads_db.py — пересобрать heads_db с sparse матрицами.
"""
import sys, os, json, math, time, pickle
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, load_npz, save_npz
from coordinate_packer import CoordinatePacker
from eva.symbolic.bpe_tokenizer import BPEVocab

packer = CoordinatePacker()
cv = BPEVocab()
V = packer.V

HIER = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\hierarchical'
SAVE = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5'

print("="*60)
print("REBUILD HEADS DB (sparse)")
print("="*60)

t0 = time.time()

# ─── 1. Load ───
print("\n[1] Loading hierarchical storage...")
sent_data = np.load(os.path.join(HIER, 'sentences.npz'))
sent_tokens = sent_data['tokens']
sent_lens = sent_data['token_lens']
sent_word_counts = sent_data['word_counts']
sent_word_spans = sent_data['word_spans']

sentences = []
ptr = 0
wptr = 0
for i in range(len(sent_lens)):
    L = int(sent_lens[i])
    nw = int(sent_word_counts[i])
    tokens = list(sent_tokens[ptr:ptr+L])
    spans = []
    for j in range(nw):
        s = int(sent_word_spans[wptr + 2*j])
        e = int(sent_word_spans[wptr + 2*j + 1])
        spans.append((s, e))
    sentences.append({'tokens': tokens, 'word_spans': spans})
    ptr += L
    wptr += 2 * nw

trans_csr = load_npz(os.path.join(HIER, 'transitions_csr.npz'))
log_prob_csr = load_npz(os.path.join(HIER, 'log_prob_csr.npz'))
token_counts = np.load(os.path.join(HIER, 'token_counts.npz'))['counts']
morph_cache = np.load(os.path.join(HIER, 'morph_cache.npz'), allow_pickle=True)
syntax_cache = np.load(os.path.join(HIER, 'syntax_cache.npz'), allow_pickle=True)
with open(os.path.join(HIER, 'morph_keys.json')) as f:
    morph_keys = json.load(f)['keys']
with open(os.path.join(HIER, 'syntax_keys.json')) as f:
    syntax_keys = json.load(f)['keys']

print(f"  {len(sentences)} sentences, {trans_csr.nnz} transitions, {time.time()-t0:.1f}s")

# ─── 2. Transition-pattern similarity (sparse cosine) ───
print("\n[2] Building transition-pattern similarity (sparse cosine)...")
row_sums = np.array(trans_csr.sum(axis=1)).flatten()
col_sums = np.array(trans_csr.sum(axis=0)).flatten()
valid_mask = (row_sums + col_sums) > 10
valid_tokens = np.where(valid_mask)[0]
n_valid = len(valid_tokens)
n_valid_bpe = int((valid_tokens >= 161).sum())
print(f"  {n_valid} total valid, {n_valid_bpe} BPE tokens")

special_set = set(range(0, 5)) | {156, 157, 158, 159, 160}

# Precompute row/col probability dicts for all valid tokens
row_prob = {}  # tid -> {dest: prob}
col_prob = {}  # tid -> {src: prob}
row_sq_sum = {}  # tid -> sum of squared probs (for denom)
col_sq_sum = {}

for tid in valid_tokens:
    if tid in special_set or tid < 161:
        continue
    total_r = int(row_sums[tid])
    total_c = int(col_sums[tid])
    if total_r > 0:
        r_coo = trans_csr[tid].tocoo()
        r_dict = {int(c): float(d) / total_r for c, d in zip(r_coo.col, r_coo.data)}
        row_prob[tid] = r_dict
        row_sq_sum[tid] = sum(v*v for v in r_dict.values())
    if total_c > 0:
        c_coo = trans_csr[:, tid].tocoo()
        c_dict = {int(r): float(d) / total_c for r, d in zip(c_coo.row, c_coo.data)}
        col_prob[tid] = c_dict
        col_sq_sum[tid] = sum(v*v for v in c_dict.values())

# Build inverted index: dest -> [tokens that transition TO dest]
dest_to_tokens = defaultdict(set)
for tid in row_prob:
    for d in row_prob[tid]:
        dest_to_tokens[d].add(tid)

print(f"  Inverted index: {len(dest_to_tokens)} destinations, computing similarities...")

trans_sim_sparse = {}
batch_size = 50
bpe_tids = [t for t in valid_tokens if t >= 161 and t in row_prob]
n_bpe = len(bpe_tids)

for bidx, tid in enumerate(bpe_tids):
    # Get all tokens that share at least one destination with this token
    candidates = set()
    for d in row_prob[tid]:
        candidates.update(dest_to_tokens[d])
    candidates.discard(tid)
    candidates -= special_set
    
    if not candidates:
        continue
    
    scores = []
    cand_list = list(candidates)
    r_dict = row_prob[tid]
    c_dict = col_prob.get(tid, {})
    r_sq = row_sq_sum.get(tid, 0)
    c_sq = col_sq_sum.get(tid, 0)
    
    for ctid in cand_list:
        # Row cosine: P(tid→*) · P(ctid→*)
        r_ct = row_prob.get(ctid, {})
        dot_r = sum(r_dict[d] * r_ct.get(d, 0) for d in r_dict if d in r_ct)
        r_ct_sq = row_sq_sum.get(ctid, 1)
        cos_r = dot_r / (math.sqrt(r_sq * r_ct_sq) + 1e-10)
        
        # Column cosine: P(*→tid) · P(*→ctid)
        c_ct = col_prob.get(ctid, {})
        dot_c = sum(c_dict[s] * c_ct.get(s, 0) for s in c_dict if s in c_ct)
        c_ct_sq = col_sq_sum.get(ctid, 1)
        cos_c = dot_c / (math.sqrt(c_sq * c_ct_sq) + 1e-10) if c_sq > 0 else 0
        
        sim = 0.5 * cos_r + 0.5 * cos_c
        if sim > 0.01:  # lower threshold for cosine
            scores.append(sim)
    
    if scores:
        top_k = min(20, len(scores))
        top_idx = np.argsort(-np.array(scores))[:top_k]
        pairs = [(int(cand_list[j]), float(scores[j])) for j in top_idx]
        if pairs:
            trans_sim_sparse[int(tid)] = pairs
    
    if (bidx + 1) % batch_size == 0:
        print(f"  {bidx+1}/{n_bpe} ({len(trans_sim_sparse)} with neighbors)")

print(f"  Done: {len(trans_sim_sparse)} tokens have neighbors ({time.time()-t0:.1f}s)")

# ─── 3. Contradiction pairs ───
print("\n[3] Computing contradiction pairs (sim>0.5, P=0)...")
contra_pairs = []
for ta in list(trans_sim_sparse.keys()):
    if ta in special_set or ta < 161:
        continue
    count_ta = int(token_counts[ta])
    if count_ta < 10:
        continue
    for tb, sim in trans_sim_sparse[ta]:
        if tb in special_set or tb < 161 or tb <= ta:
            continue
        if sim < 0.5:  # max similarity is ~0.84, so 0.5 is top 10%
            continue
        count_tb = int(token_counts[tb])
        if count_tb < 10:
            continue
        if trans_csr[ta, tb] == 0 and trans_csr[tb, ta] == 0:
            contra_pairs.append((ta, tb, sim))

print(f"  {len(contra_pairs)} clean contradict pairs ({time.time()-t0:.1f}s)")

# ─── 4. Ngram ───
print("\n[4] Computing ngram distribution...")
ngram = defaultdict(lambda: defaultdict(int))
for sent in sentences:
    t = sent['tokens']
    for i in range(2, len(t)):
        ngram[(t[i-2], t[i-1])][t[i]] += 1
ngram_sparse = {}
for (p2, p1), dist in ngram.items():
    total = sum(dist.values())
    if total < 3:
        continue
    top = sorted(dist.items(), key=lambda x: -x[1])[:10]
    ngram_sparse[f'{p2}_{p1}'] = [(int(t), int(c), c/total) for t, c in top]
print(f"  {len(ngram_sparse)} ngrams")

# ─── 5. Concept scores ───
print("\n[5] Concept scores (based on token diversity)...")
concept_scores = np.ones(V, dtype=np.float32) * 0.5
max_cnt = max(token_counts) or 1
for tid in range(V):
    freq = token_counts[tid] / max_cnt
    concept_scores[tid] = 1.0 - freq  # rare tokens = higher concept score

# ─── 6. Precompute log_prob arrays from cache ───
print("\n[6] Precomputing log_prob arrays...")
log_prior = -math.log(V)
morph_logprob = {}
morph_keys_list = json.loads(open(os.path.join(HIER, 'morph_keys.json')).read())['keys']
for key_str in morph_keys_list:
    wl, pos = int(key_str.split('_')[0]), int(key_str.split('_')[1])
    arr = np.full(V, log_prior, dtype=np.float32)
    d = morph_cache[key_str].item()
    tids, cnts, total = d['tids'], d['cnts'], d['total']
    for j in range(len(tids)):
        tid = int(tids[j])
        cnt = int(cnts[j])
        arr[tid] = math.log((cnt + 1.0) / total)
    if wl not in morph_logprob:
        morph_logprob[wl] = {}
    morph_logprob[wl][pos] = arr

syntax_logprob = {}
syntax_keys_list = json.loads(open(os.path.join(HIER, 'syntax_keys.json')).read())['keys']
for key_str in syntax_keys_list:
    wn = int(key_str)
    arr = np.full(V, log_prior, dtype=np.float32)
    d = syntax_cache[key_str].item()
    tids, cnts, total = d['tids'], d['cnts'], d['total']
    for j in range(len(tids)):
        tid = int(tids[j])
        cnt = int(cnts[j])
        arr[tid] = math.log((cnt + 1.0) / total)
    syntax_logprob[wn] = arr

# ─── 7. Save ───
print("\n[7] Saving...")
# CSR matrices already exist in hierarchical/
# Just save metadata
meta = {
    'contra_pairs': contra_pairs,
    'ngram_sparse': ngram_sparse,
    'trans_sim_sparse': trans_sim_sparse,
    'concept_scores': concept_scores,
    'token_counts': token_counts,
    'V': V,
    'morph_logprob': morph_logprob,
    'syntax_logprob': syntax_logprob,
    'stats': {
        'n_sentences': len(sentences),
        'n_valid_tokens': n_valid,
        'n_contra_pairs': len(contra_pairs),
        'n_ngrams': len(ngram_sparse),
        'n_sim_tokens': len(trans_sim_sparse),
    }
}
with open(os.path.join(SAVE, 'heads_meta.pkl'), 'wb') as f:
    pickle.dump(meta, f, protocol=5)

for f in os.listdir(SAVE):
    fp = os.path.join(SAVE, f)
    if os.path.isfile(fp):
        print(f"  {f}: {os.path.getsize(fp)/1024/1024:.1f} MB")

print(f"\n  Done in {time.time()-t0:.1f}s")
print(f"  Contra pairs: {len(contra_pairs)}")
print(f"  Ngrams: {len(ngram_sparse)}")
print(f"  Sim tokens: {len(trans_sim_sparse)}")
