"""
build_hierarchical.py — иерархическое хранение траекторий.

Читает trajectory_store один раз.
Сохраняет компактную иерархию: sentences, transitions CSR, morph/syntax cache.

Формат хранения:
  real_data/v5/hierarchical/
    metadata.json
    sentences.npz
    transitions_csr.npz
    token_counts.npz
    morph_cache.npz
    syntax_cache.npz
"""
import sys, os, json, time, pickle
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, save_npz, load_npz
from coordinate_packer import CoordinatePacker
from eva.symbolic.trajectory_store import TrajectoryStore
from eva.symbolic.bpe_tokenizer import BPEVocab
import math

packer = CoordinatePacker()
cv = BPEVocab()
V = packer.V
SAVE = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\hierarchical'
os.makedirs(SAVE, exist_ok=True)

print("="*60)
print("BUILD HIERARCHICAL STORAGE")
print("="*60)

# ─── 1. Load trajectory store ───
print("\n[1] Loading trajectory store (7.3 GB)...")
t0 = time.time()
store = TrajectoryStore()
store.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\trajectory_store.pkl')
print(f"  {store.total_stored} trajectories in {time.time()-t0:.0f}s")

# ─── 2. Extract metadata (NO full trajectories) ───
print("\n[2] Extracting metadata...")
sentences_data = []
transitions = defaultdict(int)  # (src, dst) → count
morph = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
syntax = defaultdict(lambda: defaultdict(int))
token_total = np.zeros(V, dtype=np.int32)
n_words = 0
n_tokens = 0

t0 = time.time()
for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    tokens = htraj.ids
    traj = htraj.symbol_trajectory  # need this for word_spans
    
    # Extract word_spans from trajectory flags
    word_spans = []
    in_word = False
    word_start = -1
    for t in range(len(tokens)):
        info = packer.unpack_token(traj[t])
        flags = info['flags']
        is_start = (flags >> packer.F_WORD_START) & 1
        is_end = (flags >> packer.F_WORD_END) & 1
        is_special = (flags >> packer.F_SPECIAL) & 1
        
        if is_start and not is_special:
            in_word = True
            word_start = t
        elif in_word:
            if is_end:
                word_spans.append((word_start, t))
                in_word = False
    
    # Store compact sentence data (tokens + word_spans, NO trajectory)
    sentences_data.append({
        'tokens': tokens,                    # list of int (token IDs)
        'word_spans': word_spans,            # list of (start, end)
        'n_tokens': len(tokens),
        'n_words': len(word_spans),
    })
    
    # Update transition counts (sparse)
    for t in range(len(tokens) - 1):
        src, dst = tokens[t], tokens[t + 1]
        transitions[(src, dst)] += 1
        token_total[src] += 1
    if len(tokens) > 0:
        token_total[tokens[-1]] += 1  # count last token too
    
    # Extract morph/syntax distributions
    for wi, (ws, we) in enumerate(word_spans):
        wl = we - ws + 1
        for pi in range(wl):
            tid = tokens[ws + pi]
            morph[wl][pi][tid] += 1
        # First token of each word → syntax distribution
        syntax[wi][tokens[ws]] += 1
    
    n_words += len(word_spans)
    n_tokens += len(tokens)
    
    if (idx + 1) % 5000 == 0:
        print(f"  {idx+1}/{store.total_stored} ({time.time()-t0:.0f}s)")

print(f"  Extracted: {len(sentences_data)} sentences, {n_words:,} words, {n_tokens:,} tokens")
print(f"  Unique transitions: {len(transitions)}")
print(f"  Time: {time.time()-t0:.0f}s")

# ─── 3. Build sparse transition matrix ───
print("\n[3] Building sparse transition matrix...")
t0 = time.time()

# Build CSR matrix: need data, indices, indptr
unique_pairs = list(transitions.keys())
n_pairs = len(unique_pairs)
print(f"  {n_pairs} non-zero pairs (density: {n_pairs/(V*V)*100:.4f}%)")

rows = np.array([s for s, d in unique_pairs], dtype=np.int32)
cols = np.array([d for s, d in unique_pairs], dtype=np.int32)
data = np.array([transitions[p] for p in unique_pairs], dtype=np.int32)

# Sort by row so entries for same row are contiguous
sort_idx = np.argsort(rows, kind='stable')
rows = rows[sort_idx]
cols = cols[sort_idx]
data = data[sort_idx]

# Row pointers
indptr = np.zeros(V + 1, dtype=np.int32)
for r in rows:
    indptr[r + 1] += 1
indptr = np.cumsum(indptr, dtype=np.int32)

# Sort within each row by column
for i in range(V):
    start, end = indptr[i], indptr[i+1]
    if start < end:
        idx_sorted = np.argsort(cols[start:end])
        cols[start:end] = cols[start:end][idx_sorted]
        data[start:end] = data[start:end][idx_sorted]

trans_csr = csr_matrix((data, cols, indptr), shape=(V, V), dtype=np.int32)

# Also compute log_prob
row_sums = np.array(trans_csr.sum(axis=1)).flatten()
row_sums = np.maximum(row_sums, 1)
log_prob_data = np.zeros(n_pairs, dtype=np.float32)
for i in range(n_pairs):
    r = rows[i]  # rows[i] is now the sorted row
    log_prob_data[i] = math.log(data[i] / row_sums[r]) if data[i] > 0 else -23.0

log_prob_csr = csr_matrix((log_prob_data, cols.copy(), indptr.copy()), shape=(V, V), dtype=np.float32)

print(f"  CSR size: {trans_csr.data.nbytes/1024/1024:.1f} MB + indices/indptr")
print(f"  Time: {time.time()-t0:.0f}s")

# ─── 4. Save compact storage ───
print("\n[4] Saving...")
t0 = time.time()

# Sentences: pack tokens as flat array with lengths, word_spans as flat array
sent_tokens_flat = np.concatenate([s['tokens'] for s in sentences_data]).astype(np.int16)
sent_token_lens = np.array([s['n_tokens'] for s in sentences_data], dtype=np.uint16)
sent_word_counts = np.array([s['n_words'] for s in sentences_data], dtype=np.uint16)

# Word spans: flat [start, end, start, end, ...]
word_spans_flat = np.concatenate([np.array(s['word_spans'], dtype=np.uint16).ravel() 
                                 for s in sentences_data])

np.savez_compressed(os.path.join(SAVE, 'sentences.npz'),
    tokens=sent_tokens_flat,
    token_lens=sent_token_lens,
    word_counts=sent_word_counts,
    word_spans=word_spans_flat,
)

# Sparse transition matrix
save_npz(os.path.join(SAVE, 'transitions_csr.npz'), trans_csr)
save_npz(os.path.join(SAVE, 'log_prob_csr.npz'), log_prob_csr)

# Token counts
np.savez_compressed(os.path.join(SAVE, 'token_counts.npz'), 
    counts=token_total)

# Morph cache: store as compressed numpy arrays
# morph_dist[wl][pos][tid] → build sparse matrix per (wl, pos)
morph_dict = {}
for wl in morph:
    for pos in morph[wl]:
        dist = morph[wl][pos]
        total = sum(dist.values())
        tids = np.array(list(dist.keys()), dtype=np.int16)
        cnts = np.array(list(dist.values()), dtype=np.int32)
        morph_dict[f'{wl}_{pos}'] = {'tids': tids, 'cnts': cnts, 'total': total}

np.savez_compressed(os.path.join(SAVE, 'morph_cache.npz'), **morph_dict)
with open(os.path.join(SAVE, 'morph_keys.json'), 'w') as f:
    json.dump({'keys': list(morph_dict.keys()), 'wl_range': [min(morph.keys()), max(morph.keys())]}, f)

# Syntax cache
syntax_dict = {}
for wn in syntax:
    dist = syntax[wn]
    total = sum(dist.values())
    tids = np.array(list(dist.keys()), dtype=np.int16)
    cnts = np.array(list(dist.values()), dtype=np.int32)
    syntax_dict[f'{wn}'] = {'tids': tids, 'cnts': cnts, 'total': total}

np.savez_compressed(os.path.join(SAVE, 'syntax_cache.npz'), **syntax_dict)
with open(os.path.join(SAVE, 'syntax_keys.json'), 'w') as f:
    json.dump({'keys': list(syntax_dict.keys()), 'max_wn': max(syntax.keys())}, f)

# Metadata
metadata = {
    'n_sentences': len(sentences_data),
    'n_tokens': int(n_tokens),
    'n_words': int(n_words),
    'vocab_size': V,
    'n_transitions': int(sum(transitions.values())),
    'n_unique_transitions': n_pairs,
    'density': n_pairs / (V * V),
    'morph_len_range': [int(min(morph.keys())), int(max(morph.keys()))],
    'syntax_max_word': int(max(syntax.keys())),
    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    'source': 'trajectory_store_v5.pkl -> build_hierarchical_v5.py',
    'size_mb_estimate': 0,
}

with open(os.path.join(SAVE, 'metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"  Saved in {time.time()-t0:.0f}s")
print(f"\n{'='*60}")
print("HIERARCHICAL STORAGE READY")
print(f"{'='*60}")
print(f"  Sentences: {metadata['n_sentences']:,}")
print(f"  Words: {metadata['n_words']:,}")
print(f"  Tokens: {metadata['n_tokens']:,}")
print(f"  Unique transitions: {metadata['n_unique_transitions']:,}")
print(f"  Location: {SAVE}")

# Estimate actual file sizes
total_size = 0
for fname in os.listdir(SAVE):
    fpath = os.path.join(SAVE, fname)
    size = os.path.getsize(fpath)
    total_size += size
    mb = size / 1024 / 1024
    print(f"    {fname}: {mb:.1f} MB")
print(f"  TOTAL: {total_size/1024/1024:.0f} MB")
