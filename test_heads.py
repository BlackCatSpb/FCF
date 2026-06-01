"""Demonstration of vectorized data-driven HeadsEnsemble."""
import sys, os
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
from coordinate_packer import CoordinatePacker
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.trajectory_store import TrajectoryStore
from eva.symbolic.bpe_tokenizer import BPEVocab

import time

packer = CoordinatePacker()
cv = BPEVocab()
V = packer.V

DB = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\heads_db.pkl'
print("Loading heads...", end=' ', flush=True)
t0 = time.time()
heads = HeadsEnsemble(DB)
print(f"{time.time()-t0:.1f}s")

store = TrajectoryStore()
store.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\trajectory_store_v5.pkl')

# Find a short sentence
for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    if 15 <= len(htraj.ids) <= 35:
        break

htraj = store.hierarchical[idx]
traj = htraj.symbol_trajectory
tokens = htraj.ids
L = len(tokens)
text = htraj.text

print(f"\n{'='*70}")
print(f"Sentence #{idx}: {L} tokens")
tok_text = ' '.join([cv.decode([t]) for t in tokens[:20]])
print(f"  {tok_text}...")

def ctx_at(traj, tokens, t):
    info = packer.unpack_token(traj[t])
    flags = info['flags']
    return {
        'token_id': info['token_id'],
        'pos_in_word': info['pos_in_word'],
        'word_len': info['word_len'],
        'word_num': info['word_num'],
        'sent_len': info['sent_len'],
        'flags': flags,
        'prev_token_id': tokens[t-1] if t > 0 else None,
        'context_tokens': tokens[:t],
    }

def show_step(t, label=""):
    ctx = ctx_at(traj, tokens, t)
    actual = tokens[t+1]
    pred = heads.best_token(ctx)
    scores = heads.score_all(ctx)
    top5 = heads.top_k(ctx, 5)
    
    info = packer.unpack_token(traj[t])
    flags = info['flags']
    is_special = (flags >> 5) & 1
    is_ws = (flags >> 0) & 1
    
    tags = []
    if is_special: tags.append('SPEC')
    if is_ws: tags.append('↑WORD')
    
    weights = heads.compute_weights(ctx)
    w_str = ' '.join([f"{k}={v:.1f}" for k,v in weights.items() if v > 0])
    
    rank = next((i for i, (tid,_) in enumerate(top5) if tid == actual), -1)
    mark = '✓' if pred == actual else (f'T{rank+1}' if rank >= 0 else '✗')
    
    tok_t = cv.decode([tokens[t]])
    tok_p = cv.decode([pred])
    tok_a = cv.decode([actual])
    top5_str = ', '.join([f"'{cv.decode([tid])}'" for tid,_ in top5])
    
    print(f"  [{t:2d}] {tok_t:<10} [{tags[0] if tags else 'inside':>6}] → "
          f"{mark} pred='{tok_p}' act='{tok_a}'")
    if pred != actual:
        print(f"        weights: {w_str}")
        print(f"        top-5: {top5_str}")

# ─── Show each step ───
print(f"\n{'─'*70}")
for t in range(min(8, L-1)):
    show_step(t)

# ─── Accuracy evaluation ───
print(f"\n{'─'*70}")
print("ACCURACY (1000 sentences, 15-40 tokens)")
print(f"{'─'*70}")

t0 = time.time()
results = {'correct': 0, 'total': 0, 'top5': 0, 'top3': 0}
levels = {
    'special': {'c':0,'t':0}, 'word_start': {'c':0,'t':0},
    'mid_word': {'c':0,'t':0}, 'word_end': {'c':0,'t':0},
}

count = 0
for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    L = len(htraj.ids)
    if L < 10 or L > 60:
        continue
    
    traj = htraj.symbol_trajectory
    tokens = htraj.ids
    
    for t in range(1, L-1):
        ctx = ctx_at(traj, tokens, t)
        actual = tokens[t+1]
        scores = heads.score_all(ctx)
        pred = int(np.argmax(scores))
        top3 = set(np.argsort(-scores)[:3])
        top5 = set(np.argsort(-scores)[:5])
        
        results['total'] += 1
        if pred == actual: results['correct'] += 1
        if actual in top3: results['top3'] += 1
        if actual in top5: results['top5'] += 1
        
        info = packer.unpack_token(traj[t])
        flags = info['flags']
        if (flags >> 5) & 1:
            level = 'special'
        elif (flags >> 0) & 1:
            level = 'word_start'
        elif (flags >> 1) & 1:
            level = 'word_end'
        else:
            level = 'mid_word'
        levels[level]['t'] += 1
        if pred == actual:
            levels[level]['c'] += 1
    
    count += 1
    if count >= 1000:
        break

elapsed = time.time() - t0
r = results
print(f"  Total calls: {r['total']}, time: {elapsed:.1f}s ({r['total']/elapsed:.0f} calls/s)")
print(f"  Top-1: {100*r['correct']/r['total']:.1f}% ({r['correct']}/{r['total']})")
print(f"  Top-3: {100*r['top3']/r['total']:.1f}% ({r['top3']}/{r['total']})")
print(f"  Top-5: {100*r['top5']/r['total']:.1f}% ({r['top5']}/{r['total']})")
print(f"\n  By level:")
for lv in ['special', 'word_start', 'mid_word', 'word_end']:
    d = levels[lv]
    if d['t'] > 0:
        print(f"    {lv:<15}: {100*d['c']/d['t']:.1f}% ({d['c']}/{d['t']})")

# ─── Show morph head patterns ───
print(f"\n{'─'*70}")
print("MORPHOLOGY: top-5 at each position for word_len=4")
print(f"{'─'*70}")
for pos in range(4):
    if 4 in heads.morph_logprob and pos in heads.morph_logprob[4]:
        arr = heads.morph_logprob[4][pos]
        top5 = np.argsort(-arr)[:5]
        items = [f"'{cv.decode([t])}' ({arr[t]:.1f})" for t in top5]
        print(f"  pos[{pos}]: {', '.join(items)}")

# ─── Show syntax head patterns ───
print(f"\n{'─'*70}")
print("SYNTAX: top-5 word-start tokens at word_num 0, 1, 5")
print(f"{'─'*70}")
for wn in [0, 1, 5]:
    if wn in heads.syntax_logprob:
        arr = heads.syntax_logprob[wn]
        top5 = np.argsort(-arr)[:5]
        items = [f"'{cv.decode([t])}' ({arr[t]:.1f})" for t in top5 if t < 161 or t >= 161]
        print(f"  word #{wn}: {', '.join(items)}")

print(f"\n{'─'*70}")
print("DONE")
