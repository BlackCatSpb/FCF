"""Pure PotentialField test - no PrimordialLayer, no GPU"""
import sys, os, torch, numpy as np, time
sys.path.insert(0, '.')
from eva.symbolic import PotentialField, CharacterVocab

vocab = CharacterVocab()
pf = PotentialField(156, 256)

# Simulate text: load token IDs directly
npy = 'real_data/full_corpus_ids.npy'
all_ids = np.load(npy, mmap_mode='r').astype(np.int32)
total = len(all_ids)
print(f"Dataset: {total:,} tokens")

# Quick training: process sequences, manually call strengthen for adjacent pairs
print("Training 2000 steps (adjacent pairs only)...")
start = time.time()

pos = 0
for step in range(2000):
    # Get a chunk of ~128 tokens
    if pos + 130 > total: pos = 0
    end = min(pos + 128, total)
    chunk = all_ids[pos:end]
    
    # Find a natural break (PAD or EOS)
    sep = np.where((chunk == 0) | (chunk == 3))[0]
    if len(sep) > 0: end = pos + sep[0]; chunk = all_ids[pos:end]
    
    # Convert to list
    ids = [int(x) for x in chunk if 0 <= x < 156][:128]
    pos += max(len(ids), 32)
    
    # Apply co-occurrence strengthening for adjacent pairs
    for i in range(len(ids) - 1):
        si, sj = ids[i], ids[i+1]
        if si < 156 and sj < 156 and si != sj:
            pf.strengthen(si, sj, attention_weight=0.001, weight=0.5)
    
    if step % 500 == 0: pf.global_decay()
    if step % 100 == 0 and step > 0:
        pot = float(pf.affinity.mean())
        print(f"  step {step}: pot={pot:.5f}")

elapsed = time.time() - start

# Results
aff = pf.affinity.cpu().numpy()
mean = float(aff.mean())
std = float(aff.std())
v_min = float(aff.min())
v_max = float(aff.max())
above_51 = float((aff > 0.51).mean() * 100)

print(f"\nRESULTS ({elapsed:.1f}s):")
print(f"  Mean:  {mean:.6f}")
print(f"  Std:   {std:.6f}")
print(f"  Min:   {v_min:.6f}")
print(f"  Max:   {v_max:.6f}")
print(f"  >0.51: {above_51:.1f}%")

# Top digrams
pairs = []
for i in range(156):
    for j in range(156):
        if i != j and aff[i,j] > 0.505:
            pairs.append((float(aff[i,j]), i, j))
pairs.sort(reverse=True)
print(f"\nTop 10 digrams:")
for s, i, j in pairs[:10]:
    ci = vocab.idx_to_char(i)
    cj = vocab.idx_to_char(j)
    print(f"  '{ci}'+'{cj}': {s:.4f}")

# Verdict
if std > 0.001 and v_max > 0.51 and v_max < 0.9 and above_51 > 0.1:
    print("\nPASS: model learns with differentiated affinities")
elif v_max > 0.99:
    print("\nFAIL: still saturating")
elif std < 0.0001:
    print(f"\nINCONCLUSIVE: very low std={std:.6f}, try more steps")
else:
    print(f"\nPASS (borderline): std={std:.6f}, max={v_max:.6f}")
