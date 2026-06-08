"""Reset vector space via FractalField + clear training data.

Usage:
    python reset_fractal.py
"""

import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import os, shutil
import numpy as np
from itertools import combinations
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice

model_dir = r'C:\Users\black\OneDrive\Desktop\FCF\real_data'

print("=" * 60)
print("RESET: Fractal reinit + clear training data")
print("=" * 60)

# 1. Load concept space
cs_path = os.path.join(model_dir, 'concept_space.json')
print(f"\nLoading ConceptSpace from {cs_path}...")
cs = ConceptSpace.load(cs_path)
n = len(cs.cid_list)
print(f"  Concepts: {n}, dim: {cs.dim}")

# 2. Reinitialize all vectors via fractal field
print("\nReinitializing all concept vectors via FractalField...")
cs.reinit_fractal()

# 3. Verify vector distribution
print("\nVerifying vector distribution:")
cids = cs.cid_list
sample = min(80, len(cids))
rng = np.random.RandomState(42)
idxs = rng.choice(len(cids), sample, replace=False)
sampled = [cids[i] for i in idxs]

sims = []
for i in range(len(sampled)):
    for j in range(i + 1, len(sampled)):
        va = cs.concept_vectors[sampled[i]]
        vb = cs.concept_vectors[sampled[j]]
        sim = float(np.dot(va, vb))
        sims.append(sim)
sims = np.array(sims)
print(f"  Pairwise cos sim: mean={sims.mean():.4f} std={sims.std():.4f} "
      f"min={sims.min():.4f} max={sims.max():.4f}")

# Quick specific pair check
test_pairs = [('человек', 'война'), ('любовь', 'ненависть'), ('собака', 'кошка')]
print(f"  Specific pairs:")
for a, b in test_pairs:
    cid_a = cs.word_to_cid.get(a)
    cid_b = cs.word_to_cid.get(b)
    if cid_a is not None and cid_b is not None:
        va = cs.concept_vectors.get(cid_a)
        vb = cs.concept_vectors.get(cid_b)
        if va is not None and vb is not None:
            sim = float(np.dot(va, vb))
            print(f"    sim({a}, {b}) = {sim:.4f}")

# 4. Clear concept_transitions
print("\nClearing concept transitions...")
cs.concept_transitions = None

# 5. Save concept space
cs_path_tmp = cs_path + '.tmp_reset'
cs.save(cs_path_tmp)
os.replace(cs_path_tmp, cs_path)
print(f"  Saved to {cs_path}")

# 6. Reset syntax lattice
print("\nResetting SyntaxLattice...")
lat_path = os.path.join(model_dir, 'syntax_lattice.json')
lat = SyntaxLattice()
lat.save(lat_path)
print(f"  Reset lattice saved to {lat_path}")

# 7. Clear checkpoint directories
ckpt_dir = os.path.join(model_dir, '..', 'checkpoints') if 'checkpoints' not in model_dir else None
if ckpt_dir and os.path.isdir(ckpt_dir):
    print(f"\nClearing checkpoints in {ckpt_dir}...")
    for f in os.listdir(ckpt_dir):
        fpath = os.path.join(ckpt_dir, f)
        if os.path.isfile(fpath):
            if f.endswith('.json'):
                print(f"  Removing {f}")
                os.remove(fpath)

print("\nDone! The model is now ready for fresh training with clean vectors.")
print(f"FractalField: latent_dim={cs.fractal.latent_dim}, {len(cs.fractal.codes)} codes, basis={cs.fractal.basis.shape}")
