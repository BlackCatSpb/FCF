"""Reset vector space via FractalField reinit + clear lattice.

Usage:
    python reset_fractal.py
"""

import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import os, shutil, json
import numpy as np
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
cids = list(cs.concept_vectors.keys())
print(f"  Concepts: {len(cids)}, dim: {cs.dim}")

# 2. Reinitialize all vectors via fractal field
print("\nReinitializing all concept vectors via FractalField...")
cs.reinit_fractal()

# 3. Verify vector distribution
print("\nVerifying vector distribution:")
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

# 4. Save concept space (atomic overwrite)
cs_path_tmp = cs_path + '.tmp_reset'
cs.save(cs_path_tmp)
os.replace(cs_path_tmp, cs_path)
print(f"\n  Saved to {cs_path}")

# 5. Reset syntax lattice (fresh, empty)
print("\nResetting SyntaxLattice...")
lat_path = os.path.join(model_dir, 'syntax_lattice.json')
lat = SyntaxLattice()
lat.save(lat_path)
print(f"  Reset lattice saved to {lat_path}")

# 6. Remove checkpoint state (forces full retrain)
ckpt_state = os.path.join(model_dir, 'checkpoint_state.json')
if os.path.exists(ckpt_state):
    os.remove(ckpt_state)
    print("  Removed checkpoint_state.json")

# 7. Remove old numbered checkpoints
for f in os.listdir(model_dir):
    if any(f.startswith(p) for p in ['concept_space_', 'syntax_lattice_']):
        fpath = os.path.join(model_dir, f)
        os.remove(fpath)
        print(f"  Removed {f}")

# 8. Clear optimizer state
opt_state = os.path.join(model_dir, 'concept_space.opt.json')
if os.path.exists(opt_state):
    os.remove(opt_state)
    print("  Removed optimizer state")

print("\nDone! Model is ready for fresh training with clean vectors.")
print(f"FractalField: latent_dim={cs.fractal.latent_dim}, "
      f"{len(cs.fractal.codes)} codes, basis={cs.fractal.basis.shape}")
print("Run train.bat to start from scratch.")
