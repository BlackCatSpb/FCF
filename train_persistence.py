"""
EVA — TopologicalPersistence: robustness of concepts under perturbations.

Measures which concepts survive noise, which collapse into others.
High persistence = stable feature of topology.
Low persistence = fragile/transitional concept.
"""

import torch, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()
VT = 157

print("=" * 60)
print("EVA — TopologicalPersistence")
print("=" * 60)

# ============================================================
# Load data
# ============================================================
evolved = torch.load(os.path.join(CKPT_DIR, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)
affinity = evolved['affinity']

pf_data = torch.load(os.path.join(CKPT_DIR, "potential_function.pt"), map_location='cpu', weights_only=False)
from eva.symbolic.potential_function import PotentialFunction
v_func = PotentialFunction(dim=24, hidden=128).to(DEVICE)
v_func.load_state_dict(pf_data['model'])
concept_labels = pf_data['concept_labels']
n_concepts = len(set(concept_labels))
print(f"Loaded: {n_concepts} concepts, V(z) model")

# ============================================================
# Compute concept centroids and stats
# ============================================================
centroids = []
concept_syms = []
sym_coords = coords[1:VT]  # [156, 24]

for ci in range(n_concepts):
    mask = concept_labels == ci
    indices = np.where(mask)[0]
    centroid = sym_coords[indices].mean(dim=0)
    centroids.append(centroid)
    concept_syms.append([int(s)+1 for s in indices])

centroids_t = torch.stack(centroids).to(DEVICE)  # [C, 24]

# Concept radius: max distance from centroid to member
concept_radii = []
for ci in range(n_concepts):
    members = sym_coords[concept_labels == ci]
    dists = torch.norm(members - centroids_t[ci], dim=1)
    concept_radii.append(dists.max().item())

print(f"\nConcepts and radii:")
for ci in range(n_concepts):
    chars = ''.join(cv.decode([concept_syms[ci][0]]))
    print(f"  {ci}: [{len(concept_syms[ci]):>3d}] '{chars}...' radius={concept_radii[ci]:.3f}")

# ============================================================
# Persistence test: perturb centroids, check recovery
# ============================================================
print("\n[PERSISTENCE] Testing concept stability under noise...")

n_trials = 200
noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
rng = np.random.RandomState(123)

persistence_scores = np.zeros((n_concepts, len(noise_levels)))

for ci in range(n_concepts):
    centroid = centroids_t[ci].cpu().numpy()
    
    for ni, noise_std in enumerate(noise_levels):
        recovered = 0
        
        for _ in range(n_trials):
            # Add Gaussian noise
            noise = rng.randn(24) * noise_std
            perturbed = centroid + noise
            
            # Find nearest symbol
            perturbed_t = torch.tensor(perturbed, dtype=torch.float32).to(DEVICE)
            dists = torch.norm(sym_coords - perturbed_t, dim=1)
            nearest_idx = dists.argmin().item()
            
            # Check if nearest is still in same concept
            if concept_labels[nearest_idx] == ci:
                recovered += 1
        
        persistence_scores[ci, ni] = recovered / n_trials

# ============================================================
# Results
# ============================================================
print(f"\n  Persistence scores (noise_level → recovery rate):")
print(f"  {'Concept':>8s} {'Sym':>5s}", end='')
for nl in noise_levels:
    print(f" {nl:>6.2f}", end='')
print(f" {'R_50':>6s}")

for ci in range(n_concepts):
    ch = cv.decode([concept_syms[ci][0]])
    print(f"  {ch:>8s} [{len(concept_syms[ci]):>3d}]", end='')
    for ni in range(len(noise_levels)):
        marker = '*' if persistence_scores[ci, ni] >= 0.5 else ' '
        print(f" {persistence_scores[ci, ni]:>5.2f}{marker}", end='')
    
    # Find noise level at which persistence drops below 50%
    r50 = noise_levels[-1]
    for ni in range(len(noise_levels)):
        if persistence_scores[ci, ni] < 0.5:
            r50 = noise_levels[ni-1] if ni > 0 else noise_levels[0]
            break
    print(f" {r50:>6.2f}")

# ============================================================
# Analysis: most and least persistent concepts
# ============================================================
# Persistence at noise=0.2 (moderate perturbation)
noise_idx = 3  # 0.2
scores_02 = persistence_scores[:, noise_idx]
ranking = np.argsort(scores_02)[::-1]

print(f"\n  Most persistent concepts (noise=0.2):")
for ri in range(min(5, n_concepts)):
    ci = ranking[ri]
    ch = cv.decode([concept_syms[ci][0]])
    chars_sample = ''.join(cv.decode([s]) for s in concept_syms[ci][:8])
    print(f"    {ri+1}. '{ch}...' [{len(concept_syms[ci])}]: {scores_02[ci]:.1%} "
          f"recovery, radius={concept_radii[ci]:.3f} — '{chars_sample}'")

print(f"\n  Least persistent concepts (collapsed by noise):")
for ri in range(min(5, n_concepts)):
    ci = ranking[-(ri+1)]
    ch = cv.decode([concept_syms[ci][0]])
    chars_sample = ''.join(cv.decode([s]) for s in concept_syms[ci][:8])
    print(f"    {ri+1}. '{ch}...' [{len(concept_syms[ci])}]: {scores_02[ci]:.1%} "
          f"recovery, radius={concept_radii[ci]:.3f} — '{chars_sample}'")

# ============================================================
# Test: where do perturbed points collapse to?
# ============================================================
print(f"\n  Collapse destinations (noise=0.5):")
noise_std = 0.5
collapse_counts = np.zeros((n_concepts, n_concepts))  # [from, to]

for ci in range(n_concepts):
    centroid = centroids_t[ci].cpu().numpy()
    for _ in range(n_trials):
        noise = rng.randn(24) * noise_std
        perturbed = centroid + noise
        perturbed_t = torch.tensor(perturbed, dtype=torch.float32).to(DEVICE)
        dists = torch.norm(sym_coords - perturbed_t, dim=1)
        nearest_idx = dists.argmin().item()
        collapse_counts[ci, concept_labels[nearest_idx]] += 1

# Show top collapses
for ci in range(n_concepts):
    for cj in range(n_concepts):
        if ci != cj and collapse_counts[ci, cj] > 0:
            pct = collapse_counts[ci, cj] / n_trials
            if pct > 0.05:
                chi = cv.decode([concept_syms[ci][0]])
                chj = cv.decode([concept_syms[cj][0]])
                print(f"    '{chi}' → '{chj}': {pct:.1%} of trials")

# Save
tp_path = os.path.join(CKPT_DIR, "topological_persistence.pt")
torch.save({
    'persistence_scores': persistence_scores,
    'noise_levels': noise_levels,
    'concept_radii': concept_radii,
    'collapse_counts': collapse_counts,
}, tp_path)
print(f"\nSaved: {tp_path}")
print("Done.")
