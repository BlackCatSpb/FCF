"""
EVA — GradientFlow + InstructionGenerator.

Навигация в ℝ²⁴: потенциал V(z), седловые точки, новые траектории.
1. GradientFlow: ∇V(z) → спуск к минимумам (концептам)
2. SaddlePoint: максимум V на линии между концептами → кандидат на новую связь
3. InstructionGenerator: траектория через седловую точку → новый текст
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()
VT = 157

print("=" * 60)
print("EVA — GradientFlow + InstructionGenerator")
print("=" * 60)

# ============================================================
# Load data
# ============================================================
evolved = torch.load(os.path.join(CKPT_DIR, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)  # [157, 24]
affinity = evolved['affinity']

from eva.symbolic.potential_function import PotentialFunction
pf_data = torch.load(os.path.join(CKPT_DIR, "potential_function.pt"), map_location='cpu', weights_only=False)
v_func = PotentialFunction(dim=24, hidden=128).to(DEVICE)
v_func.load_state_dict(pf_data['model'])
concept_labels = pf_data.get('concept_labels', None)
print(f"Loaded: coords {coords.shape}, V(z), concept_labels={concept_labels is not None}")

# ============================================================
# Phase 1: Find concept centroids (basins of attraction)
# ============================================================
print("\n[PHASE 1] Finding concept basins via gradient descent...")

# Use k-means labels if available, or recompute
if concept_labels is not None:
    labels = concept_labels
else:
    from sklearn.cluster import KMeans
    sym_np = coords[1:VT].cpu().numpy()
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sym_np)

n_concepts = len(set(labels))
concept_centroids = []
concept_symbols = []

with torch.no_grad():
    for ci in range(n_concepts):
        mask = labels == ci
        sym_indices = np.where(mask)[0]
        centroid = coords[1:VT][sym_indices].mean(dim=0)
        concept_centroids.append(centroid)
        concept_symbols.append([int(s) + 1 for s in sym_indices])

print(f"  Found {n_concepts} concepts:")
for ci in range(n_concepts):
    chars = ''.join(cv.decode(s) for s in concept_symbols[ci][:10])
    v_c = v_func(concept_centroids[ci].unsqueeze(0).to(DEVICE)).item()
    print(f"    Concept {ci}: [{len(concept_symbols[ci]):>3d} symbols] V={v_c:.4f} | '{chars}...'")

# ============================================================
# Phase 2: Find saddle points between concepts
# ============================================================
print("\n[PHASE 2] Finding saddle points (potential barriers)...")

saddles = []  # (ci, cj, saddle_point, barrier_height, v_saddle)

with torch.no_grad():
    for i in range(n_concepts):
        for j in range(i+1, n_concepts):
            za = concept_centroids[i].unsqueeze(0).to(DEVICE)
            zb = concept_centroids[j].unsqueeze(0).to(DEVICE)
            
            # Sample V along the line between centroids
            n_pts = 100
            t = torch.linspace(0, 1, n_pts, device=DEVICE)
            points = za + t.unsqueeze(1) * (zb - za)
            vals = v_func(points)
            
            idx_max = vals.argmax()
            v_saddle = vals[idx_max].item()
            saddle_z = points[idx_max]
            
            v_a = v_func(za).item()
            v_b = v_func(zb).item()
            barrier = v_saddle - max(v_a, v_b)
            
            saddles.append((i, j, saddle_z.cpu(), barrier, v_saddle, v_a, v_b))

# Sort saddles by barrier (highest barrier = most interesting transition)
saddles.sort(key=lambda x: x[3], reverse=True)

print(f"\n  Top saddles (highest barriers):")
for si, (ci, cj, sz, barrier, vs, va, vb) in enumerate(saddles[:10]):
    chi = cv.decode([concept_symbols[ci][0]])
    chj = cv.decode([concept_symbols[cj][0]])
    print(f"    {si}: '{chi}'↔'{chj}' barrier={barrier:.4f} "
          f"(V_a={va:.4f} V_b={vb:.4f} V_s={vs:.4f})")

# Also find LOW barrier saddles (easiest transitions = most natural)
saddles_low = sorted(saddles, key=lambda x: x[3])
print(f"\n  Easiest transitions (lowest barriers):")
for si, (ci, cj, sz, barrier, vs, va, vb) in enumerate(saddles_low[:10]):
    chi = cv.decode([concept_symbols[ci][0]])
    chj = cv.decode([concept_symbols[cj][0]])
    print(f"    {si}: '{chi}'↔'{chj}' barrier={barrier:.4f}")

# ============================================================
# Phase 3: Generate new instructions from saddle points
# ============================================================
print("\n[PHASE 3] InstructionGenerator: new trajectories from saddles...")

from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(coords.to(DEVICE))

# Load word weights
word_ckpt = torch.load(os.path.join(CKPT_DIR, "word_weights.pt"), map_location='cpu', weights_only=True)
ut.load_state_dict(word_ckpt['model'], strict=False)
ut.eval()

# For each saddle point, generate a trajectory:
# Start at concept A centroid → pass through saddle → end at concept B centroid
# Feed to transformer → see what symbols it produces

print("\n  Generating trajectories through saddle points...")

for si, (ci, cj, saddle_z, barrier, vs, va, vb) in enumerate(saddles_low[:5]):
    za = concept_centroids[ci].to(DEVICE)
    zb = concept_centroids[cj].to(DEVICE)
    sz = saddle_z.to(DEVICE)
    
    # Create trajectory: za → sz → zb (with interpolation)
    n_steps = 8
    traj = []
    for k in range(n_steps):
        if k < n_steps // 3:
            # First third: za → sz
            alpha = k / (n_steps // 3)
            traj.append(za + alpha * (sz - za))
        else:
            # Rest: sz → zb
            alpha = (k - n_steps // 3) / (n_steps - n_steps // 3)
            traj.append(sz + alpha * (zb - sz))
    traj = torch.stack(traj).unsqueeze(0).to(DEVICE)  # [1, 8, 24]
    
    # Instead of passing coordinates directly, find nearest symbol at each step
    # (the transformer expects symbol IDs, not raw coordinates)
    with torch.no_grad():
        # Find nearest symbol coordinate at each step
        dists = torch.cdist(traj[0], coords)  # [8, 157]
        nearest_ids = dists.argmin(dim=-1)  # [8]
        nearest_ids = nearest_ids.clamp(1, VT-1)  # ensure valid
        
        inp = nearest_ids.unsqueeze(0)  # [1, 8]
        _, scores = ut(inp, return_scores=True)
        pred_ids = scores[0].argmax(dim=-1).tolist()
        generated = cv.decode(pred_ids)
    
    chi = cv.decode([concept_symbols[ci][0]])
    chj = cv.decode([concept_symbols[cj][0]])
    print(f"    '{chi}'→'{chj}' (barrier={barrier:.4f}): nearest={cv.decode(nearest_ids.tolist())} | output='{generated}'")

# ============================================================
# Phase 4: Gradient navigation — walk through ℝ²⁴
# ============================================================
print("\n[PHASE 4] Gradient navigation: follow ∇V from random point to concept...")

with torch.enable_grad():
    # Start from random point on sphere
    z = torch.randn(1, 24, device=DEVICE)
    z = z / z.norm()
    z.requires_grad_(True)
    
    path = [z.detach().cpu().clone()]
    optim = torch.optim.Adam([z], lr=0.05)
    
    for step in range(30):
        optim.zero_grad()
        v = v_func(z)
        v.backward()
        optim.step()
        with torch.no_grad():
            z_norm = z.norm()
            if z_norm > 2:
                z.data = z.data / z_norm * 2.0
        path.append(z.detach().cpu().clone())
    
    path_t = torch.cat(path, dim=0)  # [31, 24]
    v_start = v_func(path_t[0:1].to(DEVICE)).item()
    v_end = v_func(path_t[-1:].to(DEVICE)).item()
    
    # Find nearest symbols at start and end
    dist_start = torch.cdist(path_t[0:1].to(DEVICE), coords)
    dist_end = torch.cdist(path_t[-1:].to(DEVICE), coords)
    nearest_start = dist_start.argmin(dim=-1).item()
    nearest_end = dist_end.argmin(dim=-1).item()
    
    path_len = sum((path_t[i] - path_t[i-1]).norm().item() for i in range(1, len(path_t)))
    
    print(f"\n  Gradient descent path:")
    print(f"    Start: V={v_start:.4f} → nearest='{cv.decode([nearest_start])}'")
    print(f"    End:   V={v_end:.4f} → nearest='{cv.decode([nearest_end])}'")
    print(f"    Path length: {path_len:.3f}, steps: {len(path)}")
    print(f"    Converged to concept: '{cv.decode([nearest_end])}'")

# Save
grad_path = os.path.join(CKPT_DIR, "gradient_flow.pt")
torch.save({
    'concept_centroids': [c.cpu() for c in concept_centroids],
    'concept_symbols': concept_symbols,
    'saddles': saddles,
}, grad_path)
print(f"\nSaved: {grad_path}")
print("Done.")
