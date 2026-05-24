"""
EVA — ConceptFinder. 
Learns potential V(z): ℝ²⁴ → ℝ over coordinate space.
Concepts = local minima of V(z) (regions where many trajectories converge).
Contradictions = regions of potential barrier between concepts.
New instructions = saddle points between minima.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, sys, os, time

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

# ============================================================
# Potential Function V(z): ℝ²⁴ → ℝ, 3-layer MLP
# ============================================================
class PotentialFunction(nn.Module):
    """Learned scalar potential over coordinate space. Low = frequent region, High = unexplored."""
    
    def __init__(self, dim=24, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, z):
        return self.net(z).squeeze(-1)  # [...] → [...] (scalar)
    
    def gradient(self, z):
        """∇V(z) via autograd."""
        z = z.detach().requires_grad_(True)
        v = self(z)
        grad = torch.autograd.grad(v.sum(), z, create_graph=True)[0]
        return grad
    
    def find_minimum(self, z0, steps=50, lr=0.01):
        """Gradient descent from z0 to nearest local minimum."""
        with torch.enable_grad():
            z = z0.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                loss = self(z).sum()
                loss.backward()
                opt.step()
        return z.detach()
    
    def find_saddle(self, za, zb, n_points=20):
        """Find the maximum potential along line za→zb (candidate saddle)."""
        t = torch.linspace(0, 1, n_points, device=za.device)
        # za, zb: [D] vectors
        za_flat = za.view(-1); zb_flat = zb.view(-1)
        points = za_flat.unsqueeze(0) + t.unsqueeze(1) * (zb_flat - za_flat).unsqueeze(0)  # [n, D]
        vals = self(points)
        idx = vals.argmax()
        return points[idx], vals[idx]

# ============================================================
# Training: V(z) low at trajectory points, high elsewhere
# ============================================================
print("=" * 60)
print("EVA — ConceptFinder: Potential V(z) over ℝ²⁴")
print("=" * 60)
print(f"Device: {DEVICE}")

# Load existing coordinates and word weights
from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()
VT = 157

# Load coordinates
coords_ckpt = torch.load(os.path.join(CKPT_DIR, "word_weights.pt"), map_location='cpu', weights_only=True)
coords = coords_ckpt['coords'].to(DEVICE)  # [157, 24]
print(f"Coords loaded: {coords.shape}")

# Load corpus for trajectory data
npy_file = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy_file):
    npy_file = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)
print(f"Corpus: {len(all_ids)/1e6:.1f}M tokens")

# Build potential model
STEPS = 5000
BATCH = 2048

pf_model = PotentialFunction(dim=24, hidden=128).to(DEVICE)
opt = torch.optim.AdamW(pf_model.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

print(f"Model: {sum(p.numel() for p in pf_model.parameters()):,} parameters")
print()

# ============================================================
# Training loop: anchored MSE — V(real) → -1, V(rand) → +1
# ============================================================
total_ids = len(all_ids)

start = time.time()
last_print = 0
rng = np.random.RandomState(99)

for step in range(1, STEPS + 1):
    starts = rng.randint(0, max(1, total_ids - 3), BATCH)
    real_ids = np.array([all_ids[s] for s in starts])
    valid = (real_ids > 0) & (real_ids < VT)
    real_ids = real_ids[valid]
    
    if len(real_ids) < 32:
        continue
    
    real_z = coords[torch.from_numpy(real_ids).long().to(DEVICE)]  # [N, 24]
    
    rand_z = torch.randn(BATCH, 24, device=DEVICE)
    rand_z = rand_z / rand_z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    v_real = pf_model(real_z[:BATCH])
    v_rand = pf_model(rand_z)
    
    # Bounded targets: V(real) → -1, V(rand) → +1
    loss_real = ((v_real + 1.0) ** 2).mean()
    loss_rand = ((v_rand - 1.0) ** 2).mean()
    loss = loss_real + loss_rand
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pf_model.parameters(), 1.0)
    opt.step()
    sch.step()
    
    now = time.time()
    if now - last_print >= 3 or step == 1 or step == STEPS:
        last_print = now
        elapsed = now - start
        eta = (elapsed / step) * (STEPS - step)
        with torch.no_grad():
            v_real_avg = pf_model(real_z[:256]).mean().item()
            v_rand_avg = pf_model(rand_z[:256]).mean().item()
            v_symbols = pf_model(coords[1:VT]).mean().item()  # avg V at symbol coords
        lr = sch.get_last_lr()[0]
        print(f"  step {step:>5d}/{STEPS} | loss={loss.item():.4f} | "
              f"V_real={v_real_avg:.3f} V_rand={v_rand_avg:.3f} V_sym={v_symbols:.3f} | "
              f"lr={lr:.6f} | {elapsed:.0f}s / eta {eta:.0f}s", flush=True)

# ============================================================
# Analysis: Direct coordinate clustering → concepts
# ============================================================
print("\n[ANALYSIS] Concept discovery from coordinate topology...")

sym_coords = coords[1:VT]  # [156, 24] — symbol coordinates
sym_coords_np = sym_coords.cpu().numpy()

# Use k-means on coordinates to find concept clusters
from sklearn.cluster import KMeans

N_CONCEPTS = 8  # target number of concepts
kmeans = KMeans(n_clusters=N_CONCEPTS, random_state=42, n_init=10)
labels = kmeans.fit_predict(sym_coords_np)

# Group symbols by concept
concept_groups = [[] for _ in range(N_CONCEPTS)]
for idx in range(156):
    concept_groups[labels[idx]].append(idx)

# Sort groups by size
concept_groups.sort(key=len, reverse=True)

print(f"\n  Found {N_CONCEPTS} concept clusters:")
for ci, group in enumerate(concept_groups):
    chars = ''.join(cv.decode([g+1]) for g in group)
    sym_ids = [g+1 for g in group]
    # Compute centroid
    centroid = sym_coords[group].mean(dim=0)
    # Average V at centroid
    with torch.no_grad():
        v_centroid = pf_model(centroid.to(DEVICE)).item()
    print(f"  Concept {ci}: [{len(group):>3d} symbols] V={v_centroid:.4f} | '{chars}'")

# ============================================================
# Saddle barriers between concepts
# ============================================================
print("\n  Potential barriers between concepts:")
with torch.no_grad():
    count = 0
    for i in range(min(6, len(concept_groups))):
        for j in range(i+1, min(i+3, len(concept_groups))):
            if len(concept_groups[i]) == 0 or len(concept_groups[j]) == 0:
                continue
            za = sym_coords[concept_groups[i][:1]].to(DEVICE)  # [1, 24]
            zb = sym_coords[concept_groups[j][:1]].to(DEVICE)  # [1, 24]
            saddle, v_sad = pf_model.find_saddle(za, zb, n_points=50)
            v_a = pf_model(za).item()
            v_b = pf_model(zb).item()
            barrier = v_sad.item() - max(v_a, v_b)
            chi = cv.decode([concept_groups[i][0]+1])
            chj = cv.decode([concept_groups[j][0]+1])
            print(f"    '{chi}' ↔ '{chj}': barrier={barrier:.4f} "
                  f"(V_a={v_a:.4f}, V_b={v_b:.4f}, V_saddle={v_sad.item():.4f})")
            count += 1
            if count >= 10:
                break
        if count >= 10:
            break

# Save
pf_path = os.path.join(CKPT_DIR, "potential_function.pt")
torch.save({'model': pf_model.state_dict(), 'coords': coords, 'concept_labels': labels}, pf_path)
print(f"\nSaved: {pf_path}")
print("Done.")
