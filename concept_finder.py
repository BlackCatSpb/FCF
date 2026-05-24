"""
EVA v8 — Step 1: ConceptFinder. 
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
        points = za.unsqueeze(0) + t.unsqueeze(1) * (zb - za).unsqueeze(0)
        vals = self(points)
        idx = vals.argmax()
        return points[idx], vals[idx]

# ============================================================
# Training: V(z) low at trajectory points, high elsewhere
# ============================================================
print("=" * 60)
print("EVA v8 — ConceptFinder: Potential V(z) over ℝ²⁴")
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
pf_model = PotentialFunction(dim=24, hidden=128).to(DEVICE)
opt = torch.optim.AdamW(pf_model.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10000)

print(f"Model: {sum(p.numel() for p in pf_model.parameters()):,} parameters")
print()

# ============================================================
# Training loop: contrastive — low V at real points, high V at random
# ============================================================
STEPS = 10000
BATCH = 2048
MARGIN = 1.0
total_ids = len(all_ids)

start = time.time()
last_print = 0
rng = np.random.RandomState(99)

for step in range(1, STEPS + 1):
    # Sample REAL trajectory points from corpus
    starts = rng.randint(0, max(1, total_ids - 3), BATCH)
    real_ids = np.array([all_ids[s] for s in starts])
    valid = (real_ids > 0) & (real_ids < VT)
    real_ids = real_ids[valid]
    
    if len(real_ids) < 32:
        continue
    
    real_z = coords[torch.from_numpy(real_ids).long().to(DEVICE)]  # [N, 24]
    
    # Sample RANDOM points in ℝ²⁴ (uniform on sphere)
    rand_z = torch.randn(BATCH, 24, device=DEVICE)
    rand_z = rand_z / rand_z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Contrastive loss: real should be BELOW margin, random ABOVE margin
    v_real = pf_model(real_z[:BATCH])
    v_rand = pf_model(rand_z)
    
    # Hinge loss: V(real) should be low, V(rand) should be high
    loss = v_real.mean() + F.relu(MARGIN - v_rand).mean()
    
    # Bonus: push real points toward each other (clustering)
    if len(real_z) >= 64:
        v_all = pf_model(real_z[:64])
        loss = loss + 0.01 * v_all.std()  # encourage similar V within batch
    
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
# Analysis: Find concepts (local minima of V)
# ============================================================
print("\n[ANALYSIS] Finding concepts in ℝ²⁴...")
pf_model.eval()

# Method 1: Find minima starting from each symbol coordinate
print("\n  Minima from symbol coordinates:")
with torch.no_grad():
    for i in range(1, min(VT, 20)):  # first 19 symbols
        z0 = coords[i:i+1].to(DEVICE)
        z_min = pf_model.find_minimum(z0, steps=30, lr=0.05)
        v_min = pf_model(z_min).item()
        v_start = pf_model(z0).item()
        char = cv.decode([i])
        moved = (z_min - z0).norm().item()
        print(f"    [{i:>3d}] '{char}' V={v_start:.3f} → min V={v_min:.4f} (moved {moved:.3f})")

# Method 2: Cluster symbols by their minima (group into concepts)
print("\n  Symbol clustering into concepts:")
all_minima = []
with torch.no_grad():
    for i in range(1, VT):
        z0 = coords[i:i+1].to(DEVICE)
        z_min = pf_model.find_minimum(z0, steps=30, lr=0.05)
        all_minima.append(z_min)

all_minima = torch.cat(all_minima, dim=0)  # [156, 24]

# Group symbols whose minima are close (< threshold)
threshold = 0.15
concept_groups = []
used = set()

for i in range(len(all_minima)):
    if i in used:
        continue
    group = [i]
    used.add(i)
    for j in range(i+1, len(all_minima)):
        if j in used:
            continue
        dist = (all_minima[i] - all_minima[j]).norm().item()
        if dist < threshold:
            group.append(j)
            used.add(j)
    concept_groups.append(group)

print(f"  Found {len(concept_groups)} concept groups (threshold={threshold}):")
for gi, group in enumerate(concept_groups[:15]):
    chars = ''.join(cv.decode([g+1]) for g in group)
    print(f"    Group {gi}: [{len(group)} symbols] '{chars}'")

# ============================================================
# Method 3: Find saddle points between concepts
# ============================================================
print("\n  Saddle points between concept groups:")
if len(concept_groups) >= 2:
    for i in range(min(3, len(concept_groups))):
        for j in range(i+1, min(i+4, len(concept_groups))):
            za = all_minima[concept_groups[i][0]:concept_groups[i][0]+1]
            zb = all_minima[concept_groups[j][0]:concept_groups[j][0]+1]
            saddle, v_sad = pf_model.find_saddle(za, zb, n_points=50)
            v_a = pf_model(za).item()
            v_b = pf_model(zb).item()
            barrier = v_sad.item() - max(v_a, v_b)
            chi = cv.decode([concept_groups[i][0]+1])
            chj = cv.decode([concept_groups[j][0]+1])
            print(f"    '{chi}' ↔ '{chj}': barrier={barrier:.4f} "
                  f"(V_a={v_a:.4f}, V_b={v_b:.4f}, V_saddle={v_sad.item():.4f})")

# Save
pf_path = os.path.join(CKPT_DIR, "potential_function.pt")
torch.save({'model': pf_model.state_dict(), 'coords': coords}, pf_path)
print(f"\nSaved: {pf_path}")
print("Done.")
