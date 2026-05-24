"""
EVA — Navigate + Generate: full navigation through ℝ²⁴ topology.

Uses:
- geodesic_navigator: find shortest paths through coordinate manifold
- symbolic_generator: generate text from coordinates
- FractalAttention transformer: execute coordinate instructions

Pipeline:
  Concept A → geodesic path → Concept B → transformer → generated text
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

cv = CharacterVocab()
VT = 157

print("=" * 60)
print("EVA — Navigate + Generate")
print("=" * 60)
print(f"Device: {DEVICE}")

# ============================================================
# Load evolved model
# ============================================================
evolved = torch.load(os.path.join(CKPT_DIR, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(coords.to(DEVICE))

ckpt = torch.load(os.path.join(CKPT_DIR, "sentence_weights.pt"), map_location='cpu', weights_only=True)
ut.load_state_dict(ckpt['model'], strict=False)
ut.eval()
print("Model loaded")

# ============================================================
# GeodesicNavigator: find paths through ℝ²⁴
# ============================================================
print("\n[NAVIGATE] Geodesic navigation in ℝ²⁴...")

from eva.symbolic.geodesic_navigator import GeodesicNavigator, TangentSpace

# Build navigator from symbol coordinates
navigator = GeodesicNavigator(coords[1:VT].cpu().numpy(), k_neighbors=5)
print(f"  Navigator: {navigator.n_points} points, k={navigator.k_neighbors}")

# Test: find geodesic between two concepts
test_pairs = [
    ("привет", "пока"),
    ("человек", "машина"),
    ("солнце", "луна"),
    ("любовь", "страх"),
]

for w1, w2 in test_pairs:
    ids1 = cv.encode(w1)[1:-1]
    ids2 = cv.encode(w2)[1:-1]
    
    if len(ids1) < 2 or len(ids2) < 2:
        continue
    
    # Compute centroids
    c1 = coords[ids1].mean(dim=0).cpu().numpy()
    c2 = coords[ids2].mean(dim=0).cpu().numpy()
    
    try:
        path, path_dist = navigator.find_path(c1, c2, max_steps=10)
        if path is not None and len(path) >= 2:
            n_steps = len(path)
            print(f"  '{w1}' → '{w2}': {n_steps} steps, dist={path_dist:.3f}")
            
            # Convert path points to nearest symbols
            path_t = torch.tensor(np.array(path), dtype=torch.float32).to(DEVICE)
            dists = torch.cdist(path_t, coords)
            nearest = dists.argmin(dim=-1).clamp(1, VT-1)
            path_text = cv.decode(nearest.tolist())
            print(f"    Path: '{path_text}'")
        else:
            print(f"  '{w1}' → '{w2}': no path found")
    except Exception as e:
        print(f"  '{w1}' → '{w2}': error — {e}")

# ============================================================
# SymbolicGenerator: generate text from coordinate paths
# ============================================================
print("\n[GENERATE] Symbolic generation from coordinate paths...")

from eva.symbolic.symbolic_generator import SymbolicGenerator

generator = SymbolicGenerator(
    char_vocab=cv,
    unified_transformer=ut,
    coords=coords,
    temperature=0.8,
    max_length=30,
)

# Generate text from a seed
test_seeds = ["привет", "человек", "знания"]

for seed in test_seeds:
    seed_ids = cv.encode(seed)[1:-1]
    if len(seed_ids) < 2:
        continue
    
    try:
        result = generator.generate(
            seed_ids=seed_ids,
            max_new_tokens=12,
            top_k=20,
            top_p=0.9,
        )
        generated = cv.decode(result)
        print(f"  '{seed}...' → '{generated}'")
    except Exception as e:
        print(f"  '{seed}...' → error: {e}")

# ============================================================
# Full cycle: navigate + generate
# ============================================================
print("\n[INTEGRATE] Navigate → Generate full cycle...")

for w1, w2 in test_pairs[:2]:
    ids1 = cv.encode(w1)[1:-1]
    ids2 = cv.encode(w2)[1:-1]
    if len(ids1) < 2 or len(ids2) < 2:
        continue
    
    c1 = coords[ids1].mean(dim=0).cpu().numpy()
    c2 = coords[ids2].mean(dim=0).cpu().numpy()
    
    try:
        path, _ = navigator.find_path(c1, c2, max_steps=8)
        if path is not None and len(path) >= 3:
            # Generate text starting from each waypoint
            path_t = torch.tensor(np.array(path), dtype=torch.float32).to(DEVICE)
            dists = torch.cdist(path_t, coords)
            waypoints = dists.argmin(dim=-1).clamp(1, VT-1).tolist()
            
            generated_parts = []
            for wp in waypoints[:5]:
                seed = [wp]
                try:
                    res = generator.generate(seed_ids=seed, max_new_tokens=8, top_k=20)
                    gen = cv.decode(res)
                    generated_parts.append(gen)
                except:
                    generated_parts.append("?")
            
            print(f"  '{w1}' → '{w2}':")
            print(f"    Path: {cv.decode(waypoints[:5])}")
            print(f"    Gen:  {' | '.join(generated_parts)}")
    except Exception as e:
        print(f"  '{w1}' → '{w2}': error — {e}")

print("\nDone.")
