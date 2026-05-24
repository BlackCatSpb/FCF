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
# SymbolicGenerator: generate text from coordinates
# ============================================================
print("\n[GENERATE] Symbolic text generation from coordinate seeds...")

from eva.symbolic.symbolic_generator import SymbolicGenerator

generator = SymbolicGenerator(
    char_vocab=cv,
    unified_transformer=ut,
    coords=coords,
    temperature=0.8,
    max_length=30,
)

test_seeds = ["привет", "человек", "знания", "метаданные", "трансформер"]

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
# Coordinate-based navigation: direct path interpolation
# ============================================================
print("\n[NAVIGATE] Direct coordinate navigation (no external deps)...")

test_pairs = [
    ("привет", "пока"),
    ("человек", "машина"),
    ("солнце", "луна"),
]

for w1, w2 in test_pairs:
    ids1 = cv.encode(w1)[1:-1]
    ids2 = cv.encode(w2)[1:-1]
    if len(ids1) < 2 or len(ids2) < 2:
        continue
    
    c1 = coords[ids1].mean(dim=0)  # centroid
    c2 = coords[ids2].mean(dim=0)
    
    # Linear path with 5 waypoints
    n_pts = 5
    path = torch.stack([c1 + (i/(n_pts-1))*(c2-c1) for i in range(n_pts)])
    
    with torch.no_grad():
        dists = torch.cdist(path.to(DEVICE), coords)
        nearest = dists.argmin(dim=-1).clamp(1, VT-1)
        path_text = cv.decode(nearest.tolist())
    
    # Generate from each waypoint
    parts = []
    for wp in nearest[:4]:
        try:
            res = generator.generate(seed_ids=[wp.item()], max_new_tokens=6, top_k=20)
            parts.append(cv.decode(res))
        except:
            parts.append("?")
    
    print(f"  '{w1}' → '{w2}':")
    print(f"    Path: '{path_text}'")
    print(f"    Gen:  {' | '.join(parts)}")

print("\nDone.")
