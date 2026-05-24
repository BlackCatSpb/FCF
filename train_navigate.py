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
# Simple text generation via transformer autoregression
# ============================================================
print("\n[GENERATE] Transformer autoregressive generation...")

def generate_text(ut, coords, seed_ids, max_new=15, temperature=0.8, top_k=20):
    """Autoregressive text generation from seed symbol IDs."""
    ids = list(seed_ids)
    ut.eval()
    
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            _, scores = ut(inp, return_scores=True)
            logits = scores[0, -1] / temperature  # last position
            
            # Top-k filtering
            topk_vals, topk_idx = torch.topk(logits, min(top_k, len(logits)))
            probs = F.softmax(topk_vals, dim=-1)
            next_token = topk_idx[torch.multinomial(probs, 1)].item()
            
            if next_token <= 0 or next_token >= VT:
                next_token = topk_idx[0].item()  # fallback to most likely
            
            ids.append(next_token)
    
    return ids

test_seeds = ["привет", "человек", "знания", "метаданные", "трансформер"]

for seed in test_seeds:
    seed_ids = cv.encode(seed)[1:-1]
    if len(seed_ids) < 2:
        continue
    
    result = generate_text(ut, coords, seed_ids, max_new=15, temperature=0.7)
    generated = cv.decode(result)
    print(f"  '{seed}...' → '{generated}'")

# ============================================================
# Navigate + Generate: direct coordinate path
# ============================================================
print("\n[NAVIGATE] Path interpolation + generation...")

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
    
    c1 = coords[ids1].mean(dim=0)
    c2 = coords[ids2].mean(dim=0)
    
    n_pts = 5
    path = torch.stack([c1 + (i/(n_pts-1))*(c2-c1) for i in range(n_pts)])
    
    with torch.no_grad():
        dists = torch.cdist(path.to(DEVICE), coords)
        nearest = dists.argmin(dim=-1).clamp(1, VT-1)
        path_text = cv.decode(nearest.tolist())
    
    # Generate from middle waypoint
    mid_idx = n_pts // 2
    mid_wp = nearest[mid_idx].item()
    gen_ids = generate_text(ut, coords, [mid_wp], max_new=12, temperature=0.7)
    gen_text = cv.decode(gen_ids)
    
    print(f"  '{w1}' → '{w2}':")
    print(f"    Path: '{path_text}'")
    print(f"    Gen:  '{gen_text}'")

print("\nDone.")
