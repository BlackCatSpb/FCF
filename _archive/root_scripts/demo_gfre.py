"""
EVA — Full GFRE Demo: reason + reflect + discover + cross-modal.

Демонстрирует все новые возможности агента.
"""

import torch, torch.nn.functional as F, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.self_reflection import SelfReflection
from eva.symbolic.active_causal import ActiveLearner, CausalDiscovery
from eva.symbolic.trajectory_store import TrajectoryStore

cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — GFRE Demonstration")
print("=" * 60)

# Load model
ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=64, num_levels=4,
    scales_per_level=4, num_layers=3, d_ff=128).to(DEVICE)

# Try GFRE weights first, then v2
for ckpt_name in ["gfre_latest.pt", "v2_latest.pt"]:
    ckpt_path = os.path.join(CKPT, ckpt_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if 'ut' in ckpt:
            ut.load_state_dict(ckpt['ut'], strict=False)
        else:
            ut.load_state_dict(ckpt['model'], strict=False)
        print(f"Loaded: {ckpt_name}")
        break

# Load coordinates
evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)
coords64 = torch.zeros(157, 64, device=DEVICE); coords64[:, :24] = coords[:, :24]
g = torch.Generator(device=DEVICE).manual_seed(42)
coords64[:, 24:] = torch.randn(157, 40, generator=g, device=DEVICE) * 0.02
coords64 = coords64 / coords64.norm(dim=-1, keepdim=True).clamp(1e-8)
ut.set_symbol_coordinates(coords64)
ut.eval()

# ============================================================
# 1. REASONING — multi-hypothesis gradient flow
# ============================================================
print("\n[1. REASONING] Multi-hypothesis query...")

queries = ["привет", "человек", "солнце", "метаданные"]
for q in queries:
    ids = cv.encode(q)[1:-1]
    if len(ids) < 2: continue
    
    result = ut.reason(ids, num_hypotheses=3, temperature=0.15, char_vocab=cv)
    print(f"  Query: '{q}'")
    print(f"    Answer:    '{result['answer']}' (conf={result['confidence']:.2f})")
    if result['alternatives']:
        print(f"    Alt:       '{result['alternatives']}'")
    print(f"    Path len:  {result['all_hypotheses'][0]['path_length']} steps")

# ============================================================
# 2. SELF-REFLECTION — trajectory quality analysis
# ============================================================
print("\n[2. SELF-REFLECTION] Trajectory diagnostics...")

reflector = SelfReflection()
# Analyze the reasoning trace from the first query
for i, h in enumerate(result['all_hypotheses'][:3]):
    diag = reflector.diagnose(h['trajectory'])
    print(f"  Hyp {i}: len={diag.length}, curv={diag.mean_curvature:.4f}, "
          f"eff={diag.efficiency:.3f}, conf={diag.confidence:.3f}")

# ============================================================
# 3. ACTIVE LEARNING — uncertainty detection
# ============================================================
print("\n[3. ACTIVE LEARNING] Uncertainty check...")

learner = ActiveLearner(entropy_threshold=2.0, confidence_threshold=0.5)
for i, h in enumerate(result['all_hypotheses'][:3]):
    diag = reflector.diagnose(h['trajectory'])
    should, urgency, reason = learner.should_query(diagnostic=diag)
    print(f"  Hyp {i}: query={'YES' if should else 'no'} (urg={urgency:.2f}) "
          f"→ {reason}")

# ============================================================
# 4. CAUSAL DISCOVERY — trajectory pattern inference
# ============================================================
print("\n[4. CAUSAL DISCOVERY] Pattern inference...")

store_path = os.path.join(CKPT, "trajectory_store.pkl")
if os.path.exists(store_path):
    store = TrajectoryStore().load(store_path)
    causal = CausalDiscovery(store)
    
    for q in queries[:2]:
        ids = cv.encode(q)[1:-1]
        links = causal.discover_from_store(ids, top_k=5)
        if links:
            print(f"  '{q}' → {links[:3]}")
        else:
            print(f"  '{q}' → no strong causal links")
else:
    print("  TrajectoryStore not found — skip")

# ============================================================
# 5. CROSS-MODAL — image/sound → ℝ²⁴ stub
# ============================================================
print("\n[5. CROSS-MODAL] ℝ²⁴ as universal embedding space...")

class CrossModalEncoder:
    """Stub: encode anything → ℝ²⁴ (text, image, sound → same space)."""
    def __init__(self, coords):
        self.coords = coords  # [V, 64]
    
    def encode_text(self, text):
        ids = cv.encode(text)[1:-1]
        return self.coords[ids].mean(dim=0)
    
    def encode_random(self, seed=0):
        """Stub for image/sound encoding: seed-based random point."""
        g = torch.Generator().manual_seed(seed)
        return torch.randn(64, generator=g).to(self.coords.device)

cm = CrossModalEncoder(coords64)

# Encode same concept as text and "image" (random seed)
text_z = cm.encode_text("солнце")
image_z = cm.encode_random(42)

# Distance between text encoding and simulated image encoding
dist = (text_z - image_z).norm().item()
print(f"  Text('солнце') vs Image(seed=42): dist={dist:.3f}")
print(f"  (Same concept → closer in ℝ²⁴ with trained cross-modal encoder)")

print("\nDone.")
