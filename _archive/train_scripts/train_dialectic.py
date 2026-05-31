"""
EVA — DialecticalSynthesis: thesis/antithesis → synthesis.

Hegelian dialectic in ℝ²⁴ coordinate space:
- Thesis = concept A (region in ℝ²⁴)
- Antithesis = concept B (region in ℝ²⁴)
- Contradiction = potential barrier between them
- Synthesis = new point beyond the saddle, creating a potential new concept
- Sublation = analysis: what is preserved, negated, transcended?

Test: syntheses that reduce barrier AND create new distinct regions are kept.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.potential_function import PotentialFunction

cv = CharacterVocab()
VT = 157

print("=" * 60)
print("EVA — DialecticalSynthesis")
print("=" * 60)

# ============================================================
# Load data
# ============================================================
evolved = torch.load(os.path.join(CKPT_DIR, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)
affinity = evolved['affinity']

pf_data = torch.load(os.path.join(CKPT_DIR, "potential_function.pt"), map_location='cpu', weights_only=False)
v_func = PotentialFunction(dim=24, hidden=128).to(DEVICE)
v_func.load_state_dict(pf_data['model'])
concept_labels = pf_data['concept_labels']
print(f"Loaded: {len(set(concept_labels))} concepts, V(z) model")

# ============================================================
# Concept centroids
# ============================================================
n_concepts = len(set(concept_labels))
centroids = []
concept_syms = []

for ci in range(n_concepts):
    mask = concept_labels == ci
    indices = np.where(mask)[0]
    centroid = coords[1:VT][indices].mean(dim=0)
    centroids.append(centroid)
    concept_syms.append([int(s)+1 for s in indices])

centroids_t = torch.stack(centroids).to(DEVICE)  # [C, 24]

print(f"\nConcepts:")
for ci in range(n_concepts):
    chars = ''.join(cv.decode([concept_syms[ci][0]]))
    v = v_func(centroids_t[ci:ci+1]).item()
    print(f"  {ci}: [{len(concept_syms[ci]):>3d}] '{chars}...' V={v:.4f}")

# ============================================================
# DialecticalSynthesis engine
# ============================================================
class DialecticalSynthesis:
    """Hegelian dialectic over coordinate space."""
    
    def __init__(self, v_func, coords, concept_centroids):
        self.v_func = v_func
        self.coords = coords
        self.centroids = concept_centroids  # [C, 24]
        self.syntheses = []  # list of (thesis, antithesis, synthesis_point, metrics)
    
    def contradiction_intensity(self, ci, cj):
        """Measure barrier height between concepts i and j."""
        za = self.centroids[ci:ci+1]
        zb = self.centroids[cj:cj+1]
        
        with torch.no_grad():
            n_pts = 50
            t = torch.linspace(0, 1, n_pts, device=za.device)
            points = za + t.unsqueeze(1) * (zb - za)
            vals = self.v_func(points)
            
            v_a = vals[0].item()
            v_b = vals[-1].item()
            v_max = vals.max().item()
            barrier = v_max - max(v_a, v_b)
            
        return barrier, v_a, v_b, v_max
    
    def synthesize(self, ci, cj, context_weight=0.5):
        """
        Generate synthesis: thesis (ci) + antithesis (cj) → synthesis.
        
        Method: start at saddle, gradient-descend to find new minimum.
        If the minimum is distinct from both thesis and antithesis, it's a synthesis.
        """
        za = self.centroids[ci:ci+1]
        zb = self.centroids[cj:cj+1]
        
        # Find saddle point
        with torch.no_grad():
            n_pts = 100
            t = torch.linspace(0, 1, n_pts, device=za.device)
            points = za + t.unsqueeze(1) * (zb - za)
            vals = self.v_func(points)
            saddle_idx = vals.argmax()
            saddle = points[saddle_idx:saddle_idx+1].clone()
        
        # Gradient descent from saddle to find minimum
        with torch.enable_grad():
            z = saddle.detach().clone().requires_grad_(True)
            optim = torch.optim.Adam([z], lr=0.03)
            history = []
            for _ in range(40):
                optim.zero_grad()
                v = self.v_func(z)
                v.backward()
                optim.step()
                history.append(z.detach().clone())
            
            synthesis = z.detach()
        
        # Metrics
        with torch.no_grad():
            v_syn = self.v_func(synthesis).item()
            v_saddle = vals[saddle_idx].item()
            v_a = self.v_func(za).item()
            v_b = self.v_func(zb).item()
            
            # Distances
            d_a = (synthesis - za).norm().item()
            d_b = (synthesis - zb).norm().item()
            d_saddle = (synthesis - saddle).norm().item()
            
            # Is synthesis distinct from both thesis and antithesis?
            distinct = (d_a > 0.3) and (d_b > 0.3)
            
            # Does synthesis improve potential (lower V)?
            improved = v_syn < min(v_a, v_b)
            
            # Sublation analysis
            preserved_a = max(0, 1.0 - d_a / (d_a + d_b + 1e-8))  # how much of A preserved
            preserved_b = max(0, 1.0 - d_b / (d_a + d_b + 1e-8))  # how much of B preserved
            transcended = d_saddle  # how far beyond saddle
            
            # Find nearest symbol
            dists_syn = torch.cdist(synthesis, self.coords)
            nearest_syn = dists_syn.argmin(dim=-1).item()
            nearest_char = cv.decode([nearest_syn])
        
        result = {
            'ci': ci, 'cj': cj,
            'synthesis': synthesis.cpu(),
            'v_syn': v_syn,
            'v_saddle': v_saddle,
            'v_a': v_a, 'v_b': v_b,
            'd_a': d_a, 'd_b': d_b,
            'd_saddle': d_saddle,
            'distinct': distinct,
            'improved': improved,
            'preserved_a': preserved_a,
            'preserved_b': preserved_b,
            'transcended': transcended,
            'nearest': nearest_syn,
            'nearest_char': nearest_char,
            'valid': distinct and improved,
        }
        
        self.syntheses.append(result)
        return result
    
    def run_all(self, min_barrier=0.3):
        """Run dialectic on all concept pairs with barrier > min_barrier."""
        results = []
        n = len(self.centroids)
        for i in range(n):
            for j in range(i+1, n):
                barrier, va, vb, vs = self.contradiction_intensity(i, j)
                if barrier > min_barrier:
                    result = self.synthesize(i, j)
                    result['barrier'] = barrier
                    results.append(result)
        return results

# ============================================================
# Run dialectic
# ============================================================
print("\n[DIALECTIC] Running thesis/antithesis → synthesis...")

dialectic = DialecticalSynthesis(v_func, coords, centroids_t)
results = dialectic.run_all(min_barrier=0.3)

# Sort by V improvement (best syntheses first)
results.sort(key=lambda r: r['v_syn'] - max(r['v_a'], r['v_b']))

print(f"\n  Found {len(results)} syntheses (barrier > 0.3):")
valid_count = 0
for ri, r in enumerate(results[:15]):
    chi = cv.decode([concept_syms[r['ci']][0]])
    chj = cv.decode([concept_syms[r['cj']][0]])
    tag = "VALID" if r['valid'] else "weak"
    if r['valid']:
        valid_count += 1
    print(f"    {ri}: '{chi}'⊕'{chj}' → '{r['nearest_char']}' "
          f"V={r['v_syn']:.3f} (barrier={r['barrier']:.3f}) "
          f"preserve={r['preserved_a']:.2f}/{r['preserved_b']:.2f} "
          f"transcend={r['transcended']:.3f} [{tag}]")

print(f"\n  Valid syntheses (distinct + improved): {valid_count}/{len(results)}")

# ============================================================
# Analysis: which syntheses are truly novel?
# ============================================================
if valid_count > 0:
    print("\n[ANALYSIS] Novel concept candidates:")
    valid_results = [r for r in results if r['valid']]
    valid_results.sort(key=lambda r: r['v_syn'])
    
    for ri, r in enumerate(valid_results[:5]):
        chi = cv.decode([concept_syms[r['ci']][0]])
        chj = cv.decode([concept_syms[r['cj']][0]])
        print(f"    {ri}: '{chi}' ⊕ '{chj}' → nearest='{r['nearest_char']}' "
              f"V_syn={r['v_syn']:.3f} V_thesis={r['v_a']:.3f} V_antithesis={r['v_b']:.3f}")
        print(f"       Preserve: {r['preserved_a']:.1%} thesis, {r['preserved_b']:.1%} antithesis")
        print(f"       Transcended: {r['transcended']:.3f} beyond saddle")

# ============================================================
# Test: feed synthesis trajectories to transformer
# ============================================================
print("\n[TEST] Synthesis → transformer generation...")

from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(coords.to(DEVICE))

word_ckpt = torch.load(os.path.join(CKPT_DIR, "word_weights.pt"), map_location='cpu', weights_only=True)
ut.load_state_dict(word_ckpt['model'], strict=False)
ut.eval()

for ri, r in enumerate(results[:5]):
    chi = cv.decode([concept_syms[r['ci']][0]])
    chj = cv.decode([concept_syms[r['cj']][0]])
    
    # Build trajectory: thesis_centroid → synthesis → antithesis_centroid
    za = centroids_t[r['ci']:r['ci']+1]
    zb = centroids_t[r['cj']:r['cj']+1]
    syn_t = r['synthesis'].view(1, -1).to(DEVICE)  # [1, 24]
    
    traj = torch.cat([za, syn_t, zb], dim=0)  # [3, 24]
    
    # Find nearest symbols along trajectory
    with torch.no_grad():
        dists = torch.cdist(traj, coords)
        nearest = dists.argmin(dim=-1).clamp(1, VT-1)
        
        inp = nearest.unsqueeze(0)  # [1, 3]
        _, scores = ut(inp, return_scores=True)
        pred = scores[0].argmax(dim=-1).tolist()
        gen = cv.decode(pred)
    
    tag = "✓" if r['valid'] else " "
    print(f"  [{tag}] '{chi}'⊕'{chj}': {cv.decode(nearest.tolist())} → '{gen}'")

# Save
syn_path = os.path.join(CKPT_DIR, "dialectical_synthesis.pt")
torch.save({
    'results': [(r['ci'], r['cj'], r['synthesis'], r['v_syn'], r['valid']) for r in results],
    'concept_centroids': [c.cpu() for c in centroids],
}, syn_path)
print(f"\nSaved: {syn_path}")
print("Done.")
