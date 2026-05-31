"""Verify all changes compile and forward works."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from eva.symbolic.phase1_model import UnifiedMultidimensionalTransformerV2
from eva.symbolic.subspace_coords import WordWeightEncoder

model = UnifiedMultidimensionalTransformerV2(vocab_size=4101)
print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')

x = torch.randint(0, 100, (2, 16), dtype=torch.long)
h, scores, weights, heads = model.forward(x, return_scores=True, return_heads=True, capture_attn=True)
print(f'Forward: h={h.shape}, scores={scores.shape}')
print(f'Heads keys: {list(heads.keys())}')
print(f'Attractors: {heads["attractor_n_attractors"]}')
print(f'Cached attn: {len(model._cached_attention)} heads')

h2, s2, w2, heads2 = model.forward(x, return_scores=True, return_heads=True, update_attractors=True)
print(f'Update attractors OK, count={heads2["attractor_n_attractors"]}')

# WordWeight with boundary_logits
wwe = WordWeightEncoder(d_model=384)
bd = torch.randn(2, 16, 3)
ww, ws, bd_out = wwe(h, boundary_logits=bd)
print(f'WordWeight: vecs={ww.shape}, weights={ws.shape}')

# WordWeight without boundary_logits
ww2, ws2, bd_out2 = wwe(h)
print(f'WordWeight (no logits): vecs={ww2.shape}, weights={ws2.shape}')

# Verify all heads produce correct shapes
assert h.shape == (2, 16, 384)
assert scores.shape == (2, 16, 4101)
assert ww.shape[-1] == 384
assert bd_out.shape == (2, 16, 3)
print('All assertions passed!')
