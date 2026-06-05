"""Analyze graph_encoder.pt — what it learned, how it maps coordinates."""
import sys, os, math, pickle
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
META = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\heads_meta.pkl'
MODEL = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\graph_encoder.pt'
OUT = r'C:\Users\black\OneDrive\Desktop\FCF\graph_encoder_analysis.txt'

# Load model
sd = torch.load(MODEL, map_location='cpu', weights_only=True)
print(f'State dict: {len(sd)} keys')
for k, v in sd.items():
    print(f'  {k}: {list(v.shape)}  ({v.sum().item():.2f})')

# Build model
class GraphEncoder(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.lin1 = torch.nn.Linear(384, 512)
        self.lin2 = torch.nn.Linear(512, 512)
        self.lin3 = torch.nn.Linear(512, 2560)
        self.proj = torch.nn.Linear(2560, 2560)
        self.load_state_dict(sd)
    
    def forward(self, x):
        h = torch.relu(self.lin1(x))
        h = torch.relu(self.lin2(h))
        h = self.lin3(h)  # no activation before proj
        h = self.proj(h)  # no activation
        return h

model = GraphEncoder(sd).to(DEVICE)
model.eval()

# Analyze embedding space: lin1.weight is the primary embedding
embed_W = sd['lin1.weight'].numpy()  # (512, 384)
embed_b = sd['lin1.bias'].numpy()    # (512,)

# Check norms, sparsity, etc
row_norms = np.linalg.norm(embed_W, axis=1)
col_norms = np.linalg.norm(embed_W, axis=0)

print(f'\nEmbedding (lin1) stats:')
print(f'  Weight: {embed_W.shape}')
print(f'  Row norms: mean={row_norms.mean():.3f} std={row_norms.std():.3f} min={row_norms.min():.3f} max={row_norms.max():.3f}')
print(f'  Col norms: mean={col_norms.mean():.3f} std={col_norms.std():.3f} min={col_norms.min():.3f} max={col_norms.max():.3f}')
print(f'  Sparsity (near-zero): {(np.abs(embed_W) < 1e-4).mean()*100:.1f}%')

# Check output projection
proj_W = sd['proj.weight'].numpy()  # (2560, 2560)
proj_b = sd['proj.bias'].numpy()    # (2560,)
print(f'\nProjection stats:')
print(f'  Weight: {proj_W.shape}')
evals = np.linalg.eigvalsh(proj_W @ proj_W.T)
print(f'  Singular values: mean={evals.mean():.3f} min={evals.min():.3f} max={evals.max():.3f}')
print(f'  Rank: {(evals > 1e-6).sum()} / {len(evals)}')

# What is the output range?
# Run random coordinates through the model
print(f'\nOutput range test:')
for _ in range(5):
    x = torch.randn(1, 384, device=DEVICE) * 10  # scaled random
    out = model(x)
    print(f'  Input norm={x.norm().item():.1f} -> Output norm={out.norm().item():.1f} mean={out.mean().item():.3f} std={out.std().item():.3f}')

# Test with REAL trajectory data
print(f'\nTesting on real trajectory data from heads_meta...')
with open(META, 'rb') as f:
    meta = pickle.load(f)
V = meta.get('V', 4101)
morph = meta.get('morph_logprob', {})
syntax = meta.get('syntax_logprob', {})

# Build a realistic coordinate using CoordinatePacker
from coordinate_packer import CoordinatePacker
packer = CoordinatePacker()

# Build a realistic coordinate for a token in context
real_coords = []
for tid in [5, 100, 500, 1000, 2000]:
    for pi in range(3):
        coord = packer.pack_token(
            token_id=tid,
            pos_in_word=pi,
            word_len=7,
            word_num=0,
            pos_in_sent=pi,
            sent_len=10,
            flags=0
        )
        real_coords.append(coord)

real_batch = np.stack(real_coords).astype(np.float32)
real_t = torch.from_numpy(real_batch).to(DEVICE)
with torch.no_grad():
    out = model(real_t)
print(f'  Tested {len(real_coords)} real coordinates')
print(f'  Output shape: {out.shape}')
print(f'  Output norm range: [{out.norm(dim=1).min().item():.1f}, {out.norm(dim=1).max().item():.1f}]')

# Check variance along batch dimension
var = out.var(dim=0)
print(f'  Output var: mean={var.mean().item():.4f} std={var.std().item():.4f}')
print(f'  Active dims (var>0.01): {(var > 0.01).sum().item()} / {var.shape[0]}')

# What dimension does the 384-dim input represent?
# Check if specific input dims dominate
print(f'\nInput sensitivity analysis:')
input_test = torch.zeros(384, 384, device=DEVICE)
for i in range(384):
    input_test[i, i] = 1.0  # one-hot per dim
with torch.no_grad():
    out_sens = model(input_test)  # (384, 2560)
col_sensitivities = out_sens.norm(dim=1).cpu().numpy()  # norm per input dim
top_dims = np.argsort(-col_sensitivities)[:20]
print(f'  Top-20 most sensitive input dims: {top_dims.tolist()}')
print(f'  Sensitivity range: [{col_sensitivities.min():.3f}, {col_sensitivities.max():.3f}]')
print(f'  High sensitivity dims (>1.0): {(col_sensitivities > 1.0).sum()} / {384}')
# Check dims 89-96 (meta section)
for d in range(89, 97):
    print(f'  dim {d} (meta): sensitivity={col_sensitivities[d]:.3f}')
# Check dims 0-87 (token/syntax section)
for d in [0, 10, 20, 40, 60, 80]:
    print(f'  dim {d} (token): sensitivity={col_sensitivities[d]:.3f}')

# Summary
print(f'\n{"="*60}')
print(f'GRAPH ENCODER ANALYSIS')
print(f'{"="*60}')
print(f'Architecture: 384 -> 512 -> 512 -> 2560 -> 2560')
n_params = sum(int(np.prod(sd[k].shape)) for k in sd)
print(f'Parameters: {n_params:,}')
print(f'Input: 384-dim coordinates (matches CoordinatePacker)')
print(f'Output: 2560-dim (projected embedding)')
print(f'Embedding dims sensitive (>1.0): {(col_sensitivities > 1.0).sum()} / 384')
print(f'Meta dims (89-96) sensitivity: {[f"{col_sensitivities[d]:.2f}" for d in range(89, 97)]}')
print(f'Output rank: {(evals > 1e-6).sum()} / 2560')
print(f'{"="*60}')
