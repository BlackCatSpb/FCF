"""
demo_full.py — Full generation pipeline demo.

Loads trained model and runs generate_text() with:
  - GradientFlowSolver (Langevin trajectory refinement)
  - KCACycle (iterative latent correction)
  - SemanticRelevanceGate (confidence evaluation)

Usage:
    python demo_full.py
    python demo_full.py --checkpoint checkpoints/v3/eva_v3_heads_latest.pt
    python demo_full.py --checkpoint checkpoints/concept_basis/concept_basis_latest.pt
"""

import torch, torch.nn.functional as F
import numpy as np, os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.char_vocab import CharacterVocab as CharVocab
from eva.symbolic.potential_fields import (
    SemanticRelevanceGate, GradientFlowSolver, KCACycle
)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str,
                    default='checkpoints/concept_basis/concept_basis_latest.pt',
                    help='Trained model checkpoint')
parser.add_argument('--prompt', type=str, default='кошка', help='Russian prompt text')
parser.add_argument('--max-new', type=int, default=64, help='Max tokens to generate')
parser.add_argument('--temperature', type=float, default=0.8)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ---- Load model ----
print(f'[Model] Loading from {args.checkpoint}')
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
model = UnifiedMultidimensionalTransformer().to(device)
model.load_state_dict(ckpt['model_state'], strict=False)
model.eval()
print(f'[Model] Loaded (step {ckpt.get("step", "?")}, '
      f'{sum(p.numel() for p in model.parameters()):,} params)')

cv = CharVocab()

# ---- Instantiate SRG, KCA, GradFlow ----
srg = SemanticRelevanceGate(w_sim=0.4, w_ent=0.3, w_eth=0.3)
grad_flow = GradientFlowSolver(
    curvature_weight=0.1, eta=0.05, D=0.01,
    max_steps=50, tol=1e-5, timeout=10.0,
)
kca = KCACycle(
    srg=srg,
    lambda_conf=1.0, lambda_kl=0.1, lambda_dist=0.5,
    eta0=0.01, rho=0.85, epsilon=1e-5, delta_srg=0.001, timeout=10.0,
)

print(f'\n[Generate] Prompt: "{args.prompt}"')
print(f'[Generate] Max new tokens: {args.max_new}, temperature: {args.temperature}')

# ---- Encode prompt ----
prompt_ids = cv.encode_with_boundaries(args.prompt)
if len(prompt_ids) < 3:
    prompt_ids = [cv.SENT_OPEN_IDX] + prompt_ids

print(f'[Generate] Prompt IDs ({len(prompt_ids)}): {prompt_ids}')

# ---- Generate ----
text, metrics = model.generate_text(
    prompt_ids=prompt_ids,
    cv=cv,
    max_new=args.max_new,
    temperature=args.temperature,
    flow_solver=grad_flow,
    kca_cycle=kca,
    srg_module=srg,
    hypothesis_buffer=None,
)

print(f'\n{"="*60}')
print(f'Generated text:')
print(f'{text}')
print(f'{"="*60}')
print(f'Metrics:')
for k, v in metrics.items():
    print(f'  {k}: {v}')
print(f'{"="*60}')
