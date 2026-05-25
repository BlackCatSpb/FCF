"""
EVA — Unified GFRE Training.

Обучает Gradient Flow Reasoning Engine:
1. CompositePotentialField (V_real + V_contr + V_curv)
2. GradientFlowSolver (Euler-Maruyama ODE)
3. Self-reflection + Active learning + Causal discovery

Training: multi-component loss — attraction + confinement + path efficiency.
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
LOG = os.path.join(os.path.dirname(__file__), "training_log_gfre.txt")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.gradient_flow import CompositePotentialField, GradientFlowSolver
from eva.symbolic.potential_function import PotentialFunction
from eva.symbolic.self_reflection import SelfReflection
from eva.symbolic.active_causal import ActiveLearner, CausalDiscovery
from eva.symbolic.trajectory_store import TrajectoryStore

cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — GFRE Training")
print("=" * 60)

# ============================================================
# Load model and data
# ============================================================
evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)

coords64 = torch.zeros(157, 64, device=DEVICE)
coords64[:, :24] = coords[:, :24]
g = torch.Generator(device=DEVICE).manual_seed(42)
coords64[:, 24:] = torch.randn(157, 40, generator=g, device=DEVICE) * 0.02
coords64 = coords64 / coords64.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=64, num_levels=4,
    scales_per_level=4, num_layers=3, d_ff=128).to(DEVICE)
ut.set_symbol_coordinates(coords64)

# Load latest v2 weights if available
v2_path = os.path.join(CKPT, "v2_latest.pt")
if os.path.exists(v2_path):
    print("Loading v2 checkpoint...")
    ckpt = torch.load(v2_path, map_location='cpu', weights_only=True)
    ut.load_state_dict(ckpt['model'], strict=False)

# Load V(z) potential function
pf_path = os.path.join(CKPT, "potential_function.pt")
if os.path.exists(pf_path):
    pf_data = torch.load(pf_path, map_location='cpu', weights_only=False)
    V_real = PotentialFunction(dim=64, hidden=128).to(DEVICE)
    # Pad weights for 64-dim
    old_weights = pf_data['model']
    new_state = {}
    for k, v in old_weights.items():
        if 'weight' in k and v.shape[-1] == 24:
            w64 = torch.zeros(v.shape[0], 64)
            w64[:, :24] = v
            new_state[k] = w64
        elif 'weight' in k and v.shape[0] == 24:
            w64 = torch.zeros(64, v.shape[1])
            w64[:24, :] = v
            new_state[k] = w64
        else:
            new_state[k] = v
    V_real.load_state_dict(new_state, strict=False)
    print("Loaded V(z) potential (padded to 64-dim)")
else:
    V_real = PotentialFunction(dim=64, hidden=128).to(DEVICE)
    print("Fresh V(z) potential")

# Load contradiction filter
contra_path = os.path.join(CKPT, "contradiction_filter.pt")
contradiction = None
if os.path.exists(contra_path):
    cd = torch.load(contra_path, map_location='cpu', weights_only=True)
    class FakeContradiction:
        def __init__(self):
            self.forbidden = {}
            self.forbidden_mask = cd.get('forbidden_mask', None)
    contradiction = FakeContradiction()
    print("Loaded contradiction filter")

# ============================================================
# Build GFRE
# ============================================================
V_composite = CompositePotentialField(V_real, contradiction, coords64).to(DEVICE)
solver = GradientFlowSolver(V_composite, ut.decoder, coords64, dt=0.05, max_steps=100)
reflector = SelfReflection(contradiction, coords64)
learner = ActiveLearner(entropy_threshold=2.5, confidence_threshold=0.4)
causal = CausalDiscovery()

print(f"GFRE built. V params: {sum(p.numel() for p in V_composite.parameters() if p.requires_grad):,}")

# ============================================================
# Training data
# ============================================================
npy = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy): npy = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)

STEPS = 50000; LR = 5e-4; B = 32; ML = 128
SAVE_EVERY = 5000
opt = torch.optim.AdamW(list(ut.parameters()) + list(V_composite.parameters()), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
rng = np.random.RandomState(42)

def log(msg):
    t = time.strftime("%H:%M:%S"); line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(line + '\n'); f.flush()

log(f"START GFRE: {STEPS} steps, dim=64, layers=3, dual loss (CE + gradient)")

# ============================================================
# Training loop
# ============================================================
t0 = time.time()
total_uncertain = 0

for s in range(1, STEPS + 1):
    lens = rng.randint(32, ML + 1, B)
    starts = rng.randint(0, max(1, total - max(lens) - 1), B)
    ml = max(lens)
    bt = torch.full((B, ml), 0, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(B, ml, device=DEVICE)
    
    for bi in range(B):
        vb = data[starts[bi]:starts[bi] + lens[bi]]
        vb = vb[(vb > 0) & (vb < VT)]
        vl = min(len(vb), ml)
        if vl >= 4: bt[bi, :vl] = torch.from_numpy(vb[:vl].astype(np.int64)).to(DEVICE); mask[bi, :vl] = 1.0
    
    if mask.sum() < 50: continue
    
    # === Causal LM loss (standard) ===
    ut.train()
    _, scores = ut(bt, return_scores=True)
    target = bt[:, 1:].clamp(1, VT - 1).contiguous()
    pred = scores[:, :-1, :].contiguous()
    t_mask = mask[:, 1:]
    
    ce_loss = F.cross_entropy(pred.view(-1, 157), target.view(-1), reduction='none')
    ce_loss = (ce_loss.view(B, ml - 1) * t_mask).sum() / (t_mask.sum() + 1e-8)
    
    # === Gradient flow loss (new) ===
    # Sample a few sequences for gradient flow training
    gf_loss = torch.tensor(0.0, device=DEVICE)
    with torch.no_grad():
        for bi in range(min(4, B)):
            ids = bt[bi][mask[bi].bool()][:16].tolist()
            if len(ids) < 4: continue
            
            z0 = ut.embed(torch.tensor([ids[:4]], dtype=torch.long, device=DEVICE)).mean(dim=1)
            
            # Run gradient flow
            hyps = solver.solve(z0, temperature=0.1, num_hypotheses=1, char_vocab=cv)
            if hyps:
                # Target: the equilibrium should be close to actual next symbol coordinate
                target_z = ut.embed(torch.tensor([[ids[4]]], dtype=torch.long, device=DEVICE)).squeeze(0).squeeze(0)
                eq_z = torch.tensor(hyps[0].equilibrium_z, device=DEVICE).float()
                gf_loss = gf_loss + F.mse_loss(eq_z, target_z)
    
    gf_loss = gf_loss / 4.0
    
    # === Self-reflection + active learning ===
    with torch.no_grad():
        uncertain = 0
        for bi in range(min(8, B)):
            ids = bt[bi][mask[bi].bool()][:12].tolist()
            if len(ids) < 4: continue
            
            z0 = ut.embed(torch.tensor([ids[:4]], dtype=torch.long, device=DEVICE)).mean(dim=1)
            hyps = solver.solve(z0, temperature=0.05, num_hypotheses=1, char_vocab=cv)
            if hyps:
                diag = reflector.diagnose(hyps[0].trajectory, ids)
                should, urg, reason = learner.should_query(diagnostic=diag)
                if should: uncertain += 1
        
        if uncertain > 2:
            total_uncertain += 1
    
    # === Combined loss ===
    loss = ce_loss + 0.01 * gf_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(ut.parameters()) + list(V_composite.parameters()), 1.0)
    opt.step()
    sch.step()
    
    if s % SAVE_EVERY == 0:
        torch.save({'ut': ut.state_dict(), 'V': V_composite.state_dict(), 'step': s},
                   os.path.join(CKPT, f"gfre_{s}.pt"))
        torch.save({'ut': ut.state_dict(), 'V': V_composite.state_dict(), 'step': s},
                   os.path.join(CKPT, "gfre_latest.pt"))
        
        with torch.no_grad():
            acc = (pred.argmax(-1) == target) & t_mask.bool()
            acc = acc.sum().item() / (t_mask.sum() + 1e-8)
        
        elapsed = time.time() - t0; eta = (elapsed / s) * (STEPS - s) if s > 0 else 0
        log(f"  step {s:>6d}/{STEPS} | CE={ce_loss.item():.4f} GF={gf_loss.item():.4f} "
            f"acc={acc:.3f} | uncert={total_uncertain} | {elapsed/60:.0f}min")
    elif s % 500 == 0:
        with torch.no_grad():
            acc = (pred.argmax(-1) == target) & t_mask.bool()
            acc = acc.sum().item() / (t_mask.sum() + 1e-8)
        elapsed = time.time() - t0
        log(f"  step {s:>6d}/{STEPS} | CE={ce_loss.item():.4f} GF={gf_loss.item():.4f} "
            f"acc={acc:.3f} | {elapsed/60:.0f}min")

log("DONE")
