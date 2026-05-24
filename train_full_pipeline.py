"""
EVA Symbolic v8 — Full Training Pipeline.

Phase 1: Affinity training (count-based, no gradients, 50K batches)
Phase 2: MDS → coordinates in ℝ¹²
Phase 3: UnifiedTransformer training (gradient descent, 30K steps)
Phase 4: Reconstruction test
"""

import sys, os, time, torch, torch.nn.functional as F, numpy as np, gc
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from eva.symbolic import *
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.primordial_layer import PrimordialLayer
from eva.config import FCFConfig

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(CKPT_DIR, exist_ok=True)

print("=" * 60)
print("EVA v8 — Full Pipeline")
print("=" * 60)
print(f"Device: {DEVICE}")

# ============================================================
# PHASE 1: Affinity Training
# ============================================================
print("\n[PHASE 1] Affinity training (count-based, no gradients)...")

config = FCFConfig(); config.d_model = 256; config.vocab_size = 156; config.num_heads = 8
layer = PrimordialLayer(config)
if DEVICE == 'cuda': layer = layer.cuda()

cv = CharacterVocab()
pf = PotentialField(156, 256)
V, PAD = 156, cv.PAD_IDX

# Check if already trained
affinity_path = os.path.join(CKPT_DIR, "final", "potential_field.pt")
if os.path.exists(affinity_path):
    print("  Loading existing affinity checkpoint...")
    pf.load_state_dict(torch.load(affinity_path, map_location='cpu', weights_only=True))
    print(f"  Affinity: mean={pf.affinity.mean():.4f} std={pf.affinity.std():.4f}")

npy_file = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy_file):
    npy_file = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)
print(f"  Dataset: {len(all_ids)/1e6:.1f}M tokens")

AFF_BATCH = 128; AFF_BLOCK = 64; AFF_STEPS = 500000  # proper training
pos = 0; start = time.time()

# Move affinity buffers to GPU for GPU-resident training
pf.affinity = pf.affinity.to(DEVICE)
pf.co_occurrence_count = pf.co_occurrence_count.to(DEVICE)

if not os.path.exists(affinity_path):
    for step in range(AFF_STEPS):
        # Batch formation
        if pos + AFF_BLOCK + 2 > len(all_ids): pos = 0
        ids_batch, lens = [], []
        for _ in range(AFF_BATCH):
            if pos + AFF_BLOCK + 2 > len(all_ids): pos = 0
            end = min(pos + AFF_BLOCK, len(all_ids))
            chunk = all_ids[pos:end]
            sep = np.where((chunk == 0) | (chunk == 3))[0]
            if len(sep) > 0 and sep[0] < AFF_BLOCK // 2:
                end = pos + sep[0] + 1; chunk = all_ids[pos:end]
            ids = [int(x) for x in chunk if x >= 0][:AFF_BLOCK]
            ids_batch.append(ids); lens.append(len(ids)); pos += max(len(ids), 32)

        ml = max(lens)
        bt = torch.full((AFF_BATCH, ml), PAD, dtype=torch.long, device=DEVICE)
        for i, ids in enumerate(ids_batch):
            bt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            layer.eval(); x = layer.embed(bt); layer.forward_transformer(x)
            attn = layer.transformer.attention.last_attention

        if attn is not None:
            # VECTORIZED: all BATCH sequences in one GPU operation
            # Extract all adjacent pairs: [B, L-1, 2]
            max_len = ml
            left_all = bt[:, :max_len-1]    # [B, L-1]
            right_all = bt[:, 1:max_len]    # [B, L-1]
            
            # Adjacent attention from all heads
            adj_attn = attn.mean(dim=1)[:, torch.arange(1, max_len, device=DEVICE), 
                                          torch.arange(max_len-1, device=DEVICE)]
            
            # Valid mask (non-PAD, valid symbol range)
            valid = (left_all > 0) & (left_all < V) & (right_all > 0) & (right_all < V)
            
            # Flatten batch and positions
            i_flat = left_all[valid].long()   # [N]
            j_flat = right_all[valid].long()  # [N]
            w_flat = adj_attn[valid].float()  # [N]
            
            if len(i_flat) > 0:
                flat_idx = i_flat * V + j_flat
                inc = torch.ones_like(w_flat).to(pf.co_occurrence_count.dtype)
                pf.co_occurrence_count.view(-1).scatter_add_(0, flat_idx, inc)
                
                # Update affinity for changed pairs only
                uf = flat_idx.unique(); ui = uf // V; uj = uf % V
                raw = pf.co_occurrence_count[ui, uj] / 100000.0
                pf.affinity[ui, uj] = (0.5 + 0.5 * torch.clamp(raw, 0.0, 1.0)).float()

    if step % 100 == 0:  # ~3 sec updates
        elapsed = time.time() - start
        aps = step * AFF_BATCH / max(elapsed, 0.01)
        pct = step * 100 // AFF_STEPS
        bar = '#' * (pct // 4) + '-' * (25 - pct // 4)
        print(f"\r  [{bar}] {pct}% | {aps:.0f} a/s | pot={pf.affinity.mean():.4f}", end='', flush=True)

    # Move back to CPU for saving
    pf.affinity = pf.affinity.cpu()
    pf.co_occurrence_count = pf.co_occurrence_count.cpu()
    os.makedirs(os.path.join(CKPT_DIR, "final"), exist_ok=True)
    torch.save(pf.state_dict(), affinity_path)
    print(f"\n  Done. Affinity: mean={pf.affinity.mean():.4f} std={pf.affinity.std():.4f}")

# ============================================================
# PHASE 2: MDS → Coordinates
# ============================================================
print("\n[PHASE 2] MDS → Coordinates in ℝ¹²...")

from eva.symbolic.topological_field import TopologicalField
topo = TopologicalField(pf, coord_dim=24)
topo._compute_coordinates_from_affinity()
coords = topo.coordinates[:156, :24].clone()
# Diagnostic: check coordinate quality
import numpy as np
topo_aff = pf.affinity.cpu().numpy()
n = 156; D_mds = 1.0 - topo_aff[:n,:n]; np.fill_diagonal(D_mds, 0.0)
J = np.eye(n) - np.ones((n,n))/n; B = -0.5*J@(D_mds*D_mds)@J
eigvals = np.linalg.eigh(B)[0]; eigvals = np.sort(eigvals)[::-1]
eff_dim = (eigvals[:24].sum() / eigvals[eigvals>0].sum()) if (eigvals>0).sum()>0 else 0
unique_coords = len(np.unique(coords.numpy().round(decimals=6), axis=0))
print(f"  MDS: eff_dim(24)={eff_dim:.1%}, unique_coords={unique_coords}/{n}, "
      f"eig_range=[{eigvals[0]:.2f}, {eigvals[:24].min():.4f}]")
print(f"  Coordinates: {coords.shape}")

# ============================================================
# PHASE 3: Symbol reproduction — ALL 156 symbols, GPU only
# ============================================================
print("\n[PHASE 3] UnifiedTransformer — SYMBOL REPRODUCTION (156 symbols, GPU)")
print("  Goal: reproduce ALL 156 symbols from their coordinates")
print("  No sequences. No text. Just symbols.")

ut = UnifiedMultidimensionalTransformer(vocab_size=156, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(coords.to(DEVICE))
print(f"  {ut.summary()}")

UT_STEPS = 50000; UT_LR = 1e-3
opt = torch.optim.AdamW(ut.parameters(), lr=UT_LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UT_STEPS)

# Constant batch: all 156 symbols (skip PAD=0)
all_symbols = torch.arange(1, 157, dtype=torch.long, device=DEVICE)  # 1..156
symbol_batch = all_symbols.unsqueeze(0)  # [1, 156]

start = time.time()

for step in range(1, UT_STEPS + 1):
    ut.train()
    _, scores = ut(symbol_batch, return_scores=True)  # [1, 156, V]

    loss = F.cross_entropy(scores.view(-1, V), symbol_batch.view(-1))

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    opt.step()
    sch.step()

    if step % 500 == 0 or step == 1:
        elapsed = time.time() - start
        eta = (elapsed / step) * (UT_STEPS - step)
        with torch.no_grad():
            predicted = scores[0].argmax(dim=-1)  # [156]
            correct = (predicted == all_symbols).sum().item()
        lr = sch.get_last_lr()[0]
        sys.stdout.write(f"\r  step {step}/{UT_STEPS} | loss={loss.item():.4f} | "
                         f"correct={correct}/156 ({correct/156:.0%}) | lr={lr:.6f} | "
                         f"elapsed={elapsed:.0f}s eta={eta:.0f}s")
        if step % 500 == 0:
            sys.stdout.write("\n")

# ============================================================
# PHASE 4: Final symbol test — 100% or fail
# ============================================================
print("\n[PHASE 4] FINAL TEST: reproduce ALL 156 symbols")
ut.eval()
with torch.no_grad():
    _, scores = ut(symbol_batch, return_scores=True)
    predicted = scores[0].argmax(dim=-1)
    correct = (predicted == all_symbols).sum().item()

print(f"  Result: {correct}/156 ({correct/156:.0%})")
if correct == 156:
    print("  SUCCESS: all 156 symbols reproduced")
else:
    failed = [(i, cv.id_to_char[i+1], cv.id_to_char[p.item()]) 
              for i, p in enumerate(predicted) if p.item() != (i+1)]
    print(f"  FAIL: {156 - correct} symbols wrong")
    for fid, expected, got in failed[:10]:
        print(f"    id={fid+1} expected='{expected}' got='{got}'")
    print(f"  Checkpoint saved for debugging")
    torch.save({'model': ut.state_dict(), 'coords': coords}, 'PHASE3_FAIL.pt')

print("\nDone.")
