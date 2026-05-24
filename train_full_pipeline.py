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

AFF_BATCH = 128; AFF_BLOCK = 64; AFF_STEPS = 200000
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
                inc = (1.0 + w_flat).to(pf.co_occurrence_count.dtype)
                pf.co_occurrence_count.view(-1).scatter_add_(0, flat_idx, inc)
                
                # Update affinity for changed pairs only
                uf = flat_idx.unique(); ui = uf // V; uj = uf % V
                raw = pf.co_occurrence_count[ui, uj] / 100000.0
                pf.affinity[ui, uj] = (0.5 + 0.5 * torch.clamp(raw, 0.0, 1.0)).float()

        if step % 5000 == 0 and step > 0:
            elapsed = time.time() - start
            aps = step * AFF_BATCH / max(elapsed, 0.01)
            pct = step * 100 // AFF_STEPS
            bar = '#' * (pct // 4) + '-' * (25 - pct // 4)
            print(f"\r  [{bar}] {pct}% | {aps:.0f} a/s | pot={pf.affinity.mean():.4f}", end='', flush=True)
        if step % 10000 == 0 and step > 0:
            print()

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
print(f"  Coordinates: {coords.shape}")

# ============================================================
# PHASE 3: UnifiedTransformer — affinity distillation
# ============================================================
print("\n[PHASE 3] UnifiedTransformer training (affinity distillation)...")

ut = UnifiedMultidimensionalTransformer(vocab_size=156, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(coords)  # CRITICAL: inject MDS coordinates
print(f"  {ut.summary()}")

UT_BATCH = 256; UT_BLOCK = 64; UT_STEPS = 50000; UT_LR = 1e-3
opt = torch.optim.AdamW(ut.parameters(), lr=UT_LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UT_STEPS)

# Affinity-based soft targets
aff_tgt = pf.affinity.to(DEVICE)  # [V, V]
aff_tgt = aff_tgt / (aff_tgt.sum(dim=-1, keepdim=True) + 1e-8)  # normalize

pos = 0; start = time.time()

for step in range(1, UT_STEPS + 1):
    if pos + UT_BLOCK + 2 > len(all_ids): pos = 0
    ids_batch, lens = [], []
    for _ in range(UT_BATCH):
        if pos + UT_BLOCK + 2 > len(all_ids): pos = 0
        end = min(pos + UT_BLOCK, len(all_ids))
        chunk = all_ids[pos:end]
        sep = np.where((chunk == 0) | (chunk == 3))[0]
        if len(sep) > 0 and sep[0] < UT_BLOCK // 2:
            end = pos + sep[0] + 1; chunk = all_ids[pos:end]
        ids = [int(x) for x in chunk if x >= 0][:UT_BLOCK]
        ids_batch.append(ids); lens.append(len(ids)); pos += max(len(ids), 32)

    ml = max(lens)
    bt = torch.full((UT_BATCH, ml), PAD, dtype=torch.long, device=DEVICE)
    for i, ids in enumerate(ids_batch):
        bt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=DEVICE)

    ut.train()
    _, scores = ut(bt, return_scores=True)
    
    # Target: affinity[token] — what the symbolic layer knows
    tgt = aff_tgt[bt.clamp(0, V-1)]  # [B, L, V]
    mask = (bt != PAD).float()  # [B, L]
    
    # KL divergence: transformer output should match affinity distribution
    log_probs = F.log_softmax(scores, dim=-1)
    loss = -(tgt * log_probs).sum(dim=-1)  # [B, L]
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    opt.step()
    sch.step()

    if step % 5000 == 0:
        elapsed = time.time() - start
        bps = step / max(elapsed, 0.01)
        pct = step * 100 // UT_STEPS
        bar = '#' * (pct // 4) + '-' * (25 - pct // 4)
        print(f"\r  [{bar}] {pct}% | {bps:.0f} b/s | loss={loss.item():.6f}", end='', flush=True)
        gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()

print()
os.makedirs(os.path.join(CKPT_DIR, "unified"), exist_ok=True)
torch.save(ut.state_dict(), os.path.join(CKPT_DIR, "unified", "transformer_final.pt"))
print(f"  Done. Final loss: {loss.item():.6f}")

# ============================================================
# PHASE 4: Reconstruction Test
# ============================================================
print("\n[PHASE 4] Reconstruction test...")
ut.eval()

test_sentences = [
    "привет мир",
    "человек идёт",
    "солнце светит",
    "мама мыла раму",
    "я люблю программирование",
]

for sentence in test_sentences:
    ids = cv.encode(sentence)
    inp = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _, scores = ut(inp, return_scores=True)
    predicted = torch.argmax(scores[0], dim=-1).tolist()
    generated = cv.decode(predicted)
    accuracy = sum(1 for p, t in zip(predicted, ids[1:]) if p == t) / max(len(ids)-1, 1)
    print(f"  '{sentence}' → '{generated[:60]}' ({accuracy:.0%})")

print("\nDone.")
