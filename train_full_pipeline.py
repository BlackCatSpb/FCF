"""
EVA Symbolic v8 — Full Training Pipeline.

Phase 1: Affinity training (count-based, no gradients, 50K batches)
Phase 2: MDS → coordinates in ℝ¹²
Phase 3: UnifiedTransformer training (gradient descent, 30K steps)
Phase 4: Reconstruction test
"""

import sys, os, time, torch, numpy as np, gc
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

npy_file = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy_file):
    npy_file = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)
print(f"  Dataset: {len(all_ids)/1e6:.1f}M tokens")

AFF_BATCH = 128; AFF_BLOCK = 64; AFF_STEPS = 50000
pos = 0; start = time.time()

for step in range(AFF_STEPS):
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
        for i in range(AFF_BATCH):
            L_i = lens[i]
            if L_i < 2: continue
            left = bt[i, :L_i - 1]; right = bt[i, 1:L_i]
            adj = attn[i].mean(dim=0)[torch.arange(1, L_i), torch.arange(L_i - 1)]
            valid = (left < V) & (right < V) & (left != PAD) & (right != PAD)
            i_idx = left[valid].long().cpu(); j_idx = right[valid].long().cpu()
            w = adj[valid].float().cpu()
            if len(i_idx) > 0:
                flat = i_idx * V + j_idx
                inc = (1.0 + w).to(pf.co_occurrence_count.dtype)
                pf.co_occurrence_count.view(-1).scatter_add_(0, flat, inc)
                uf = flat.unique(); ui = uf // V; uj = uf % V
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

os.makedirs(os.path.join(CKPT_DIR, "final"), exist_ok=True)
torch.save(pf.state_dict(), os.path.join(CKPT_DIR, "final", "potential_field.pt"))
print(f"  Done. Affinity: mean={pf.affinity.mean():.4f} std={pf.affinity.std():.4f}")

# ============================================================
# PHASE 2: MDS → Coordinates
# ============================================================
print("\n[PHASE 2] MDS → Coordinates in ℝ¹²...")

from eva.symbolic.topological_field import TopologicalField
topo = TopologicalField(pf, coord_dim=12)
topo._compute_coordinates_from_affinity()
coords = topo.coordinates[:156, :12].clone()
print(f"  Coordinates: {coords.shape}")

# ============================================================
# PHASE 3: UnifiedTransformer Training
# ============================================================
print("\n[PHASE 3] UnifiedTransformer training (gradient descent)...")

ut = UnifiedMultidimensionalTransformer(vocab_size=156, coord_dim=12)
ut.set_symbol_coordinates(coords)
if DEVICE == 'cuda':
    ut = ut.cuda()
print(f"  {ut.summary()}")

UT_BATCH = 256; UT_BLOCK = 64; UT_STEPS = 200000; UT_LR = 5e-4
opt = torch.optim.AdamW(ut.parameters(), lr=UT_LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UT_STEPS)

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

    target = torch.roll(bt, -1, dims=1)
    mask = (bt != PAD).float()

    ut.train()
    loss = ut.compute_loss(bt, target, mask)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    opt.step()
    sch.step()

    if step % 3000 == 0:
        elapsed = time.time() - start
        bps = step / max(elapsed, 0.01)
        pct = step * 100 // UT_STEPS
        bar = '#' * (pct // 4) + '-' * (25 - pct // 4)
        print(f"\r  [{bar}] {pct}% | {bps:.0f} b/s | loss={loss.item():.4f}", end='', flush=True)
        gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()

os.makedirs(os.path.join(CKPT_DIR, "unified"), exist_ok=True)
torch.save(ut.state_dict(), os.path.join(CKPT_DIR, "unified", "transformer_final.pt"))
print(f"  Done. Final loss: {loss.item():.4f}")

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
