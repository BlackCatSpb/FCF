"""
EVA — Coordinate Prediction Training.

Обучает трансформер предсказывать СЛЕДУЮЩИЕ КООРДИНАТЫ (не символы).

Вход:  координаты символов [pos 0..L-2]
Цель:  координаты символов [pos 1..L-1] — следующий шаг траектории
Loss:  MSE(предсказанные_координаты, истинные_следующие_координаты)

Генерация: seed → координаты → предсказать → decode → повторить.
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
print("EVA — Coordinate Prediction Training")
print("=" * 60)
print(f"Device: {DEVICE}")

# ============================================================
# Load coordinates
# ============================================================
evolved = torch.load(os.path.join(CKPT_DIR, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)
affinity = evolved['affinity']
print(f"Coordinates: {coords.shape}")

# ============================================================
# Transformer for coordinate prediction
# ============================================================
ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(coords.to(DEVICE))
print(f"  {ut.summary()}")

# ============================================================
# Training: predict NEXT coordinate
# ============================================================
npy_file = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy_file):
    npy_file = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)

UT_STEPS = 10000; UT_LR = 1e-3; UT_BATCH = 64; MAX_LEN = 64
opt = torch.optim.AdamW(ut.parameters(), lr=UT_LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UT_STEPS)

rng = np.random.RandomState(777)
total_ids = len(all_ids)
start_t = time.time()
last_print_t = 0

print("\n[TRAIN] Predicting NEXT coordinates (not same)...")

for step in range(1, UT_STEPS + 1):
    lengths = rng.randint(16, MAX_LEN + 1, UT_BATCH)
    starts = rng.randint(0, max(1, total_ids - max(lengths) - 1), UT_BATCH)
    max_len = max(lengths)
    
    bt = torch.full((UT_BATCH, max_len), 0, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(UT_BATCH, max_len, device=DEVICE)
    for bi in range(UT_BATCH):
        s, l = starts[bi], lengths[bi]
        block = all_ids[s:s+l]
        valid = (block > 0) & (block < VT)
        vb = block[valid]
        vl = min(len(vb), max_len)
        if vl >= 4:
            bt[bi, :vl] = torch.from_numpy(vb[:vl].astype(np.int64)).to(DEVICE)
            mask[bi, :vl] = 1.0
    
    if mask.sum() < 50:
        continue
    
    ut.train()
    pred_coords, pred_scores = ut(bt, return_scores=True)  # [B, L, 24], [B, L, 157]
    
    # TARGET: next position coordinates (shift right by 1)
    target_ids = bt[:, 1:].clamp(1, VT-1)  # [B, L-1]
    target_coords = ut.embed(target_ids).detach()  # [B, L-1, 24]
    
    # LOSS 1: Coordinate MSE — predicted position should be close to next symbol
    coord_loss = F.mse_loss(
        pred_coords[:, :-1, :],  # positions 0..L-2 predict 1..L-1
        target_coords,
        reduction='none'
    ).mean(dim=-1)  # [B, L-1]
    coord_loss = (coord_loss * mask[:, 1:]).sum() / (mask[:, 1:].sum() + 1e-8)
    
    # LOSS 2: Symbol CE — for stability, also predict correct next symbol
    sym_loss = F.cross_entropy(
        pred_scores[:, :-1, :].contiguous().view(-1, 157),
        target_ids.contiguous().view(-1),
        reduction='none'
    ).view(UT_BATCH, max_len - 1)
    sym_loss = (sym_loss * mask[:, 1:]).sum() / (mask[:, 1:].sum() + 1e-8)
    
    loss = coord_loss + 0.1 * sym_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    opt.step()
    sch.step()
    
    now = time.time()
    if now - last_print_t >= 5 or step == 1 or step == UT_STEPS:
        last_print_t = now
        elapsed = now - start_t
        eta = (elapsed / step) * (UT_STEPS - step)
        with torch.no_grad():
            pred_ids = pred_scores[:, :-1, :].argmax(dim=-1)
            sym_acc = ((pred_ids == target_ids) & mask[:, 1:].bool()).sum().item()
            sym_acc /= (mask[:, 1:].sum() + 1e-8)
        lr = sch.get_last_lr()[0]
        print(f"  step {step:>5d}/{UT_STEPS} | loss={loss.item():.4f} "
              f"(coord={coord_loss.item():.3f} sym={sym_loss.item():.3f}) "
              f"acc={sym_acc:.3f} | {elapsed:.0f}s", flush=True)

# ============================================================
# Test: autoregressive coordinate prediction → generation
# ============================================================
print("\n[TEST] Coordinate-based generation...")

def generate_coord(ut, coords, affinity, seed_ids, max_new=25, temp=0.8):
    """Generate by predicting next coordinates, decoding to symbols."""
    ids = list(seed_ids)
    ut.eval()
    
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            pred_c, pred_s = ut(inp, return_scores=True)
            
            # Last predicted coordinate
            next_coord = pred_c[0, -1]  # [24]
            
            # Find nearest symbol
            dists = torch.norm(coords - next_coord, dim=-1)
            dists[0] = 1e9  # block PAD
            
            # Top-k nearest, weighted by affinity
            k = 10
            topk_dists, topk_idx = torch.topk(dists, k, largest=False)
            
            # Convert distances to scores (closer = higher)
            scores = 1.0 / (topk_dists + 0.01)
            
            # Affinity bonus from previous token
            if len(ids) > 0 and 0 < ids[-1] < VT:
                aff_bonus = affinity[ids[-1]][topk_idx].to(DEVICE)
                aff_bonus = aff_bonus / aff_bonus.max().clamp(min=1e-8)
                scores = scores + aff_bonus * 2.0
            
            # Repetition penalty
            for t in set(ids[-3:]):
                mask_r = (topk_idx == t).nonzero(as_tuple=True)[0]
                if len(mask_r) > 0:
                    scores[mask_r] *= 0.1
            
            probs = F.softmax(scores / temp, dim=-1)
            next_tok = topk_idx[torch.multinomial(probs, 1)].item()
            
            if next_tok <= 0 or next_tok >= VT:
                next_tok = topk_idx[0].item()
            
            ids.append(next_tok)
    
    return ids

test_seeds = ["привет", "человек", "солнце", "я люблю", "сегодня", "метаданные"]

for seed in test_seeds:
    ids = cv.encode(seed)[1:-1]
    if len(ids) < 2: continue
    result = generate_coord(ut, coords, affinity, ids, max_new=25, temp=0.8)
    text = cv.decode(result)
    print(f"  '{seed}' → '{text}'")

# Save
cp_path = os.path.join(CKPT_DIR, "coord_predict.pt")
torch.save({'model': ut.state_dict(), 'coords': coords}, cp_path)
print(f"\nSaved: {cp_path}")
print("Done.")
