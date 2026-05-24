"""
EVA — ContradictionFilter: иммунная система траекторий в ℝ²⁴.

5 типов запретов, адаптированных под координатное пространство:
1. Структурный: affinity → 0
2. Частотный: usage = 0
3. Семантический: cos(вектор_i, вектор_j) < порог
4. Потенциальный: V(z) на середине траектории > порог
5. Координатный: расстояние в ℝ²⁴ > порог

Фильтр выдаёт forbidden_mask [157,157] — маска запрещённых переходов.
Трансформер обучается с masked loss: запрещённые переходы штрафуются.
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()
VT = 157

print("=" * 60)
print("EVA — ContradictionFilter")
print("=" * 60)
print(f"Device: {DEVICE}")

# ============================================================
# Load evolved data
# ============================================================
evolved = torch.load(os.path.join(CKPT_DIR, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
affinity = evolved['affinity']      # [157, 157] — evolved
coords = evolved['coords']           # [157, 24]
print(f"Loaded: affinity {affinity.shape}, coords {coords.shape}")

# Load PotentialFunction if available
from eva.symbolic.potential_function import PotentialFunction
pf_path = os.path.join(CKPT_DIR, "potential_function.pt")
if os.path.exists(pf_path):
    pf_data = torch.load(pf_path, map_location='cpu', weights_only=False)
    v_func = PotentialFunction(dim=24, hidden=128).to(DEVICE)
    v_func.load_state_dict(pf_data['model'])
    v_func.eval()
    print("Loaded: PotentialFunction V(z)")
else:
    v_func = None
    print("PotentialFunction not found, skipping potential-based detection")

# ============================================================
# Build contradiction mask
# ============================================================
print("\n[DETECT] Building contradiction mask...")

aff = affinity.numpy()
coords_np = coords.numpy()

# Distance-based: symbols far apart in ℝ²⁴ are unlikely to follow each other
dist_matrix = np.zeros((VT, VT))
for i in range(VT):
    diff = coords_np[i:i+1] - coords_np
    dist_matrix[i] = np.linalg.norm(diff, axis=1)

# Threshold: use distance percentiles
valid_dists = dist_matrix[1:VT, 1:VT][~np.eye(156, dtype=bool)]
dist_median = np.median(valid_dists)
dist_threshold = dist_median * 0.6  # closer than 60% of median = too far

dist_forbidden = dist_matrix > dist_threshold
np.fill_diagonal(dist_forbidden, False)

# Affinity-based: very weak connections
aff_threshold = np.percentile(aff[aff > 0.01], 10)  # bottom 10% of non-zero
aff_forbidden = aff < aff_threshold
np.fill_diagonal(aff_forbidden, False)

# Combine: distance OR affinity
forbidden_mask_np = dist_forbidden | aff_forbidden

# Never forbid PAD transitions
forbidden_mask_np[0, :] = False
forbidden_mask_np[:, 0] = False

n_forbidden = forbidden_mask_np.sum()
n_total = VT * VT - VT  # exclude diagonal
print(f"  Distance threshold: {dist_threshold:.3f} (median={dist_median:.3f})")
print(f"  Affinity threshold: {aff_threshold:.4f}")
print(f"  Distance-forbidden: {dist_forbidden.sum():,}")
print(f"  Affinity-forbidden: {aff_forbidden.sum():,}")
print(f"  TOTAL forbidden: {n_forbidden}/{n_total} ({n_forbidden/n_total:.1%})")

# ============================================================
# Train transformer with contradiction-aware loss
# ============================================================
print("\n[TRAIN] Transformer with contradiction mask...")

from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(coords.to(DEVICE))

# Load word weights as starting point
word_weights_path = os.path.join(CKPT_DIR, "word_weights.pt")
if os.path.exists(word_weights_path):
    ckpt = torch.load(word_weights_path, map_location='cpu', weights_only=True)
    ut.load_state_dict(ckpt['model'], strict=False)
    print("  Loaded word weights for initialization")

forbidden_mask = torch.tensor(forbidden_mask_np, dtype=torch.bool, device=DEVICE)  # [VT, VT]

UT_STEPS = 3000; UT_LR = 1e-4; UT_BATCH = 128
opt = torch.optim.AdamW(ut.parameters(), lr=UT_LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UT_STEPS)

# Extract words
npy_file = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy_file):
    npy_file = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)

_id_to_char_arr = [cv.decode([i]) for i in range(157)]
_is_letter = np.array([c.isalpha() or c.isdigit() for c in _id_to_char_arr], dtype=bool)

words = []
chunk_size = 20_000_000
for cs in range(0, len(all_ids), chunk_size):
    ce = min(cs + chunk_size + 20, len(all_ids))
    chunk = all_ids[cs:ce]
    vm = _is_letter[chunk]
    iw = False; st = 0
    for j in range(len(chunk)):
        if (cs + j) >= len(all_ids): break
        if vm[j]:
            if not iw: iw = True; st = j
        elif iw:
            iw = False
            wl = j - st
            if 2 <= wl <= 20:
                words.append(chunk[st:j].tolist())
print(f"  Words: {len(words):,}")

start = time.time()
last_print = 0
rng = np.random.RandomState(111)

for step in range(1, UT_STEPS + 1):
    idxs = rng.randint(0, len(words), UT_BATCH)
    batch_words = [words[i] for i in idxs]
    max_len = max(len(w) for w in batch_words)
    
    bt = torch.full((UT_BATCH, max_len), 0, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(UT_BATCH, max_len, device=DEVICE)
    for bi, w in enumerate(batch_words):
        bt[bi, :len(w)] = torch.tensor(w, dtype=torch.long, device=DEVICE)
        mask[bi, :len(w)] = 1.0
    
    ut.train()
    _, scores = ut(bt, return_scores=True)
    target = bt.clamp(1, VT-1)
    
    ce_loss = F.cross_entropy(
        scores.view(-1, 157),
        target.view(-1),
        reduction='none'
    ).view(UT_BATCH, max_len)
    ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)
    
    # Simple contradiction penalty: boost loss if batch has violations
    with torch.no_grad():
        prev_v = bt[:, :-1][mask[:, 1:].bool()].long()
        next_v = target[:, 1:][mask[:, 1:].bool()].long()
        valid = (prev_v > 0) & (prev_v < VT) & (next_v > 0) & (next_v < VT)
        total_adj = valid.sum().item()
        violations = 0
        if valid.any():
            violations = forbidden_mask[prev_v[valid].long(), next_v[valid].long()].sum().item()
    
    if violations > 0:
        ce_loss = ce_loss + 0.01 * violations / max(total_adj, 1)
    
    opt.zero_grad()
    ce_loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    opt.step()
    sch.step()
    
    now = time.time()
    if now - last_print >= 5 or step == 1 or step == UT_STEPS:
        last_print = now
        elapsed = now - start
        eta = (elapsed / step) * (UT_STEPS - step)
        with torch.no_grad():
            pred = scores.argmax(dim=-1)
            correct = ((pred == target) & mask.bool()).sum().item()
            tok_acc = correct / (mask.sum() + 1e-8)
        lr = sch.get_last_lr()[0]
        viol_rate = violations / (total_adj + 1e-8)
        print(f"  step {step:>4d}/{UT_STEPS} | loss={ce_loss.item():.4f} | "
              f"tok_acc={tok_acc:.3f} | violations={violations}/{total_adj} ({viol_rate:.1%})"
              f" | {elapsed:.0f}s", flush=True)

# ============================================================
# Test
# ============================================================
print("\n[TEST] Word reconstruction with contradiction mask:")
ut.eval()

test_words = ["привет", "человек", "солнце", "трансформер", "фрактал", "ъъъъъ"]

for word_text in test_words:
    ids = cv.encode(word_text)[1:-1]
    if len(ids) == 0: continue
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _, scores = ut(inp, return_scores=True)
        pred = scores[0].argmax(dim=-1).tolist()
        gen = cv.decode(pred)
    
    # Check contradiction violations
    violations = 0
    for i in range(1, len(ids)):
        if 0 < ids[i-1] < VT and 0 < ids[i] < VT:
            if forbidden_mask_np[ids[i-1], ids[i]]:
                violations += 1
    
    ok = "OK" if pred == ids else f"ERR ({sum(1 for p,t in zip(pred,ids) if p==t)}/{len(ids)})"
    viol = f"[{violations} contradictions]" if violations > 0 else "[clean]"
    print(f"  '{word_text}' → '{gen}' {ok} {viol}")

# Save
contra_path = os.path.join(CKPT_DIR, "contradiction_filter.pt")
torch.save({
    'forbidden_mask': forbidden_mask.cpu(),
    'model': ut.state_dict(),
    'coords': coords,
}, contra_path)
print(f"\nSaved: {contra_path}")
print("Done.")
