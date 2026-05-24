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
pf_path = os.path.join(CKPT_DIR, "potential_function.pt")
if os.path.exists(pf_path):
    pf_data = torch.load(pf_path, map_location='cpu', weights_only=False)
    from concept_finder import PotentialFunction
    v_func = PotentialFunction(dim=24, hidden=128).to(DEVICE)
    v_func.load_state_dict(pf_data['model'])
    v_func.eval()
    print("Loaded: PotentialFunction V(z)")
else:
    v_func = None
    print("PotentialFunction not found, skipping potential-based detection")

# ============================================================
# Build contradiction types
# ============================================================
print("\n[DETECT] Building contradiction mask...")

aff = affinity.numpy()
coords_np = coords.numpy()

forbidden = np.zeros((VT, VT), dtype=bool)

# Type 1: Structural — very low affinity
struct_threshold = 0.1
struct_forbidden = aff < struct_threshold
np.fill_diagonal(struct_forbidden, False)  # self not forbidden
forbidden |= struct_forbidden
n_struct = struct_forbidden.sum()
print(f"  Structural (aff < {struct_threshold}): {n_struct} forbidden pairs")

# Type 2: Semantic — opposed continuation vectors
# Normalize each row to get continuation distribution
aff_norm = aff / (aff.sum(axis=1, keepdims=True) + 1e-8)
sem_threshold = 0.1
sem_forbidden = np.zeros((VT, VT), dtype=bool)
for i in range(1, VT):  # skip PAD
    for j in range(1, VT):
        if i == j: continue
        cos_sim = np.dot(aff_norm[i], aff_norm[j]) / (np.linalg.norm(aff_norm[i]) * np.linalg.norm(aff_norm[j]) + 1e-8)
        if cos_sim < sem_threshold and aff[i, j] < 0.5:
            sem_forbidden[i, j] = True

forbidden |= sem_forbidden
print(f"  Semantic (cos < {sem_threshold}): {sem_forbidden.sum()} forbidden pairs")

# Type 3: Coordinate distance — very far in ℝ²⁴
dist_matrix = np.zeros((VT, VT))
for i in range(VT):
    diff = coords_np[i:i+1] - coords_np
    dist_matrix[i] = np.linalg.norm(diff, axis=1)

dist_threshold = 1.5  # max distance on unit sphere is 2.0
dist_forbidden = dist_matrix > dist_threshold
np.fill_diagonal(dist_forbidden, False)
forbidden |= dist_forbidden
print(f"  Distance (> {dist_threshold}): {dist_forbidden.sum()} forbidden pairs")

# Type 4: Potential barrier (if V(z) available)
if v_func is not None:
    with torch.no_grad():
        pot_forbidden = np.zeros((VT, VT), dtype=bool)
        # Sample pairs and check V at midpoint
        for i in range(1, min(VT, 50)):
            for j in range(1, min(VT, 50)):
                if i == j: continue
                za = coords[i:i+1].to(DEVICE)
                zb = coords[j:j+1].to(DEVICE)
                mid = (za + zb) / 2.0
                v_mid = v_func(mid).item()
                if v_mid > 1.5:  # high potential = forbidden region
                    pot_forbidden[i, j] = True
        forbidden |= pot_forbidden
        print(f"  Potential (V_mid > 1.5): {pot_forbidden.sum()} forbidden pairs")
else:
    print("  Potential: skipped (no V(z) model)")

total_forbidden = forbidden.sum()
total_pairs = VT * VT - VT  # exclude diagonal
print(f"\n  TOTAL forbidden: {total_forbidden}/{total_pairs} ({total_forbidden/total_pairs:.1%})")

# Expand full mask for all pairs (batch computation for speed)
# For pairs not explicitly tested, use affinity threshold
full_forbidden = aff < 0.05  # very low affinity always forbidden
full_forbidden |= (aff < 0.2) & (dist_matrix > 1.8)  # far + weak
np.fill_diagonal(full_forbidden, False)
print(f"  Full mask (expanded): {full_forbidden.sum()}/{total_pairs} forbidden")

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

forbidden_mask = torch.tensor(full_forbidden, dtype=torch.bool, device=DEVICE)  # [VT, VT]

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
    _, scores = ut(bt, return_scores=True)  # [B, L, 157]
    target = bt.clamp(1, VT-1)
    
    # Standard CE
    ce_loss = F.cross_entropy(
        scores.view(-1, 157),
        target.view(-1),
        reduction='none'
    ).view(UT_BATCH, max_len)
    ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)
    
    # Contradiction penalty: if prediction would be forbidden, penalize
    with torch.no_grad():
        pred = scores.argmax(dim=-1)  # [B, L]
        # For each position, check if predicted transition is forbidden
        # (target is the actual next symbol, not predicted)
        # Check: is target symbol forbidden from context?
        for bi in range(min(UT_BATCH, 32)):  # sample batch for speed
            for pos in range(1, max_len):
                if mask[bi, pos] == 0: continue
                prev = bt[bi, pos-1].item()
                curr = target[bi, pos].item()
                if 0 < prev < VT and 0 < curr < VT:
                    if forbidden_mask[prev, curr]:
                        # Add penalty for predicting forbidden transition
                        ce_loss = ce_loss + 0.5 * ce_loss  # boost loss
    
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
            
            # Count contradiction violations
            violations = 0; total_adj = 0
            for bi in range(min(UT_BATCH, 32)):
                for pos in range(1, max_len):
                    if mask[bi, pos] == 0: continue
                    p = bt[bi, pos-1].item()
                    c = target[bi, pos].item()
                    if 0 < p < VT and 0 < c < VT:
                        total_adj += 1
                        if forbidden_mask[p, c]:
                            violations += 1
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
            if forbidden_mask[ids[i-1], ids[i]].item():
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
