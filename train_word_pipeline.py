"""
EVA — Word-Level Pipeline.

Phase 1: Affinity matrix from text (co-occurrence counting)
Phase 2: MDS → topological coordinates in ℝ²⁴
Phase 3: Word autoencoding (coordinates trajectory → word symbols)
Phase 4: Word reconstruction test (matrix instructs → transformer assembles)
"""

import sys, os, time, torch, torch.nn.functional as F, numpy as np, gc, math
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from eva.symbolic import *
from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(CKPT_DIR, exist_ok=True)

cv = CharacterVocab()
VT = 157  # total vocab: PAD(0) + 156 symbols
PAD = cv.PAD_IDX  # 0

print("=" * 60)
print("EVA — Word-Level Pipeline")
print("=" * 60)
print(f"Device: {DEVICE}")

# ============================================================
# Load text corpus
# ============================================================
npy_file = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy_file):
    npy_file = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)
print(f"\nCorpus: {len(all_ids)/1e6:.1f}M tokens")

# ============================================================
# Pre-extract words from corpus
# ============================================================
print("Extracting words from corpus...")

# Precompute which IDs are letters or digits (fast vectorized check)
_id_to_char_arr = [cv.decode([i]) for i in range(157)]
_is_letter_digit = np.array([c.isalpha() or c.isdigit() for c in _id_to_char_arr], dtype=bool)

# Extract words: contiguous letter/digit sequences (min length 2, max length 20)
words = []
i = 0
total = len(all_ids)
chunk_size = 10_000_000
for chunk_start in range(0, total, chunk_size):
    chunk_end = min(chunk_start + chunk_size + 20, total)  # overlap for boundaries
    chunk = all_ids[chunk_start:chunk_end]
    valid = _is_letter_digit[chunk]
    
    # Find word boundaries
    in_word = False; start = 0
    for j in range(len(chunk)):
        if (chunk_start + j) >= total:
            break
        if valid[j]:
            if not in_word:
                in_word = True; start = j
        else:
            if in_word:
                in_word = False
                word_len = j - start
                if 2 <= word_len <= 20:
                    words.append(chunk[start:j].tolist())
    
    if (chunk_start + chunk_size) % 50_000_000 < chunk_size:
        print(f"  ... {(chunk_start+chunk_size)/1e6:.0f}M tokens, {len(words):,} words")

print(f"  Words extracted: {len(words):,}")

# ============================================================
# PHASE 1: Affinity matrix — pure co-occurrence counting
# ============================================================
print("\n[PHASE 1] Affinity matrix (pure co-occurrence, GPU)...")

affinity_path = os.path.join(CKPT_DIR, "affinity_word.pt")

if os.path.exists(affinity_path):
    print("  Loading existing affinity...")
    aff_data = torch.load(affinity_path, map_location='cpu', weights_only=True)
    co_occurrence = aff_data['co_occurrence'].to(torch.float64)
    affinity = aff_data['affinity'].to(torch.float32)
    print(f"  Affinity mean={affinity.mean():.4f} std={affinity.std():.4f}")
else:
    co_occurrence = torch.zeros(VT, VT, dtype=torch.float64, device=DEVICE)
    affinity = torch.full((VT, VT), 0.5, dtype=torch.float32, device=DEVICE)
    
    AFF_STEPS = 50000
    pos = 0
    start = time.time()
    last_print = 0
    
    # Pre-compute valid range mask
    valid_range = (np.arange(len(all_ids)) < len(all_ids) - 1)
    
    for step in range(1, AFF_STEPS + 1):
        # Chunk-based counting: process ~1M adjacent pairs per step
        chunk_size = min(1000000, len(all_ids) - pos - 1)
        if chunk_size < 1000:
            pos = 0
            chunk_size = min(1000000, len(all_ids) - pos - 1)
        
        end = pos + chunk_size
        left = all_ids[pos:end]
        right = all_ids[pos+1:end+1]
        
        valid = (left > 0) & (left < VT) & (right > 0) & (right < VT)
        lv = left[valid]; rv = right[valid]
        
        if len(lv) > 0:
            lv_t = torch.from_numpy(lv).long().to(DEVICE)
            rv_t = torch.from_numpy(rv).long().to(DEVICE)
            flat_idx = lv_t * VT + rv_t
            inc = torch.ones(len(flat_idx), dtype=torch.float64, device=DEVICE)
            co_occurrence.view(-1).index_add_(0, flat_idx, inc)
        
        pos += chunk_size
        
        now = time.time()
        if now - last_print >= 3 or step == 1:
            last_print = now
            elapsed = now - start
            aps = step * chunk_size / max(elapsed, 0.01)
            total_pairs = co_occurrence.sum().item()
            # Update affinity
            raw = co_occurrence / 10000.0  # lower threshold for better signal
            affinity = (0.5 + 0.5 * torch.clamp(raw, 0.0, 1.0)).float()
            pot = affinity.mean().item()
            print(f"  step {step:>5d}/{AFF_STEPS} | {aps:,.0f} pairs/s | "
                  f"pairs={total_pairs:,.0f} | pot={pot:.4f} | {elapsed:.0f}s", flush=True)
    
    # Save
    co_occurrence_cpu = co_occurrence.cpu()
    affinity_cpu = affinity.cpu()
    torch.save({'co_occurrence': co_occurrence_cpu, 'affinity': affinity_cpu}, affinity_path)
    print(f"  Saved: {affinity_path}")
    
    co_occurrence = co_occurrence_cpu
    affinity = affinity_cpu

# ============================================================
# PHASE 2: MDS → topological coordinates
# ============================================================
print("\n[PHASE 2] MDS → topological coordinates in ℝ²⁴...")

from eva.symbolic.topological_field import TopologicalField
from eva.symbolic.potential_field import PotentialField

pf = PotentialField(VT, 256)
pf.affinity = torch.nn.Parameter(affinity, requires_grad=False)
pf.co_occurrence_count = torch.nn.Parameter(co_occurrence, requires_grad=False)

topo = TopologicalField(pf, coord_dim=24)
topo._compute_coordinates_from_affinity()
coords_full = topo.coordinates[:VT, :24].clone()  # [157, 24] — includes PAD

# Diagnostic (on symbols only, excl PAD)
sym_coords = coords_full[1:VT].clone()  # [156, 24]
sym_np = sym_coords.cpu().numpy()
n = 156
topo_aff = affinity.cpu().numpy()
# MDS quality: classical scaling on 156×156 submatrix (excl PAD)
aff_156 = topo_aff[1:VT, 1:VT]
D_mds = 1.0 - aff_156; np.fill_diagonal(D_mds, 0.0)
J = np.eye(n) - np.ones((n,n))/n; B = -0.5*J@(D_mds*D_mds)@J
eigvals = np.linalg.eigh(B)[0]; eigvals = np.sort(eigvals)[::-1]
eff_dim = (eigvals[:24].sum() / eigvals[eigvals>0].sum()) if (eigvals>0).sum()>0 else 0
unique_coords = len(np.unique(sym_coords.numpy().round(decimals=6), axis=0))
print(f"  MDS: eff_dim(24)={eff_dim:.1%}, unique={unique_coords}/{n}")
print(f"  Coordinates: {coords_full.shape}")

# ============================================================
# PHASE 3: Word autoencoding (coordinates trajectory → words)
# ============================================================
print("\n[PHASE 3] Transformer — word autoencoding (GPU)")
print("  Goal: learn to assemble words from coordinate trajectories")
print("  100% per-word accuracy required.")

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(coords_full.to(DEVICE))
print(f"  {ut.summary()}")

# Check for existing weights
word_weights_path = os.path.join(CKPT_DIR, "word_weights.pt")
if os.path.exists(word_weights_path):
    print("  Loading existing word weights, skipping training...")
    ckpt = torch.load(word_weights_path, map_location='cpu', weights_only=True)
    ut.load_state_dict(ckpt['model'])
    ut.set_symbol_coordinates(ckpt['coords'].to(DEVICE))
    ut = ut.to(DEVICE)
else:
    print(f"  {ut.summary()}")
    UT_STEPS = 30000; UT_LR = 1e-3; UT_BATCH = 128
    opt = torch.optim.AdamW(ut.parameters(), lr=UT_LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UT_STEPS)
    
    start = time.time()
    last_print = 0
    rng = np.random.RandomState(42)
    
    for step in range(1, UT_STEPS + 1):
        idxs = rng.randint(0, len(words), UT_BATCH)
        batch_words = [words[i] for i in idxs]
        max_len = max(len(w) for w in batch_words)
        
        bt = torch.full((UT_BATCH, max_len), PAD, dtype=torch.long, device=DEVICE)
        mask = torch.zeros(UT_BATCH, max_len, device=DEVICE)
        for bi, w in enumerate(batch_words):
            bt[bi, :len(w)] = torch.tensor(w, dtype=torch.long, device=DEVICE)
            mask[bi, :len(w)] = 1.0
        
        ut.train()
        _, scores = ut(bt, return_scores=True)
        
        target = bt.clamp(1, VT-1)
        
        loss = F.cross_entropy(
            scores.view(-1, 157),
            target.view(-1),
            reduction='none'
        ).view(UT_BATCH, max_len)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
        opt.step()
        sch.step()
        
        now = time.time()
        if now - last_print >= 3 or step == 1 or step == UT_STEPS:
            last_print = now
            elapsed = now - start
            eta = (elapsed / step) * (UT_STEPS - step)
            with torch.no_grad():
                pred = scores.argmax(dim=-1)
                correct_tokens = ((pred == target) * mask).sum().item()
                total_tokens = mask.sum().item()
                per_word = ((pred == target) | (mask == 0)).all(dim=1).sum().item()
            lr = sch.get_last_lr()[0]
            print(f"  step {step:>5d}/{UT_STEPS} | loss={loss.item():.4f} | "
                  f"tok_acc={correct_tokens/total_tokens:.3f} | "
                  f"word_acc={per_word}/{UT_BATCH} | lr={lr:.6f} | "
                  f"{elapsed:.0f}s / eta {eta:.0f}s", flush=True)
        
        if step > 500 and loss.item() < 0.01:
            print(f"\n  Early exit at step {step}: loss stable.")
            break
    
    torch.save({'model': ut.state_dict(), 'coords': coords_full}, word_weights_path)
    print(f"  Saved: {word_weights_path}")

# ============================================================
# PHASE 4: Word reconstruction test
# ============================================================
print("\n[PHASE 4] Word reconstruction test")
print("  Matrix instructs → transformer assembles word")

ut.eval()
test_words_list = [
    "привет", "человек", "солнце", "программа", "обучение",
    "трансформер", "метаданные", "инструкция", "фрактал",
    "мир", "дом", "кот", "год", "весна",
]

for word_text in test_words_list:
    ids = cv.encode(word_text)[1:-1]  # strip BOS/EOS — training uses raw IDs
    if len(ids) == 0:
        print(f"  '{word_text}' → SKIP (empty)")
        continue
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _, scores = ut(inp, return_scores=True)
        pred = scores[0].argmax(dim=-1).tolist()
        generated = cv.decode(pred)
    correct = sum(1 for p, t in zip(pred, ids) if p == t)
    acc = correct / max(len(ids), 1)
    status = "OK" if acc == 1.0 else f"ERR ({correct}/{len(ids)})"
    print(f"  '{word_text}' → '{generated}' [{status}]")

# ============================================================
# PHASE 5: Sentence autoencoding (longer sequences)
# ============================================================
print("\n[PHASE 5] Transformer — sentence autoencoding (GPU)")
print("  Goal: reproduce sentences from coordinate trajectories")
print("  Training on random blocks from corpus (len 10-128)")

sent_weights_path = os.path.join(CKPT_DIR, "sentence_weights.pt")
if os.path.exists(sent_weights_path):
    print("  Loading existing sentence weights, skipping training...")
    ckpt = torch.load(sent_weights_path, map_location='cpu', weights_only=True)
    ut.load_state_dict(ckpt['model'])
    ut.set_symbol_coordinates(ckpt['coords'].to(DEVICE))
    ut = ut.to(DEVICE)
else:
    UT_SENT_STEPS = 20000; UT_LR = 1e-4; UT_BATCH = 64; MAX_LEN = 128
    opt = torch.optim.AdamW(ut.parameters(), lr=UT_LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UT_SENT_STEPS)
    
    start = time.time()
    last_print = 0
    rng = np.random.RandomState(123)
    total_ids = len(all_ids)
    
    for step in range(1, UT_SENT_STEPS + 1):
        # Random blocks from corpus
        lengths = rng.randint(10, MAX_LEN + 1, UT_BATCH)
        starts = rng.randint(0, max(1, total_ids - max(lengths) - 1), UT_BATCH)
        
        max_len = max(lengths)
        bt = torch.full((UT_BATCH, max_len), PAD, dtype=torch.long, device=DEVICE)
        mask = torch.zeros(UT_BATCH, max_len, device=DEVICE)
        
        for bi in range(UT_BATCH):
            s, l = starts[bi], lengths[bi]
            block = all_ids[s:s+l]
            valid = (block > 0) & (block < VT)
            valid_block = block[valid]
            vl = min(len(valid_block), max_len)
            if vl < 3:
                vl = 0
            if vl > 0:
                bt[bi, :vl] = torch.from_numpy(valid_block[:vl].astype(np.int64)).to(DEVICE)
                mask[bi, :vl] = 1.0
        
        mask_sum = mask.sum()
        if mask_sum < 10:
            continue
        
        ut.train()
        _, scores = ut(bt, return_scores=True)
        
        target = bt.clamp(1, VT-1)
        
        loss = F.cross_entropy(
            scores.view(-1, 157),
            target.view(-1),
            reduction='none'
        ).view(UT_BATCH, max_len)
        loss = (loss * mask).sum() / (mask_sum + 1e-8)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
        opt.step()
        sch.step()
        
        now = time.time()
        if now - last_print >= 3 or step == 1 or step == UT_SENT_STEPS:
            last_print = now
            elapsed = now - start
            eta = (elapsed / step) * (UT_SENT_STEPS - step)
            with torch.no_grad():
                pred = scores.argmax(dim=-1)
                correct_tokens = ((pred == target) * mask).sum().item()
                total_tokens = mask_sum.item()
                per_block = ((pred == target) | (mask == 0)).all(dim=1).sum().item()
            lr = sch.get_last_lr()[0]
            print(f"  step {step:>5d}/{UT_SENT_STEPS} | loss={loss.item():.4f} | "
                  f"tok_acc={correct_tokens/total_tokens:.3f} | "
                  f"block_acc={per_block}/{UT_BATCH} | lr={lr:.6f} | "
                  f"{elapsed:.0f}s / eta {eta:.0f}s", flush=True)
        
        if step > 500 and loss.item() < 0.01:
            print(f"\n  Early exit at step {step}: loss stable.")
            break
    
    torch.save({'model': ut.state_dict(), 'coords': coords_full}, sent_weights_path)
    print(f"  Saved: {sent_weights_path}")

# ============================================================
# PHASE 6: Sentence reconstruction test
# ============================================================
print("\n[PHASE 6] Sentence reconstruction test")
print("  Matrix instructs → transformer assembles sentence")

ut.eval()
test_sentences = [
    "привет мир",
    "человек идёт",
    "солнце светит ярко",
    "мама мыла раму",
    "я люблю программирование",
    "трансформер понимает текст",
    "метаданные хранят порядок",
]

for sentence in test_sentences:
    ids = cv.encode(sentence)[1:-1]
    if len(ids) == 0:
        print(f"  '{sentence}' → SKIP")
        continue
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _, scores = ut(inp, return_scores=True)
        pred = scores[0].argmax(dim=-1).tolist()
        generated = cv.decode(pred)
    correct = sum(1 for p, t in zip(pred, ids) if p == t)
    acc = correct / max(len(ids), 1)
    status = "OK" if acc == 1.0 else f"ERR ({correct}/{len(ids)})"
    print(f"  '{sentence}' → '{generated}' [{status}]")

print("\nDone.")
