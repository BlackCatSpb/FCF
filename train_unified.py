"""
EVA Symbolic — Обучение UnifiedMultidimensionalTransformer.

Символ ≡ позиция в ℝ¹². Трансформер учится навигировать по координатам.
Loss = MSE(предсказанная_координата, координата_правильного_символа).
"""

import sys, os, time, torch, numpy as np, gc
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic import *
from eva.symbolic.advanced_methods import NGramContext

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 256; BLOCK = 64
STEPS = 2000; LR = 1e-3

print("=" * 60)
print("EVA Symbolic — Unified Transformer Training")
print("=" * 60)
print(f"Device: {DEVICE}, Batch: {BATCH}, Block: {BLOCK}")

# === 1. LOAD AFFINITY + COMPUTE COORDINATES ===
print("\n[1] Loading affinity + computing coordinates...")
pf = PotentialField(156, 256)
pf_path = os.path.join(CKPT_DIR, "final", "potential_field.pt")
if not os.path.exists(pf_path):
    pf_path = os.path.join(CKPT_DIR, "step_880000", "potential_field.pt")
if not os.path.exists(pf_path):
    pf_path = os.path.join(CKPT_DIR, "step_320000", "potential_field.pt")
if not os.path.exists(pf_path):
    print("No trained affinity found! Run train_to_convergence.py first.")
    sys.exit(1)

pf.load_state_dict(torch.load(pf_path, map_location='cpu', weights_only=True))
print(f"  Affinity: mean={pf.affinity.mean():.4f}, std={pf.affinity.std():.4f}")

# Compute MDS coordinates for symbols
from eva.symbolic.topological_field import TopologicalField
topo = TopologicalField(pf, coord_dim=12)
topo._compute_coordinates_from_affinity()
coords = topo.coordinates[:156, :12].clone()
print(f"  Coordinates: {coords.shape}")

# === 2. CREATE TRANSFORMER ===
print("\n[2] Creating UnifiedMultidimensionalTransformer...")
ut = UnifiedMultidimensionalTransformer(vocab_size=156, coord_dim=12)
ut.set_symbol_coordinates(coords)
if DEVICE == 'cuda':
    ut = ut.cuda()
    coords = coords.cuda()
print(f"  {ut.summary()}")

# === 3. DATASET ===
print("\n[3] Loading dataset...")
char_vocab = CharacterVocab()
npy_file = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy_file):
    npy_file = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)
print(f"  Dataset: {len(all_ids)/1e6:.1f}M tokens")

# === 4. TRAINING ===
print(f"\n[4] Training {STEPS} batches...")
opt = torch.optim.AdamW(ut.parameters(), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

pos = 0; start = time.time(); total_loss = 0
PAD = char_vocab.PAD_IDX

for step in range(1, STEPS + 1):
    if pos + BLOCK + 2 > len(all_ids): pos = 0
    ids_batch, lens = [], []
    for _ in range(BATCH):
        if pos + BLOCK + 2 > len(all_ids): pos = 0
        end = min(pos + BLOCK, len(all_ids))
        chunk = all_ids[pos:end]
        sep = np.where((chunk == 0) | (chunk == 3))[0]
        if len(sep) > 0 and sep[0] < BLOCK // 2:
            end = pos + sep[0] + 1; chunk = all_ids[pos:end]
        ids = [int(x) for x in chunk if x >= 0][:BLOCK]
        ids_batch.append(ids); lens.append(len(ids))
        pos += max(len(ids), 32)

    ml = max(lens)
    bt = torch.full((BATCH, ml), PAD, dtype=torch.long, device=DEVICE)
    for i, ids in enumerate(ids_batch):
        bt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=DEVICE)

    # Target: shift right by 1
    target = torch.roll(bt, -1, dims=1)
    # Mask: non-PAD positions
    mask = (bt != PAD).float()

    ut.train()
    loss = ut.compute_loss(bt, target, mask)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    opt.step()
    sch.step()
    total_loss += loss.item()

    if step % 1000 == 0:
        elapsed = time.time() - start
        avg_l = total_loss / 1000; total_loss = 0
        bps = step / max(elapsed, 0.01)
        print(f"  step={step} | {bps:.0f} b/s | loss={avg_l:.4f} | {elapsed/3600:.1f}h")
        gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()

    if step % 10000 == 0:
        os.makedirs(os.path.join(CKPT_DIR, "unified"), exist_ok=True)
        torch.save(ut.state_dict(), os.path.join(CKPT_DIR, "unified", "transformer.pt"))
        print(f"  [Saved]")

elapsed = time.time() - start
print(f"\nDone: {STEPS} steps in {elapsed:.0f}s | loss={loss.item():.4f}")
os.makedirs(os.path.join(CKPT_DIR, "unified"), exist_ok=True)
torch.save(ut.state_dict(), os.path.join(CKPT_DIR, "unified", "transformer_final.pt"))

# === 5. RECONSTRUCTION TEST ===
print("\n[5] Reconstruction test...")
char_vocab_inv = CharacterVocab()
ut.eval()

for test_word in ["привет", "человек", "мир"]:
    ids = char_vocab_inv.encode(test_word)
    inp = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)  # без EOS
    with torch.no_grad():
        _, scores = ut(inp, return_scores=True)
    predicted = torch.argmax(scores[0], dim=-1).tolist()
    generated = char_vocab_inv.decode(predicted)
    print(f"  '{test_word}' → '{generated}'")

print("\nDone.")
