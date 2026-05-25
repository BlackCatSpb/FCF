"""
EVA — Local 128-dim training (resume from Yandex checkpoint).
Fits in 2.1 GB VRAM: batch=16, block=64.
"""
import torch, torch.nn.functional as F, numpy as np, sys, os, time, random
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
LOG = os.path.join(os.path.dirname(__file__), "local128_log.txt")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.self_reflection import SelfReflection
from eva.symbolic.active_causal import ActiveLearner
from eva.symbolic.trajectory_store import TrajectoryStore
from train_genetics import fitness, mutate, crossover
cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — Local 128-dim (from Yandex)")
print("=" * 60)

# Load Yandex checkpoint
evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)
c128 = torch.zeros(157, 128, device=DEVICE); c128[:, :24] = coords[:, :24]
g = torch.Generator(device=DEVICE).manual_seed(42)
c128[:, 24:] = torch.randn(157, 104, generator=g, device=DEVICE) * 0.02
c128 = c128 / c128.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=128, num_levels=8,
    scales_per_level=4, num_layers=6, d_ff=512).to(DEVICE)
ut.set_symbol_coordinates(c128)

yandex_ckpt = os.path.join(CKPT, "yandex_backup.pt")
ut.load_state_dict(torch.load(yandex_ckpt, map_location='cpu', weights_only=True)['ut'], strict=False)
print(f"Resumed from: yandex_backup.pt (step 2500)")

store_path = os.path.join(CKPT, "trajectory_store.pkl")
store = TrajectoryStore()
if os.path.exists(store_path): store.load(store_path)

npy = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy): npy = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)

# Extract words for word-aligned sampling
print("Extracting words for boundary-aligned sampling...")
_id_to_char = [cv.decode([i]) for i in range(157)]
_is_letter = np.array([c.isalpha() or c.isdigit() for c in _id_to_char], dtype=bool)

words_list = []
i = 0
chunk_size = 20_000_000
for cs in range(0, total, chunk_size):
    ce = min(cs + chunk_size + 20, total)
    chunk = data[cs:ce]
    vm = _is_letter[chunk]
    in_word = False; st = 0
    for j in range(len(chunk)):
        if (cs + j) >= total: break
        if vm[j]:
            if not in_word: in_word = True; st = j
        elif in_word:
            in_word = False
            wl = j - st
            if 2 <= wl <= 20:
                words_list.append(chunk[st:j].tolist())
print(f"  Words: {len(words_list):,}")

STEPS = 100000; LR = 3e-4; B = 32; ML = 64
SAVE_EVERY = 5000; THINK_EVERY = 1000; GEN_EVERY = 5000
opt = torch.optim.AdamW(ut.parameters(), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
rng = np.random.RandomState(42)

def log(msg):
    t = time.strftime("%H:%M:%S"); line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(line+'\n'); f.flush()

log(f"START: {STEPS} steps, 128-dim, 1.6M params, batch={B}")

t0 = time.time()
for s in range(1, STEPS + 1):
    # Sample words, concatenate to block length
    ids_flat = []
    while len(ids_flat) < ML:
        widx = rng.randint(0, len(words_list))
        ids_flat.extend(words_list[widx])
        ids_flat.append(7)  # space
    ids_flat = ids_flat[:ML]
    
    bt = torch.full((B, ML), 0, dtype=torch.long, device=DEVICE)
    mask = torch.ones(B, ML, device=DEVICE)
    bt[0, :len(ids_flat)] = torch.tensor(ids_flat, dtype=torch.long, device=DEVICE)
    
    # Pad shorter sequences
    for bi in range(1, B):
        ids2 = []
        while len(ids2) < ML:
            widx = rng.randint(0, len(words_list))
            ids2.extend(words_list[widx])
            ids2.append(7)
        ids2 = ids2[:ML]
        bt[bi, :len(ids2)] = torch.tensor(ids2, dtype=torch.long, device=DEVICE)
    
    if mask.sum() < 20: continue
    
    ut.train(); _, scores = ut(bt, return_scores=True)
    target = bt[:,1:].clamp(1,VT-1).contiguous(); pred = scores[:,:-1].contiguous(); tm = mask[:,1:]
    loss = F.cross_entropy(pred.view(-1,157), target.view(-1), reduction='none')
    loss = (loss.view(B, ML-1)*tm).sum()/(tm.sum()+1e-8)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ut.parameters(),1.0); opt.step(); sch.step()
    
    if s % SAVE_EVERY == 0:
        torch.save({'ut': ut.state_dict(), 'step': 2500+s}, os.path.join(CKPT, f"local128_{2500+s}.pt"))
        torch.save({'ut': ut.state_dict(), 'step': 2500+s}, os.path.join(CKPT, "local128_latest.pt"))
        with torch.no_grad(): acc = ((pred.argmax(-1)==target)&tm.bool()).sum().item()/(tm.sum()+1e-8)
        elapsed = time.time()-t0
        log(f"  step {2500+s:>7d} | loss={loss.item():.4f} acc={acc:.3f} | {elapsed/60:.0f}min")
    elif s % 500 == 0:
        with torch.no_grad(): acc = ((pred.argmax(-1)==target)&tm.bool()).sum().item()/(tm.sum()+1e-8)
        log(f"  step {2500+s:>7d} | loss={loss.item():.4f} acc={acc:.3f} | {(time.time()-t0)/60:.0f}min")

log("DONE")
