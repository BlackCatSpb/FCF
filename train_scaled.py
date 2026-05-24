"""
EVA — Scaled Causal Training.

Масштабированное обучение:
- 100K шагов (вместо 8K)
- Batch size 128, block size 96
- Сохранение каждые 5K шагов
- Логирование в файл
- Использование всей GPU памяти
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
LOG_FILE = os.path.join(os.path.dirname(__file__), "training_log.txt")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — Scaled Causal Training")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Log: {LOG_FILE}")

# Load coordinates
evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24).to(DEVICE)
ut.set_symbol_coordinates(coords)

# Optionally load existing causal weights as starting point
causal_path = os.path.join(CKPT, "causal_weights.pt")
if os.path.exists(causal_path):
    print("Loading existing causal weights as starting point...")
    ut.load_state_dict(torch.load(causal_path, map_location='cpu', weights_only=True)['model'], strict=False)

# Load data
npy = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy):
    npy = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32)
total = len(data)
print(f"Corpus: {total/1e6:.1f}M tokens")

STEPS = 100000; LR = 5e-4; B = 128; ML = 96
SAVE_EVERY = 5000
opt = torch.optim.AdamW(ut.parameters(), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
rng = np.random.RandomState(42)

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

log(f"START: {STEPS} steps, batch={B}, block={ML}, lr={LR}")
log(f"Model: {sum(p.numel() for p in ut.parameters()):,} params")
t0 = time.time()

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
        if vl >= 4:
            bt[bi, :vl] = torch.from_numpy(vb[:vl].astype(np.int64)).to(DEVICE)
            mask[bi, :vl] = 1.0
    
    if mask.sum() < 50:
        continue
    
    ut.train()
    _, scores = ut(bt, return_scores=True)
    target = bt[:, 1:].clamp(1, VT - 1).contiguous()
    pred = scores[:, :-1, :].contiguous()
    t_mask = mask[:, 1:]
    
    loss = F.cross_entropy(pred.view(-1, 157), target.view(-1), reduction='none')
    loss = (loss.view(B, ml - 1) * t_mask).sum() / (t_mask.sum() + 1e-8)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    opt.step()
    sch.step()
    
    if s % SAVE_EVERY == 0:
        ckpt_path = os.path.join(CKPT, f"causal_{s}.pt")
        torch.save({'model': ut.state_dict(), 'step': s, 'loss': loss.item()}, ckpt_path)
        torch.save({'model': ut.state_dict(), 'step': s, 'loss': loss.item()},
                   os.path.join(CKPT, "causal_latest.pt"))
        
        with torch.no_grad():
            acc = (pred.argmax(-1) == target) & t_mask.bool()
            acc_val = acc.sum().item() / (t_mask.sum() + 1e-8)
        
        elapsed = time.time() - t0
        eta = (elapsed / s) * (STEPS - s) if s > 0 else 0
        log(f"  step {s:>6d}/{STEPS} | loss={loss.item():.4f} | acc={acc_val:.3f} | "
            f"{elapsed/60:.0f}min | eta {eta/60:.0f}min | saved")
    elif s % 500 == 0:
        with torch.no_grad():
            acc = (pred.argmax(-1) == target) & t_mask.bool()
            acc_val = acc.sum().item() / (t_mask.sum() + 1e-8)
        elapsed = time.time() - t0
        log(f"  step {s:>6d}/{STEPS} | loss={loss.item():.4f} | acc={acc_val:.3f} | "
            f"{elapsed/60:.0f}min")

log("DONE")
