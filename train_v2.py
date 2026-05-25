"""
EVA — Training v2: 64-dim, 16 heads, 3 layers, fp16 mixed precision.
"""
import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
LOG = os.path.join(os.path.dirname(__file__), "training_log_v2.txt")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
cv = CharacterVocab(); VT = 157

evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)

# Initialize coordinates: 24 evolved dims + 40 seeded random dims
coords_64 = torch.zeros(157, 64, device=DEVICE)
coords_64[:, :24] = coords[:, :24]
g = torch.Generator(device=DEVICE).manual_seed(42)
coords_64[:, 24:] = torch.randn(157, 40, generator=g, device=DEVICE) * 0.02
# Make unit-norm in 64D
coords_64 = coords_64 / coords_64.norm(dim=-1, keepdim=True).clamp(min=1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=64, num_levels=4, scales_per_level=4,
                                        num_layers=3, d_ff=128).to(DEVICE)
ut.set_symbol_coordinates(coords_64)

npy = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy): npy = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)

STEPS = 100000; LR = 1e-3; B = 96; ML = 128
SAVE_EVERY = 10000
opt = torch.optim.AdamW(ut.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None
rng = np.random.RandomState(42)

def log(msg):
    t = time.strftime("%H:%M:%S"); line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
        f.flush()

log(f"START v2: {STEPS} steps, dim=64, heads=16, layers=3, fp16")
log(f"Model: {sum(p.numel() for p in ut.parameters()):,} params, batch={B}, block={ML}")
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
        if vl >= 4: bt[bi, :vl] = torch.from_numpy(vb[:vl].astype(np.int64)).to(DEVICE); mask[bi, :vl] = 1.0
    
    if mask.sum() < 50: continue
    
    with torch.amp.autocast('cuda'):
        ut.train()
        _, scores = ut(bt, return_scores=True)
        target = bt[:, 1:].clamp(1, VT - 1).contiguous()
        pred = scores[:, :-1, :].contiguous()
        t_mask = mask[:, 1:]
        
        loss = F.cross_entropy(pred.view(-1, 157), target.view(-1), reduction='none')
        loss = (loss.view(B, ml - 1) * t_mask).sum() / (t_mask.sum() + 1e-8)
    
    opt.zero_grad()
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
        opt.step()
    sch.step()
    
    if s % SAVE_EVERY == 0:
        with torch.no_grad():
            acc = (pred.argmax(-1) == target) & t_mask.bool()
            acc = acc.sum().item() / (t_mask.sum() + 1e-8)
        torch.save({'model': ut.state_dict(), 'step': s, 'loss': loss.item()},
                   os.path.join(CKPT, f"v2_{s}.pt"))
        torch.save({'model': ut.state_dict(), 'step': s, 'loss': loss.item()},
                   os.path.join(CKPT, "v2_latest.pt"))
        elapsed = time.time() - t0; eta = (elapsed / s) * (STEPS - s) if s > 0 else 0
        log(f"  step {s:>6d}/{STEPS} | loss={loss.item():.4f} | acc={acc:.3f} | {elapsed/60:.0f}min | eta {eta/60:.0f}min")
    elif s % 500 == 0:
        with torch.no_grad():
            acc = (pred.argmax(-1) == target) & t_mask.bool()
            acc = acc.sum().item() / (t_mask.sum() + 1e-8)
        elapsed = time.time() - t0
        log(f"  step {s:>6d}/{STEPS} | loss={loss.item():.4f} | acc={acc:.3f} | {elapsed/60:.0f}min")

log("DONE")
