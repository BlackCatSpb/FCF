"""
EVA — Extended Causal Training (resume from 100K).

Продолжение масштабированного обучения:
- +100K шагов (всего 200K)
- Больше блок (128 вместо 96)
- Сохранение каждые 10K шагов
"""
import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
LOG = os.path.join(os.path.dirname(__file__), "training_log2.txt")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
cv = CharacterVocab(); VT = 157

evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24).to(DEVICE)
ut.set_symbol_coordinates(coords)

# Resume from latest
latest = os.path.join(CKPT, "causal_latest.pt")
ckpt = torch.load(latest, map_location='cpu', weights_only=True)
ut.load_state_dict(ckpt['model'], strict=False)
prev_step = ckpt.get('step', 100000)
print(f"Resumed from step {prev_step}")

npy = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy): npy = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)

EXTRA = 100000; TOTAL = prev_step + EXTRA; LR = 3e-4; B = 128; ML = 128
SAVE_EVERY = 10000
opt = torch.optim.AdamW(ut.parameters(), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EXTRA)
rng = np.random.RandomState(123)

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(line+'\n')

log(f"RESUME: +{EXTRA} steps (total {TOTAL}), batch={B}, block={ML}")
t0 = time.time()

for s in range(1, EXTRA + 1):
    lens = rng.randint(32, ML+1, B)
    starts = rng.randint(0, max(1, total - max(lens) - 1), B)
    ml = max(lens)
    bt = torch.full((B, ml), 0, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(B, ml, device=DEVICE)
    
    for bi in range(B):
        vb = data[starts[bi]:starts[bi]+lens[bi]]
        vb = vb[(vb>0)&(vb<VT)]
        vl = min(len(vb), ml)
        if vl >= 4: bt[bi,:vl] = torch.from_numpy(vb[:vl].astype(np.int64)).to(DEVICE); mask[bi,:vl] = 1.0
    
    if mask.sum() < 50: continue
    
    ut.train(); _, scores = ut(bt, return_scores=True)
    target = bt[:,1:].clamp(1,VT-1).contiguous()
    pred = scores[:,:-1,:].contiguous(); t_mask = mask[:,1:]
    
    loss = F.cross_entropy(pred.view(-1,157), target.view(-1), reduction='none')
    loss = (loss.view(B, ml-1) * t_mask).sum() / (t_mask.sum() + 1e-8)
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    opt.step(); sch.step()
    
    if s % SAVE_EVERY == 0:
        step_total = prev_step + s
        with torch.no_grad():
            acc = (pred.argmax(-1)==target) & t_mask.bool()
            acc = acc.sum().item()/(t_mask.sum()+1e-8)
        torch.save({'model':ut.state_dict(),'step':step_total,'loss':loss.item()},
                   os.path.join(CKPT, f"causal_{step_total}.pt"))
        torch.save({'model':ut.state_dict(),'step':step_total,'loss':loss.item()},
                   os.path.join(CKPT, "causal_latest.pt"))
        elapsed = time.time()-t0
        eta = (elapsed/s)*(EXTRA-s) if s>0 else 0
        log(f"  step {step_total}/{TOTAL} | loss={loss.item():.4f} | acc={acc:.3f} | {elapsed/60:.0f}min")
    elif s % 1000 == 0:
        step_total = prev_step + s
        with torch.no_grad():
            acc = (pred.argmax(-1)==target) & t_mask.bool()
            acc = acc.sum().item()/(t_mask.sum()+1e-8)
        elapsed = time.time()-t0
        log(f"  step {step_total}/{TOTAL} | loss={loss.item():.4f} | acc={acc:.3f} | {elapsed/60:.0f}min")

log("DONE")
