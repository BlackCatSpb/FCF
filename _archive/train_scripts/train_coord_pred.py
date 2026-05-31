"""EVA — Coordinate Prediction: train + save + test."""
import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — Coordinate Prediction")
print("=" * 60)

evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE); aff = evolved['affinity']

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24).to(DEVICE)
ut.set_symbol_coordinates(coords)

CP = os.path.join(CKPT, "coord_predict.pt")
if os.path.exists(CP):
    print("Loading checkpoint, skip training")
    ut.load_state_dict(torch.load(CP, map_location='cpu', weights_only=True)['model'], strict=False)
else:
    print("Training...")
    npy = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
    if not os.path.exists(npy): npy = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
    data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)
    STEPS, LR, B = 5000, 1e-3, 64
    opt = torch.optim.AdamW(ut.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    rng = np.random.RandomState(777); t0 = time.time()
    
    for s in range(1, STEPS+1):
        lens = rng.randint(16, 65, B); starts = rng.randint(0, max(1, total - max(lens) - 1), B)
        ml = max(lens); bt = torch.full((B, ml), 0, dtype=torch.long, device=DEVICE)
        mask = torch.zeros(B, ml, device=DEVICE)
        for bi in range(B):
            sl, st = lens[bi], starts[bi]
            vb = data[st:st+sl]; vb = vb[(vb>0)&(vb<VT)]
            vl = min(len(vb), ml)
            if vl >= 4: bt[bi,:vl] = torch.from_numpy(vb[:vl].astype(np.int64)).to(DEVICE); mask[bi,:vl] = 1.0
        if mask.sum() < 50: continue
        
        ut.train(); pc, ps = ut(bt, return_scores=True)
        tc = ut.embed(bt[:,1:].clamp(1,VT-1)).detach()
        cl = F.mse_loss(pc[:,:-1], tc, reduction='none').mean(-1)
        cl = (cl * mask[:,1:]).sum() / (mask[:,1:].sum() + 1e-8)
        sl = F.cross_entropy(ps[:,:-1].contiguous().view(-1,157), bt[:,1:].clamp(1,VT-1).contiguous().view(-1), reduction='none')
        sl = sl.view(B, ml-1); sl = (sl * mask[:,1:]).sum() / (mask[:,1:].sum() + 1e-8)
        loss = cl + 0.1 * sl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0); opt.step(); sch.step()
        
        if s % 500 == 0 or s == 1:
            with torch.no_grad(): acc = (ps[:,:-1].argmax(-1) == bt[:,1:].clamp(1,VT-1)) & mask[:,1:].bool(); acc = acc.sum().item() / (mask[:,1:].sum() + 1e-8)
            print(f"  {s}/{STEPS} loss={loss.item():.3f} acc={acc:.3f}  {time.time()-t0:.0f}s", flush=True)
    
    torch.save({'model': ut.state_dict(), 'coords': coords}, CP); print(f"Saved: {CP}")

# Test generation
print("\n[TEST] Coordinate generation:")
ut.eval()

def gen(ids, n=20, T=0.8):
    ids = list(ids)
    with torch.no_grad():
        for _ in range(n):
            pc, _ = ut(torch.tensor([ids], dtype=torch.long, device=DEVICE), return_scores=True)
            nc = pc[0,-1]; d = torch.norm(coords - nc, dim=-1); d[0] = 1e9
            k = 15; td, ti = torch.topk(d, k, largest=False)
            sc = 1.0 / (td + 0.01)
            if 0 < ids[-1] < VT:
                ab = aff[ids[-1]][ti.cpu()].to(DEVICE); sc = sc + ab/ab.max().clamp(1e-8) * 2.0
            for t in set(ids[-3:]): m = (ti==t).nonzero(as_tuple=True)[0]; sc[m] *= 0.1 if len(m)>0 else 1.0
            pr = F.softmax(sc/T, dim=-1); nt = ti[torch.multinomial(pr,1)].item()
            if nt<=0 or nt>=VT: nt=ti[0].item()
            ids.append(nt)
    return ids

for w in ['привет','человек','солнце','сегодня','метаданные','трансформер']:
    ids = cv.encode(w)[1:-1]
    if len(ids)>=2: print(f"  '{w}' -> '{cv.decode(gen(ids,20,0.8))}'")
print("Done.")
