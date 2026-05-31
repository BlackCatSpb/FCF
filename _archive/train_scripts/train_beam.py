"""
EVA — Trajectory Planning + Beam Search Generation.

B) Trajectory planning: predict MULTIPLE future coordinates at once
C) Beam search: keep top-K candidates, score by whole-sequence coherence

Training: predict next 1,2,4,8 coordinates (multi-step prediction)
Generation: beam search with affinity scoring
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — Trajectory Planning + Beam Search")
print("=" * 60)

evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE); aff = evolved['affinity']

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24).to(DEVICE)
ut.set_symbol_coordinates(coords)

# Load ConceptNet weights (best available)
cn_path = os.path.join(CKPT, "conceptnet_weights.pt")
if os.path.exists(cn_path):
    ckpt = torch.load(cn_path, map_location='cpu', weights_only=True)
    ut.load_state_dict(ckpt['model'], strict=False)
    if 'coords' in ckpt:
        ut.set_symbol_coordinates(ckpt['coords'].to(DEVICE))
    print("Loaded: ConceptNet weights")
else:
    ckpt = torch.load(os.path.join(CKPT, "sentence_weights.pt"), map_location='cpu', weights_only=True)
    ut.load_state_dict(ckpt['model'], strict=False)
    print("Loaded: sentence weights")

# ============================================================
# Train: multi-step coordinate prediction
# ============================================================
npy = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy): npy = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)

STEPS = 5000; LR = 1e-3; B = 32; ML = 48
opt = torch.optim.AdamW(ut.parameters(), lr=LR); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
rng = np.random.RandomState(888); t0 = time.time()

print(f"\n[TRAIN] Multi-step coordinate prediction ({STEPS} steps)...")

for s in range(1, STEPS+1):
    lens = rng.randint(16, ML+1, B); starts = rng.randint(0, max(1, total - max(lens) - 1), B)
    ml = max(lens); bt = torch.full((B, ml), 0, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(B, ml, device=DEVICE)
    for bi in range(B):
        vb = data[starts[bi]:starts[bi]+lens[bi]]; vb = vb[(vb>0)&(vb<VT)]
        vl = min(len(vb), ml)
        if vl >= 4: bt[bi,:vl] = torch.from_numpy(vb[:vl].astype(np.int64)).to(DEVICE); mask[bi,:vl] = 1.0
    if mask.sum() < 50: continue
    
    ut.train(); pc, ps = ut(bt, return_scores=True)
    
    # Multi-step loss: predict 1, 2, 4 steps ahead
    total_loss = 0
    for horizon in [1, 2, 4]:
        if ml <= horizon: continue
        tc = ut.embed(bt[:,horizon:].clamp(1,VT-1)).detach()
        cl = F.mse_loss(pc[:,:-horizon], tc, reduction='none').mean(-1)
        cl = (cl * mask[:,horizon:]).sum() / (mask[:,horizon:].sum() + 1e-8)
        total_loss = total_loss + cl * (1.0 / horizon)  # weight by inverse horizon
    
    sl = F.cross_entropy(ps[:,:-1].contiguous().view(-1,157), bt[:,1:].clamp(1,VT-1).contiguous().view(-1), reduction='none')
    sl = (sl.view(B, ml-1) * mask[:,1:]).sum() / (mask[:,1:].sum() + 1e-8)
    loss = total_loss + 0.05 * sl
    
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0); opt.step(); sch.step()
    
    if s % 500 == 0 or s == 1:
        with torch.no_grad(): acc = (ps[:,:-1].argmax(-1) == bt[:,1:].clamp(1,VT-1)) & mask[:,1:].bool(); acc = acc.sum().item()/(mask[:,1:].sum()+1e-8)
        print(f"  {s}/{STEPS} loss={loss.item():.3f} acc={acc:.3f}  {time.time()-t0:.0f}s", flush=True)

torch.save({'model': ut.state_dict()}, os.path.join(CKPT, "trajectory_planner.pt"))
print("Saved.")

# ============================================================
# Beam search generation
# ============================================================
print("\n[TEST] Beam search generation...")
ut.eval()

def beam_generate(seed_ids, beam_width=5, max_len=25, temp=0.7):
    """Beam search: keep top-K candidates, score by affinity + coherence."""
    beams = [(list(seed_ids), 0.0)]  # (ids, score)
    
    with torch.no_grad():
        for _ in range(max_len):
            candidates = []
            for ids, cum_score in beams:
                if len(ids) < 2: continue
                
                inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                pc, ps = ut(inp, return_scores=True)
                nc = pc[0, -1]  # predicted next coordinate
                
                d = torch.norm(coords - nc, dim=-1); d[0] = 1e9
                k = min(beam_width * 3, VT-1)
                td, ti = torch.topk(d, k, largest=False)
                
                # Score: coordinate proximity + affinity + diversity
                scores = -td  # closer = higher score
                
                if 0 < ids[-1] < VT:
                    ab = aff[ids[-1]][ti.cpu()].to(DEVICE)
                    scores = scores + torch.log(ab.clamp(min=0.01)) * 0.5
                
                # Repetition penalty
                for t in set(ids[-4:]):
                    m = (ti == t).nonzero(as_tuple=True)[0]
                    if len(m) > 0: scores[m] -= 2.0
                
                probs = F.softmax(scores / temp, dim=-1)
                topk_p, topk_i = torch.topk(probs, beam_width)
                
                for pk, ix in zip(topk_p, topk_i):
                    new_ids = ids + [ti[ix].item()]
                    new_score = cum_score + torch.log(pk + 1e-8).item()
                    candidates.append((new_ids, new_score))
            
            # Keep top beam_width
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]
    
    return beams[0][0]  # best beam

for w in ['привет', 'человек', 'солнце', 'сегодня утром']:
    ids = cv.encode(w)[1:-1]
    if len(ids) >= 2:
        result = beam_generate(ids, beam_width=8, max_len=20, temp=0.6)
        print(f"  '{w}' -> '{cv.decode(result)}'")

print("Done.")
