"""
EVA — Causal Next-Token Training.

Стандартный causal language modeling:
- Вход: [s0, s1, ..., sN]
- Цель: [s1, s2, ..., sN+1] (следующий токен)
- Causal mask: нельзя смотреть в будущее
- Loss: CrossEntropy

После обучения — генерация работает.
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — Causal Training")
print("=" * 60)

evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE); aff = evolved['affinity']

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24).to(DEVICE)
ut.set_symbol_coordinates(coords)

npy = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy): npy = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)

STEPS = 8000; LR = 1e-3; B = 64; ML = 64
opt = torch.optim.AdamW(ut.parameters(), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
rng = np.random.RandomState(42); t0 = time.time()

print(f"\n[TRAIN] Causal next-token prediction ({STEPS} steps)...")

for s in range(1, STEPS+1):
    lens = rng.randint(16, ML+1, B); starts = rng.randint(0, max(1, total - max(lens) - 1), B)
    ml = max(lens); bt = torch.full((B, ml), 0, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(B, ml, device=DEVICE)
    for bi in range(B):
        vb = data[starts[bi]:starts[bi]+lens[bi]]; vb = vb[(vb>0)&(vb<VT)]
        vl = min(len(vb), ml)
        if vl >= 4: bt[bi,:vl] = torch.from_numpy(vb[:vl].astype(np.int64)).to(DEVICE); mask[bi,:vl] = 1.0
    if mask.sum() < 50: continue
    
    ut.train(); _, scores = ut(bt, return_scores=True)
    
    # Causal: pos i predicts pos i+1
    target = bt[:, 1:].clamp(1, VT-1).contiguous()
    pred = scores[:, :-1, :].contiguous()
    t_mask = mask[:, 1:]
    
    loss = F.cross_entropy(pred.view(-1, 157), target.view(-1), reduction='none')
    loss = (loss.view(B, ml-1) * t_mask).sum() / (t_mask.sum() + 1e-8)
    
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0); opt.step(); sch.step()
    
    if s % 500 == 0 or s == 1:
        with torch.no_grad(): 
            acc = (pred.argmax(-1) == target) & t_mask.bool(); acc = acc.sum().item()/(t_mask.sum()+1e-8)
        print(f"  {s}/{STEPS} loss={loss.item():.4f} acc={acc:.3f}  {time.time()-t0:.0f}s", flush=True)

torch.save({'model': ut.state_dict(), 'coords': coords}, os.path.join(CKPT, "causal_weights.pt"))
print("Saved.")

# ============================================================
# Autoregressive generation
# ============================================================
print("\n[GENERATE] Autoregressive text generation...")
ut.eval()

def generate(ids, max_new=20, temp=0.8, top_k=30, top_p=0.95):
    ids = list(ids)
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            _, scores = ut(inp, return_scores=True)
            logits = scores[0, -1] / temp
            
            # Top-p (nucleus) filtering
            sorted_logits, sorted_idx = logits.sort(descending=True)
            cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            cutoff = (cumprobs > top_p).nonzero(as_tuple=True)[0]
            k = cutoff[0].item() + 1 if len(cutoff) > 0 else top_k
            k = min(max(k, 3), top_k)
            
            vals, idx = logits.topk(k)
            probs = F.softmax(vals, dim=-1)
            
            # Repetition penalty
            for t in set(ids[-5:]):
                m = (idx == t).nonzero(as_tuple=True)[0]
                if len(m) > 0: probs[m] *= 0.3
            
            probs = probs / probs.sum()
            next_tok = idx[torch.multinomial(probs, 1)].item()
            if next_tok <= 0 or next_tok >= VT: next_tok = idx[0].item()
            ids.append(next_tok)
    return ids

tests = ["привет", "человек идет", "солнце светит", "сегодня хорошая", "я люблю", "метаданные"]
for w in tests:
    ids = cv.encode(w)[1:-1]
    if len(ids) >= 2:
        r = generate(ids, 20, 0.8)
        print(f"  '{w}' -> '{cv.decode(r)}'")

print("Done.")
