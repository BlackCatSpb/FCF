"""
EVA — Hierarchical Training on War & Peace.
Boundary tokens <W></W> <S></S>, adaptive levels, word weights.
"""
import torch, torch.nn.functional as F, numpy as np, sys, os, time, re
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
os.makedirs(CKPT, exist_ok=True)

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
cv = CharacterVocab(); VT = cv.vocab_size

print("=" * 60)
print("EVA — Hierarchical Training (War & Peace)")
print("=" * 60)
print(f"Vocab: {VT} tokens")

# Coordinates
c128 = torch.zeros(VT, 128, device=DEVICE)
g = torch.Generator(device=DEVICE).manual_seed(42)
c128[:, :] = torch.randn(VT, 128, generator=g, device=DEVICE) * 0.02
c128 = c128 / c128.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=VT, coord_dim=128, max_levels=8,
    total_heads=32, num_layers=6, d_ff=128).to(DEVICE)
ut.set_symbol_coordinates(c128)
print(f"Model: {sum(p.numel() for p in ut.parameters()):,} params")

# Load & encode War & Peace with boundaries
print("Encoding War & Peace with boundary tokens...")
all_ids = []
for book in [1, 2]:
    path = rf"C:\Users\black\OneDrive\Desktop\Толстой Лев. Война и мир. Книга {book} - royallib.ru.txt"
    with open(path, 'r', encoding='windows-1251') as f:
        raw = f.read()
    raw = re.sub(r'\r\n|\r', '\n', raw)
    sents = re.split(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z])', raw)
    for s in sents:
        s = s.strip()
        if len(s) >= 4:
            ids = cv.encode_with_boundaries(s)
            if len(ids) >= 5:
                all_ids.extend(ids)
    print(f"  Book {book}: {len(sents):,} sentences")

data = np.array(all_ids, dtype=np.int32)
total = len(data)
print(f"Total: {total/1e6:.2f}M tokens")

# Extract boundary-delimited blocks for sequential training
blocks = []
i = 0
while i < total - 1:
    start = i
    while i < total and data[i] != cv.SENT_CLOSE_IDX:
        i += 1
    if i < total: 
        blocks.append(data[start:i+1].tolist())
        i += 1
print(f"Blocks: {len(blocks):,}")
sent_ptr = 0

STEPS = 100000; LR = 5e-3; B = 32; ML = 96
opt = torch.optim.AdamW(ut.parameters(), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

def gen_text(ids, n=40, T=0.6):
    ids = list(ids)
    mask = torch.zeros(VT, device=DEVICE)
    for i in range(VT):
        ch = cv.decode([i])
        if ch and (ch.isalpha() and ord(ch) > 127 or ch in ' ,.!?;:()-…«»\"\'\n' or '<W>' in ch or '</W>' in ch or '<S>' in ch or '</S>' in ch):
            mask[i] = 1
    mask[0] = 0
    with torch.no_grad():
        for _ in range(n):
            _, sc = ut(torch.tensor([ids], dtype=torch.long, device=DEVICE), return_scores=True)
            logits = sc[0, -1] / T
            logits = logits + (mask - 1) * 1e9
            sl, si = logits.sort(descending=True); cp = F.softmax(sl, dim=-1).cumsum(dim=-1)
            cut = (cp > 0.95).nonzero(as_tuple=True)[0]
            k = cut[0].item() + 1 if len(cut) > 0 else 20; k = min(max(k, 3), 40)
            v, idx = logits.topk(k); p = F.softmax(v, dim=-1)
            for t in set(ids[-5:]): m = (idx == t).nonzero(as_tuple=True)[0]; p[m] *= 0.3
            p /= p.sum(); nt = idx[torch.multinomial(p, 1)].item()
            if nt <= 0 or nt >= VT: nt = idx[0].item()
            ids.append(nt)
    return cv.decode(ids)

t0 = time.time()
for s in range(1, STEPS + 1):
    bt = torch.zeros(B, ML, dtype=torch.long, device=DEVICE)
    mask = torch.ones(B, ML, device=DEVICE)
    for bi in range(B):
        ids_flat = []
        while len(ids_flat) < ML:
            ids_flat.extend(blocks[sent_ptr % len(blocks)])
            sent_ptr += 1
        ids_flat = ids_flat[:ML]
        bt[bi, :len(ids_flat)] = torch.tensor(ids_flat, dtype=torch.long, device=DEVICE)
    
    ut.train(); _, scores = ut(bt, return_scores=True)
    target = bt[:, 1:].clamp(1, VT-1).contiguous(); pred = scores[:, :-1].contiguous(); tm = mask[:, 1:]
    loss = F.cross_entropy(pred.view(-1, VT), target.view(-1), reduction='none')
    loss = (loss.view(B, ML-1) * tm).sum() / (tm.sum() + 1e-8)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0); opt.step(); sch.step()
    
    if s % 50 == 0:
        with torch.no_grad(): acc = ((pred.argmax(-1) == target) & tm.bool()).sum().item() / (tm.sum() + 1e-8)
        print(f"  {s:>6d} | loss={loss.item():.4f} acc={acc:.3f} | {int((time.time()-t0)/60)}min", flush=True)
    
    if s % 500 == 0:
        torch.save({'ut': ut.state_dict(), 'step': s}, os.path.join(CKPT, "wp_latest.pt"))
    
    if s % 5000 == 0:
        ut.eval()
        print(f"\n  GEN @ {s}")
        for w in ['<S><W>привет</W>', '<S><W>князь</W>', '<S><W>Наташа</W>', '<S><W>война</W>', '<S><W>Пьер</W>']:
            ids = cv.encode(w)[1:-1]
            if len(ids) >= 2:
                print(f"  {w[:20]} -> {gen_text(ids, 35, 0.6)}")
        print()
        ut.train()

print("Done.")
