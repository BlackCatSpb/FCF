"""
EVA — Train on War & Peace. Test generation every 5000 steps.
Status every 50 steps. Fresh start, no prior checkpoints.
"""
import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
os.makedirs(CKPT, exist_ok=True)

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
cv = CharacterVocab(); VT = cv.vocab_size

print("=" * 60)
print("EVA — War & Peace Training")
print("=" * 60)

# Fresh coordinates
c128 = torch.zeros(VT, 128, device=DEVICE)
g = torch.Generator(device=DEVICE).manual_seed(42)
c128[:, :] = torch.randn(VT, 128, generator=g, device=DEVICE) * 0.02
c128 = c128 / c128.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=VT, coord_dim=128, num_levels=8,
    scales_per_level=4, num_layers=6, d_ff=512).to(DEVICE)
ut.set_symbol_coordinates(c128)
print(f"Model: {sum(p.numel() for p in ut.parameters()):,} params")

# Load War & Peace
npy = os.path.join(os.path.dirname(__file__), "real_data", "war_and_peace.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)
print(f"Data: {total/1e6:.2f}M tokens ({total/len(data)*1e6:.0f} sentences)")

STEPS = 100000; LR = 5e-3; B = 32; ML = 64
opt = torch.optim.AdamW(ut.parameters(), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
rng = np.random.RandomState(42)

# Sequential with random offset per block: covers ALL text
sentences = []
i = 0
while i < total - 1:
    if data[i] == cv.EOS_IDX:
        start = i + 1
    else:
        start = i
    # Find EOS
    while i < total and data[i] != cv.EOS_IDX:
        i += 1
    sent = data[start:i]
    valid = sent[(sent > 0) & (sent < VT)]
    if len(valid) >= 4:
        sentences.append(valid.tolist())
    i += 1

print(f"Sentences: {len(sentences):,}")
sent_ptr = 0

def gen_text(ids, n=40, T=0.6):
    ids = list(ids)
    # Cyrillic-only mask: block Latin, digits, special chars during generation
    cyrillic_mask = torch.zeros(VT, device=DEVICE)
    for i in range(VT):
        ch = cv.decode([i])
        if ch and (ch.isalpha() and ord(ch) > 127 or ch in ' ,.!?;:()-—…«»\"\'\n'):
            cyrillic_mask[i] = 1
    cyrillic_mask[0] = 0  # block PAD
    
    with torch.no_grad():
        for _ in range(n):
            _, sc = ut(torch.tensor([ids], dtype=torch.long, device=DEVICE), return_scores=True)
            logits = sc[0, -1] / T
            # Block non-Cyrillic
            logits = logits + (cyrillic_mask - 1) * 1e9
            
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
    # Sequential sentence-based sampling (covers ALL text in order)
    bt = torch.zeros(B, ML, dtype=torch.long, device=DEVICE)
    mask = torch.ones(B, ML, device=DEVICE)
    for bi in range(B):
        ids_flat = []
        while len(ids_flat) < ML:
            ids_flat.extend(sentences[sent_ptr % len(sentences)])
            ids_flat.append(cv.EOS_IDX)
            sent_ptr += 1
        ids_flat = ids_flat[:ML]
        bt[bi, :len(ids_flat)] = torch.tensor(ids_flat, dtype=torch.long, device=DEVICE)
    
    ut.train(); _, scores = ut(bt, return_scores=True)
    target = bt[:, 1:].clamp(1, VT-1).contiguous(); pred = scores[:, :-1].contiguous(); tm = mask[:, 1:]
    loss = F.cross_entropy(pred.view(-1, 157), target.view(-1), reduction='none')
    loss = (loss.view(B, ML-1) * tm).sum() / (tm.sum() + 1e-8)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0); opt.step(); sch.step()
    
    if s % 50 == 0:
        with torch.no_grad(): acc = ((pred.argmax(-1) == target) & tm.bool()).sum().item() / (tm.sum() + 1e-8)
        print(f"  {s:>6d} | loss={loss.item():.4f} acc={acc:.3f} | {int((time.time()-t0)/60)}min", flush=True)
    
    if s % 500 == 0:
        torch.save({'ut': ut.state_dict(), 'step': s}, os.path.join(CKPT, "wp_latest.pt"))
    
    if s % 5000 == 0:
        ut.eval()
        print(f"\n  ── GEN @ step {s} ──")
        for w in ['привет', 'князь Андрей', 'Наташа', 'война', 'Пьер']:
            ids = cv.encode(w)[1:-1]
            if len(ids) >= 2:
                gtxt = gen_text(ids, 35, 0.8)
                print(f"  '{w}' → '{gtxt}'")
        print()
        ut.train()

print("Done.")
