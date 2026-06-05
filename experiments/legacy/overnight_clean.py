"""
overnight_clean.py — CE + nxt + SRG на clean corpus (full_corpus_ids.npy, NO boundary tokens).
Train 100K steps at B=8, L=64.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, sys, os, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.char_vocab import CharacterVocab as CharVocab
from eva.symbolic.train_v3 import safe_load_state_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

DATA_PATH = 'real_data/full_corpus_ids.npy'
CKPT_DIR = 'checkpoints/v3'
N_STEPS = 100000
B, L = 8, 64  # safe for MX550 2.1GB
LR = 3e-4
WARMUP = 1000
LOG_EVERY = 50
SAVE_EVERY = 10000
W_CE = 1.0
W_NXT = 0.1
W_SRG = 0.05

data = np.load(DATA_PATH)
data = np.array(data, dtype=np.int64)
N = len(data)
print(f'Data: {DATA_PATH} — {N:,} tokens, max={data.max()}')

model = UnifiedMultidimensionalTransformer().to(device)
total_params = sum(p.numel() for p in model.parameters())
model.train()
print(f'Model: {total_params:,} params')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS)

t0 = time.time()
loss_log = defaultdict(list)

print(f'\nClean training (CE+nxt+SRG) for {N_STEPS} steps...')
for step in range(N_STEPS):
    # LR warmup
    if step < WARMUP:
        lr_scale = step / WARMUP
        for pg in optimizer.param_groups:
            pg['lr'] = LR * lr_scale
    
    idx = np.random.randint(0, N - L - 1, size=B)
    x = torch.tensor([data[i:i+L] for i in idx], dtype=torch.long, device=device)
    targets = torch.tensor([data[i+1:i+L+1] for i in idx], dtype=torch.long, device=device)

    optimizer.zero_grad()
    h, scores, weights, heads_out = model.forward(x, return_scores=True, return_heads=True, capture_attn=False)

    # 1. CE loss
    logits = scores
    ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    # 2. nxt-loss
    nxt = heads_out.get('boundary_next')
    if nxt is not None and h.shape[1] > 1:
        delta = h[:, 1:] - h[:, :-1]
        nxt_pred = nxt[:, :-1]
        nxt_loss = F.mse_loss(nxt_pred, delta)
    else:
        nxt_loss = torch.tensor(0.0, device=h.device)

    # 3. SRG loss
    srg_loss = model.temporal_smoothness_loss(h)

    total = W_CE * ce_loss + W_NXT * nxt_loss + W_SRG * srg_loss
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step >= WARMUP:
        scheduler.step()

    acc = (logits.argmax(-1) == targets).float().mean().item()
    loss_log['ce'].append(ce_loss.item())
    loss_log['nxt'].append(nxt_loss.item())
    loss_log['srg'].append(srg_loss.item())

    if step % LOG_EVERY == 0:
        avg = {k: np.mean(v[-LOG_EVERY:]) for k, v in loss_log.items() if v}
        elapsed = time.time() - t0
        tokens_seen = step * B * L
        print(f'[CLEAN {step}/{N_STEPS}] ce={avg.get("ce",0):.4f} '
              f'nxt={avg.get("nxt",0):.4f} srg={avg.get("srg",0):.4f} '
              f'acc={acc:.3f} tokens={tokens_seen/1e6:.1f}M | {elapsed:.0f}s')

    if step > 0 and step % SAVE_EVERY == 0:
        out_path = f'{CKPT_DIR}/clean_step_{step}.pt'
        tmp = out_path + '.tmp'
        torch.save({'step': step, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, tmp)
        os.replace(tmp, out_path)
        print(f'  Saved {out_path}')

elapsed = time.time() - t0
out_path = f'{CKPT_DIR}/clean_final.pt'
tmp = out_path + '.tmp'
torch.save({'step': N_STEPS, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, tmp)
os.replace(tmp, out_path)
print(f'\nDone. {N_STEPS} steps, {elapsed:.0f}s ({elapsed/60:.1f} min).')
print(f'Final CE loss: {np.mean(loss_log["ce"][-100:]):.4f}')
print(f'Saved to {out_path}')

# Quick test
print('\n--- Generation test ---')
model.eval()
cv = CharVocab()
for p in ['привет ', 'как дела ', 'это ']:
    ids = cv.encode(p)
    prompt_ids = [i for i in ids if i not in (cv.BOS_IDX, cv.EOS_IDX)]
    try:
        t, _ = model.generate_text(prompt_ids, cv, max_new=48)
        print(f'  {p!r} -> {t[:80]!r}')
    except Exception as e:
        print(f'  {p!r} -> ERROR: {e}')
