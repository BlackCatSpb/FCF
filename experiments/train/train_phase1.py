"""
Phase 1 (revised): Train 384-dim / 12-layer on char-level data.
20M params, 200K steps, B=8 L=64 for MX550 2.1GB.
Fixed: SequentialLR warmup, BoundaryDetectionHead import, L_align loss.

Usage: python train_phase1.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.phase1_model import UnifiedMultidimensionalTransformerV2
from eva.symbolic.bpe_tokenizer import BPEVocab
from eva.symbolic.heads import BoundaryDetectionHead

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ─── Config ───
DATA_PATH = 'real_data/full_corpus_ids.npy'
CKPT_DIR = 'checkpoints/v4'
os.makedirs(CKPT_DIR, exist_ok=True)

N_STEPS = 200000
B, L = 8, 64
LR = 3e-4
WARMUP = 2000
LOG_EVERY = 100
SAVE_EVERY = 20000

W_CE = 1.0
W_NXT = 0.05
W_BOUNDARY = 0.1
W_ALIGN = 0.05
UPDATE_ATTRACTORS_EVERY = 10

VOCAB_SIZE = 4101
# Char data: only tokens 0-160 appear, boundary tokens 4096-4100 never present
SPECIAL_IDS = {0, 1, 2, 3, 156, 157, 158, 159, 160}

data = np.load(DATA_PATH).astype(np.int64)
N = len(data)
print(f'Data: {DATA_PATH} — {N:,} tokens')

model = UnifiedMultidimensionalTransformerV2(vocab_size=VOCAB_SIZE).to(device)
total_params = sum(p.numel() for p in model.parameters())
model.train()
print(f'Model: {total_params:,} params')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, total_iters=WARMUP)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS - WARMUP)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP])

cv = BPEVocab()

t0 = time.time()
print(f'\nPhase 1 training: {N_STEPS} steps...')

for step in range(N_STEPS):
    idx = np.random.randint(0, N - L - 1, size=B)
    batch = np.stack([data[i:i+L] for i in idx])
    x = torch.tensor(batch, dtype=torch.long, device=device)
    targets = torch.tensor(
        np.stack([data[i+1:i+L+1] for i in idx]),
        dtype=torch.long, device=device)

    optimizer.zero_grad()
    h, scores, weights, heads_out = model.forward(
        x, return_scores=True, return_heads=True, capture_attn=False,
        update_attractors=(step % UPDATE_ATTRACTORS_EVERY == 0))

    # 1. CE loss — mask special tokens
    logits = scores
    special_t = torch.tensor(list(SPECIAL_IDS), device=device)
    ce_mask = ~torch.isin(targets, special_t)
    if ce_mask.any():
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1])[ce_mask.reshape(-1)],
            targets.reshape(-1)[ce_mask.reshape(-1)])
    else:
        ce_loss = torch.tensor(0.0, device=device)

    # 2. nxt-loss (trajectory smoothness)
    nxt = heads_out.get('boundary_next')
    if nxt is not None and h.shape[1] > 1:
        delta = h[:, 1:] - h[:, :-1]
        nxt_loss = F.mse_loss(nxt[:, :-1], delta)
    else:
        nxt_loss = torch.tensor(0.0, device=device)

    # 3. Boundary loss (via BoundaryDetectionHead)
    boundary_logits = heads_out.get('boundary_detect')
    if boundary_logits is not None:
        boundary_labels = BoundaryDetectionHead.make_labels(
            x, WORD_OPEN=157, WORD_CLOSE=158, SENT_OPEN=159, SENT_CLOSE=160)
        boundary_loss_val = F.cross_entropy(
            boundary_logits.reshape(-1, 3),
            boundary_labels.reshape(-1),
            ignore_index=-100)
    else:
        boundary_loss_val = torch.tensor(0.0, device=device)

    # 4. L_align — cross-level consistency
    boundary_probs = boundary_logits.softmax(-1) if boundary_logits is not None else None
    if boundary_probs is not None and weights is not None:
        char_inside = boundary_probs[..., 1]  # [B, L]
        word_pooled = heads_out.get('boundary_end', h.mean(dim=-1, keepdim=True).expand(-1, -1, 384))
        word_avg = word_pooled.mean(dim=-1).sigmoid()  # [B, L]
        align_loss = F.mse_loss(char_inside.reshape(-1), word_avg.reshape(-1))
    else:
        align_loss = torch.tensor(0.0, device=device)

    total = (W_CE * ce_loss + W_NXT * nxt_loss +
             W_BOUNDARY * boundary_loss_val + W_ALIGN * align_loss)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if ce_mask.any():
        acc = (logits.argmax(-1)[ce_mask] == targets[ce_mask]).float().mean().item()
    else:
        acc = 0.0

    if step % LOG_EVERY == 0:
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        print(f'[PHASE1 {step}/{N_STEPS}] ce={ce_loss.item():.4f} '
              f'nxt={nxt_loss.item():.4f} bc={boundary_loss_val.item():.4f} '
              f'align={align_loss.item():.4f} acc={acc:.3f} lr={lr_now:.2e} '
              f'| {elapsed:.0f}s ({elapsed/60:.1f}min)')

    if step > 0 and step % SAVE_EVERY == 0:
        out_path = f'{CKPT_DIR}/phase1_step_{step}.pt'
        tmp = out_path + '.tmp'
        torch.save({
            'step': step, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, tmp)
        os.replace(tmp, out_path)
        print(f'  Saved {out_path}')

elapsed = time.time() - t0
out_path = f'{CKPT_DIR}/phase1_final.pt'
tmp = out_path + '.tmp'
torch.save({
    'step': N_STEPS, 'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}, tmp)
os.replace(tmp, out_path)
print(f'\nDone. {N_STEPS} steps, {elapsed:.0f}s ({elapsed/60:.1f} min).')
print(f'Saved to {out_path}')
