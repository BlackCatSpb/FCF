"""
Phase 0: BoundaryDetectionHead + masked CE on full_corpus_encoded.npy.

Runs 20K fine-tuning steps from clean_step_40000.pt.
Boundary tokens (157-160) are MASKED in CE loss.
Boundary loss supervises BoundaryDetectionHead via WORD_OPEN/WORD_CLOSE labels.

Usage: python train_phase0.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.heads import BoundaryDetectionHead

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

DATA_PATH = 'real_data/full_corpus_encoded.npy'
CKPT_DIR = 'checkpoints/v3'
RESUME = f'{CKPT_DIR}/clean_step_40000.pt'

N_STEPS = 20000
B, L = 8, 64
LR = 1e-4
WARMUP = 500
LOG_EVERY = 50
SAVE_EVERY = 5000

W_CE = 1.0
W_NXT = 0.05
W_BOUNDARY = 0.2

VOCAB_SIZE = 161
# Boundary tokens to mask in CE loss
BOUNDARY_IDS = {157, 158, 159, 160}
GAP_FILLER = 156
SPECIAL_IDS = {0, 1, 2, 3, GAP_FILLER} | BOUNDARY_IDS

data = np.load(DATA_PATH)
data = np.array(data, dtype=np.int64)
N = len(data)
print(f'Data: {DATA_PATH} — {N:,} tokens, max={data.max()}')

model = UnifiedMultidimensionalTransformer(vocab_size=VOCAB_SIZE).to(device)

if os.path.exists(RESUME):
    ckpt = torch.load(RESUME, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'], strict=False)
    start_step = ckpt.get('step', 0)
    print(f'Resumed from {RESUME} (step {start_step})')
else:
    start_step = 0
    print(f'Starting from scratch (no checkpoint at {RESUME})')

total_params = sum(p.numel() for p in model.parameters())
model.train()
print(f'Model: {total_params:,} params')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS)

cv = CharacterVocab()

t0 = time.time()
print(f'\nPhase 0 training: {N_STEPS} steps...')

for step in range(N_STEPS):
    if step < WARMUP:
        lr_scale = step / max(WARMUP, 1)
        for pg in optimizer.param_groups:
            pg['lr'] = LR * lr_scale

    idx = np.random.randint(0, N - L - 1, size=B)
    batch = np.stack([data[i:i+L] for i in idx])
    x = torch.tensor(batch, dtype=torch.long, device=device)
    batch_t = np.stack([data[i+1:i+L+1] for i in idx])
    targets = torch.tensor(batch_t, dtype=torch.long, device=device)

    optimizer.zero_grad()
    h, scores, weights, heads_out = model.forward(
        x, return_scores=True, return_heads=True, capture_attn=False)

    # 1. CE loss — mask boundary + special tokens
    logits = scores
    ce_mask = ~torch.isin(targets, torch.tensor(list(SPECIAL_IDS), device=device))
    if ce_mask.any():
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        ce_mask_flat = ce_mask.reshape(-1)
        ce_loss = F.cross_entropy(
            logits_flat[ce_mask_flat], targets_flat[ce_mask_flat])
    else:
        ce_loss = torch.tensor(0.0, device=device)

    # 2. nxt-loss
    nxt = heads_out.get('boundary_next')
    if nxt is not None and h.shape[1] > 1:
        delta = h[:, 1:] - h[:, :-1]
        nxt_pred = nxt[:, :-1]
        nxt_loss = F.mse_loss(nxt_pred, delta)
    else:
        nxt_loss = torch.tensor(0.0, device=device)

    # 3. Boundary loss
    boundary_logits = heads_out.get('boundary_detect')
    if boundary_logits is not None:
        boundary_loss_val = BoundaryDetectionHead.boundary_loss(
            boundary_logits, x,
            WORD_OPEN=cv.WORD_OPEN_IDX, WORD_CLOSE=cv.WORD_CLOSE_IDX,
            SENT_OPEN=cv.SENT_OPEN_IDX, SENT_CLOSE=cv.SENT_CLOSE_IDX)
    else:
        boundary_loss_val = torch.tensor(0.0, device=device)

    total = W_CE * ce_loss + W_NXT * nxt_loss + W_BOUNDARY * boundary_loss_val
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step >= WARMUP:
        scheduler.step()

    # Accuracy: only on non-masked tokens
    if ce_mask.any():
        acc = (logits.argmax(-1)[ce_mask] == targets[ce_mask]).float().mean().item()
    else:
        acc = 0.0

    if step % LOG_EVERY == 0:
        elapsed = time.time() - t0
        tokens_seen = step * B * L
        print(f'[PHASE0 {step}/{N_STEPS}] ce={ce_loss.item():.4f} '
              f'nxt={nxt_loss.item():.4f} bc={boundary_loss_val.item():.4f} '
              f'acc={acc:.3f} | {elapsed:.0f}s')

    if step > 0 and step % SAVE_EVERY == 0:
        out_path = f'{CKPT_DIR}/phase0_step_{step}.pt'
        tmp = out_path + '.tmp'
        torch.save({
            'step': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, tmp)
        os.replace(tmp, out_path)
        print(f'  Saved {out_path}')

# Final save
elapsed = time.time() - t0
out_path = f'{CKPT_DIR}/phase0_final.pt'
tmp = out_path + '.tmp'
torch.save({
    'step': N_STEPS,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}, tmp)
os.replace(tmp, out_path)
print(f'\nDone. {N_STEPS} steps, {elapsed:.0f}s ({elapsed/60:.1f} min).')
print(f'Saved to {out_path}')
