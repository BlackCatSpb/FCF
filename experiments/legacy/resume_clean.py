"""
resume_clean.py — Resume clean training from latest checkpoint.
Continues from clean_step_40000.pt to 100000 steps.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, sys, os, time, glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.char_vocab import CharacterVocab as CharVocab
from eva.symbolic.train_v3 import safe_load_state_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

DATA_PATH = 'real_data/full_corpus_ids.npy'
CKPT_DIR = 'checkpoints/v3'
TARGET_STEPS = 100000
B, L = 8, 64
LR = 3e-4
LOG_EVERY = 50
SAVE_EVERY = 10000
W_CE = 1.0
W_NXT = 0.1
W_SRG = 0.05

data = np.load(DATA_PATH)
data = np.array(data, dtype=np.int64)
N = len(data)
print(f'Data: {DATA_PATH} — {N:,} tokens')

# Find latest checkpoint
ckpts = sorted(glob.glob(f'{CKPT_DIR}/clean_step_*.pt'))
if not ckpts:
    print('No checkpoint found!')
    sys.exit(1)
latest = ckpts[-1]
print(f'Loading {latest}')

model = UnifiedMultidimensionalTransformer().to(device)
ckpt = torch.load(latest, map_location=device, weights_only=True)
safe_load_state_dict(model, ckpt['model_state'])
start_step = ckpt.get('step', 0)
model.train()
total_params = sum(p.numel() for p in model.parameters())
print(f'Model: {total_params:,} params, resuming from step {start_step}')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
if 'optimizer_state' in ckpt:
    try:
        optimizer.load_state_dict(ckpt['optimizer_state'])
        print('Optimizer state restored')
    except Exception as e:
        print(f'Optimizer restore failed: {e}')

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TARGET_STEPS)

t0 = time.time()
loss_log = defaultdict(list)

remaining = TARGET_STEPS - start_step
print(f'\nResuming for {remaining} steps (target: {TARGET_STEPS})...')
for raw_step in range(start_step, TARGET_STEPS):
    idx = np.random.randint(0, N - L - 1, size=B)
    x = torch.tensor([data[i:i+L] for i in idx], dtype=torch.long, device=device)
    targets = torch.tensor([data[i+1:i+L+1] for i in idx], dtype=torch.long, device=device)

    optimizer.zero_grad()
    h, scores, weights, heads_out = model.forward(
        x, return_scores=True, return_heads=True, capture_attn=False
    )

    logits = scores
    ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    nxt = heads_out.get('boundary_next')
    if nxt is not None and h.shape[1] > 1:
        delta = h[:, 1:] - h[:, :-1]
        nxt_pred = nxt[:, :-1]
        nxt_loss = F.mse_loss(nxt_pred, delta)
    else:
        nxt_loss = torch.tensor(0.0, device=h.device)

    srg_loss = model.temporal_smoothness_loss(h)
    total = W_CE * ce_loss + W_NXT * nxt_loss + W_SRG * srg_loss
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    acc = (logits.argmax(-1) == targets).float().mean().item()
    loss_log['ce'].append(ce_loss.item())
    loss_log['nxt'].append(nxt_loss.item())
    loss_log['srg'].append(srg_loss.item())

    if raw_step % LOG_EVERY == 0:
        avg = {k: np.mean(v[-LOG_EVERY:]) for k, v in loss_log.items() if v}
        elapsed = time.time() - t0
        tokens_seen = raw_step * B * L
        print(f'[CLEAN {raw_step}/{TARGET_STEPS}] ce={avg.get("ce",0):.4f} '
              f'nxt={avg.get("nxt",0):.4f} srg={avg.get("srg",0):.4f} '
              f'acc={acc:.3f} tokens={tokens_seen/1e6:.1f}M | {elapsed:.0f}s')

    if raw_step > 0 and raw_step % SAVE_EVERY == 0:
        out_path = f'{CKPT_DIR}/clean_step_{raw_step}.pt'
        tmp = out_path + '.tmp'
        torch.save({
            'step': raw_step, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, tmp)
        os.replace(tmp, out_path)
        print(f'  Saved {out_path}')

elapsed = time.time() - t0
out_path = f'{CKPT_DIR}/clean_final.pt'
tmp = out_path + '.tmp'
torch.save({
    'step': TARGET_STEPS, 'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}, tmp)
os.replace(tmp, out_path)
print(f'\nDone. Target {TARGET_STEPS} reached in {elapsed:.0f}s ({elapsed/60:.1f} min).')
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
