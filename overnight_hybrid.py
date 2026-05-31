"""
overnight_hybrid.py — CE+nxt+SRG hybrid training for EVA.
Loads hybrid_final.pt and runs 30000 more steps.
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

DATA_PATH = 'real_data/full_corpus_encoded.npy'
CKPT_DIR = 'checkpoints/v3'
N_STEPS = 30000
B, L = 8, 64
LR = 3e-4
LOG_EVERY = 50
SAVE_EVERY = 5000
W_CE = 1.0
W_NXT = 0.1
W_SRG = 0.05

# Load checkpoint
ckpt_path = f'{CKPT_DIR}/hybrid_final.pt'
print(f'Loading {ckpt_path}')
model = UnifiedMultidimensionalTransformer().to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
safe_load_state_dict(model, ckpt['model_state'])
start_step = ckpt.get('step', 0)
model.train()
total_params = sum(p.numel() for p in model.parameters())
print(f'Model: {total_params:,} params, resuming from step {start_step}')

data = np.load(DATA_PATH)
data = np.array(data, dtype=np.int64)
N = len(data)
print(f'Data: {N:,} tokens')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS)

# Resume optimizer if available
if 'optimizer_state' in ckpt:
    try:
        optimizer.load_state_dict(ckpt['optimizer_state'])
        print('Optimizer state restored')
    except Exception as e:
        print(f'Optimizer restore failed (fresh start): {e}')

t0 = time.time()
loss_log = defaultdict(list)

print(f'\nHybrid training (CE+nxt+SRG) for {N_STEPS} additional steps...')
for raw_step in range(N_STEPS):
    step = start_step + raw_step
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
        print(f'[HYBRID {raw_step}/{N_STEPS}] ce={avg.get("ce",0):.4f} '
              f'nxt={avg.get("nxt",0):.4f} srg={avg.get("srg",0):.4f} '
              f'acc={acc:.3f} | {elapsed:.0f}s')

    if raw_step > 0 and raw_step % SAVE_EVERY == 0:
        out_path = f'{CKPT_DIR}/overnight_step_{step}.pt'
        tmp = out_path + '.tmp'
        torch.save({
            'step': step, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, tmp)
        os.replace(tmp, out_path)
        print(f'  Saved {out_path}')

# Final save
elapsed = time.time() - t0
final_step = start_step + N_STEPS
out_path = f'{CKPT_DIR}/overnight_final_step_{final_step}.pt'
tmp = out_path + '.tmp'
torch.save({
    'step': final_step, 'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}, tmp)
os.replace(tmp, out_path)
print(f'\nDone. {N_STEPS} steps in {elapsed:.0f}s ({elapsed/60:.1f} min).')
print(f'Final CE loss: {np.mean(loss_log["ce"][-100:]):.4f}')
print(f'Saved to {out_path}')

# Quick Russian generation test
print('\n--- Russian generation test ---')
model.eval()
cv = CharVocab()
ru_prompts = ['привет ', 'как дела ', 'это ']
for p in ru_prompts:
    ids = cv.encode(p)
    prompt_ids = [i for i in ids if i not in (cv.BOS_IDX, cv.EOS_IDX)]
    t, meta = model.generate_text(prompt_ids, cv, max_new=48)
    decoded = cv.decode([i for i in cv.encode(t) if i not in (cv.BOS_IDX, cv.EOS_IDX, cv.PAD_IDX)])
    print(f'  {p!r} -> {decoded[:80]!r}')
