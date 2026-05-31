"""
hybrid_train.py — CE + nxt-loss hybrid training for EVA.

Trains the ENTIRE model (unlike decoder_finetune.py which froze everything).
Uses the existing checkpoint as starting point with CE loss enabled.
Goal: calibrate decoder + transformer for next-token prediction.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, sys, os, time, glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.char_vocab import CharacterVocab as CharVocab
from eva.symbolic.train_v3 import safe_load_state_dict, TrainingConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ---- Config ----
DATA_PATH = 'real_data/full_corpus_encoded.npy'
CKPT_DIR = 'checkpoints/v3'
N_STEPS = 5000
B, L = 8, 64
LR = 3e-4
LOG_EVERY = 50
SAVE_EVERY = 1000

# ---- Load checkpoint ----
ckpts = sorted(glob.glob(f'{CKPT_DIR}/train_v3_step_*.pt'))
latest = ckpts[-1] if ckpts else None
if latest:
    print(f'Loading {latest}')
    model = UnifiedMultidimensionalTransformer().to(device)
    ckpt = torch.load(latest, map_location=device, weights_only=True)
    safe_load_state_dict(model, ckpt['model_state'])
    start_step = ckpt.get('step', 0)
else:
    print('No checkpoint, starting from scratch')
    model = UnifiedMultidimensionalTransformer().to(device)
    start_step = 0

model.train()
total_params = sum(p.numel() for p in model.parameters())
print(f'Model: {total_params:,} params, starting step {start_step}')

# ---- Data ----
print(f'Loading {DATA_PATH}')
data = np.load(DATA_PATH)
data = np.array(data, dtype=np.int64)
N = len(data)
print(f'Data: {N:,} tokens')

# ---- Optimizer ----
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS)

# ---- Loss weights ----
W_CE = 1.0
W_NXT = 0.1
W_SRG = 0.05

# ---- Training ----
t0 = time.time()
loss_log = defaultdict(list)

print(f'\nHybrid training (CE+nxt+SRG) for {N_STEPS} steps...')
for step in range(N_STEPS):
    idx = np.random.randint(0, N - L - 1, size=B)
    x = torch.tensor([data[i:i+L] for i in idx], dtype=torch.long, device=device)
    targets = torch.tensor([data[i+1:i+L+1] for i in idx], dtype=torch.long, device=device)
    
    optimizer.zero_grad()
    
    h, scores, weights, heads_out = model.forward(
        x, return_scores=True, return_heads=True, capture_attn=False
    )
    
    # 1. CE loss (next-token prediction)
    logits = scores  # [B, L, V]
    ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    
    # 2. nxt-loss (coordinate shift prediction)
    nxt = heads_out.get('boundary_next')
    if nxt is not None and h.shape[1] > 1:
        delta = h[:, 1:] - h[:, :-1]
        nxt_pred = nxt[:, :-1]
        nxt_loss = F.mse_loss(nxt_pred, delta)
    else:
        nxt_loss = torch.tensor(0.0, device=h.device)
    
    # 3. SRG loss (topological smoothness)
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
    loss_log['total'].append(total.item())
    
    if step % LOG_EVERY == 0:
        avg = {k: np.mean(v[-LOG_EVERY:]) for k, v in loss_log.items() if v}
        elapsed = time.time() - t0
        print(f'[HYBRID {step}/{N_STEPS}] ce={avg.get("ce",0):.4f} '
              f'nxt={avg.get("nxt",0):.4f} srg={avg.get("srg",0):.4f} '
              f'acc={acc:.3f} | {elapsed:.0f}s')
    
    if step % SAVE_EVERY == 0 and step > 0:
        out_path = f'{CKPT_DIR}/hybrid_step_{start_step + step}.pt'
        tmp = out_path + '.tmp'
        torch.save({
            'step': start_step + step, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, tmp)
        os.replace(tmp, out_path)
        print(f'  Saved {out_path}')

# ---- Final save ----
elapsed = time.time() - t0
out_path = f'{CKPT_DIR}/hybrid_final.pt'
tmp = out_path + '.tmp'
torch.save({
    'step': start_step + N_STEPS, 'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}, tmp)
os.replace(tmp, out_path)
print(f'\nDone. {N_STEPS} steps, {elapsed:.0f}s. Final CE loss: {np.mean(loss_log["ce"][-100:]):.4f}')
print(f'Saved to {out_path}')

# ---- Generation test ----
print('\n--- Generation test ---')
model.eval()
cv = CharVocab()
prompts = ['the ', 'once ', 'i am ', 'she was ']
for p in prompts:
    inp = cv.encode(p)
    prompt_ids = [i for i in inp if i not in (cv.BOS_IDX, cv.EOS_IDX)]
    t, meta = model.generate_text(prompt_ids, cv, max_new=48)
    print(f'  {repr(p):>12} -> {repr(t)}')
