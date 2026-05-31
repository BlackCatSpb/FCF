"""
decoder_finetune.py — Быстрая калибровка decoder.linear через CE-loss.

Фиксит главную проблему: decoder.linear не обучен предсказывать next-token.
Тренирует только decoder.linear + embed.coordinates, всё остальное frozen.
500 шагов ~5 минут на MX550.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, sys, os, time, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.char_vocab import CharacterVocab as CharVocab
from eva.symbolic.train_v3 import safe_load_state_dict, TrainingConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ---- Config ----
DATA_PATH = 'real_data/full_corpus_encoded.npy'
CKPT_DIR = 'checkpoints/v3'
N_STEPS = 2000
B, L = 8, 64           # small batch for speed on 2GB GPU
LR = 5e-4
SAVE_EVERY = 500

# ---- Load checkpoint ----
ckpts = sorted(glob.glob(f'{CKPT_DIR}/train_v3_step_*.pt'))
if not ckpts:
    print(f'No checkpoints in {CKPT_DIR}')
    sys.exit(1)
latest = ckpts[-1]
print(f'Loading {latest}')

model = UnifiedMultidimensionalTransformer().to(device)
ckpt = torch.load(latest, map_location=device, weights_only=True)
safe_load_state_dict(model, ckpt['model_state'])
model.train()
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

# ---- FREEZE: only train decoder.linear (+temperature), keep coordinates frozen ----
for name, p in model.named_parameters():
    if 'decoder.linear' in name or 'decoder.temperature' in name:
        p.requires_grad = True
        print(f'  [TRAIN] {name} ({p.numel():,} params)')
    else:
        p.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable params: {trainable:,} (decoder.linear={161*128+161:,} + coords={161*128:,})')

# ---- Data ----
print(f'Loading {DATA_PATH}')
data = np.load(DATA_PATH)
data = np.array(data, dtype=np.int64)
print(f'Data shape: {data.shape}')

# ---- Optimizer (only trainable params) ----
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS)

# ---- Training loop ----
N = len(data)
t0 = time.time()
losses = []
print(f'\nTraining decoder.linear with CE loss for {N_STEPS} steps...')

for step in range(N_STEPS):
    # Sample random batch
    idx = np.random.randint(0, N - L - 1, size=B)
    x = torch.tensor([data[i:i+L] for i in idx], dtype=torch.long, device=device)
    targets = torch.tensor([data[i+1:i+L+1] for i in idx], dtype=torch.long, device=device)
    
    optimizer.zero_grad()
    
    # Forward: need return_scores=True to get decoder logits
    h, scores, weights, heads_out = model.forward(
        x, return_scores=True, return_heads=True, capture_attn=False
    )
    
    # CE loss on decoder logits
    logits = scores  # [B, L, V]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    losses.append(loss.item())
    
    if step % 100 == 0:
        avg = np.mean(losses[-100:]) if losses else loss.item()
        acc = (logits.argmax(-1) == targets).float().mean().item()
        elapsed = time.time() - t0
        print(f'[CE {step}/{N_STEPS}] loss={avg:.4f} acc={acc:.3f} | {elapsed:.0f}s')
    
    if step > 0 and step % SAVE_EVERY == 0:
        out_path = f'{CKPT_DIR}/decoder_finetune_step_{step}.pt'
        tmp_path = out_path + '.tmp'
        torch.save({
            'step': step, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, tmp_path)
        os.replace(tmp_path, out_path)
        print(f'[Save] {out_path}')

# ---- Final save ----
out_path = f'{CKPT_DIR}/decoder_finetune_final.pt'
tmp_path = out_path + '.tmp'
torch.save({
    'step': N_STEPS, 'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}, tmp_path)
os.replace(tmp_path, out_path)

elapsed = time.time() - t0
final_loss = np.mean(losses[-100:]) if losses else 0
print(f'\nDone. {N_STEPS} steps, {elapsed:.0f}s, final CE loss={final_loss:.4f}')
print(f'Saved to {out_path}')

# ---- Quick generation test ----
print('\n--- Generation test ---')
model.eval()
cv = CharVocab()

prompts = ['the ', 'once ', 'i am ', 'she was ']
for p in prompts:
    ids = cv.encode(p)
    gen_ids = [i for i in ids if i not in (cv.BOS_IDX, cv.EOS_IDX)]
    t, meta = model.generate_text(gen_ids, cv, max_new=32)
    print(f'  {repr(p):>12} -> {repr(t)}')
