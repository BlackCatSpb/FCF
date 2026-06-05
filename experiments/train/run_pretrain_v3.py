"""
run_pretrain_v3.py — Полный pre-training pipeline.

1. MiniBERT MLM pre-train
2. EVA with MiniBERT teacher + multi-task + distillation

Usage:
    python run_pretrain_v3.py --mb-steps 10000 --eva-steps 10000
    python run_pretrain_v3.py  # only teacher, ~500 min for 50K eva
"""
import torch, sys, os, time, numpy as np, signal, argparse
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')

parser = argparse.ArgumentParser()
parser.add_argument('--mb-steps', type=int, default=10000)
parser.add_argument('--eva-steps', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--seq-len', type=int, default=128)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--drop-ce', action='store_true', default=True,
                    help='Head-only EVA training (no next-token prediction)')
parser.add_argument('--teacher-only', action='store_true',
                    help='Only train MiniBERT teacher, skip EVA')
parser.add_argument('--hybrid', action='store_true',
                    help='Include CE loss in EVA training (legacy mode)')
args = parser.parse_args()

INTERRUPTED = False
def handler(signum, frame):
    global INTERRUPTED
    print('\n[Signal] Interrupted, saving checkpoint...')
    INTERRUPTED = True
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

DATA_PATH = 'real_data/full_corpus_encoded.npy'
CKPT_DIR = 'checkpoints/v3'
os.makedirs(CKPT_DIR, exist_ok=True)

# --- Load data ---
print(f'[Data] Loading {DATA_PATH}')
data = np.load(DATA_PATH)
print(f'[Data] Shape: {data.shape}, dtype: {data.dtype}')
data = np.array(data, dtype=np.int64)

N_TOKENS = len(data)
B, L = args.batch_size, args.seq_len
TOKENS_PER_STEP = B * L

print(f'[Config] MiniBERT: {args.mb_steps} steps | EVA: {args.eva_steps} steps')
print(f'[Config] Batch={B} Seq={L} LR={args.lr}')

# ============================================================
# Phase 1: MiniBERT MLM pre-train
# ============================================================
from eva.symbolic.teacher_models.minibert import MiniBERT

minibert = MiniBERT(vocab_size=161, d_model=128).to(device)
N_STEPS_MB = args.mb_steps
SAVE_EVERY_MB = max(1000, N_STEPS_MB // 10)
LOG_EVERY_MB = 100
LR_MB = args.lr

optim_mb = torch.optim.AdamW(minibert.parameters(), lr=LR_MB, weight_decay=0.01)
scheduler_mb = torch.optim.lr_scheduler.CosineAnnealingLR(optim_mb, T_max=N_STEPS_MB)

# Check resume
mb_start = 0
mb_resume = f'{CKPT_DIR}/minibert_latest.pt'
if os.path.exists(mb_resume):
    ckpt = torch.load(mb_resume, map_location=device, weights_only=True)
    minibert.load_state_dict(ckpt['model_state'])
    optim_mb.load_state_dict(ckpt['optimizer_state'])
    mb_start = ckpt['step']
    print(f'[MiniBERT] Resumed from step {mb_start}')

from eva.symbolic.train_v3 import create_batch, MultiTaskLoss, TrainingConfig, safe_load_state_dict

mb_config = TrainingConfig(batch_size=B, seq_len=L, lr=LR_MB)

minibert.train()
print(f'[MiniBERT] Training {N_STEPS_MB} steps...')
t0 = time.time() if N_STEPS_MB > 0 else 0
mb_losses = []
for step in range(mb_start, N_STEPS_MB):
    if INTERRUPTED:
        print('[MiniBERT] Interrupted')
        break
    batch = create_batch(data, mb_config, device)
    if batch is None:
        break
    optim_mb.zero_grad()
    masked, mask = minibert.generate_mlm_batch(batch['input_ids'])
    loss, acc = minibert.mlm_loss(masked, mask)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(minibert.parameters(), 1.0)
    optim_mb.step()
    scheduler_mb.step()
    mb_losses.append(loss.item())

    if step % LOG_EVERY_MB == 0:
        avg = np.mean(mb_losses[-LOG_EVERY_MB:])
        elapsed = time.time() - t0
        tokens_seen = step * TOKENS_PER_STEP
        print(f'[MB {step}/{N_STEPS_MB}] loss={avg:.4f} acc={acc.item():.3f} '
              f'tokens={tokens_seen/1e6:.1f}M | {elapsed:.0f}s')

    if step % SAVE_EVERY_MB == 0 and step > 0:
        tmp_path = mb_resume + '.tmp'
        torch.save({
            'step': step, 'model_state': minibert.state_dict(),
            'optimizer_state': optim_mb.state_dict(),
        }, tmp_path)
        os.replace(tmp_path, mb_resume)
        print(f'[MiniBERT] Saved to {mb_resume}')

# Final save (handles case where step not defined when loop didn't execute)
try:
    final_mb_steps = step + 1
except NameError:
    final_mb_steps = mb_start
tmp_path = mb_resume + '.tmp'
torch.save({
    'step': final_mb_steps, 'model_state': minibert.state_dict(),
    'optimizer_state': optim_mb.state_dict(),
}, tmp_path)
os.replace(tmp_path, mb_resume)
minibert.eval()
print(f'[MiniBERT] Done. {final_mb_steps} steps, {time.time()-t0:.0f}s')

if args.teacher_only:
    print('[Done — teacher-only mode]')
    sys.exit(0)

# ============================================================
# Phase 2: EVA pre-train with MiniBERT teacher
# ============================================================
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

eva = UnifiedMultidimensionalTransformer(vocab_size=161, coord_dim=128).to(device)
eva.set_teacher(minibert, teacher_hidden_dim=128)
print(f'[EVA] {eva.summary()}')

N_STEPS_EVA = args.eva_steps
SAVE_EVERY_EVA = max(1000, N_STEPS_EVA // 10)
LOG_EVERY_EVA = 100
LR_EVA = args.lr

optim = torch.optim.AdamW(eva.parameters(), lr=LR_EVA, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_STEPS_EVA)

drop_ce = args.drop_ce and not args.hybrid

eva_config = TrainingConfig(
    batch_size=B, seq_len=L, lr=LR_EVA,
    drop_ce=drop_ce,
    use_weight_context=True,
    update_weight_every=50,
)
# Head-only: SRG главный loss, остальные вспомогательные
if drop_ce:
    eva_config.w_srg = 0.5          # топологическая гладкость → основа обучения
    eva_config.w_concept = 0.15
    eva_config.w_contra = 0.15
    eva_config.w_uncertainty = 0.1
    eva_config.w_residual = 0.15
    eva_config.w_head_consistency = 0.15
    eva_config.w_self_distill = 0.3
    eva_config.w_distill = 0.5
    eva_config.w_attn_entropy = 0.05
    eva_config.w_cross_gen = 0.0    # отключено для скорости; включить в continuous
else:
    # Legacy hybrid mode
    eva_config.w_ce = 1.0
    eva_config.w_concept = 0.1
    eva_config.w_contra = 0.1
    eva_config.w_uncertainty = 0.1
    eva_config.w_contrastive = 0.05
    eva_config.w_distill = 1.0

loss_fn = MultiTaskLoss(eva_config)

# Check resume
eva_start = 0
eva_resume = f'{CKPT_DIR}/eva_v3_latest.pt'
if os.path.exists(eva_resume):
    ckpt = torch.load(eva_resume, map_location=device, weights_only=True)
    safe_load_state_dict(eva, ckpt['model_state'])
    optim.load_state_dict(ckpt['optimizer_state'])
    eva_start = ckpt['step']
    print(f'[EVA] Resumed from step {eva_start}')

from collections import defaultdict

eva.train()
print(f'[EVA] Training {N_STEPS_EVA} steps...')
t0 = time.time() if N_STEPS_EVA > 0 else 0
recent = defaultdict(list)
for step in range(eva_start, N_STEPS_EVA):
    if INTERRUPTED:
        print('[EVA] Interrupted')
        break
    batch = create_batch(data, mb_config, device)
    if batch is None:
        break

    optim.zero_grad()
    model_out = eva.forward(batch['input_ids'],
                             return_scores=not drop_ce,
                             return_heads=True,
                             use_weight=True,
                             capture_attn=True)
    h, scores, weights, heads_out = model_out

    intrinsic = {
        'contra': eva._intrinsic_contra_labels(h, batch['input_ids']),
        'concept': eva._intrinsic_concept_labels(h, batch['input_ids']),
    }

    losses = loss_fn(model_out, batch, intrinsic, h2=None)

    # Head-only losses (если drop_ce)
    if drop_ce:
        losses['srg'] = eva.srg_loss(h)
        losses['attn_entropy'] = eva.attention_entropy()
        losses['head_consistency'] = eva.head_consistency_loss(heads_out)
        try:
            losses['self_distill'] = eva.self_distill_thought(heads_out, h)
        except Exception:
            losses['self_distill'] = torch.tensor(0.0, device=h.device)

        # Weighted sum (head-only)
        total = (eva_config.w_concept * losses['concept'] +
                 eva_config.w_contra * losses['contradiction'] +
                 eva_config.w_uncertainty * losses['uncertainty'] +
                 eva_config.w_boundary * losses['boundary'] +
                 eva_config.w_residual * losses['residual'] +
                 eva_config.w_srg * losses['srg'] +
                 eva_config.w_attn_entropy * losses['attn_entropy'] +
                 eva_config.w_head_consistency * losses['head_consistency'] +
                 eva_config.w_self_distill * losses['self_distill'] +
                 eva_config.w_contrastive * losses['contrastive'])
    else:
        total = losses['total']

    # Distillation (всегда)
    d_loss = eva.distill_loss(h, heads_out)
    losses['distill'] = d_loss
    total = total + eva_config.w_distill * d_loss

    total.backward()
    torch.nn.utils.clip_grad_norm_(eva.parameters(), 1.0)
    optim.step()
    scheduler.step()

    loss_items = {k: v.item() if torch.is_tensor(v) else v
                  for k, v in losses.items()}
    for k, v in loss_items.items():
        recent[k].append(v)
    recent['total'].append(total.item())

    if step % 50 == 0:
        eva.update_weight_token()

    if step % LOG_EVERY_EVA == 0:
        avg = {k: np.mean(v[-LOG_EVERY_EVA:]) for k, v in recent.items() if v}
        elapsed = time.time() - t0
        tokens_seen = step * TOKENS_PER_STEP
        mode = 'HEAD' if drop_ce else 'HYBRID'
        print(f'[{mode} {step}/{N_STEPS_EVA}] tot={avg["total"]:.4f} '
              f'conc={avg.get("concept",0):.4f} ctr={avg.get("contradiction",0):.4f} '
              f'srg={avg.get("srg",0):.4f} attn={avg.get("attn_entropy",0):.4f} '
              f'cons={avg.get("head_consistency",0):.4f} res={avg.get("residual",0):.4f} '
              f'dist={avg.get("distill",0):.4f} sd={avg.get("self_distill",0):.4f} '
              f'ce={avg.get("ce",0):.4f} | {elapsed:.0f}s')

    if step % SAVE_EVERY_EVA == 0 and step > 0:
        tmp_path = eva_resume + '.tmp'
        torch.save({
            'step': step, 'model_state': eva.state_dict(),
            'optimizer_state': optim.state_dict(),
        }, tmp_path)
        os.replace(tmp_path, eva_resume)
        print(f'[EVA] Saved to {eva_resume}')

# Final save
try:
    final_eva_steps = step + 1
except NameError:
    final_eva_steps = eva_start
tmp_path = eva_resume + '.tmp'
torch.save({
    'step': final_eva_steps, 'model_state': eva.state_dict(),
    'optimizer_state': optim.state_dict(),
}, tmp_path)
os.replace(tmp_path, eva_resume)
print(f'[EVA] Done. {final_eva_steps} steps, {time.time()-t0:.0f}s')

# ============================================================
# Summary
# ============================================================
print()
print('=== PRETRAIN COMPLETE ===')
print(f'MiniBERT: {sum(p.numel() for p in minibert.parameters()):,} params, {final_mb_steps} steps')
print(f'EVA:      {sum(p.numel() for p in eva.parameters()):,} params, {final_eva_steps} steps')
print(f'Mode:     {"HEAD-ONLY" if drop_ce else "HYBRID (CE+heads)"}')
print(f'Teacher:  MiniBERT ({args.teacher_dim}-dim)')
print(f'Checkpoints: {CKPT_DIR}/')
if INTERRUPTED:
    print('(interrupted — resume by re-running)')
