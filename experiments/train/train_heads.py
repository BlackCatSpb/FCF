"""
train_heads.py — Bridge: load concept basis → train heads on real text.

Pipeline:
1. Load concept basis checkpoint (trained on ConceptNet, step ~56K)
2. Load MiniBERT teacher (distillation)
3. Set teacher + weight context on model
4. Train heads on full_corpus_encoded.npy with drop_ce=True
5. Save to checkpoints/v3/

Losses:
  - concept + contradiction + uncertainty + boundary + residual
  - SRG (topological smoothness)
  - attention_entropy + head_consistency
  - self_distill (thought loop)
  - distill (teacher alignment)
  - kca_aux

Usage:
    python train_heads.py --steps 10000
    python train_heads.py --steps 10000 --lr 3e-4 --batch-size 8
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, time, signal, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.char_vocab import CharacterVocab as CharVocab
from eva.symbolic.train_v3 import TrainingConfig, MultiTaskLoss, create_batch, save_checkpoint, safe_load_state_dict
from eva.symbolic.teacher_models.minibert import MiniBERT
from eva.symbolic.continuous_runtime import (
    continuous_learning_loop, RuntimeConfig, DataSource, ThoughtLoopConfig
)
from eva.symbolic.trajectory_store import TrajectoryStore

parser = argparse.ArgumentParser()
parser.add_argument('--concept-basis', type=str,
                    default='checkpoints/concept_basis/concept_basis_latest.pt')
parser.add_argument('--teacher', type=str, default='checkpoints/v3/minibert_latest.pt')
parser.add_argument('--teacher-dim', type=int, default=128)
parser.add_argument('--data', type=str, default='real_data/full_corpus_encoded.npy')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/v3')
parser.add_argument('--steps', type=int, default=10000, help='Training steps')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--seq-len', type=int, default=128)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--warmup', type=int, default=500)
parser.add_argument('--save-every', type=int, default=1000)
parser.add_argument('--log-every', type=int, default=50)
parser.add_argument('--no-teacher', action='store_true',
                    help='Skip teacher distillation')
parser.add_argument('--resume', type=str, default=None,
                    help='Resume from checkpoint (overrides concept-basis)')
parser.add_argument('--continuous', action='store_true',
                    help='Run continuous self-supervised learning loop (generate→evaluate→H2K→learn)')
parser.add_argument('--runtime-cycles', type=int, default=50,
                    help='Number of continuous learning cycles')
parser.add_argument('--h2k-path', type=str, default='h2k_hypotheses.jsonl',
                    help='Path to store H2K hypotheses')
parser.add_argument('--trajectory-store', type=str, default='',
                    help='Path to save/load TrajectoryStore')
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

# ---- Load model ----
if args.resume:
    print(f'[Model] Resuming from {args.resume}')
    resume_ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    model = UnifiedMultidimensionalTransformer().to(device)
    if not args.no_teacher and os.path.exists(args.teacher):
        print(f'[Teacher] Loading MiniBERT from {args.teacher}')
        teacher_ckpt = torch.load(args.teacher, map_location=device, weights_only=True)
        teacher = MiniBERT(vocab_size=161, d_model=128).to(device)
        teacher.load_state_dict(teacher_ckpt['model_state'])
        model.set_teacher(teacher, args.teacher_dim)
    safe_load_state_dict(model, resume_ckpt['model_state'])
    start_step = resume_ckpt.get('step', 0)
    print(f'[Model] Resumed at step {start_step}')
else:
    print(f'[Model] Loading concept basis from {args.concept_basis}')
    concept_ckpt = torch.load(args.concept_basis, map_location=device, weights_only=True)
    model = UnifiedMultidimensionalTransformer().to(device)
    safe_load_state_dict(model, concept_ckpt['model_state'])
    print(f'[Model] Concept basis loaded (step {concept_ckpt.get("step", "?")})')
    start_step = 0
    if not args.no_teacher and os.path.exists(args.teacher):
        print(f'[Teacher] Loading MiniBERT from {args.teacher}')
        teacher_ckpt = torch.load(args.teacher, map_location=device, weights_only=True)
        teacher = MiniBERT(vocab_size=161, d_model=128).to(device)
        teacher.load_state_dict(teacher_ckpt['model_state'])
        model.set_teacher(teacher, args.teacher_dim)
        print(f'[Teacher] Distillation active (dim={args.teacher_dim})')
    else:
        print('[Teacher] Skipped')

# ---- Config ----
config = TrainingConfig(
    lr=args.lr,
    batch_size=args.batch_size,
    seq_len=args.seq_len,
    warmup_steps=args.warmup,
    save_every=args.save_every,
    log_every=args.log_every,
    checkpoint_dir=args.checkpoint_dir,
    drop_ce=True,
    use_weight_context=True,
    update_weight_every=50,
    cross_gen_every=999999,
)

config.w_srg = 0.4
config.w_nxt = 0.3
config.w_concept = 0.15
config.w_contra = 0.15
config.w_uncertainty = 0.05
config.w_boundary = 0.05
config.w_boundary_valid = 0.05
config.w_residual = 0.05
config.w_head_consistency = 0.1
config.w_self_distill = 0.2
config.w_distill = 0.3
config.w_attn_entropy = 0.05
config.w_cross_gen = 0.0
config.w_kca = 0.05
config.w_ce = 0.0

loss_fn = MultiTaskLoss(config)

# ---- Optimizer ----
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

# ---- Resume optimizer state ----
if args.resume and 'optimizer_state' in resume_ckpt:
    try:
        optimizer.load_state_dict(resume_ckpt['optimizer_state'])
        # Move optimizer state to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f'[Optimizer] State restored')
    except Exception as e:
        print(f'[Optimizer] Warning: could not restore state ({e}), using fresh')

# ---- Data ----
print(f'[Data] Loading {args.data}')
data = np.load(args.data)
print(f'[Data] Shape: {data.shape}, dtype: {data.dtype}')
data = np.array(data, dtype=np.int64)

# ---- Training mode: continuous self-supervised or batch ----
if args.continuous:
    # ---- Continuous self-supervised learning ----
    model.train()
    if config.use_weight_context:
        model.set_weight_context(True)
        model.update_weight_token()

    cv = CharVocab()
    data_source = DataSource(args.data.replace('.npy', '.txt')) if os.path.exists(args.data.replace('.npy', '.txt')) else None
    if data_source:
        print(f'[Continuous] DataSource ready ({len(data_source.texts)} prompts)')
    else:
        print('[Continuous] No text source for prompts, using empty prompt')

    runtime_cfg = RuntimeConfig(
        device=str(device),
        max_tokens_per_cycle=128,
        max_generations=5,
        h2k_buffer_size=10,
        train_batch_size=config.batch_size,
        train_steps_per_cycle=5,
        h2k_path=args.h2k_path,
        trajectory_store_path=args.trajectory_store,
        log_every=1,
    )
    thought_cfg = ThoughtLoopConfig()
    thought_cfg.max_iterations = 3

    print(f'[Continuous] Starting {args.runtime_cycles} self-supervised cycles...')
    t0 = time.time()
    results = continuous_learning_loop(
        model, cv, optimizer, loss_fn,
        n_cycles=args.runtime_cycles,
        runtime_cfg=runtime_cfg,
        thought_cfg=thought_cfg,
        train_cfg=config,
        data_source=data_source,
        trajectory_store_path=args.trajectory_store,
    )
    elapsed = time.time() - t0

    avg_best = np.mean([r['best_composite'] for r in results]) if results else 0
    avg_train_loss = np.mean([r['train_loss'] for r in results]) if results else 0
    final_traj = results[-1]['trajectories'] if results else 0
    print(f'[Continuous] Done. {args.runtime_cycles} cycles, {elapsed:.0f}s')
    print(f'[Continuous] avg_best_composite={avg_best:.3f} avg_train_loss={avg_train_loss:.4f} trajectories={final_traj}')

    # Save final checkpoint
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, 'eva_v3_heads_latest.pt')
    tmp_path = path + '.tmp'
    torch.save({
        'step': args.runtime_cycles,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, tmp_path)
    os.replace(tmp_path, path)
    print(f'[Save] Final checkpoint saved to {path}')

else:
    # ---- Batch training loop ----
    model.train()
    if config.use_weight_context:
        model.set_weight_context(True)
        model.update_weight_token()

    loss_keys = ['total', 'concept', 'contradiction', 'uncertainty',
                 'boundary', 'boundary_valid', 'residual', 'distill',
                 'srg', 'srg_real', 'attn_entropy', 'head_consistency', 'cross_gen', 'self_distill', 'kca', 'nxt', 'meta_kl']
    total_losses = {k: 0.0 for k in loss_keys}
    n_batches = 0
    t0 = time.time()

    trajectory_store = None
    if args.trajectory_store:
        trajectory_store = TrajectoryStore(max_trajectories=100000)
        if os.path.exists(args.trajectory_store):
            try:
                trajectory_store.load(args.trajectory_store)
                print(f'[TrajectoryStore] Loaded {trajectory_store.total_stored} trajectories')
            except Exception:
                pass

    print(f'[Train] NAVIGATOR (head-only + nxt-loss) training {args.steps} steps (starting from step {start_step})...')
    for step in range(start_step, start_step + args.steps):
        if INTERRUPTED:
            break

        batch = create_batch(data, config, device)
        if batch is None:
            print('[Train] Data exhausted')
            break

        optimizer.zero_grad()

        # Forward
        model_out = model.forward(
            batch['input_ids'],
            return_scores=False,
            return_heads=True,
            use_weight=config.use_weight_context,
            capture_attn=True,
        )
        h, _, _, heads_out = model_out

        # Intrinsic labels
        intrinsic_labels = {
            'contra': model._intrinsic_contra_labels(h, batch['input_ids']),
            'concept': model._intrinsic_concept_labels(h, batch['input_ids']),
        }

        losses = loss_fn(model_out, batch, intrinsic_labels, h2=None)

        # nxt_all_loss: predict coordinate delta at EVERY position
        nxt = heads_out.get('boundary_next')
        if nxt is not None and h.shape[1] > 1:
            delta_actual = h[:, 1:] - h[:, :-1]
            nxt_pred = nxt[:, :-1]
            losses['nxt'] = F.mse_loss(nxt_pred, delta_actual)
        else:
            losses['nxt'] = torch.tensor(0.0, device=h.device)

        # Head-only losses
        losses['srg'] = model.temporal_smoothness_loss(h)
        losses['srg_real'] = model.srg_loss(h)
        losses['attn_entropy'] = model.attention_entropy()
        losses['head_consistency'] = model.head_consistency_loss(heads_out)
        try:
            losses['self_distill'] = model.self_distill_thought(heads_out, h)
        except Exception:
            losses['self_distill'] = torch.tensor(0.0, device=h.device)
        if config.w_kca > 0:
            losses['kca'] = model.kca_aux_loss(h, heads_out)
        else:
            losses['kca'] = torch.tensor(0.0, device=h.device)

        # MetaWeighter KL divergence
        context = h.mean(dim=1)
        losses['meta_kl'] = model.meta_weighter.kl_loss(context)

        distill_val = model.distill_loss(h, heads_out)
        losses['distill'] = distill_val

        head_only_keys = {
            'srg': config.w_srg,
            'srg_real': config.w_srg_real,
            'attn_entropy': config.w_attn_entropy,
            'head_consistency': config.w_head_consistency,
            'cross_gen': config.w_cross_gen,
            'self_distill': config.w_self_distill,
            'meta_kl': config.w_meta_kl,
        }
        total = (config.w_concept * losses['concept'] +
                 config.w_contra * losses['contradiction'] +
                 config.w_uncertainty * losses['uncertainty'] +
                 config.w_boundary * losses['boundary'] +
                 config.w_boundary_valid * losses['boundary_valid'] +
                 config.w_residual * losses['residual'] +
                 config.w_distill * distill_val +
                 config.w_kca * losses['kca'] +
                 config.w_nxt * losses['nxt'])
        for k, w in head_only_keys.items():
            total = total + w * losses.get(k, torch.tensor(0.0, device=h.device))

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        if config.use_weight_context and step % config.update_weight_every == 0:
            model.update_weight_token()

        # Store trajectory (every 10 steps to keep store manageable)
        if trajectory_store is not None and step % 10 == 0:
            traj_np = h[0].cpu().numpy().astype(np.float32)
            ids_list = batch['input_ids'][0].tolist()
            trajectory_store.store('', ids_list, traj_np)

        for k in total_losses:
            total_losses[k] += losses.get(k, torch.tensor(0.0, device=h.device)).item()
        total_losses['total'] += total.item()
        n_batches += 1

        if step % config.log_every == 0 and step > start_step:
            avg = {k: v / n_batches for k, v in total_losses.items()}
            elapsed = time.time() - t0
            traj_str = f' traj={trajectory_store.total_stored}' if trajectory_store else ''
            print(f'[NAV Step {step}/{start_step + args.steps}] tot={avg["total"]:.4f} '
                  f'conc={avg["concept"]:.4f} ctr={avg["contradiction"]:.4f} '
                  f'srg={avg["srg"]:.4f} srg_r={avg["srg_real"]:.4f} nxt={avg["nxt"]:.4f} '
                  f'cons={avg["head_consistency"]:.4f} res={avg["residual"]:.4f} '
                  f'bnd={avg["boundary"]:.4f} dist={avg["distill"]:.4f} '
                  f'sd={avg["self_distill"]:.4f} kca={avg["kca"]:.4f} mkl={avg["meta_kl"]:.4f}'
                  f'{traj_str} | {elapsed:.0f}s')

        if step % config.save_every == 0 and step > start_step:
            save_checkpoint(model, optimizer, step, config)

    # Save TrajectoryStore
    if trajectory_store is not None and args.trajectory_store:
        trajectory_store.save(args.trajectory_store)
        print(f'[TrajectoryStore] Saved {trajectory_store.total_stored} trajectories to {args.trajectory_store}')

    # ---- Final save ----
    final_step = step + 1 if not INTERRUPTED else step
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, 'eva_v3_heads_latest.pt')
    tmp_path = path + '.tmp'
    torch.save({
        'step': final_step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, tmp_path)
    os.replace(tmp_path, path)
    print(f'[Save] Final checkpoint saved to {path}')

    elapsed = time.time() - t0
    print(f'[Train] Done. {final_step} steps, {elapsed:.0f}s')
    print(f'[Train] Model: {sum(p.numel() for p in model.parameters()):,} params')
