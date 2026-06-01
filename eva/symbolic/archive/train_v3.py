"""
train_v3.py — Multi-task training for EVA Symbolic v3.

Режимы:
1. Head-only (drop_ce=True): только головы, без CE (новый режим)
2. Hybrid (drop_ce=False): CE + головы (старый режим)

Loss (head-only):
  concept + contradiction + uncertainty + residual + boundary
  + SRG (топологическая гладкость h)
  + attention_entropy (острота внимания)
  + head_consistency (concept ≈ 1 - contradiction)
  + cross_gen (alignment разных генераций)
  + self_distill (thought loop → direct output)
  + distill (teacher alignment)
  + contrastive (SimCSE)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import os, time, json
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    batch_size: int = 8
    seq_len: int = 128
    num_epochs: int = 1
    warmup_steps: int = 500
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 50
    clip_grad_norm: float = 1.0

    drop_ce: bool = True            # выключить next-token prediction

    # loss weights — head-only режим
    w_concept: float = 0.1
    w_contra: float = 0.1
    w_uncertainty: float = 0.1
    w_boundary: float = 0.05
    w_boundary_valid: float = 0.05
    w_residual: float = 0.1
    w_srg: float = 0.3             # топологическая гладкость (главный loss!)
    w_attn_entropy: float = 0.05   # острота внимания
    w_head_consistency: float = 0.1  # concept ≈ 1-contra
    w_cross_gen: float = 0.1       # alignment разных генераций
    w_self_distill: float = 0.2    # thought loop → direct output
    w_contrastive: float = 0.05
    w_distill: float = 0.1
    w_ewc: float = 0.01
    w_kca: float = 0.05           # differentiable KCA aux loss
    w_nxt: float = 0.3            # nxt head: coordinate delta prediction
    w_srg_real: float = 0.1        # SRG loss: 1.0 - SRG(query, response)
    w_meta_kl: float = 0.01        # MetaWeighter KL(weights || uniform)

    # legacy CE weights (используются только если drop_ce=False)
    w_ce: float = 1.0

    # continuous learning
    continuous_mode: bool = False
    h2k_every: int = 10
    cross_gen_every: int = 5       # как часто делать cross-gen (дорого)

    # weight context
    use_weight_context: bool = True
    update_weight_every: int = 10

    # checkpoint
    checkpoint_dir: str = "checkpoints/v3"
    resume: Optional[str] = None


class SimCSELoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temp = temperature

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        B, L, D = h1.shape
        h1_flat = F.normalize(h1.reshape(B * L, D), dim=-1)
        h2_flat = F.normalize(h2.reshape(B * L, D), dim=-1)
        if (h1 - h2).abs().max().item() < 1e-8:
            h2_flat = h2_flat + torch.randn_like(h2_flat) * 1e-6
        sim = h1_flat @ h2_flat.T / self.temp
        labels = torch.arange(B * L, device=h1.device)
        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
        return loss * 0.5


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss: головы + опционально CE (legacy).
    """
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss() if not config.drop_ce else None
        self.simcse = SimCSELoss()

    def forward(self, model_out, batch, intrinsic_labels, h2=None):
        h, scores, weights, heads_out = model_out
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        B, L = input_ids.shape

        contra_labels = intrinsic_labels['contra']
        conc_labels = intrinsic_labels['concept']

        loss_dict = {}

        # 1. CE (legacy) — включается только если drop_ce=False
        if not self.config.drop_ce and scores is not None and self.ce is not None:
            loss_dict['ce'] = self.ce(scores.view(-1, scores.shape[-1]), target_ids.view(-1))
        else:
            loss_dict['ce'] = torch.tensor(0.0, device=h.device)

        # 2. ConceptHead
        loss_dict['concept'] = self.bce(heads_out['concept'], conc_labels)

        # 3. ContradictionHead
        loss_dict['contradiction'] = self.bce(heads_out['contradiction'], contra_labels)

        # 4. UncertaintyHead
        with torch.no_grad():
            h_pred = h[:, :-1]
            h_next = h[:, 1:]
            actual_mse = F.pad((h_pred - h_next) ** 2, (0, 0, 0, 1))
        loss_dict['uncertainty'] = self.mse(heads_out['uncertainty'], actual_mse)

        # 5. Boundary (predictor + validator)
        end, nxt, conn = heads_out.get('boundary_end'), heads_out.get('boundary_next'), heads_out.get('boundary_conn')
        bv = heads_out.get('boundary_valid')
        if end is not None and nxt is not None:
            with torch.no_grad():
                word_open_mask = (input_ids == 157).float().unsqueeze(-1)
                word_close_mask = (input_ids == 158).float().unsqueeze(-1)
                # end ≈ h at word_close; nxt ≈ h at word_open (shifted for next word start)
                end_target = h * word_close_mask
                nxt_target = h * word_open_mask.roll(1, dims=1)
                mse_end = self.mse(end * word_close_mask, end_target)
                mse_nxt = self.mse(nxt * word_open_mask.roll(1, dims=1), nxt_target)
                # Trajectory change-point signal
                if h.shape[1] > 1:
                    h_diffs = torch.norm(h[:, 1:] - h[:, :-1], dim=-1)
                    thresh = h_diffs.mean(dim=-1, keepdim=True) + h_diffs.std(dim=-1, keepdim=True)
                    traj_bnd = F.pad((h_diffs > thresh).float().unsqueeze(-1), (0, 0, 0, 1))
                    end_t = self.mse(end * traj_bnd, h * traj_bnd)
                    nxt_t = self.mse(nxt * traj_bnd.roll(1, dims=1), h * traj_bnd.roll(1, dims=1))
                else:
                    end_t = nxt_t = torch.tensor(0.0, device=h.device)
            loss_dict['boundary'] = (mse_end + mse_nxt + end_t + nxt_t) * 0.25
        else:
            loss_dict['boundary'] = torch.tensor(0.0, device=h.device)

        # 5b. BoundaryValidator loss (validates boundary quality)
        if bv is not None:
            with torch.no_grad():
                word_bnd = ((input_ids == 157) | (input_ids == 158)).float()
                sent_bnd = ((input_ids == 159) | (input_ids == 160)).float()
                bv_target_word = torch.stack([1.0 - word_bnd, word_bnd], dim=-1)
                bv_target_sent = torch.stack([1.0 - sent_bnd, sent_bnd], dim=-1)
            loss_dict['boundary_valid'] = (F.mse_loss(bv, bv_target_word) + F.mse_loss(bv, bv_target_sent)) * 0.5
        else:
            loss_dict['boundary_valid'] = torch.tensor(0.0, device=h.device)

        # 6. Residual
        if 'residual_error' in heads_out:
            loss_dict['residual'] = heads_out['residual_error'].mean()
        else:
            loss_dict['residual'] = torch.tensor(0.0, device=h.device)

        # 7. SimCSE
        if h2 is not None:
            loss_dict['contrastive'] = self.simcse(h, h2)
        else:
            loss_dict['contrastive'] = torch.tensor(0.0, device=h.device)

        return loss_dict


def create_batch(data, config: TrainingConfig, device):
    """batch без target_ids если drop_ce=True."""
    B, L = config.batch_size, config.seq_len
    total_len = len(data) - L - 1
    if total_len <= 0:
        return None
    starts = np.random.randint(0, total_len, size=B)
    batch_ids = np.stack([data[s:s+L+1] for s in starts])
    input_ids = torch.tensor(batch_ids[:, :L], dtype=torch.long, device=device)
    target_ids = torch.tensor(batch_ids[:, 1:L+1], dtype=torch.long, device=device)
    return {'input_ids': input_ids, 'target_ids': target_ids}


def train_epoch(model, data, config: TrainingConfig, optimizer, scheduler,
                loss_fn: MultiTaskLoss, device, ewc=None, start_step=0):
    """
    Training loop.
    Head-only mode (drop_ce=True):
      - без decoder forward
      - SRG + attention_entropy + head_consistency + cross_gen + self_distill
    """
    model.train()
    loss_keys = ['total', 'ce', 'concept', 'contradiction', 'uncertainty',
                 'boundary', 'boundary_valid', 'residual', 'contrastive', 'distill',
                 'srg', 'attn_entropy', 'head_consistency', 'cross_gen', 'self_distill', 'kca']
    total_losses = {k: 0.0 for k in loss_keys}
    n_batches = 0
    step = start_step
    t0 = time.time()

    data = np.array(data, dtype=np.int64)

    if config.use_weight_context:
        model.set_weight_context(True)
        model.update_weight_token()

    while True:
        batch = create_batch(data, config, device)
        if batch is None:
            break

        optimizer.zero_grad()
        use_w = config.use_weight_context

        # ---- Forward pass 1 ----
        model_out = model.forward(
            batch['input_ids'],
            return_scores=not config.drop_ce,  # без decoder если drop_ce
            return_heads=True,
            use_weight=use_w,
            capture_attn=True,
        )
        h, scores, weights, heads_out = model_out

        # ---- Forward pass 2 (SimCSE) ----
        h2 = None
        if config.w_contrastive > 0:
            model_out2 = model.forward(
                batch['input_ids'],
                return_scores=False,
                return_heads=True,
                use_weight=use_w,
            )
            h2 = model_out2[0]

        # ---- Intrinsic labels ----
        intrinsic_labels = {
            'contra': model._intrinsic_contra_labels(h, batch['input_ids']),
            'concept': model._intrinsic_concept_labels(h, batch['input_ids']),
        }

        # ---- Core losses (MultiTaskLoss) ----
        losses = loss_fn(model_out, batch, intrinsic_labels, h2)

        # ---- Head-only дополнительные потери ----
        # SRG: топологическая гладкость
        losses['srg'] = model.temporal_smoothness_loss(h)

        # Attention entropy: острота внимания
        losses['attn_entropy'] = model.attention_entropy()

        # Head consistency: concept ≈ 1 - contradiction
        losses['head_consistency'] = model.head_consistency_loss(heads_out)

        # Cross-generation contrastive (каждые cross_gen_every шагов)
        if (step > 0 and config.w_cross_gen > 0
                and step % config.cross_gen_every == 0):
            from eva.symbolic.char_vocab import CharacterVocab as CharVocab
            prompt_ids = batch['input_ids'][0, :8].tolist()
            cv = CharVocab()
            cg_loss = model.cross_gen_contrastive(
                [cv.SENT_OPEN_IDX] + prompt_ids, cv, max_new=16)
            losses['cross_gen'] = cg_loss

        # Self-distillation: thought loop → direct output
        if config.w_self_distill > 0 and heads_out is not None:
            try:
                losses['self_distill'] = model.self_distill_thought(heads_out, h)
            except Exception:
                losses['self_distill'] = torch.tensor(0.0, device=h.device)

        # ---- KCA auxiliary loss ----
        if config.w_kca > 0:
            losses['kca'] = model.kca_aux_loss(h, heads_out)
        else:
            losses['kca'] = torch.tensor(0.0, device=h.device)

        # ---- Distillation loss ----
        distill_val = model.distill_loss(h, heads_out)
        losses['distill'] = distill_val

        # ---- Weighted sum ----
        head_only_keys = {
            'srg': config.w_srg,
            'attn_entropy': config.w_attn_entropy,
            'head_consistency': config.w_head_consistency,
            'cross_gen': config.w_cross_gen,
            'self_distill': config.w_self_distill,
        }
        total = (config.w_concept * losses['concept'] +
                 config.w_contra * losses['contradiction'] +
                 config.w_uncertainty * losses['uncertainty'] +
                 config.w_boundary * losses['boundary'] +
                 config.w_boundary_valid * losses['boundary_valid'] +
                 config.w_residual * losses['residual'] +
                 config.w_contrastive * losses['contrastive'] +
                 config.w_distill * distill_val +
                 config.w_kca * losses['kca'] +
                 config.w_ce * losses['ce'])

        for k, w in head_only_keys.items():
            total = total + w * losses.get(k, torch.tensor(0.0, device=h.device))

        # EWC
        if ewc is not None:
            total = total + config.w_ewc * ewc.ewc_loss()

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        if use_w and step % config.update_weight_every == 0:
            model.update_weight_token()

        for k in total_losses:
            total_losses[k] += losses.get(k, torch.tensor(0.0, device=h.device)).item()
        total_losses['total'] += total.item()
        n_batches += 1

        if step % config.log_every == 0 and step > start_step:
            avg = {k: v / n_batches for k, v in total_losses.items()}
            elapsed = time.time() - t0
            mode = 'HEAD' if config.drop_ce else 'HYBRID'
            print(f'[{mode} Step {step}] tot={avg["total"]:.4f} '
                  f'conc={avg["concept"]:.4f} ctr={avg["contradiction"]:.4f} '
                  f'srg={avg["srg"]:.4f} attn={avg["attn_entropy"]:.4f} '
                  f'cons={avg["head_consistency"]:.4f} '
                  f'res={avg["residual"]:.4f} bnd={avg["boundary"]:.4f} '
                  f'dist={avg["distill"]:.4f} '
                  f'cg={avg["cross_gen"]:.4f} sd={avg["self_distill"]:.4f} '
                  f'ce={avg["ce"]:.4f} | {elapsed:.0f}s')

        if step % config.save_every == 0 and step > start_step:
            save_checkpoint(model, optimizer, step, config)

        step += 1

    return step


def save_checkpoint(model, optimizer, step, config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f'train_v3_step_{step}.pt')
    tmp_path = path + '.tmp'
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, tmp_path)
    os.replace(tmp_path, path)
    print(f'[Save] Checkpoint saved to {path}')


def safe_load_state_dict(model, state_dict, log_prefix='[Load]'):
    """
    Загружает state_dict в model, конвертируя size mismatch.
    
    Специальная обработка:
    - MetaWeighter 4→3: отбрасывает decoder weight (index 0), сохраняет know/conc/contr
    - Все остальные size mismatch: пропускаются с предупреждением
    
    Returns: (missing_keys, skipped_or_converted_keys)
    """
    model_state = model.state_dict()
    converted = []
    skipped = []
    filtered = {}
    
    for key, val in state_dict.items():
        if key not in model_state:
            filtered[key] = val
            continue
        
        if val.shape == model_state[key].shape:
            filtered[key] = val
            continue
        
        # --- Special conversions ---
        if key == 'meta_weighter._bias' and val.shape == (4,) and model_state[key].shape == (3,):
            filtered[key] = val[1:]  # drop decoder index 0, keep [know, conc, contr]
            converted.append((key, '(4->3, dropped decoder)'))
            continue
        
        if key == 'meta_weighter.weight_net.weight' and val.shape == (4, 64) and model_state[key].shape == (3, 64):
            filtered[key] = val[1:]  # drop decoder row
            converted.append((key, '(4x64->3x64, dropped decoder row)'))
            continue
        
        if key == 'meta_weighter.weight_net.bias' and val.shape == (4,) and model_state[key].shape == (3,):
            filtered[key] = val[1:]  # drop decoder bias
            converted.append((key, '(4->3, dropped decoder bias)'))
            continue
        
        skipped.append((key, val.shape, model_state[key].shape))
    
    if converted:
        for key, note in converted:
            print(f'{log_prefix} Converted {key} {note}')
    if skipped:
        for key, old_s, new_s in skipped:
            print(f'{log_prefix} Skipping {key}: checkpoint {old_s} != model {new_s}')
    
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        still_missing = [k for k in missing if k not in [c[0] for c in converted]]
        if still_missing:
            print(f'{log_prefix} Missing keys ({len(still_missing)}): {still_missing[:3]}...')
    if unexpected:
        print(f'{log_prefix} Unexpected keys ({len(unexpected)}): {unexpected[:3]}...')
    return missing, [k for k, _, _ in skipped]


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    safe_load_state_dict(model, ckpt['model_state'])
    if 'optimizer_state' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except Exception as e:
            print(f'[Load] Warning: optimizer state not restored ({e})')
    print(f'[Load] Resumed from {path} (step {ckpt.get("step", "?")})')
    return ckpt.get('step', 0)


def continuous_learning_step(model, token_sequence, optimizer, h2k_writer,
                              config: TrainingConfig, device):
    model.train()
    optimizer.zero_grad()
    inp = torch.tensor([token_sequence], dtype=torch.long, device=device)
    model_out = model.forward(inp, return_scores=False, return_heads=True,
                               capture_attn=True)
    h, _, weights, heads_out = model_out

    intrinsic_labels = {
        'contra': model._intrinsic_contra_labels(h, inp),
        'concept': model._intrinsic_concept_labels(h, inp),
    }
    batch = {'input_ids': inp, 'target_ids': inp.clone()}
    loss_fn = MultiTaskLoss(config)
    losses = loss_fn(model_out, batch, intrinsic_labels)

    # Head-only losses
    losses['srg'] = model.temporal_smoothness_loss(h)
    losses['attn_entropy'] = model.attention_entropy()
    losses['head_consistency'] = model.head_consistency_loss(heads_out)
    if config.w_self_distill > 0:
        try:
            losses['self_distill'] = model.self_distill_thought(heads_out, h)
        except Exception:
            losses['self_distill'] = torch.tensor(0.0, device=h.device)

    head_only_w = {'srg': config.w_srg, 'attn_entropy': config.w_attn_entropy,
                   'head_consistency': config.w_head_consistency,
                   'self_distill': config.w_self_distill}
    total = (config.w_concept * losses['concept'] +
             config.w_contra * losses['contradiction'] +
             config.w_uncertainty * losses['uncertainty'] +
             config.w_boundary * losses['boundary'] +
             config.w_residual * losses['residual'] +
             config.w_distill * model.distill_loss(h, heads_out))
    for k, w in head_only_w.items():
        total = total + w * losses.get(k, torch.tensor(0.0, device=h.device))
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
    optimizer.step()

    if h2k_writer is not None:
        from .h2k_pipeline import Hypothesis
        from .h2k_pipeline import HypothesisValidator
        validator = HypothesisValidator()
        hyp = Hypothesis(
            tokens=token_sequence,
            hidden_states=h[0].detach().cpu(),
            srg_val=validator.compute_srg(h[0]),
            concept_val=heads_out['concept'][0].mean().item(),
            contra_val=heads_out['contradiction'][0].mean().item(),
        )
        hyp.combined_score = validator.score(hyp)
        h2k_writer.write(hyp)

    return losses


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='real_data/full_corpus_encoded.npy')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/v3')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--drop-ce', action='store_true', default=True,
                        help='Head-only training (no next-token prediction)')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--h2k-path', type=str, default='h2k_trajectories.jsonl')
    parser.add_argument('--teacher', type=str, default=None)
    parser.add_argument('--teacher-dim', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    from .unified_transformer import UnifiedMultidimensionalTransformer
    model = UnifiedMultidimensionalTransformer().to(device)

    if args.teacher:
        print(f'[Teacher] Loading from {args.teacher}')
        teacher = torch.load(args.teacher, map_location=device, weights_only=False)
        model.set_teacher(teacher, args.teacher_dim)
        print(f'[Teacher] Distillation active (dim={args.teacher_dim})')

    config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        continuous_mode=args.continuous,
        drop_ce=args.drop_ce,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
    loss_fn = MultiTaskLoss(config)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, device)

    if args.continuous:
        from .h2k_pipeline import HypothesisWriter
        h2k_writer = HypothesisWriter(args.h2k_path)
        print('[Continuous] Ready.')
    else:
        print(f'[Train] Loading data from {args.data}')
        data = np.load(args.data)
        mode_str = 'HEAD' if config.drop_ce else 'HYBRID'
        print(f'[Train] Data shape: {data.shape}, mode={mode_str}')

        step = train_epoch(model, data, config, optimizer, scheduler, loss_fn,
                           device, start_step=start_step)
        save_checkpoint(model, optimizer, step, config)
        print(f'[Train] Done. Final step: {step}')
