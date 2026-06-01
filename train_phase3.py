"""
Phase 3: Continue from Phase 2 — MetaWeighter training + SRG-validated attractors.
Loss: CE + nxt + boundary + align + attractor + meta_KL + flow_smooth.

Integrates:
  - MetaWeighter.kl_loss (heads.py:226)     — учит взвешивать know/conc/contr
  - SemanticRelevanceGate (potential_fields:386) — валидация качества траектории
  - Confidence-based attractor storage      — только хорошие треки → в память
  - Trajectory smoothness regularization    — штраф за резкие повороты nxt

Usage: python train_phase3.py [--resume checkpoints/v4/phase2_step_N.pt]
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, time, argparse, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.phase1_model import UnifiedMultidimensionalTransformerV2, D_MODEL
from eva.symbolic.bpe_tokenizer import BPEVocab
from eva.symbolic.potential_fields import SemanticRelevanceGate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ─── Paths ───
DATA_IDS = 'real_data/full_corpus_bpe_boundary.npy'
DATA_LABELS = 'real_data/full_corpus_bpe_labels.npy'
CKPT_DIR = 'checkpoints/v4'
os.makedirs(CKPT_DIR, exist_ok=True)

# ─── Config ───
N_STEPS = 200000
B, L = 8, 64
LR = 3e-4
WARMUP = 4000
LOG_EVERY = 200
SAVE_EVERY = 20000

# Loss weights — Phase 2 base
W_CE = 1.0
W_NXT = 0.05
W_BOUNDARY = 0.1
W_ALIGN = 0.05

# Loss weights — Phase 3 new
W_ATTRACTOR = 0.03       # ↑ с 0.01 — сильнее притяжение к центрам
W_DIVERSITY = 0.02       # отдельный вес diversity (был частью W_ATTRACTOR)
W_META = 0.01            # KL loss для MetaWeighter
W_FLOW = 0.001           # smoothness: MSE соседних nxt-векторов
W_HAF = 0.001            # HAF multi-path loss (Phase 2 compatibility)

# Attractor config
UPDATE_ATTRACTORS_EVERY = 10
ATTRACTOR_WARMUP = 1000
ATTRACTOR_CONFIDENCE_THRESHOLD = 0.3  # мин. confidence для сохранения трека
HAF_WARMUP = 1000

VOCAB = 4101
SPECIAL_IDS = {0, 1, 2, 3, 4096, 4099, 4100}

# ─── Data ───
ids = np.load(DATA_IDS).astype(np.int64)
labels = np.load(DATA_LABELS).astype(np.int64)
N = len(ids)
print(f'Data: {N:,} tokens')

# ─── Model ───
model = UnifiedMultidimensionalTransformerV2(vocab_size=VOCAB).to(device)
parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str,
                    default='checkpoints/v4/phase2_step_20000.pt')
args = parser.parse_args()

if args.resume and os.path.exists(args.resume):
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'], strict=False)
    print(f'Resumed from {args.resume} (step {ckpt.get("step","?")})')
    missing = [k for k in model.state_dict() if k not in ckpt['model_state']]
    extra = [k for k in ckpt['model_state'] if k not in model.state_dict()]
    if missing:
        print(f'  Missing keys: {len(missing)} (HAF/heads mismatch — expected)')
    if extra:
        print(f'  Extra keys: {len(extra)} (from different checkpoint — expected)')
else:
    print('No checkpoint, training from scratch')

total_params = sum(p.numel() for p in model.parameters())
model.train()
print(f'Model: {total_params:,} params')

# ─── Optimizer (fresh — Phase 3 restarts LR schedule) ───
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, total_iters=WARMUP)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS - WARMUP)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP])

# ─── SRG — quality gate for attractors ───
srg = SemanticRelevanceGate(w_sim=0.3, w_ent=0.7, w_eth=0.0)
cv = BPEVocab()

def batch_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Per-position confidence = 1 - normalized entropy.
    logits: [B, L, V] → confidence: [B, L] in [0, 1]."""
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    max_ent = math.log(logits.shape[-1])
    return 1.0 - entropy / max_ent

def target_from_confidence(conf: torch.Tensor) -> torch.Tensor:
    """MetaWeighter target: [B, 3] — softmax over [know_score, conc_score, contr_score].
    High confidence → know dominates. Low confidence → contr rises."""
    know = conf.sigmoid()                               # [B]
    contr = (1 - know) * 0.5                             # [B]
    conc = (1 - know) * 0.5                              # [B]
    target = torch.stack([know, conc, contr], dim=-1)   # [B, 3]
    return target / (target.sum(dim=-1, keepdim=True) + 1e-10)

log_file = f'{CKPT_DIR}/phase3.log'
print(f'\nPhase 3 training: {N_STEPS} steps (log -> {log_file})')
t0 = time.time()

for step in range(N_STEPS):
    idx = np.random.randint(0, N - L - 1, size=B)
    batch_ids = np.stack([ids[i:i+L] for i in idx])
    batch_labels = np.stack([labels[i:i+L] for i in idx])
    x = torch.tensor(batch_ids, dtype=torch.long, device=device)
    y_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    targets = torch.tensor(
        np.stack([ids[i+1:i+L+1] for i in idx]),
        dtype=torch.long, device=device)

    optimizer.zero_grad()

    # ── Forward (no auto attractor update — handled below) ──
    h, scores, weights, heads_out = model.forward(
        x, return_scores=True, return_heads=True, capture_attn=False,
        update_attractors=False)

    logits = scores
    special_t = torch.tensor(list(SPECIAL_IDS), device=device)
    ce_mask = ~torch.isin(targets, special_t)

    # 1. CE
    if ce_mask.any():
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1])[ce_mask.reshape(-1)],
            targets.reshape(-1)[ce_mask.reshape(-1)])
    else:
        ce_loss = torch.tensor(0.0, device=device)

    # 2. NXT
    nxt = heads_out.get('boundary_next')
    if nxt is not None and h.shape[1] > 1:
        delta = h[:, 1:] - h[:, :-1]
        nxt_loss = F.mse_loss(nxt[:, :-1], delta)
    else:
        nxt_loss = torch.tensor(0.0, device=device)

    # 3. Boundary
    boundary_logits = heads_out.get('boundary_detect')
    if boundary_logits is not None:
        boundary_mask = y_labels >= 0
        boundary_loss_val = F.cross_entropy(
            boundary_logits.reshape(-1, 3)[boundary_mask.reshape(-1)],
            y_labels.reshape(-1)[boundary_mask.reshape(-1)])
    else:
        boundary_loss_val = torch.tensor(0.0, device=device)

    # 4. L_align
    boundary_probs = boundary_logits.softmax(-1) if boundary_logits is not None else None
    if boundary_probs is not None and weights is not None:
        char_inside = boundary_probs[..., 1]
        word_pooled = heads_out.get('boundary_end', h.mean(dim=-1, keepdim=True).expand(-1, -1, D_MODEL))
        word_avg = word_pooled.mean(dim=-1).sigmoid()
        align_loss = F.mse_loss(char_inside.reshape(-1), word_avg.reshape(-1))
    else:
        align_loss = torch.tensor(0.0, device=device)

    # 5. AttractorField consistency + diversity
    af = model.attractor_field
    if step > ATTRACTOR_WARMUP and af.n_attractors > 0:
        valid = af.valid_mask[:af.n_attractors]
        if valid.any():
            centers = af.centers[:af.n_attractors][valid]
            z_flat = h.reshape(-1, D_MODEL)
            dists = torch.cdist(z_flat, centers)
            nearest = dists.argmin(dim=-1)
            attractor_loss = F.mse_loss(z_flat, centers[nearest].detach())

            valid_n = int(valid.sum().item())
            if valid_n > 1:
                c_norm = F.normalize(centers, dim=-1)
                cos_sim = c_norm @ c_norm.T
                mask = 1.0 - torch.eye(valid_n, device=device)
                # Mean cosine² — stronger diversity signal
                diversity_loss = (cos_sim * mask).pow(2).mean()
            else:
                diversity_loss = torch.tensor(0.0, device=device)
        else:
            attractor_loss = torch.tensor(0.0, device=device)
            diversity_loss = torch.tensor(0.0, device=device)
    else:
        attractor_loss = torch.tensor(0.0, device=device)
        diversity_loss = torch.tensor(0.0, device=device)

    # 6. MetaWeighter KL loss
    context = h.mean(dim=1)  # [B, D]
    meta_weights = model.meta_weighter(context)  # [B, 3]
    conf = batch_confidence(logits)  # [B, L]
    batch_conf = conf.mean(dim=1)    # [B]
    meta_target = target_from_confidence(batch_conf)
    meta_loss = model.meta_weighter.kl_loss(context, prior=meta_target)

    # 7. Trajectory smoothness regularization
    if nxt is not None and h.shape[1] > 2:
        # Penalise large changes in nxt direction between consecutive positions
        nxt_normed = F.normalize(nxt[:, :-1], dim=-1)  # [B, L-1, D]
        cos_nxt = (nxt_normed[:, :-1] * nxt_normed[:, 1:]).sum(dim=-1)  # [B, L-2]
        flow_loss = (1.0 - cos_nxt).mean()
    else:
        flow_loss = torch.tensor(0.0, device=device)

    # 8. HAF multi-path loss
    haf = model.haf
    if step > HAF_WARMUP:
        z_pooled = h.mean(dim=(0, 1))
        haf_loss_dict = haf.multi_path_loss(
            z_pooled, n_paths=2, w_cross=0.05, w_sparsity=0.005)
        haf_loss = haf_loss_dict['total']
        haf_K = heads_out.get('haf_K', 0)
        haf_res = heads_out.get('haf_residual', 0.0)
    else:
        haf_loss = torch.tensor(0.0, device=device)
        haf_K = 0
        haf_res = 0.0

    # ── Total loss ──
    total = (W_CE * ce_loss + W_NXT * nxt_loss +
             W_BOUNDARY * boundary_loss_val + W_ALIGN * align_loss +
             W_ATTRACTOR * attractor_loss + W_DIVERSITY * diversity_loss +
             W_META * meta_loss + W_FLOW * flow_loss +
             W_HAF * haf_loss)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    # ── Confidence-based attractor update ──
    if step > ATTRACTOR_WARMUP and step % UPDATE_ATTRACTORS_EVERY == 0:
        with torch.no_grad():
            pos_conf = batch_confidence(logits)  # [B, L]
            for b in range(B):
                for pos in range(1, L):
                    if pos_conf[b, pos].item() > ATTRACTOR_CONFIDENCE_THRESHOLD:
                        model.attractor_field.hebbian_update(h[b, pos-1], h[b, pos])

    # ── Metrics ──
    if ce_mask.any():
        acc = (logits.argmax(-1)[ce_mask] == targets[ce_mask]).float().mean().item()
    else:
        acc = 0.0
    if boundary_logits is not None:
        bm = y_labels >= 0
        b_acc = (boundary_logits.argmax(-1)[bm] == y_labels[bm]).float().mean().item()
    else:
        b_acc = 0.0

    if step % LOG_EVERY == 0:
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        n_att = model.attractor_field.n_attractors
        haf_n_att = model.haf.attractors.n_attractors
        steps_per_sec = (step + 1) / (elapsed + 1e-8)
        eta = (N_STEPS - step) / (steps_per_sec + 1e-8) / 3600
        mw = meta_weights[0].tolist()
        msg = (f'[PHASE3 {step}/{N_STEPS}] ce={ce_loss.item():.3f} '
               f'nxt={nxt_loss.item():.3f} bc={boundary_loss_val.item():.3f} '
               f'ac={attractor_loss.item():.3f} dv={diversity_loss.item():.3f} '
               f'meta={meta_loss.item():.4f} flow={flow_loss.item():.4f} '
               f'hf={haf_loss.item():.4f} hk={haf_K} hr={haf_res:.3f} '
               f'acc={acc:.3f} b_acc={b_acc:.3f} att={n_att} haf_att={haf_n_att} '
               f'mw=[{mw[0]:.2f},{mw[1]:.2f},{mw[2]:.2f}] '
               f'steps/s={steps_per_sec:.1f} ETA={eta:.1f}h'
               f' | {elapsed/60:.0f}min')
        print(msg)

    if step > 0 and step % SAVE_EVERY == 0:
        out_path = f'{CKPT_DIR}/phase3_step_{step}.pt'
        tmp = out_path + '.tmp'
        torch.save({
            'step': step, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, tmp)
        os.replace(tmp, out_path)
        print(f'  Saved {out_path}')

elapsed = time.time() - t0
out_path = f'{CKPT_DIR}/phase3_final.pt'
torch.save({
    'step': N_STEPS, 'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}, out_path)
print(f'\nDone. {N_STEPS} steps in {elapsed/60:.1f} min. -> {out_path}')
