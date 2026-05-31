"""
Phase 2: Train 384-dim / 12-layer on BPE boundary corpus.
20M params, 200K steps, B=8 L=64 for MX550 2.1GB.
Loss: CE + nxt + boundary (bpe_labels) + L_align + attractor.

Usage: python train_phase2.py [--resume checkpoints/v4/phase1_step_N.pt]
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.phase1_model import UnifiedMultidimensionalTransformerV2
from eva.symbolic.bpe_tokenizer import BPEVocab
from eva.symbolic.heads import BoundaryDetectionHead
from eva.symbolic.phase1_model import D_MODEL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ─── Config ───
DATA_IDS = 'real_data/full_corpus_bpe_boundary.npy'
DATA_LABELS = 'real_data/full_corpus_bpe_labels.npy'
CKPT_DIR = 'checkpoints/v4'
os.makedirs(CKPT_DIR, exist_ok=True)

N_STEPS = 200000
B, L = 8, 64
LR = 3e-4
WARMUP = 4000
LOG_EVERY = 100
SAVE_EVERY = 20000

W_CE = 1.0
W_NXT = 0.05
W_BOUNDARY = 0.1
W_ALIGN = 0.05
W_ATTRACTOR = 0.01
ATTRACTOR_WARMUP = 1000
W_HAF = 0.001
HAF_WARMUP = 1000
UPDATE_ATTRACTORS_EVERY = 10

VOCAB = 4101
# BPE boundary data: tokens 6-4095 (BPE) + 4097-4098 (WO/WC)
SPECIAL_IDS = {0, 1, 2, 3, 4096, 4099, 4100}  # PAD,UNK,BOS,EOS,GAP,SO,SC

# ─── Data ───
ids = np.load(DATA_IDS).astype(np.int64)
labels = np.load(DATA_LABELS).astype(np.int64)
assert len(ids) == len(labels), f'ids={len(ids)} != labels={len(labels)}'
N = len(ids)
print(f'Data: {DATA_IDS} — {N:,} tokens (vocab={int(ids.max())+1})')

# ─── Model ───
model = UnifiedMultidimensionalTransformerV2(vocab_size=VOCAB).to(device)
parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None)
args = parser.parse_args()
if args.resume:
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'], strict=False)
    print(f'Resumed from {args.resume} (step {ckpt.get("step","?")})')

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
print(f'\nPhase 2 training: {N_STEPS} steps...')

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

    # 3. Boundary loss — from pre-computed boundary labels
    boundary_logits = heads_out.get('boundary_detect')
    if boundary_logits is not None:
        boundary_mask = y_labels >= 0
        boundary_loss_val = F.cross_entropy(
            boundary_logits.reshape(-1, 3)[boundary_mask.reshape(-1)],
            y_labels.reshape(-1)[boundary_mask.reshape(-1)])
    else:
        boundary_loss_val = torch.tensor(0.0, device=device)

    # 4. L_align — boundary consistency across levels
    boundary_probs = boundary_logits.softmax(-1) if boundary_logits is not None else None
    if boundary_probs is not None and weights is not None:
        char_inside = boundary_probs[..., 1]  # [B, L]
        word_pooled = heads_out.get('boundary_end', h.mean(dim=-1, keepdim=True).expand(-1, -1, 384))
        word_avg = word_pooled.mean(dim=-1).sigmoid()  # [B, L]
        align_loss = F.mse_loss(char_inside.reshape(-1), word_avg.reshape(-1))
    else:
        align_loss = torch.tensor(0.0, device=device)

    # 5. AttractorField consistency + diversity loss
    af = model.attractor_field
    if step > ATTRACTOR_WARMUP and af.n_attractors > 0:
        valid = af.valid_mask[:af.n_attractors]
        if valid.any():
            centers = af.centers[:af.n_attractors][valid]
            z_flat = h.reshape(-1, D_MODEL)
            dists = torch.cdist(z_flat, centers)
            nearest = dists.argmin(dim=-1)
            nearest_center = centers[nearest]
            attractor_loss = F.mse_loss(z_flat, nearest_center.detach())

            valid_n = int(valid.sum().item())
            if af.n_attractors > 0 and valid_n > 1:
                c_norm = F.normalize(centers, dim=-1)
                cos_sim = c_norm @ c_norm.T
                mask = 1.0 - torch.eye(valid_n, device=device)
                diversity_loss = (cos_sim * mask).pow(2).mean()
            else:
                diversity_loss = torch.tensor(0.0, device=device)
        else:
            attractor_loss = torch.tensor(0.0, device=device)
            diversity_loss = torch.tensor(0.0, device=device)
    else:
        attractor_loss = torch.tensor(0.0, device=device)
        diversity_loss = torch.tensor(0.0, device=device)

    # 6. HAF (HierarchicalAdditiveField) loss
    haf = model.haf
    if step > HAF_WARMUP:
        z_pooled = h.mean(dim=(0, 1))  # [D] — global mean
        haf_loss_dict = haf.multi_path_loss(
            z_pooled, n_paths=2, w_cross=0.05, w_sparsity=0.005)
        haf_loss = haf_loss_dict['total']
        haf_K = heads_out.get('haf_K', 0)
        haf_res = heads_out.get('haf_residual', 0.0)
    else:
        haf_loss = torch.tensor(0.0, device=device)
        haf_K = 0
        haf_res = 0.0

    total = (W_CE * ce_loss + W_NXT * nxt_loss +
             W_BOUNDARY * boundary_loss_val + W_ALIGN * align_loss +
             W_ATTRACTOR * attractor_loss + W_ATTRACTOR * diversity_loss +
             W_HAF * haf_loss)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if ce_mask.any():
        acc = (logits.argmax(-1)[ce_mask] == targets[ce_mask]).float().mean().item()
    else:
        acc = 0.0

    # Boundary accuracy
    if boundary_logits is not None:
        bm = y_labels >= 0
        b_acc = (boundary_logits.argmax(-1)[bm] == y_labels[bm]).float().mean().item()
    else:
        b_acc = 0.0

    if step % LOG_EVERY == 0:
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        n_att = heads_out.get('attractor_n_attractors', 0)
        haf_n_att = heads_out.get('haf_n_attractors', 0)
        print(f'[PHASE2 {step}/{N_STEPS}] ce={ce_loss.item():.4f} '
              f'nxt={nxt_loss.item():.4f} bc={boundary_loss_val.item():.4f} '
              f'align={align_loss.item():.4f} ac={attractor_loss.item():.4f} '
              f'dv={diversity_loss.item():.4f} hf={haf_loss.item():.4f} '
              f'hk={haf_K} hr={haf_res:.3f} '
              f'acc={acc:.3f} b_acc={b_acc:.3f} '
              f'att={n_att} haf_att={haf_n_att} lr={lr_now:.2e} | '
              f'{elapsed:.0f}s ({elapsed/60:.1f}min)')

    if step > 0 and step % SAVE_EVERY == 0:
        out_path = f'{CKPT_DIR}/phase2_step_{step}.pt'
        tmp = out_path + '.tmp'
        torch.save({
            'step': step, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, tmp)
        os.replace(tmp, out_path)
        print(f'  Saved {out_path}')

elapsed = time.time() - t0
out_path = f'{CKPT_DIR}/phase2_final.pt'
tmp = out_path + '.tmp'
torch.save({
    'step': N_STEPS, 'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}, tmp)
os.replace(tmp, out_path)
print(f'\nDone. {N_STEPS} steps, {elapsed:.0f}s ({elapsed/60:.1f} min).')
print(f'Saved to {out_path}')
