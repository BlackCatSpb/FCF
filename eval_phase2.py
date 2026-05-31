"""Eval Phase 2: generation + boundary accuracy + attractor analysis.
Usage: python eval_phase2.py checkpoints/v4/phase2_step_N.pt
"""
import torch, torch.nn.functional as F, numpy as np, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.phase1_model import UnifiedMultidimensionalTransformerV2
from eva.symbolic.bpe_tokenizer import BPEVocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cv = BPEVocab()
VOCAB = 4101

ckpt = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/v4/phase2_step_20000.pt'
print(f'Loading {ckpt}...')
state = torch.load(ckpt, map_location=device)
model = UnifiedMultidimensionalTransformerV2(vocab_size=VOCAB).to(device)
model.load_state_dict(state['model_state'])
model.eval()
step = state.get('step', '?')
print(f'Loaded step {step} — {sum(p.numel() for p in model.parameters()):,} params')

# ─── 1. Boundary accuracy on held-out labelled data ───
print('\n=== 1. Boundary Accuracy ===')
DATA = 'real_data/full_corpus_bpe_boundary.npy'
LABELS = 'real_data/full_corpus_bpe_labels.npy'
ids = np.load(DATA).astype(np.int64)
labels = np.load(LABELS).astype(np.int64)
N = len(ids)
B, L = 8, 64

rng = np.random.RandomState(42)
n_batches = 50
correct, total = 0, 0
all_bounds = []
wall_start = time.time()
for _ in range(n_batches):
    idx = rng.randint(0, N - L - 1, size=B)
    batch = np.stack([ids[i:i+L] for i in idx])
    batch_labels = np.stack([labels[i:i+L] for i in idx])
    x = torch.tensor(batch, dtype=torch.long, device=device)
    y = torch.tensor(batch_labels, dtype=torch.long, device=device)
    with torch.no_grad():
        h, scores, weights, heads = model.forward(x, return_heads=True)
    bd_logits = heads.get('boundary_detect')
    if bd_logits is not None:
        bd_pred = bd_logits.argmax(dim=-1)
        valid = y >= 0
        correct += (bd_pred[valid] == y[valid]).sum().item()
        total += valid.sum().item()
        pred_probs = F.softmax(bd_logits, dim=-1)
        all_bounds.append(pred_probs[..., 1].cpu().numpy())

b_acc = correct / total if total > 0 else 0.0
print(f'  Boundary accuracy: {b_acc:.4f} ({correct}/{total} tokens)')

# ─── 2. Token prediction accuracy on held-out BPE data ───
print('\n=== 2. Token Prediction Accuracy ===')
correct, total = 0, 0
SPECIAL_IDS = {0, 1, 2, 3, 4096, 4099, 4100}
for _ in range(n_batches):
    idx = rng.randint(0, N - L - 2, size=B)
    batch = np.stack([ids[i:i+L] for i in idx])
    targets = np.stack([ids[i+1:i+L+1] for i in idx])
    x = torch.tensor(batch, dtype=torch.long, device=device)
    y = torch.tensor(targets, dtype=torch.long, device=device)
    with torch.no_grad():
        h, logits, weights, heads = model.forward(x, return_scores=True, return_heads=True)
    special_t = torch.tensor(list(SPECIAL_IDS), device=device)
    mask = ~torch.isin(y, special_t)
    if mask.any():
        pred = logits.argmax(dim=-1)
        correct += (pred[mask] == y[mask]).sum().item()
        total += mask.sum().item()
tok_acc = correct / total if total > 0 else 0.0
print(f'  Token accuracy: {tok_acc:.4f} ({correct}/{total} tokens)')

# ─── 3. Attractor state ───
print('\n=== 3. Attractor Field State ===')
af = model.attractor_field
n_att = af.n_attractors
print(f'  Attractors: {n_att}/{af.max_attractors}')
if n_att > 1:
    valid = af.valid_mask[:n_att]
    centers = af.centers[:n_att][valid]
    counts = af.counts[:n_att][valid]
    print(f'  Count range: {counts.min().item():.1f}..{counts.max().item():.1f}')
    print(f'  Count mean: {counts.mean().item():.1f}')
    print(f'  Dead (count<0.1): {(counts < 0.1).sum().item()}')
    pairwise = torch.cdist(centers, centers, p=2)
    triu = torch.triu(pairwise, diagonal=1)
    n_pairs = (triu > 0).sum().item()
    if n_pairs > 0:
        print(f'  Mean inter-attractor dist: {triu[triu>0].mean().item():.4f}')
        print(f'  Min inter-attractor dist: {triu[triu>0].min().item():.4f}')

# ─── 4. Generation (standard mode) ───
print('\n=== 4. Generation (standard) ===')
prompts = [
    cv.encode('Литва'),
    cv.encode('Россия'),
    cv.encode('Квантовая'),
]
def generate(model, prompt_ids, max_new=64, temp=0.8, use_attractors=False):
    model.eval()
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids[-64:]], dtype=torch.long, device=device)
            h, _, _, heads_out = model.forward(inp, return_heads=True, capture_attn=True)
            z_curr = h[0, -1]
            if use_attractors and model.attractor_field.n_attractors > 0:
                nxt_dir = model.attractor_field.nxt_direction(z_curr.unsqueeze(0))[0]
                z_pred = z_curr + nxt_dir
            else:
                end, nxt, conn = model.boundary_predictor(h[:, -1:])
                z_pred = z_curr + nxt[0, 0]
            logits_know = model.decoder(z_pred.unsqueeze(0).unsqueeze(0))[0, 0]
            sym_coords = model.embed.weight
            dists = -torch.cdist(z_pred.unsqueeze(0), sym_coords, p=2).squeeze(0)
            meta_w = model.meta_weighter(h.mean(dim=1))[0]
            concept_score = heads_out['concept'][0, -1].item()
            contra_score = heads_out['contradiction'][0, -1].item()
            logits_conc = dists * (1.0 + concept_score)
            logits_contr = dists * (1.0 - contra_score * 0.5)
            final = (meta_w[0]*logits_know + meta_w[1]*logits_conc + meta_w[2]*logits_contr) / temp
            final[:4] = -float('inf')
            for special_idx in [157, 158, 159, 160, cv.GAP_FILLER_IDX]:
                if special_idx < len(final):
                    final[special_idx] = -float('inf')
            freq_counts = {t: ids.count(t) for t in set(ids)}
            for t, c in freq_counts.items():
                if t < len(final):
                    final[t] -= c * 0.5
            sv, si = final.sort(descending=True)
            top_k = min(20, len(sv))
            p = F.softmax(sv[:top_k], dim=-1)
            nt = si[:top_k][torch.multinomial(p, 1)].item()
            ids.append(nt)
            if nt in {cv.EOS_IDX, cv.SENT_CLOSE_IDX}:
                break
    return cv.decode(ids), {}
for p in prompts:
    text, _ = generate(model, p, max_new=64, use_attractors=False)
    print(f'  {text[:150]}')

# ─── 5. Attention analysis ───
print('\n=== 5. Attention Pattern Analysis ===')
idx = rng.randint(0, N - L - 1, size=B)
batch = np.stack([ids[i:i+L] for i in idx])
x = torch.tensor(batch, dtype=torch.long, device=device)
with torch.no_grad():
    h, scores, weights, heads = model.forward(x, return_heads=True, capture_attn=True)
attn_mats = model._cached_attention
if attn_mats:
    print(f'  Captured {len(attn_mats)} attention matrices (last layer, 24 heads)')
    avg_attn = torch.stack([a.mean(dim=0) for a in attn_mats]).mean(dim=0)
    diagonal = avg_attn.diag().mean().item()
    off_diag = (avg_attn - torch.diag(avg_attn.diag())).mean().item()
    print(f'  Mean diagonal weight: {diagonal:.4f}')
    print(f'  Mean off-diagonal weight: {off_diag:.4f}')

# ─── 6. Dimension utilization ───
print('\n=== 6. Dimension Analysis ===')
h_np = h.cpu().numpy()  # (B, L, 384)
dim_var = h_np.var(axis=(0, 1))
mean_var = dim_var.mean()
active = (dim_var > mean_var * 0.5).sum()
print(f'  Active dims (var>0.5*mean): {active}/384')
print(f'  Var range: {dim_var.min():.4f}..{dim_var.max():.4f}')
print(f'  Mean var: {mean_var:.4f}')

wall_elapsed = time.time() - wall_start
print(f'\nEval done in {wall_elapsed:.0f}s')
