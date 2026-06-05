"""
train_residual_v3.py — Train ResidualTransformer with 4-head ensemble prior.

Heads: transition (2.0), semantic (0.5), concept (0.2), contra (-0.5)
All computed as vectorized GPU operations — no Python per-position loops.
"""
import sys, os, time, pickle
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import load_npz

from eva.symbolic.residual_transformer import ResidualTransformer

# ─── Config ───
META_DIR = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5'
META = os.path.join(META_DIR, 'heads_meta_hierarchical.pkl')
CSR_FILE = os.path.join(META_DIR, 'hierarchical', 'log_prob_csr.npz')
OUT_DIR = r'C:\Users\black\OneDrive\Desktop\FCF\models'
os.makedirs(OUT_DIR, exist_ok=True)
PREFIX = 'residual_v3'  # checkpoint prefix

SOURCES = {
    'wp':  os.path.join(META_DIR, 'hierarchical', 'sentences.npz'),
    'wiki': os.path.join(META_DIR, 'wikipedia', 'sentences.npz'),
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
SEQ_LEN = 64
LR = 3e-4
N_EPOCHS = 20
N_LAYERS = 3
D_MODEL = 128
N_HEADS = 4
MAX_SENTENCES = None

# ─── Head weights (matching HeadsEnsemble defaults) ───
W_TRANS = 2.0
W_SEM = 0.5
W_CONCEPT = 0.2
W_CONTRA = -1.0  # matches original: scores -= w*penalty*2 with default w=0.5

print("="*60)
print("RESIDUAL TRANSFORMER V3 — 4-HEAD ENSEMBLE PRIOR")
print("="*60)
print("Device:", DEVICE)
print("Weights: trans=%.1f sem=%.1f concept=%.1f contra=%.1f" % (
    W_TRANS, W_SEM, W_CONCEPT, W_CONTRA))

# ─── Load head data and push to GPU as dense tensors ───
print("\nLoading heads...")
with open(META, 'rb') as f:
    meta = pickle.load(f)
V = meta.get('V', 4101)

# Transition CSR → dense
print("  Transition...")
csr = load_npz(CSR_FILE)
td = np.full((V, V), -10.0, dtype=np.float32)
coo = csr.tocoo()
for i, j, v in zip(coo.row, coo.col, coo.data):
    td[i, j] = float(v)
t_transition = torch.from_numpy(td).to(DEVICE)
del csr, coo, td

# Semantic similarity
print("  Semantic...")
sd = np.zeros((V, V), dtype=np.float32)
for tid, neighbors in meta.get('trans_sim_sparse', {}).items():
    for nid, sim in neighbors:
        sd[tid, nid] = sim
t_semantic = torch.from_numpy(sd).to(DEVICE)
del sd

# Contradiction penalty
print("  Contradiction...")
cd = np.zeros((V, V), dtype=np.float32)
for ta, tb, s in meta.get('contra_pairs', []):
    cd[ta, tb] = float(s); cd[tb, ta] = float(s)
t_contra = torch.from_numpy(cd).to(DEVICE)

# Concept scores
t_concept = torch.from_numpy(
    np.asarray(meta.get('concept_scores', np.ones(V) * 0.5), dtype=np.float32)
).to(DEVICE)

print("  GPU memory: %.0f MB" % (torch.cuda.memory_allocated() / 1024 / 1024))

# ─── Dataset (no word boundaries needed) ───
class SentenceDataset(Dataset):
    def __init__(self, sources, seq_len=64, max_sentences=None):
        self.seq_len = seq_len
        self.samples = []
        for name, path in sources.items():
            s = np.load(path)
            tf = s['tokens']; tl = s['token_lens']
            cursor = 0; loaded = 0
            for si in range(len(tl)):
                nt = int(tl[si])
                sent = tf[cursor:cursor+nt].tolist()
                cursor += nt
                # Truncate to seq_len (instead of skipping long sentences)
                if len(sent) > seq_len:
                    sent = sent[:seq_len]
                self.samples.append(sent)
                loaded += 1
                if max_sentences and loaded >= max_sentences:
                    break
            print("  %s: loaded %d sents" % (name, loaded))
        print("  Total: %d sentences" % len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sent = self.samples[idx]
        if len(sent) > self.seq_len:
            sent = sent[:self.seq_len]
        if len(sent) < self.seq_len:
            sent = sent + [0] * (self.seq_len - len(sent))
        return torch.tensor(sent, dtype=torch.long)

print("\nLoading dataset...")
dataset = SentenceDataset(SOURCES, seq_len=SEQ_LEN, max_sentences=MAX_SENTENCES)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ─── Model ───
model = ResidualTransformer(vocab_size=V, d_model=D_MODEL,
    n_layers=N_LAYERS, n_heads=N_HEADS, ff_dim=D_MODEL*4, max_len=SEQ_LEN).to(DEVICE)
print("Model: %.2fM params" % (model.get_num_params() / 1e6))
optimizer = model.configure_optimizers(lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=len(loader) * N_EPOCHS)

# ─── Vectorized head prior ───
def compute_head_prior(x):
    """
    x: [B, T] token IDs
    Returns: [B, T, V] = transition*2 + semantic*0.5 + concept*0.2 + contra*(-0.5)
    """
    B, T = x.shape
    prior = torch.zeros(B, T, V, device=DEVICE, dtype=torch.float32)

    # 1. Transition head (weight 2.0)
    prior += W_TRANS * t_transition[x]

    # 2. Concept head (constant)
    prior += W_CONCEPT * t_concept.unsqueeze(0).unsqueeze(0)

    # 3. Semantic head: sum over previous 1, 2, 3 tokens
    if W_SEM != 0.0:
        pad = torch.zeros(B, 1, dtype=torch.long, device=DEVICE)
        for ctx_src in [x, torch.cat([pad, x[:, :-1]], dim=1),
                        torch.cat([pad, pad, x[:, :-2]], dim=1)]:
            mask = (ctx_src != 0).unsqueeze(-1).float()
            prior += W_SEM * t_semantic[ctx_src] * mask

    # 4. Contradiction penalty: max penalty value over prev 1/2/3 tokens
    if W_CONTRA != 0.0:
        pad = torch.zeros(B, 1, dtype=torch.long, device=DEVICE)
        penalty_max = torch.zeros(B, T, V, device=DEVICE)
        for ctx_src in [x, torch.cat([pad, x[:, :-1]], dim=1),
                        torch.cat([pad, pad, x[:, :-2]], dim=1)]:
            mask = (ctx_src != 0).unsqueeze(-1).float()
            penalty_max = torch.maximum(penalty_max, t_contra[ctx_src] * mask)
        prior += W_CONTRA * penalty_max

    return prior

# ─── Training ───
n_batches = len(loader)
print("Batches per epoch:", n_batches)
best_loss = float('inf')
for epoch in range(N_EPOCHS):
    t0 = time.time()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    model.train()
    for bi, batch in enumerate(loader):
        batch = batch.to(DEVICE)
        x = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.no_grad():
            head_prior = compute_head_prior(x)

        optimizer.zero_grad()
        logits = model(x, head_prior_logits=head_prior)

        # Loss over all tokens. Boundary tokens (157-160) are part of the natural
        # token sequence and the model learns to predict them. The generation loop
        # handles boundary insertion/blocking at inference time.
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        mask = targets != 0
        n_correct += ((preds == targets) & mask).sum().item()
        n_total += mask.sum().item()

        if (bi + 1) % 500 == 0:
            print("  epoch %d, batch %d/%d: loss=%.4f, acc=%.2f%%, lr=%.2e" % (
                epoch + 1, bi + 1, n_batches,
                total_loss / (bi + 1), n_correct / max(n_total, 1) * 100,
                scheduler.get_last_lr()[0]))

    avg_loss = total_loss / n_batches
    acc = n_correct / max(n_total, 1) * 100
    elapsed = time.time() - t0
    print("Epoch %d: loss=%.4f, acc=%.2f%%, time=%.0fs" % (
        epoch + 1, avg_loss, acc, elapsed))

    ckpt = os.path.join(OUT_DIR, '%s_epoch%d.pt' % (PREFIX, epoch + 1))
    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss, 'acc': acc,
        'config': {'vocab_size': V, 'd_model': D_MODEL,
            'n_layers': N_LAYERS, 'n_heads': N_HEADS, 'seq_len': SEQ_LEN,
            'ff_dim': D_MODEL*4,
            'heads': 'trans+sem+concept+contra'},
    }, ckpt)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(OUT_DIR, '%s_best.pt' % PREFIX))

print("\nDone! Best loss: %.4f" % best_loss)
