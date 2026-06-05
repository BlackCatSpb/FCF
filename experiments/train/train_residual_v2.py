"""
train_residual_v2.py — Train ResidualTransformer WITH head prior conditioning.

Key change: transition CSR is converted to dense (67 MB) for fast batch index lookup.
  head_prior_logits[b, t] = transition_dense[prev_token_id]
  final_logits = transformer(x) + head_prior_logits
  Loss = CE(final_logits, targets)

The transformer learns the RESIDUAL error of the transition head.
Now trained on ALL available sentences (not just 100K), for 10+ epochs.
"""
import sys, os, time, math, pickle
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

SOURCES = {
    'wp':        os.path.join(META_DIR, 'hierarchical', 'sentences.npz'),
    'wikipedia': os.path.join(META_DIR, 'wikipedia', 'sentences.npz'),
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
SEQ_LEN = 64
LR = 3e-4
N_EPOCHS = 10
N_LAYERS = 3
D_MODEL = 128
N_HEADS = 4
MAX_SENTENCES = None  # ALL sentences

print("=" * 60)
print("RESIDUAL TRANSFORMER V2 — WITH HEAD PRIOR CONDITIONING")
print("=" * 60)
print("Device:", DEVICE)
print("Model: %d layers, d=%d, h=%d" % (N_LAYERS, D_MODEL, N_HEADS))
print("Seq len:", SEQ_LEN, "Batch:", BATCH_SIZE)
print("Max sentences:", MAX_SENTENCES or "ALL")
print()

# ─── Pre-compute dense transition matrix for fast batch lookup ───
print("Loading transition CSR...")
csr = load_npz(CSR_FILE)
V = csr.shape[0]
print("  CSR shape: %s, nnz=%d (%.2f%% dense)" % (
    str(csr.shape), csr.nnz, csr.nnz / (V * V) * 100))

# Convert to dense for fast batch indexing: [V, V] float32
# Preserve all 40,970 real CSR entries (incl. 279 values == 0.0 = deterministic transitions)
# Fill non-stored entries with fallback=-10.0 (matching HeadsEnsemble._lookup_csr_subword)
print("  Converting to dense (67 MB)...")
transition_dense = np.full((V, V), -10.0, dtype=np.float32)
coo = csr.tocoo()
for i, j, v in zip(coo.row, coo.col, coo.data):
    transition_dense[i, j] = float(v)
print("  Dense shape:", transition_dense.shape)
print("  Values: min=%.2f, max=%.2f, fallback=%.1f" % (
    transition_dense.min(), transition_dense.max(), -10.0))
del csr, coo

# ─── Dataset ───
class SentenceDataset(Dataset):
    def __init__(self, sources, seq_len=64, max_sentences=None):
        self.seq_len = seq_len
        self.samples = []
        total_sents = 0
        for name, path in sources.items():
            print("  Loading %s..." % name)
            s = np.load(path)
            tf = s['tokens']
            tl = s['token_lens']
            n = len(tl)
            cursor = 0
            for si in range(n):
                nt = int(tl[si])
                if nt > seq_len + 5:
                    cursor += nt
                    continue
                sent = tf[cursor:cursor + nt].tolist()
                cursor += nt
                self.samples.append(sent)
                total_sents += 1
                if max_sentences and total_sents >= max_sentences:
                    break
            if max_sentences and total_sents >= max_sentences:
                break
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

print("Loading dataset (ALL sentences)...")
dataset = SentenceDataset(SOURCES, seq_len=SEQ_LEN, max_sentences=MAX_SENTENCES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print()

# ─── Model ───
vocab_size = V
model = ResidualTransformer(
    vocab_size=vocab_size, d_model=D_MODEL,
    n_layers=N_LAYERS, n_heads=N_HEADS, ff_dim=D_MODEL * 4,
    max_len=SEQ_LEN,
).to(DEVICE)
print("Model params: %.2fM" % (model.get_num_params() / 1e6))

optimizer = model.configure_optimizers(lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=len(dataloader) * N_EPOCHS)

# ─── Transition prior helper ───
# Convert to torch tensor on device for fast indexing
transition_t = torch.from_numpy(transition_dense).to(DEVICE)  # [V, V]

def compute_head_prior(x):
    """x: [B, T] — compute transition-based head prior for each position.
    head_prior_logits[b, t] = transition_dense[x[b, t]]  (the transition FROM prev token)
    """
    return transition_t[x]  # [B, T, V]

# ─── Training ───
n_batches = len(dataloader)
print("Batches per epoch:", n_batches)
print()

best_loss = float('inf')
for epoch in range(N_EPOCHS):
    t0 = time.time()
    total_loss = 0.0
    n_correct = 0
    n_total = 0

    model.train()
    for bi, batch in enumerate(dataloader):
        batch = batch.to(DEVICE)  # [B, T]

        # Input: all tokens except last; Target: all tokens except first
        x = batch[:, :-1]       # [B, T-1]
        targets = batch[:, 1:]  # [B, T-1]

        # ─── Head prior: transition from previous token ───
        with torch.no_grad():
            head_prior = compute_head_prior(x)  # [B, T-1, V]

        optimizer.zero_grad()
        # Forward: transformer logits + head_prior = final logits
        final_logits = model(x, head_prior_logits=head_prior)  # [B, T-1, V]

        loss = F.cross_entropy(
            final_logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=0,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Accuracy (on non-pad positions)
        preds = final_logits.argmax(dim=-1)
        mask = targets != 0
        n_correct += ((preds == targets) & mask).sum().item()
        n_total += mask.sum().item()

        if (bi + 1) % 500 == 0:
            avg_loss = total_loss / (bi + 1)
            acc = n_correct / max(n_total, 1) * 100
            lr_now = scheduler.get_last_lr()[0]
            print("  epoch %d, batch %d/%d: loss=%.4f, acc=%.2f%%, lr=%.2e" % (
                epoch + 1, bi + 1, n_batches, avg_loss, acc, lr_now))

    avg_loss = total_loss / n_batches
    acc = n_correct / max(n_total, 1) * 100
    elapsed = time.time() - t0
    print("Epoch %d done: loss=%.4f, acc=%.2f%%, time=%.0fs" % (
        epoch + 1, avg_loss, acc, elapsed))

    # Save checkpoint
    ckpt = os.path.join(OUT_DIR, 'residual_transformer_v2_epoch%d.pt' % (epoch + 1))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'acc': acc,
        'config': {
            'vocab_size': vocab_size, 'd_model': D_MODEL,
            'n_layers': N_LAYERS, 'n_heads': N_HEADS,
            'seq_len': SEQ_LEN, 'with_head_prior': True,
        },
    }, ckpt)
    print("  Saved:", ckpt)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(OUT_DIR, 'residual_transformer_v2_best.pt'))
        print("  New best loss! Saved.")

print("\nTraining complete! Best loss: %.4f" % best_loss)
