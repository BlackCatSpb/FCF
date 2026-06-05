"""
train_residual.py — Train ResidualTransformer on real data (WP + Wikipedia).
Loss = cross-entropy(residual_logits + head_prior_logits, target_tokens).

Key idea: transformer learns residuals — corrections to head priors.
If heads already give P=0.8 to correct token, transformer only needs +0.2.
"""
import sys, os, time, math, pickle
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from coordinate_packer import CoordinatePacker
from eva.symbolic.residual_transformer import ResidualTransformer
from eva.symbolic.heads import HeadsEnsemble, id_level, id_value, LEVEL_SUBWORD

cp = CoordinatePacker()

# ─── Config ───
META = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\heads_meta_hierarchical.pkl'
OUT_DIR = r'C:\Users\black\OneDrive\Desktop\FCF\models'
os.makedirs(OUT_DIR, exist_ok=True)

SOURCES = {
    'wp':        r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\hierarchical\sentences.npz',
    'wikipedia': r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\wikipedia\sentences.npz',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
SEQ_LEN = 64
LR = 3e-4
N_EPOCHS = 3
N_LAYERS = 3
D_MODEL = 128
N_HEADS = 4
MAX_SENTENCES = 100000  # 100K sentences (~30 min on GPU)

print("="*60)
print("RESIDUAL TRANSFORMER TRAINING")
print("="*60)
print("Device:", DEVICE)
print("Model: %d layers, d=%d, h=%d" % (N_LAYERS, D_MODEL, N_HEADS))
print("Seq len:", SEQ_LEN, "Batch:", BATCH_SIZE)
print()

# ─── Load heads for scoring ───
print("Loading heads...")
heads = HeadsEnsemble(META, csr_path=os.path.dirname(META))
vocab_size = heads.V

# ─── Dataset ───
class SentenceDataset(Dataset):
    def __init__(self, sources, seq_len=64, max_sentences=None):
        self.seq_len = seq_len
        self.samples = []  # list of token sequences (as flat lists)
        
        total_sents = 0
        for name, path in sources.items():
            s = np.load(path)
            tf = s['tokens']
            tl = s['token_lens']
            n = len(tl)
            
            cursor = 0
            for si in range(n):
                nt = int(tl[si])
                if nt > seq_len + 5:  # too long, skip
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
        
        print("  Loaded %d sentences" % len(self.samples))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sent = self.samples[idx]
        # Truncate or pad to seq_len
        if len(sent) > self.seq_len:
            sent = sent[:self.seq_len]
        # Pad with PAD=0
        if len(sent) < self.seq_len:
            sent = sent + [0] * (self.seq_len - len(sent))
        x = torch.tensor(sent, dtype=torch.long)
        return x

print("Loading dataset...")
dataset = SentenceDataset(SOURCES, seq_len=SEQ_LEN, max_sentences=MAX_SENTENCES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ─── Model ───
model = ResidualTransformer(
    vocab_size=vocab_size, d_model=D_MODEL,
    n_layers=N_LAYERS, n_heads=N_HEADS, ff_dim=D_MODEL * 4,
    max_len=SEQ_LEN,
).to(DEVICE)
print("Model params: %.2fM" % (model.get_num_params() / 1e6))
print()

optimizer = model.configure_optimizers(lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * N_EPOCHS)

# ─── Training ───
n_batches = len(dataloader)
print("Batches per epoch:", n_batches)
print()

for epoch in range(N_EPOCHS):
    t0 = time.time()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    
    model.train()
    for bi, batch in enumerate(dataloader):
        batch = batch.to(DEVICE)  # [B, T]
        B, T = batch.shape
        
        # Input: all tokens except last; Target: all tokens except first
        x = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Compute head priors for each position
        # (In practice, we'd cache these or compute on-the-fly)
        # For now, skip head priors and let transformer learn from scratch
        # head_prior_logits = None
        
        optimizer.zero_grad()
        
        residual_logits = model(x)  # [B, T-1, V]
        
        loss = F.cross_entropy(
            residual_logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=0,
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Accuracy
        preds = residual_logits.argmax(dim=-1)
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
    ckpt = os.path.join(OUT_DIR, 'residual_transformer_epoch%d.pt' % (epoch + 1))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'acc': acc,
        'config': {
            'vocab_size': vocab_size, 'd_model': D_MODEL,
            'n_layers': N_LAYERS, 'n_heads': N_HEADS,
            'seq_len': SEQ_LEN,
        },
    }, ckpt)
    print("  Saved:", ckpt)

print("\nTraining complete!")

