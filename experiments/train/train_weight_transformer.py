"""
train_weight_transformer.py — train WeightTransformer to optimize head weights.

Training signal: maximize likelihood of actual next token by learning to weight 6 heads.
Heads are frozen (parameterless data lookups). Only ~13K transformer params learned.

Usage:
    python train_weight_transformer.py [--epochs 200] [--samples-per-epoch 10000]
"""
import sys, os, json, time, math, random, argparse
import numpy as np

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.weight_transformer import WeightTransformer, count_params

# ─── Paths ───
HIER = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\hierarchical'
META = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\heads_meta.pkl'
SAVEDIR = r'C:\Users\black\OneDrive\Desktop\FCF\models'

# ─── Config ───
V = 4101
NORM = {  # max values for normalization
    'word_len': 19,
    'pos_in_word': 18,
    'word_num': 275,
    'pos_in_sent': 587,
    'sent_len': 587,
}


class TrajectoryDataset:
    """Stream trajectory positions on the fly. No precomputed head scores."""

    def __init__(self, hier_path: str, heads: HeadsEnsemble):
        # Load sentence data
        sent_data = np.load(os.path.join(hier_path, 'sentences.npz'))
        self.sent_tokens = sent_data['tokens']
        self.sent_lens = sent_data['token_lens']
        self.sent_word_counts = sent_data['word_counts']
        self.sent_word_spans = sent_data['word_spans']

        # Build sentence list
        self.sentences = []
        ptr = 0
        wptr = 0
        for i in range(len(self.sent_lens)):
            L = int(self.sent_lens[i])
            nw = int(self.sent_word_counts[i])
            tokens = list(self.sent_tokens[ptr:ptr + L])
            spans = []
            for j in range(nw):
                s = int(self.sent_word_spans[wptr + 2 * j])
                e = int(self.sent_word_spans[wptr + 2 * j + 1])
                spans.append((s, e))
            self.sentences.append({'tokens': tokens, 'word_spans': spans, 'length': L})
            ptr += L
            wptr += 2 * nw

        # Compute max sent_len from data
        max_sl = max(s['length'] for s in self.sentences) if self.sentences else 1
        NORM['pos_in_sent'] = max_sl - 1
        NORM['sent_len'] = max_sl

        self.heads = heads
        self.V = V

        # Stats
        print(f"  Dataset: {len(self.sentences)} sentences, {sum(s['length'] for s in self.sentences)} tokens")
        print(f"  Max sent_len: {NORM['pos_in_sent']+1}")

    def sample_positions(self, n: int, rng=None):
        """Sample n positions from random sentences. Returns (contexts, next_tokens)."""
        if rng is None:
            rng = random.Random()

        contexts = []
        next_tokens = []

        while len(contexts) < n:
            sent = rng.choice(self.sentences)
            if sent['length'] < 2:
                continue
            # Pick a random token position (not the last one, since we need next)
            pos = rng.randint(0, sent['length'] - 2)
            tok_id = sent['tokens'][pos]
            next_id = sent['tokens'][pos + 1]

            # Find word containing this token
            word_idx = None
            for wi, (ws, we) in enumerate(sent['word_spans']):
                if ws <= pos <= we:
                    word_idx = wi
                    piw = pos - ws
                    wl = we - ws + 1
                    break

            if word_idx is None:
                continue

            wn = word_idx
            pis = pos
            sl = sent['length']
            flags = 0
            if piw == 0:
                flags |= 1  # word_start
            if piw == wl - 1:
                flags |= 2  # word_end

            # Previous token (or SENT_OPEN if first)
            prev_tok = sent['tokens'][pos - 1] if pos > 0 else 0

            # Context tokens (up to 3 previous)
            ctx_toks = sent['tokens'][max(0, pos - 3):pos]

            ctx = {
                'token_id': tok_id,
                'pos_in_word': piw,
                'word_len': wl,
                'word_num': wn,
                'pos_in_sent': pis,
                'sent_len': sl,
                'prev_token_id': prev_tok,
                'flags': flags,
                'context_tokens': ctx_toks,
            }
            contexts.append(ctx)
            next_tokens.append(next_id)

        return contexts, next_tokens

    def context_to_tensor(self, ctx: dict) -> dict:
        """Convert context to normalized tensor features for transformer."""
        return {
            'prev_token_id': torch.tensor([ctx['prev_token_id']], dtype=torch.long),
            'word_len': torch.tensor([ctx['word_len'] / NORM['word_len']], dtype=torch.float32),
            'pos_in_word': torch.tensor([ctx['pos_in_word'] / NORM['pos_in_word']], dtype=torch.float32) if NORM['pos_in_word'] > 0 else torch.zeros(1),
            'word_num': torch.tensor([ctx['word_num'] / NORM['word_num']], dtype=torch.float32) if NORM['word_num'] > 0 else torch.zeros(1),
            'pos_in_sent': torch.tensor([ctx['pos_in_sent'] / NORM['pos_in_sent']], dtype=torch.float32) if NORM['pos_in_sent'] > 0 else torch.zeros(1),
            'sent_len': torch.tensor([ctx['sent_len'] / NORM['sent_len']], dtype=torch.float32),
            'flags': torch.tensor([ctx['flags'] / 255.0], dtype=torch.float32),  # normalize flags
        }

    def batch_to_tensors(self, contexts, next_tokens):
        """Convert batch of contexts to transformer input tensors + targets."""
        batch_size = len(contexts)
        prev_ids = torch.zeros(batch_size, dtype=torch.long)
        wl = torch.zeros(batch_size, dtype=torch.float32)
        piw = torch.zeros(batch_size, dtype=torch.float32)
        wn = torch.zeros(batch_size, dtype=torch.float32)
        pis = torch.zeros(batch_size, dtype=torch.float32)
        sl = torch.zeros(batch_size, dtype=torch.float32)
        fl = torch.zeros(batch_size, dtype=torch.float32)
        targets = torch.zeros(batch_size, dtype=torch.long)

        for i, ctx in enumerate(contexts):
            prev_ids[i] = ctx['prev_token_id']
            wl[i] = ctx['word_len'] / NORM['word_len']
            piw[i] = ctx['pos_in_word'] / NORM['pos_in_word'] if NORM['pos_in_word'] > 0 else 0
            wn[i] = ctx['word_num'] / NORM['word_num'] if NORM['word_num'] > 0 else 0
            pis[i] = ctx['pos_in_sent'] / NORM['pos_in_sent'] if NORM['pos_in_sent'] > 0 else 0
            sl[i] = ctx['sent_len'] / NORM['sent_len']
            fl[i] = ctx['flags'] / 255.0
            targets[i] = next_tokens[i]

        return prev_ids, wl, piw, wn, pis, sl, fl, targets


def compute_batch_head_scores(heads: HeadsEnsemble, contexts: list) -> torch.Tensor:
    """Compute (batch, 6, V) head score tensor for a batch of contexts."""
    batch_size = len(contexts)
    scores = np.zeros((batch_size, 6, V), dtype=np.float32)
    for i, ctx in enumerate(contexts):
        scores[i] = heads.individual_scores(ctx)
    return torch.from_numpy(scores)


def evaluate(model, heads, dataset, n_samples=500):
    """Evaluate accuracy on held-out samples."""
    model.eval()
    ctxs, next_toks = dataset.sample_positions(n_samples)
    head_scores = compute_batch_head_scores(heads, ctxs)
    prev_ids, wl, piw, wn, pis, sl, fl, targets = dataset.batch_to_tensors(ctxs, next_toks)

    with torch.no_grad():
        weights = model(prev_ids, wl, piw, wn, pis, sl, fl)
        final = torch.einsum('bi,biv->bv', weights, head_scores)
        preds = final.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        loss = F.cross_entropy(final, targets).item()

    model.train()
    return acc, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--samples-per-epoch', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval-every', type=int, default=10)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"TRAIN WEIGHT TRANSFORMER")
    print(f"{'='*60}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Samples/epoch: {args.samples_per_epoch}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")

    os.makedirs(SAVEDIR, exist_ok=True)

    # ─── Load heads & data ───
    print("\n[1] Loading heads and data...")
    t0 = time.time()
    heads = HeadsEnsemble(META, HIER)
    dataset = TrajectoryDataset(HIER, heads)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ─── Create transformer ───
    print("\n[2] Creating WeightTransformer...")
    model = WeightTransformer(vocab_size=V)
    n_params = count_params(model)
    print(f"  Parameters: {n_params:,}")
    assert n_params < 60000, f"Too many params: {n_params}"
    print("  Architecture:")
    print(f"    token_embed: {V} → 32")
    print(f"    MLP: 40 → 128 → 64 → 6 (Softplus)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ─── Training loop ───
    print("\n[3] Training...")
    best_acc = 0.0
    train_t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches = max(1, args.samples_per_epoch // args.batch_size)
        epoch_t0 = time.time()

        for batch_idx in range(n_batches):
            # Sample positions
            ctxs, next_toks = dataset.sample_positions(args.batch_size)

            # Compute head scores (batch, 6, V)
            head_scores = compute_batch_head_scores(heads, ctxs)

            # Prepare transformer inputs
            prev_ids, wl, piw, wn, pis, sl, fl, targets = dataset.batch_to_tensors(ctxs, next_toks)

            # Forward
            weights = model(prev_ids, wl, piw, wn, pis, sl, fl)
            final = torch.einsum('bi,biv->bv', weights, head_scores)
            loss = F.cross_entropy(final, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Epoch metrics
        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_t0

        if epoch % args.eval_every == 0 or epoch == 1:
            val_acc, val_loss = evaluate(model, heads, dataset, n_samples=500)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(SAVEDIR, 'weight_transformer_best.pt'))
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  " +
                  f"time={epoch_time:.1f}s  best_acc={best_acc:.4f}")
        else:
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}  time={epoch_time:.1f}s")

    # ─── Final ───
    total_time = time.time() - train_t0
    print(f"\n[4] Done in {total_time:.1f}s")
    print(f"  Best val_acc: {best_acc:.4f}")
    print(f"  Model: {os.path.join(SAVEDIR, 'weight_transformer_best.pt')}")

    # Load best and final eval
    model.load_state_dict(torch.load(os.path.join(SAVEDIR, 'weight_transformer_best.pt')))
    final_acc, final_loss = evaluate(model, heads, dataset, n_samples=2000)
    print(f"  Final eval ({2000} samples): acc={final_acc:.4f}, loss={final_loss:.4f}")

    # Compare with rule-based weights
    print("\n[5] Comparing with rule-based weights...")
    ctxs, next_toks = dataset.sample_positions(1000)
    head_scores = compute_batch_head_scores(heads, ctxs)

    # Rule-based (from compute_weights)
    rule_acc = 0.0
    for i, ctx in enumerate(ctxs):
        w = heads.compute_weights(ctx)
        w_vec = np.array([w.get(k, 0) for k in ['morph', 'syntax', 'transition', 'semantic', 'concept', 'contra']],
                         dtype=np.float32)
        final = np.dot(w_vec, head_scores[i].numpy())
        pred = np.argmax(final)
        if pred == next_toks[i]:
            rule_acc += 1.0
    rule_acc /= 1000

    print(f"  Rule-based accuracy: {rule_acc:.4f}")
    print(f"  Learned accuracy:    {final_acc:.4f}")
    print(f"  Improvement:         {final_acc - rule_acc:+.4f}")


if __name__ == '__main__':
    main()
