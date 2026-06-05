"""
Check why round-trip fails: test if it's the decoder or the coordinates.
1. Next-token accuracy (what model was trained for)
2. Auto-encoding accuracy (what user wants)
3. Coordinate similarity analysis
"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

cv = CharacterVocab(); V = cv.vocab_size; D = 128

ut = UnifiedMultidimensionalTransformer(vocab_size=V, coord_dim=D, max_levels=8,
    total_heads=32, num_layers=6, d_ff=128).to(DEVICE)
ckpt = torch.load('checkpoints/symbolic/full_best.pt', map_location=DEVICE, weights_only=False)
sd = ckpt['ut']
for k in list(sd.keys()):
    if k in ut.state_dict() and sd[k].shape != ut.state_dict()[k].shape:
        del sd[k]
ut.load_state_dict(sd, strict=False)
ut.eval()
print(f'Loaded step {ckpt.get("step", "?")}')

# Load data
data = np.load('real_data/full_corpus_encoded.npy', mmap_mode='r').astype(np.int32)
SENT_CLOSE = cv.SENT_CLOSE_IDX
total = len(data)
print(f'Data: {total/1e6:.2f}M tokens')

# Test next-token prediction accuracy
B, ML = 8, 64
blocks = []
i = 0
while i < total:
    start = i
    while i < total and data[i] != SENT_CLOSE:
        i += 1
    if i < total:
        blocks.append(data[start:i+1].tolist())
        i += 1
    else:
        break

bt = torch.zeros(B, ML, dtype=torch.long, device=DEVICE)
mask = torch.ones(B, ML, device=DEVICE)
sent_ptr = 60000
rng = np.random.RandomState(seed=0)

for bi in range(B):
    ids_flat = []
    while len(ids_flat) < ML:
        ids_flat.extend(blocks[(sent_ptr + bi * 100) % len(blocks)])
        sent_ptr += 1
    ids_flat = ids_flat[:ML]
    sent_cut = 0; word_cut = 0
    for cut in range(len(ids_flat), max(len(ids_flat) - 50, 0), -1):
        if ids_flat[cut - 1] == cv.SENT_CLOSE_IDX and sent_cut == 0:
            sent_cut = cut
        if ids_flat[cut - 1] == cv.WORD_CLOSE_IDX and word_cut == 0:
            word_cut = cut
    if sent_cut > 0: ids_flat = ids_flat[:sent_cut]
    elif word_cut > 0: ids_flat = ids_flat[:word_cut]
    bt[bi, :len(ids_flat)] = torch.tensor(ids_flat, dtype=torch.long, device=DEVICE)

with torch.no_grad():
    h, scores = ut(bt, return_scores=True)

# 1. Next-token prediction: h[t] → token[t+1]
target = bt[:, 1:].clamp(1, V-1)
pred = scores[:, :-1]
tm = mask[:, 1:]
acc_next = ((pred.argmax(-1) == target) & tm.bool()).sum().item() / tm.sum().item()
print(f'\nNext-token accuracy: {acc_next*100:.1f}%')

# 2. Auto-encoding: h[t] → token[t]
target_ae = bt.clamp(1, V-1)
logits_ae = ut.decoder(h)
pred_ae = logits_ae.argmax(-1)
acc_ae = ((pred_ae == target_ae) & mask.bool()).sum().item() / mask.sum().item()
print(f'Auto-encode accuracy: {acc_ae*100:.1f}%')

# 3. Coordinate similarity analysis
coords = ut.embed.coordinates  # [V, D]
sim = torch.mm(coords, coords.T)  # cosine sim (already normalized)
# For each token, how many other tokens are within 0.1 cosine distance?
close_count = (sim > 0.9).sum(dim=-1).float()
print(f'\nCoordinate similarity:')
print(f'  Mean self-sim: {sim.diag().mean():.3f}')
print(f'  Mean pairwise sim: {(sim.sum() - sim.diag().sum()) / (V*(V-1)):.3f}')
print(f'  Tokens with >5 neighbors (cos>0.9): {(close_count > 5).sum().item()}/{V}')

# Show some nearest neighbors
print(f'\nNearest neighbors for a few tokens:')
test_tokens = [cv._char_to_idx.get(c, 1) for c in ['П', 'р', 'и', 'в', 'е', 'т', ' ', 'м']]
for t_id in test_tokens:
    if t_id < V:
        dists = torch.cdist(coords[t_id:t_id+1], coords, p=2)[0]
        nn = dists.argsort()[:4]
        nn_chars = [f'{cv.idx_to_char(n.item())!r}({n.item()})' for n in nn]
        print(f'  {cv.idx_to_char(t_id)!r}({t_id}): {nn_chars}')

# 4. Decoder weights analysis
print(f'\nDecoder linear weight norm: {ut.decoder.linear.weight.norm(dim=-1).mean():.3f}')
print(f'Decoder nn_weight: {ut.decoder.nn_weight.item():.3f}')
print(f'Decoder temperature: {ut.decoder.temperature.item():.3f}')

# 5. Test: does the decoder training loss make sense?
# The linear weight projects 128→V: this is a linear classifier for coordinates
# If coordinates are informative, the linear layer should work
# Let's check if linear projection alone does better
with torch.no_grad():
    lin_logits = ut.decoder.linear(h) / ut.decoder.temperature.clamp(min=0.1)
pred_lin = lin_logits.argmax(-1)
acc_lin = ((pred_lin == target_ae) & mask.bool()).sum().item() / mask.sum().item()
print(f'\nLinear-only auto-encode accuracy: {acc_lin*100:.1f}%')

# 6. Check: are the decoder output distributions sharp or flat?
with torch.no_grad():
    probs = torch.softmax(logits_ae[0, :5], dim=-1)
    top_p, top_i = probs.topk(5, dim=-1)
    print(f'\nDecoder output samples (first 5 tokens):')
    for pos in range(5):
        chars = [f'{cv.idx_to_char(i.item())!r}({p.item():.3f})' for p,i in zip(top_p[pos], top_i[pos])]
        print(f'  Pos {pos}: {chars}')

print('\nDone.')
