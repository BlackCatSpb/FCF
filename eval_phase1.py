"""
Eval Phase 1 checkpoint: generation + topology analysis.
Usage: python eval_phase1.py [checkpoint_path]
"""
import torch, numpy as np, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.phase1_model import UnifiedMultidimensionalTransformerV2
from eva.symbolic.bpe_tokenizer import BPEVocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cv = BPEVocab()
VOCAB = 4101

ckpt = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/v4/phase1_step_20000.pt'
print(f'Loading {ckpt}...')
state = torch.load(ckpt, map_location=device)

model = UnifiedMultidimensionalTransformerV2(vocab_size=VOCAB).to(device)
model.load_state_dict(state['model_state'])
model.eval()
print(f'Loaded step {state.get("step", "?")}')

def generate(prompt_ids, max_len=64, temp=0.8, top_k=40):
    model.eval()
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([ids[-64:]], dtype=torch.long, device=device)
            h, logits, _, _ = model.forward(x, return_scores=True, return_heads=True)
            logits = logits[0, -1, :] / temp

            # top-k filtering
            top_vals, top_idx = torch.topk(logits, top_k)
            probs = torch.softmax(top_vals, -1)
            next_id = top_idx[torch.multinomial(probs, 1)].item()

            if next_id in {cv.EOS_IDX, cv.SENT_CLOSE_IDX}:
                break
            ids.append(next_id)
    return ids

# Test generation
print('\n--- Generation ---')
prompts = [
    cv.encode('Литва'),
    cv.encode('Россия'),
    cv.encode('Искусственный'),
]
for p in prompts:
    gen = generate(p, max_len=48)
    text = cv.decode(gen, skip_special=True)
    print(f'  {text[:120]}')

# Topology analysis — check hidden state clustering
print('\n--- Topology (coord trajectory clustering) ---')
data = np.load('real_data/full_corpus_bpe.npy')
N = len(data)
B, L = 8, 64
idx = np.random.randint(0, N - L - 1, size=B)
batch = np.stack([data[i:i+L] for i in idx])
x = torch.tensor(batch, dtype=torch.long, device=device)

with torch.no_grad():
    h, _, _, heads = model.forward(x, return_scores=True, return_heads=True)
h = h.cpu().numpy()  # (B, L, 384)

# Per-token-type clustering
from collections import defaultdict
vecs_by_type = defaultdict(list)
for b in range(B):
    for pos in range(L):
        tid = batch[b, pos]
        if tid < 4096:
            vecs_by_type[tid].append(h[b, pos])

# Check intra/inter variance for frequent tokens
freq_tokens = [t for t, v in vecs_by_type.items() if len(v) >= 8]
if freq_tokens:
    all_vecs = np.concatenate([np.array(vecs_by_type[t]) for t in freq_tokens])
    global_mean = all_vecs.mean(0)
    global_var = all_vecs.var(0).mean()

    intra_dists = []
    for t in freq_tokens[:20]:
        vecs = np.array(vecs_by_type[t])
        centroid = vecs.mean(0)
        dists = np.linalg.norm(vecs - centroid[None, :], axis=-1)
        intra_dists.append(dists.mean())
    mean_intra = np.mean(intra_dists)

    # Inter-dist: distance between centroids of different tokens
    centroids = np.array([np.array(vecs_by_type[t]).mean(0) for t in freq_tokens[:20]])
    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    mean_inter = np.mean(inter_dists)

    dim_var = all_vecs.var(0)
    active_dims = (dim_var > dim_var.mean() * 0.5).sum()

    print(f'  Tokens with >=8 samples: {len(freq_tokens)}')
    print(f'  Mean intra-token dist: {mean_intra:.4f}')
    print(f'  Mean inter-token dist: {mean_inter:.4f}')
    print(f'  Ratio (intra/inter): {mean_intra/mean_inter:.4f}')
    print(f'  Active dims (var>0.5*mean): {active_dims}/384')
    print(f'  Global var: {global_var:.4f}')
else:
    print('  Not enough samples per token type')

dim_range = h.var(0)
print(f'  Dim variance range: {dim_range.min():.4f}..{dim_range.max():.4f}')
print(f'  Dims above mean: {(dim_range > dim_range.mean()).sum()}/384')
