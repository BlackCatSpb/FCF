"""
Compare round-trip accuracy at different checkpoints.
Test step 60000 (best) vs step 100000 (latest).
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
special_set = {cv.PAD_IDX, cv.UNK_IDX, cv.BOS_IDX, cv.EOS_IDX,
               cv.WORD_OPEN_IDX, cv.WORD_CLOSE_IDX,
               cv.SENT_OPEN_IDX, cv.SENT_CLOSE_IDX}

def load_model(ckpt_path):
    ut = UnifiedMultidimensionalTransformer(vocab_size=V, coord_dim=D, max_levels=8,
        total_heads=32, num_layers=6, d_ff=128).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt['ut']
    for k in list(sd.keys()):
        if k in ut.state_dict() and sd[k].shape != ut.state_dict()[k].shape:
            del sd[k]
    ut.load_state_dict(sd, strict=False)
    ut.eval()
    return ut, ckpt.get('step', 0)

# Load data
data = np.load('real_data/full_corpus_encoded.npy', mmap_mode='r').astype(np.int32)
SENT_CLOSE = cv.SENT_CLOSE_IDX
blocks = []
i = 0
while i < len(data):
    start = i
    while i < len(data) and data[i] != SENT_CLOSE:
        i += 1
    if i < len(data):
        blocks.append(data[start:i+1].tolist())
        i += 1
    else:
        break
print(f'Blocks: {len(blocks):,}')

# Pick 20 random blocks for evaluation
rng = np.random.RandomState(seed=42)
test_blocks = [blocks[rng.randint(0, len(blocks))] for _ in range(20)]

checkpoints = [
    'checkpoints/symbolic/full_latest.pt',
    'checkpoints/symbolic/full_best.pt',
]

print(f'\n{"="*70}')
print(f'BENCHMARK: next-token prediction accuracy across checkpoints')
print(f'{"="*70}')

for ckpt_path in checkpoints:
    if not os.path.exists(ckpt_path):
        print(f'\n{ckpt_path} — NOT FOUND, skipping')
        continue

    ut, step = load_model(ckpt_path)
    print(f'\n--- Checkpoint step {step} ({os.path.basename(ckpt_path)}) ---')

    total_tokens = 0
    correct_tokens = 0
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for blk in test_blocks:
            ids = blk
            if len(ids) < 3:
                continue

            inp = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)
            h, scores = ut(inp, return_scores=True)
            pred = scores.argmax(dim=-1).squeeze(0).tolist()
            recon = [ids[0]] + pred

            # All tokens
            for a, b in zip(ids, recon):
                total_tokens += 1
                if a == b:
                    correct_tokens += 1

            # Chars only
            for a, b in zip(ids, recon):
                if a not in special_set and b not in special_set:
                    total_chars += 1
                    if a == b:
                        correct_chars += 1

    acc_tok = correct_tokens / total_tokens * 100
    acc_char = correct_chars / total_chars * 100 if total_chars > 0 else 0
    print(f'  All tokens: {correct_tokens}/{total_tokens} = {acc_tok:.1f}%')
    print(f'  Chars only: {correct_chars}/{total_chars} = {acc_char:.1f}%')

print('\nDone.')
