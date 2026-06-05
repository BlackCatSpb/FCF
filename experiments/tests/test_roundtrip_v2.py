"""
Round-trip test v2: uses full transformer to see if hidden states decode correctly.
Also tests raw coordinate path for baseline comparison.
"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

cv = CharacterVocab()
V = cv.vocab_size  # 160
D = 128

# Build full transformer
ut = UnifiedMultidimensionalTransformer(vocab_size=V, coord_dim=D, max_levels=8,
    total_heads=32, num_layers=6, d_ff=128).to(DEVICE)

# Load checkpoint
ckpt = torch.load('checkpoints/symbolic/full_best.pt', map_location=DEVICE, weights_only=False)
sd = ckpt['ut']
# Remove shape mismatches
for k in list(sd.keys()):
    if k in ut.state_dict() and sd[k].shape != ut.state_dict()[k].shape:
        print(f'  Skip shape mismatch: {k} {sd[k].shape} vs {ut.state_dict()[k].shape}')
        del sd[k]
ut.load_state_dict(sd, strict=False)
print(f'Loaded step {ckpt.get("step", "?")}')
ut.eval()

test_texts = [
    "Привет мир",
    "Солнце встаёт на востоке",
    "Война и мир",
]

print('\n' + '=' * 70)
print('ROUND-TRIP v2 — with full transformer')
print('=' * 70)

with torch.no_grad():
    for text in test_texts:
        ids = cv.encode(text)
        t = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        # A) Raw coordinate path (no transformer)
        raw_coords = ut.embed(t)
        logits_raw = ut.decoder(raw_coords)
        pred_raw = logits_raw.argmax(dim=-1).squeeze(0).tolist()

        # B) Full transformer path
        h, scores = ut(t, return_scores=True)
        logits_tf = scores
        pred_tf = logits_tf.argmax(dim=-1).squeeze(0).tolist()

        # C) From scores (already contains decoder output)
        pred_score = scores.argmax(dim=-1).squeeze(0).tolist()

        original = cv.decode(ids, skip_special=True)
        recon_raw = cv.decode(pred_raw, skip_special=True)
        recon_tf = cv.decode(pred_tf, skip_special=True)

        clean = [i for i in ids if i not in (cv.PAD_IDX, cv.UNK_IDX, cv.BOS_IDX, cv.EOS_IDX)]
        clean_raw = [i for i in pred_raw if i not in (cv.PAD_IDX, cv.UNK_IDX, cv.BOS_IDX, cv.EOS_IDX)]
        clean_tf = [i for i in pred_tf if i not in (cv.PAD_IDX, cv.UNK_IDX, cv.BOS_IDX, cv.EOS_IDX)]

        acc_raw = sum(1 for a,b in zip(clean, clean_raw) if a==b) / len(clean) * 100
        acc_tf = sum(1 for a,b in zip(clean, clean_tf) if a==b) / len(clean) * 100

        print(f'\n--- {text} ---')
        print(f'  Raw coord accuracy:  {acc_raw:.1f}%')
        print(f'  Transformer acc:     {acc_tf:.1f}%')
        print(f'  Original:  {original}')
        print(f'  Raw coords: {recon_raw}')
        print(f'  TF output:  {recon_tf}')

        # Show transformer final hidden norms
        h_norm = h.norm(dim=-1).squeeze(0)
        print(f'  Hidden norms: min={h_norm.min().item():.3f}, mean={h_norm.mean().item():.3f}, max={h_norm.max().item():.3f}')

print('\nDone.')
