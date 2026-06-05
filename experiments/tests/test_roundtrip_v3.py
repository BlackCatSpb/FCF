"""
Round-trip v3: correct understanding.
The decoder maps h[t] → token[t+1] (next-token prediction).
So to reconstruct sequence [t1, t2, ..., tn]:
  feed [BOS, t1, ..., tn-1] → transformer → h[0..n-1] → decoder → [t1, t2, ..., tn]
"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

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

test_texts = [
    "Привет мир",
    "Солнце встаёт на востоке",
    "Война и мир — великий роман",
]

print('\n' + '=' * 70)
print('ROUND-TRIP v3 — next-token reconstruction (correct formulation)')
print('=' * 70)

with torch.no_grad():
    for text in test_texts:
        ids = cv.encode(text)  # [BOS, chars..., EOS]
        # feed [BOS, t1, ..., tn-1] → predict [t1, t2, ..., tn]
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)
        h, scores = ut(inp, return_scores=True)

        pred_ids = scores.argmax(dim=-1).squeeze(0).tolist()

        # Full reconstructed sequence: [pred_tokens] + [last_token from original]
        # Actually: pred_ids[0] corresponds to ids[1], etc.
        # So: reconstructed = [ids[0]] + pred_ids (should be [BOS, t1', t2', ..., tn'])
        recon_full = [ids[0]] + pred_ids
        recon_text = cv.decode(recon_full, skip_special=True)

        # Clean comparison (exclude BOS/EOS)
        clean_ids = [i for i in ids if i not in (cv.PAD_IDX, cv.UNK_IDX, cv.BOS_IDX, cv.EOS_IDX)]
        clean_pred = [i for i in pred_ids if i not in (cv.PAD_IDX, cv.UNK_IDX, cv.BOS_IDX, cv.EOS_IDX)]

        matches = sum(1 for a, b in zip(clean_ids, clean_pred) if a == b)
        total = len(clean_ids)
        accuracy = matches / total * 100 if total > 0 else 0

        original = cv.decode(ids, skip_special=True)

        print(f'\n--- {text} ---')
        print(f'  Original:  {original}')
        print(f'  Reconstructed: {recon_text}')
        print(f'  Accuracy: {matches}/{total} = {accuracy:.1f}%')

        # Show per-token breakdown for short texts
        if total <= 15:
            for pos in range(min(total, len(pred_ids))):
                orig = ids[1 + pos] if 1 + pos < len(ids) else None
                pred = pred_ids[pos] if pos < len(pred_ids) else None
                if orig is not None and pred is not None and orig != pred:
                    o_char = cv.idx_to_char(orig) if orig < V else '?'
                    p_char = cv.idx_to_char(pred) if pred is not None and pred < V else '?'
                    print(f'    Pos {pos}: {o_char!r}({orig}) → {p_char!r}({pred})')

        # Show random reconstruction (generate FROM the hidden states)
        print(f'  Model-generated continuation:')
        gen = ut.enhanced_generate(ids[:4], cv, max_new=20, temperature=0.8)
        print(f'    {gen}')

print('\nDone.')
