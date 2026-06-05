"""
Round-trip v4: use boundary encoding (same format as training data).
feed tokens[:-1] → predict tokens[1:]. Compare full sequences.
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
    "Как дела",
    "Солнце встаёт на востоке",
]

print('\n' + '=' * 70)
print('ROUND-TRIP v4 — boundary format, full sequence comparison')
print('=' * 70)

with torch.no_grad():
    for text in test_texts:
        # Encode with word/sentence boundaries (matching training format)
        ids = cv.encode_with_boundaries(text)  # [<S>, <W>, chars..., </W>, ..., </S>]
        if not ids:
            print(f'\n--- {text} --- EMPTY')
            continue

        print(f'\n--- {text} --- (len={len(ids)})')
        print(f'  Original tokens: {ids}')

        # Next-token prediction: feed [0..n-1] → predict [1..n]
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)
        h, scores = ut(inp, return_scores=True)
        pred = scores.argmax(dim=-1).squeeze(0).tolist()

        # pred[t] = predicted token at position t+1 (i.e., pred[0] corresponds to ids[1])
        # So reconstructed sequence = [ids[0]] + pred
        recon = [ids[0]] + pred

        # Remove <S>...</S> wrappers for clean display
        def strip_special(seq):
            return '|'.join(cv.idx_to_char(t) if t < V else '?' for t in seq)

        print(f'  Target:   {strip_special(ids)}')
        print(f'  Reconst:  {strip_special(recon)}')

        # Token-level accuracy (all tokens, including special)
        matches = sum(1 for a, b in zip(ids, recon) if a == b)
        acc = matches / len(ids) * 100
        print(f'  Accuracy (all tokens): {matches}/{len(ids)} = {acc:.1f}%')

        # Character-only accuracy (skip special tokens)
        special = {cv.PAD_IDX, cv.UNK_IDX, cv.BOS_IDX, cv.EOS_IDX,
                   cv.WORD_OPEN_IDX, cv.WORD_CLOSE_IDX,
                   cv.SENT_OPEN_IDX, cv.SENT_CLOSE_IDX}
        clean_target = [t for t in ids if t not in special]
        clean_pred = [t for t in recon if t not in special]
        if clean_target and clean_pred:
            m2 = sum(1 for a, b in zip(clean_target, clean_pred) if a == b)
            acc2 = m2 / len(clean_target) * 100
            print(f'  Accuracy (chars only): {m2}/{len(clean_target)} = {acc2:.1f}%')

        # Show errors
        if len(ids) <= 25:
            for i, (a, b) in enumerate(zip(ids, recon)):
                if a != b:
                    print(f'    Pos {i}: {cv.idx_to_char(a)!r}({a}) → {cv.idx_to_char(b)!r}({b})')

print('\nDone.')
