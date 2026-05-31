"""
Round-trip test: text → IDs → coordinates → IDs → text.
Uses only CoordinateEmbedding + CoordinateDecoder, NO transformer.
"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import CoordinateEmbedding, CoordinateDecoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

cv = CharacterVocab()
V = cv.vocab_size  # 160
D = 128

# Build pure encode-decode modules (no transformer)
embed = CoordinateEmbedding(V, D).to(DEVICE)
decoder = CoordinateDecoder(embed).to(DEVICE)

# Load checkpoint
ckpt = torch.load('checkpoints/symbolic/full_best.pt', map_location=DEVICE, weights_only=False)
sd = ckpt['ut']
print(f'Loaded step {ckpt.get("step", "?")}')

# Copy only embed + decoder weights
embed.load_state_dict({k.replace('embed.', ''): v for k, v in sd.items() if k.startswith('embed.')})
decoder.load_state_dict({k.replace('decoder.', ''): v for k, v in sd.items() if k.startswith('decoder.')})
print('Weights loaded')

# Normalise coord lookup for NN branch
with torch.no_grad():
    embed.coordinates.data = embed.coordinates / embed.coordinates.norm(dim=-1, keepdim=True).clamp(1e-8)

# Test texts
test_texts = [
    "Привет мир",
    "Как дела",
    "Солнце встаёт на востоке",
    "Нейронные сети изучают мир",
    "Война и мир — великий роман",
]

print('\n' + '=' * 70)
print('ROUND-TRIP TEST (Text → IDs → Coords → IDs → Text)')
print('=' * 70)

for text in test_texts:
    # 1. Encode text to IDs
    ids = cv.encode(text)
    t = torch.tensor([ids], dtype=torch.long, device=DEVICE)  # [1, L]

    # 2. Look up coordinates
    with torch.no_grad():
        coords = embed(t)  # [1, L, D]

    # 3. Decode coordinates back to IDs
    with torch.no_grad():
        logits = decoder(coords)  # [1, L, V]
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()  # [L]

    # 4. Decode both to text
    original = cv.decode(ids, skip_special=True)
    reconstructed = cv.decode(pred_ids, skip_special=True)

    # 5. Compare token-by-token (skip special tokens like BOS, EOS)
    clean_ids = [i for i in ids if i not in (cv.PAD_IDX, cv.UNK_IDX, cv.BOS_IDX, cv.EOS_IDX)]
    clean_pred = [i for i in pred_ids if i not in (cv.PAD_IDX, cv.UNK_IDX, cv.BOS_IDX, cv.EOS_IDX)]

    matches = sum(1 for a, b in zip(clean_ids, clean_pred) if a == b)
    total = len(clean_ids)
    accuracy = matches / total * 100 if total > 0 else 0

    print(f'\n--- {text} ---')
    print(f'  Original:       {original}')
    print(f'  Reconstructed:  {reconstructed}')
    print(f'  Token accuracy: {matches}/{total} = {accuracy:.1f}%')
    if matches < total:
        for i, (a, b) in enumerate(zip(clean_ids, clean_pred)):
            if a != b:
                print(f'    Pos {i}: {cv.idx_to_char(a)!r}({a}) → {cv.idx_to_char(b)!r}({b})')

print('\nDone.')
