"""Verify saved symbol weights — reproduce all 156 symbols."""
import torch, os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
ckpt_path = os.path.join(CKPT_DIR, "symbol_weights.pt")

print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
ut.load_state_dict(ckpt['model'])
ut.set_symbol_coordinates(ckpt['coords'].to(DEVICE))
ut = ut.to(DEVICE)
ut.eval()

cv = CharacterVocab()
all_symbols = torch.arange(1, 157, dtype=torch.long, device=DEVICE)
symbol_batch = all_symbols.unsqueeze(0)  # [1, 156]

with torch.no_grad():
    _, scores = ut(symbol_batch, return_scores=True)
    predicted = scores[0].argmax(dim=-1)
    correct = (predicted == all_symbols).sum().item()

print(f"Correct: {correct}/156 ({correct/156:.0%})")
print()

if correct == 156:
    print("ALL 156 SYMBOLS CORRECT:")
else:
    print(f"FAIL: {156 - correct} errors")

for i, p in enumerate(predicted):
    char = cv.decode([p.item()])
    marker = "" if p.item() == (i+1) else " <-- WRONG"
    print(f"  {i+1:>3d}: '{char}'{marker}")

print(f"\nSymbols as string: {cv.decode(predicted.tolist())}")
