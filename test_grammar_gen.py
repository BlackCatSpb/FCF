"""EVA — GrammarHead generation test with affinity constraint."""
import torch, torch.nn.functional as F, sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.grammar_head import GrammarHead

cv = CharacterVocab(); VT = 157

evolved = torch.load(os.path.join(CKPT_DIR, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)
affinity = evolved['affinity']

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24).to(DEVICE)
ut.set_symbol_coordinates(coords)
ckpt = torch.load(os.path.join(CKPT_DIR, "conceptnet_weights.pt"), map_location='cpu', weights_only=True)
ut.load_state_dict(ckpt['model'], strict=False)
for p in ut.parameters(): p.requires_grad = False
ut.eval()

gh = GrammarHead().to(DEVICE)
gh.load_state_dict(torch.load(os.path.join(CKPT_DIR, "grammar_head.pt"), map_location='cpu', weights_only=True)['model'])
gh.eval()

def generate(seed_ids, max_new=25, temp=0.7, top_k=30):
    ids = list(seed_ids)
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            ut_coords, _ = ut(inp, return_scores=True)
            logits = gh(ut_coords)[0, -1] / temp
            
            # Affinity constraint
            prev = ids[-1]
            if 0 < prev < VT:
                aff_boost = affinity[prev].to(DEVICE).clone()
                aff_boost[0] = -1e9
                aff_boost = aff_boost / aff_boost.max().clamp(min=1e-8)
                logits = logits + aff_boost * 3.0
            
            k = min(top_k, len(logits) - 1)
            vals, idx = torch.topk(logits, k)
            probs = F.softmax(vals, dim=-1)
            
            for t in set(ids[-4:]):
                mask = (idx == t).nonzero(as_tuple=True)[0]
                if len(mask) > 0: probs[mask] *= 0.05
            
            probs = probs / probs.sum()
            next_tok = idx[torch.multinomial(probs, 1)].item()
            if next_tok <= 0 or next_tok >= VT:
                next_tok = idx[0].item()
            ids.append(next_tok)
    return ids

print("EVA — GrammarHead + Affinity Generation")
print("=" * 60)

tests = ["привет", "человек", "солнце", "я люблю", "метаданные", "трансформер", "сегодня"]

for seed in tests:
    ids = cv.encode(seed)[1:-1]
    if len(ids) < 2: continue
    result = generate(ids, max_new=20, temp=0.7)
    text = cv.decode(result)
    print(f"  '{seed}' → '{text}'")

print("Done.")
