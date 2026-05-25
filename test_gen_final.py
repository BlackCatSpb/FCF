"""Test generation with scaled causal model."""
import torch, torch.nn.functional as F, sys, os
sys.path.insert(0, os.path.dirname(__file__))
DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
cv = CharacterVocab(); VT = 157

evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE); aff = evolved['affinity']

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24).to(DEVICE)
ut.set_symbol_coordinates(coords)
ut.load_state_dict(torch.load(os.path.join(CKPT, "causal_latest.pt"), map_location='cpu', weights_only=True)['model'], strict=False)
ut.eval()

def gen(ids, n=30, T=0.8):
    ids=list(ids)
    with torch.no_grad():
        for _ in range(n):
            _, scores = ut(torch.tensor([ids],dtype=torch.long,device=DEVICE), return_scores=True)
            logits = scores[0,-1]/T
            sorted_l, sorted_i = logits.sort(descending=True)
            cumprobs = F.softmax(sorted_l,dim=-1).cumsum(dim=-1)
            cutoff = (cumprobs>0.95).nonzero(as_tuple=True)[0]
            k = cutoff[0].item()+1 if len(cutoff)>0 else 30; k = min(max(k,3),40)
            vals, idx = logits.topk(k); probs = F.softmax(vals,dim=-1)
            for t in set(ids[-5:]): m=(idx==t).nonzero(as_tuple=True)[0]; probs[m]*=0.2 if len(m)>0 else 1.0
            probs/=probs.sum(); nt=idx[torch.multinomial(probs,1)].item()
            if nt<=0 or nt>=VT: nt=idx[0].item()
            ids.append(nt)
    return ids

tests = [
    "привет", "человек идет по", "солнце светит", "сегодня хорошая погода",
    "я люблю", "метаданные хранят", "трансформер понимает",
    "программа работает", "знания это сила", "один два три"
]

print("EVA — Generation Test (100K steps)")
print("=" * 50)
for w in tests:
    ids = cv.encode(w)[1:-1]
    if len(ids)>=2:
        r=gen(ids,25,0.8)
        print(f"  {w} -> {cv.decode(r)}")
print("Done.")
