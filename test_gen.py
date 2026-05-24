"""Quick generation test — coordinate prediction."""
import torch, torch.nn.functional as F, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
DEVICE = 'cuda'
CKPT = 'checkpoints/symbolic'

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

cv = CharacterVocab(); VT = 157
evolved = torch.load(f'{CKPT}/evolved_affinity.pt', map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE); affinity = evolved['affinity']

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24).to(DEVICE)
ut.set_symbol_coordinates(coords)
ut.load_state_dict(torch.load(f'{CKPT}/coord_predict.pt', map_location='cpu', weights_only=True)['model'], strict=False)
ut.eval()

def gen(ids, max_new=20, temp=0.8):
    ids = list(ids)
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            pc, _ = ut(inp, return_scores=True)
            nc = pc[0, -1]
            dists = torch.norm(coords - nc, dim=-1); dists[0] = 1e9
            k = 15; td, ti = torch.topk(dists, k, largest=False)
            scores = 1.0 / (td + 0.01)
            if 0 < ids[-1] < VT:
                ab = affinity[ids[-1]][ti.cpu()].to(DEVICE)
                scores = scores + ab / ab.max().clamp(min=1e-8) * 2.0
            for t in set(ids[-3:]):
                m = (ti == t).nonzero(as_tuple=True)[0]
                if len(m) > 0: scores[m] *= 0.1
            probs = F.softmax(scores / temp, dim=-1)
            nt = ti[torch.multinomial(probs, 1)].item()
            if nt <= 0 or nt >= VT: nt = ti[0].item()
            ids.append(nt)
    return ids

for s in ['привет', 'человек', 'солнце', 'сегодня', 'я люблю']:
    ids = cv.encode(s)[1:-1]
    r = gen(ids, 15, 0.8)
    print(f"'{s}' -> '{cv.decode(r)}'")
print("Done")
