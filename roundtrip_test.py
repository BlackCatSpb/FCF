"""Fixed roundtrip: NGgram-based sentence reconstruction"""
import sys, os, torch, numpy as np, json
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from eva.symbolic import *
from eva.symbolic.advanced_methods import NGramContext

pf = PotentialField(156, 256)
pf.load_state_dict(torch.load('checkpoints/symbolic/final/potential_field.pt', map_location='cpu', weights_only=True))
vocab = CharacterVocab()
aff = pf.affinity.cpu().numpy()
ng = NGramContext(pf, max_context=4, decay=0.5)

def sample(cont, t=0.5, top_k=15):
    c = cont.copy()
    if top_k > 0:
        tk = min(top_k, len(c)); ti = np.argpartition(c, -tk)[-tk:]
        m = np.ones(len(c), dtype=bool); m[ti] = False; c[m] = -1e10
    c = c / max(t, 0.1); p = np.exp(c - c.max()); p /= p.sum()
    return int(np.random.choice(len(p), p=p))

def reconstruct(sentence, prompt_ratio=0.4, n_trials=50):
    ids = vocab.encode(sentence)
    if len(ids) < 4: return 0
    n_prompt = max(2, int(len(ids) * prompt_ratio))
    prompt, expected = ids[:n_prompt], ids[n_prompt:]
    
    best = 0
    for _ in range(n_trials):
        ctx = list(prompt); correct = 0
        for i in range(len(expected)):
            cont = ng.get_continuation(ctx)
            ns = sample(cont, t=0.5)
            if ns == expected[i]: correct += 1
            ctx.append(ns)
        acc = correct / len(expected)
        if acc > best: best = acc
    return best

print("=" * 60)
print("ROUND-TRIP (NGram, temperature)")
print("=" * 60)

tests = [
    "мама мыла раму", "привет мир", "человек идёт по улице",
    "солнце светит ярко", "кошка спит на окне",
    "я люблю программирование", "сегодня хорошая погода",
]
for s in tests:
    acc = reconstruct(s, prompt_ratio=0.4)
    print(f"  {acc:.0%} | {s}")

print("\nWORD RECONSTRUCTION:")
words = ["привет", "человек", "математика", "природа", "компьютер", "история", "вселенная", "космонавт"]
for w in words:
    ids = vocab.encode(w)[1:-1]
    mid = len(ids)//2
    correct = 0
    ctx = list(ids[:mid])
    for i in range(len(ids[mid:])):
        cont = ng.get_continuation(ctx)
        ns = sample(cont, t=0.4, top_k=8)
        if ns == ids[mid+i]: correct += 1
        ctx.append(ns)
    print(f"  {vocab.decode(ids[:mid])}... : {correct}/{len(ids)-mid} = {correct/max(len(ids)-mid,1):.0%}")

# Save
with open("roundtrip_result.json", "w", encoding="utf-8") as f:
    json.dump({"avg_accuracy": float(np.mean([reconstruct(s, prompt_ratio=0.4, n_trials=30) for s in tests]))}, f)
print("\nSaved roundtrip_result.json")
