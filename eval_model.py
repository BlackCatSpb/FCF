import torch, numpy as np, json, sys, os
sys.path.insert(0, '.')
from eva.symbolic import CharacterVocab

pf = torch.load('checkpoints/symbolic/final/potential_field.pt', map_location='cpu', weights_only=True)
aff = pf['affinity'].cpu().numpy()
count = pf['co_occurrence_count'].cpu().numpy()
vocab = CharacterVocab()

out = {"pairs": [], "char_order": [], "word_completion": []}

# Russian pair affinities
tst = ["\u043f\u0440","\u0440\u0438","\u0438\u0432","\u043d\u043e","\u0441\u0442","\u043b\u043e","\u0437\u0430","\u043c\u0430","\u0432\u043e","\u043d\u0430"]
for pair_str in tst:
    ids = vocab.encode(pair_str); i, j = ids[1], ids[2]
    out["pairs"].append({
        "pair": pair_str, "count": int(count[i,j]),
        "affinity": float(aff[i,j])
    })

# Character order
for w, name in [("\u043f\u0440\u0438\u0432\u0435\u0442","privet"), ("\u0447\u0435\u043b\u043e\u0432\u0435\u043a","chelovek"), ("\u043f\u0440\u0438\u0440\u043e\u0434\u0430","priroda")]:
    ids = vocab.encode(w)[1:-1]
    ranks = []
    for k in range(len(ids)-1):
        row = aff[ids[k]]; ranks.append(int(np.sum(row > row[ids[k+1]])))
    out["char_order"].append({"word": name, "avg_rank": float(np.mean(ranks)), "ranks": [int(r) for r in ranks]})

# Word completion
prefixes = [("\u043c\u0430\u043c","mam"),("\u043f\u0430\u043f","pap"),("\u043a\u043d\u0438\u0433","knig"),("\u0447\u0435\u043b\u043e","chelo"),("\u0437\u0435\u043c\u043b","zeml"),("\u0432\u043e\u0434","vod"),("\u0440\u0443\u043a","ruk"),("\u0434\u043e\u043c","dom"),("\u0441\u0442\u043e\u043b","stol")]
for ru, latin in prefixes:
    ids = vocab.encode(ru)[1:-1]
    if not ids: continue
    row = aff[ids[-1]]
    top5 = np.argsort(row)[-5:][::-1]
    chars = [vocab.idx_to_char(int(i)) for i in top5]
    out["word_completion"].append({"prefix": latin, "top5": chars})

with open("eval_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("Saved eval_result.json")
