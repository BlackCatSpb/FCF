import sys, numpy as np
sys.path.insert(0, '.')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from eva.symbolic.concept_space import ConceptSpace

print('Loading checkpoint...')
cs = ConceptSpace.load('checkpoints/concept_space.json')
print(f'Vectors: {len(cs.concept_vectors)}, dim={cs.dim}')

# Check before
orig = {c: v for c, v in cs.concept_vectors.items() if c < 100000}
import random
vecs = list(orig.values())
n = len(vecs)
sims = []
for _ in range(2000):
    i = random.randint(0, n-1)
    j = random.randint(0, n-1)
    if i != j:
        sims.append(float(np.dot(vecs[i], vecs[j])))
print(f'Before: mean_sim={np.mean(sims):.4f}')

print()
print('Normalizing...')
cs.normalize_vectors()

# Check after
vecs2 = [cs.concept_vectors[c] for c in orig]
sims2 = []
for _ in range(2000):
    i = random.randint(0, n-1)
    j = random.randint(0, n-1)
    if i != j:
        sims2.append(float(np.dot(vecs2[i], vecs2[j])))
print(f'After:  mean_sim={np.mean(sims2):.4f}')

# Check specific pairs
pairs = [('\u0447\u0435\u043b\u043e\u0432\u0435\u043a', '\u0432\u043e\u0439\u043d\u0430'),
         ('\u0447\u0435\u043b\u043e\u0432\u0435\u043a', '\u0441\u043e\u0431\u0430\u043a\u0430'),
         ('\u0447\u0435\u043b\u043e\u0432\u0435\u043a', '\u0432\u0440\u0435\u043c\u044f'),
         ('\u0447\u0435\u043b\u043e\u0432\u0435\u043a', '\u0440\u0430\u0431\u043e\u0442\u0430')]
print()
for a, b in pairs:
    ca = cs.word_to_cid.get(a)
    cb = cs.word_to_cid.get(b)
    if ca and cb and ca in cs.concept_vectors and cb in cs.concept_vectors:
        sim = float(np.dot(cs.concept_vectors[ca], cs.concept_vectors[cb]))
        print(f'  sim({a}, {b}) = {sim:.4f}')

print()
print('Saving...')
cs.save('checkpoints/concept_space.json', include_morph=False)
