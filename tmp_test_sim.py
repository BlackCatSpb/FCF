import sys, math, numpy as np, random
sys.path.insert(0, '.')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from eva.symbolic.concept_space import ConceptSpace

cs = ConceptSpace.load('checkpoints/concept_space.json')
print(f'Concepts: {len(cs.concept_vectors)}')

orig = {c: v for c, v in cs.concept_vectors.items() if c < 100000}
print(f'Original: {len(orig)}')

norms = [np.linalg.norm(v) for v in orig.values()]
print(f'Norms: mean={np.mean(norms):.3f} std={np.std(norms):.3f} min={np.min(norms):.3f} max={np.max(norms):.3f}')

vecs = list(orig.values())
n = len(vecs)
sims = []
for _ in range(20000):
    i = random.randint(0, n-1)
    j = random.randint(0, n-1)
    if i != j:
        sims.append(float(np.dot(vecs[i], vecs[j])))

print(f'Random pair sim: mean={np.mean(sims):.4f} std={np.std(sims):.4f}')
print(f'  50pct={np.percentile(sims, 50):.4f} 90pct={np.percentile(sims, 90):.4f} 99pct={np.percentile(sims, 99):.4f}')

# Mean vector of all concept vectors
mean_v = np.mean(vecs, axis=0)
mean_norm = np.linalg.norm(mean_v)
print(f'\nMean vector norm: {mean_norm:.4f} (if 0 -> well-centered sphere)')

# Check if vectors point toward mean direction
cents = [float(np.dot(v, mean_v / mean_norm)) for v in vecs[:1000]]
print(f'Cos to centroid: mean={np.mean(cents):.4f} std={np.std(cents):.4f}')

# Check specific concept distances
pairs = [('человек', 'война'), ('человек', 'собака'), ('человек', 'время')]
for a, b in pairs:
    ca = cs.word_to_cid.get(a)
    cb = cs.word_to_cid.get(b)
    if ca and cb and ca in cs.concept_vectors and cb in cs.concept_vectors:
        sim = float(np.dot(cs.concept_vectors[ca], cs.concept_vectors[cb]))
        print(f'  sim({a}, {b}) = {sim:.4f}')
