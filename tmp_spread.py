import sys, numpy as np
sys.path.insert(0, '.')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from eva.symbolic.concept_space import ConceptSpace

cs = ConceptSpace.load('checkpoints/concept_space.json')
print(f'Vectors: {len(cs.concept_vectors)}')
cs.normalize_vectors()
print()

cs.contrastive_spread(target_sim=0.5, lr=0.15, epochs=6)
print()

pairs = [('человек', 'война'), ('человек', 'собака'), ('человек', 'время'),
         ('человек', 'работа'), ('война', 'мир'), ('любовь', 'ненависть')]
for a, b in pairs:
    ca = cs.word_to_cid.get(a)
    cb = cs.word_to_cid.get(b)
    if ca and cb and ca in cs.concept_vectors and cb in cs.concept_vectors:
        sim = float(np.dot(cs.concept_vectors[ca], cs.concept_vectors[cb]))
        print(f'  sim({a}, {b}) = {sim:.4f}')

print()
print('Saving...')
cs.save('checkpoints/concept_space.json', include_morph=False)
