import sys, numpy as np
sys.path.insert(0, '.')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from eva.symbolic.concept_space import ConceptSpace

cs = ConceptSpace.load('checkpoints/concept_space.json')

pairs = [('человек', 'война'), ('человек', 'собака')]
for a, b in pairs:
    ca = cs.word_to_cid.get(a)
    cb = cs.word_to_cid.get(b)
    va = cs.concept_vectors.get(ca)
    vb = cs.concept_vectors.get(cb)
    print(f'{a}: cid={ca}, norm={np.linalg.norm(va):.4f}')
    print(f'{b}: cid={cb}, norm={np.linalg.norm(vb):.4f}')
    print(f'  sim(original) = {float(np.dot(va, vb)):.4f}')
    print()

# Check: are these adaptive concepts?
print(f'человек cid<100000: {ca < 100000}')
print(f'война cid<100000: {cb < 100000}')

# Count original vs adaptive
orig = sum(1 for c in cs.concept_vectors if c < 100000)
adap = sum(1 for c in cs.concept_vectors if c >= 100000)
print(f'\nOriginal: {orig}, Adaptive: {adap}')
