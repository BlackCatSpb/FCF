import sys, numpy as np
sys.path.insert(0, '.')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from eva.symbolic.concept_space import ConceptSpace

cs = ConceptSpace.load('checkpoints/concept_space.json')

for w in ['человек', 'война', 'собака', 'время', 'мир', 'любовь', 'ненависть', 'работа']:
    cid = cs.word_to_cid.get(w)
    v = cs.concept_vectors.get(cid)
    print(f'{w:12s} -> cid={cid:8d}  norm={np.linalg.norm(v):.4f}  vector[:5]={v[:5] if v is not None else "NONE"}')
