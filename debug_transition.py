import sys, numpy as np
sys.path.insert(0, '.')
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.association_graph import AssociationGraph
from eva.symbolic.vector_space import VectorGenerator

hv = HierarchicalVocab()
head = HeadsEnsemble('real_data/v8/heads_meta_merged.pkl', 'real_data/v8')
ag = AssociationGraph(n_clusters=48, n_metas=12)
ag.build(head.log_prob_csr, hv.token_type, decode_fn=hv.decode)
vg = VectorGenerator(head, ag, hv)
vg.load_refined_vectors('real_data/v8/vectors_refined.pkl')

# Check: is 358 in structural[475]?
struct_475 = vg.structural.get(475, set())
print('Is 358 in structural[475]?', 358 in struct_475)
print('476 in structural[475]?', 476 in struct_475)
print('Total transitions from 475:', len(struct_475))

# Check weight
w = vg.structural_weights.get(475, {}).get(358, None)
if w is not None:
    print('Weight(475->358):', w)
    print('Prob:', np.exp(min(w,0)))
else:
    print('No weight for 475->358')

# Check target tid for 'on'
enc = hv.encode(' on')
print('Encoded "on":', enc, [hv.decode([t]) for t in enc])
