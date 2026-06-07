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

sent = 'сказал он'
words = sent.split()
result = vg.generate(seed_word=words[0], target_text=sent, max_tokens=60, temperature=0.3)
gen_text = result['text']
matched = vg._target_matches

# Show all token types
tokens = result['tokens']
print('Tokens:', tokens)
types = [vg.tt[t] if t < len(vg.tt) else '?' for t in tokens]
print('Types:', types)
decoded = [hv.decode([t]).strip() for t in tokens]
print('Decoded:', decoded)

print('\nTARGET: %s' % sent)
print('GEN:    %s' % gen_text[:150])
print('MATCH:  %d/%d words' % (matched, len(words)))
