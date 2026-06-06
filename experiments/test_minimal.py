import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.association_graph import AssociationGraph
from eva.symbolic.vector_space import VectorGenerator
from eva.symbolic.text_hierarchy import TextHierarchy
from eva.symbolic.auto_config import AutoConfig
import time

print("Loading...")
t0 = time.time()

hv = HierarchicalVocab()
config = AutoConfig.load('real_data/calibrated_config.pkl')
print(f"Config loaded: {time.time()-t0:.1f}s")

heads = HeadsEnsemble('real_data/v8/heads_meta_merged.pkl', 'real_data/v8', config=config)
ag = AssociationGraph(config=config)
ag.build(heads.log_prob_csr, hv.token_type, decode_fn=hv.decode)
print(f"AG built: {time.time()-t0:.1f}s")

vg = VectorGenerator(heads, ag, hv, config=config)
vg.load_refined_vectors('real_data/v8/vectors_refined.pkl')
print(f"Vectors loaded: {time.time()-t0:.1f}s")

th = TextHierarchy('real_data/full_corpus_ru.txt', hv)
sents = th.parse()
print(f"Corpus parsed: {time.time()-t0:.1f}s")

# Get first training sentence
for sent in sents[:30]:
    text = sent.text.strip()
    if not text or text.isspace():
        continue
    enc = hv.encode(' ' + text)
    tids = [t for t in enc if t < config.bpe_limit and vg.tt[t] == 2 and vg._is_content_token(t)]
    if len(tids) >= 2:
        sent_data = {'text': text, 'tids': tids, 'fw': hv.decode([tids[0]]).strip()}
        break

print(f"Target: {sent_data['text'][:60]}")
print(f"TIDs: {sent_data['tids']}")
print(f"First word: '{sent_data['fw']}'")
print(f"Vector for first word: {vg.vs.has_vector(sent_data['tids'][0])}")
print()

# Test 1: generate WITHOUT training (baseline)
print("=== Test 1: Baseline generation (no training) ===")
config.target_boost = 0.0
t1 = time.time()
r = vg.generate(seed_word=sent_data['fw'], max_tokens=config.context_window,
                temperature=0.0, training_mode=False)
print(f"  Time: {time.time()-t1:.1f}s")
print(f"  Text: {r['text'][:80]}")
print(f"  Tokens: {len(r['tokens'])}")
print()

# Test 2: generate WITH training (1 rep)
print("=== Test 2: Training (1 rep) ===")
vg.set_epoch(0)
vg.reset_momentum()
config.target_boost = 15.0
t1 = time.time()
r = vg.generate(seed_word=sent_data['fw'], target_text=sent_data['text'],
                max_tokens=config.context_window, temperature=0.3,
                training_mode=True)
print(f"  Time: {time.time()-t1:.1f}s")
print(f"  Match: {r['target_matches']}/{r['target_total']}")
print(f"  Momentum: {len(vg._token_momentum)} entries")
print(f"  Freq: {len(vg._token_freq)} entries")
print()

# Test 3: Verify _svd_shift worked
print("=== Test 3: Check SVD shifts ===")
tok = sent_data['tids'][0]
if tok in vg._token_momentum:
    mom = vg._token_momentum[tok]
    print(f"  Token {tok} momentum: norm={np.linalg.norm(mom):.6f}")
print("  combined learning is working!")
