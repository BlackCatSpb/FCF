import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.association_graph import AssociationGraph
from eva.symbolic.vector_space import VectorGenerator
from eva.symbolic.text_hierarchy import TextHierarchy

hv = HierarchicalVocab()
heads = HeadsEnsemble('real_data/v8/heads_meta_merged.pkl', 'real_data/v8')
ag = AssociationGraph(n_clusters=48, n_metas=12)
ag.build(heads.log_prob_csr, hv.token_type, decode_fn=hv.decode)
vg = VectorGenerator(heads, ag, hv)
vg.load_refined_vectors('real_data/v8/vectors_refined.pkl')
vg.load_gates('real_data/v8/gates')

th = TextHierarchy('real_data/full_corpus_ru.txt', hv)
sents = th.parse()

# Prepare sentence list
sentences = []
for sent in sents[:30]:
    text = sent.text.strip()
    if not text or text.isspace(): continue
    enc = hv.encode(' ' + text)
    tids = [t for t in enc if t < 4096 and vg.tt[t] == 2 and vg._is_content_token(t)]
    if len(tids) < 2: continue
    fw = hv.decode([tids[0]]).strip()
    sentences.append((text, fw, tids, len(tids)))

# Phase 1: 10 passes WITH forced selection (accumulate SVD shifts)
print("=== PHASE 1: TRAINING WITH FORCED SELECTION ===")
for p in range(10):
    total_m, total_t = 0, 0
    for text, fw, tids, nt in sentences:
        r = vg.generate(seed_word=fw, target_text=text, max_tokens=80, temperature=0.3)
        m = vg._target_matches
        total_m += m; total_t += nt
    print(f'  Pass {p+1}: {total_m}/{total_t} ({100*total_m/max(1,total_t):.1f}%)')

# Save shifted vectors
out_dir = os.path.join(os.path.dirname(__file__), '..', 'real_data', 'v9')
os.makedirs(out_dir, exist_ok=True)
vg.save_refined_vectors(os.path.join(out_dir, 'vectors_trained.pkl'))
print(f'\nSaved trained vectors to {out_dir}\\vectors_trained.pkl')

# Phase 2: Reload fresh vectors and test WITHOUT forced selection
print("\n=== PHASE 2: TEST WITHOUT FORCED SELECTION ===")
# Disable forced selection by temporarily removing it... actually, let me just
# reload fresh vectors and note that forced selection is still in the code.
# We need to verify: do the SVD-shifted vectors make the model naturally
# generate the target sequence?
# 
# Since forced selection is the LAST thing before select_token, and it
# returns before scoring runs, disabling it means scoring runs normally.
# The SVD shifts should make target pairs have higher semantic scores.
# Let me instead test by disabling forced selection temporarily:
# We'll use a modified approach - compare pre-training vs post-training
# semantic similarities for target pairs.

# Phase 3: Analyze SVD shift impact on key transitions
print("\n=== PHASE 3: SEMANTIC SIMILARITY BEFORE/AFTER ===")
# Load original vectors for comparison
vg_pre = VectorGenerator(heads, ag, hv)
vg_post = VectorGenerator(heads, ag, hv)
vg_pre.load_refined_vectors('real_data/v8/vectors_refined.pkl')
vg_post.load_refined_vectors(os.path.join(out_dir, 'vectors_trained.pkl'))

total_sim_pre, total_sim_post = 0.0, 0.0
n_pairs = 0
analyzed = []
for text, fw, tids, nt in sentences[:10]:
    for j in range(len(tids)-1):
        prev, nxt = tids[j], tids[j+1]
        if vg_pre.vs.has_vector(prev) and vg_pre.vs.has_vector(nxt) and vg_post.vs.has_vector(prev) and vg_post.vs.has_vector(nxt):
            sim_pre = vg_pre.vs.similarity(prev, nxt)
            sim_post = vg_post.vs.similarity(prev, nxt)
            total_sim_pre += sim_pre
            total_sim_post += sim_post
            n_pairs += 1
            if abs(sim_post - sim_pre) > 0.05:
                analyzed.append((hv.decode([prev]).strip(), hv.decode([nxt]).strip(), round(sim_pre,3), round(sim_post,3)))

print(f'  Average semantic similarity for target pairs:')
print(f'    Before: {total_sim_pre/max(1,n_pairs):.4f} ({n_pairs} pairs)')
print(f'    After:  {total_sim_post/max(1,n_pairs):.4f}')
print(f'    Delta:  {(total_sim_post-total_sim_pre)/max(1,n_pairs):.4f}')
print(f'  Pairs with >0.05 delta: {len(analyzed)}')
for w1, w2, s_pre, s_post in analyzed[:20]:
    arrow = '↑' if s_post > s_pre else '↓'
    print(f'    {w1:>12} -> {w2:<12}: {s_pre:.3f} -> {s_post:.3f} {arrow}')

# Phase 4: Generate WITHOUT forced selection using trained vectors
print("\n=== PHASE 4: GENERATION WITH TRAINED VECTORS (forced DISABLED temporarily) ===")
# Create a copy of the generate_step that doesn't force-select but uses the boosted
# semantic similarity from trained vectors
vg_test = VectorGenerator(heads, ag, hv)
vg_test.load_refined_vectors(os.path.join(out_dir, 'vectors_trained.pkl'))
vg_test.load_gates('real_data/v8/gates')

# Manually disable forced: set _target_tokens to None so forced doesn't fire
# But we want to test if natural generation follows the learned transitions.
# The only difference between vg_test and original is the SVD-shifted vectors.
# So generating WITH target but WITHOUT forced will show if the shifts matter.

# We need to temporarily block forced selection. Easiest way: pass target_text=None
# and instead just generate freely from seed.
total_m2, total_t2 = 0, 0
for text, fw, tids, nt in sentences[:20]:
    # Generate WITHOUT target (pure generation from seed)
    r = vg_test.generate(seed_word=fw, max_tokens=30, temperature=0.3)
    gen_tids = [t for t in hv.encode(r['text']) if t < 4096 and vg_test.tt[t] == 2 and vg_test._is_content_token(t)]
    # Count how many match target sequence
    matches = 0
    ti = 0
    for gt in gen_tids:
        if ti < len(tids) and gt == tids[ti]:
            matches += 1
            ti += 1
    total_m2 += matches; total_t2 += nt
    print(f'    {nt} target words: {matches} natural matches')
print(f'  Natural generation match: {total_m2}/{total_t2} ({100*total_m2/max(1,total_t2):.1f}%)')

# Compare with original vectors (fresh load)
vg_orig = VectorGenerator(heads, ag, hv)
vg_orig.load_refined_vectors('real_data/v8/vectors_refined.pkl')
vg_orig.load_gates('real_data/v8/gates')
total_m3, total_t3 = 0, 0
for text, fw, tids, nt in sentences[:20]:
    r = vg_orig.generate(seed_word=fw, max_tokens=30, temperature=0.3)
    gen_tids = [t for t in hv.encode(r['text']) if t < 4096 and vg_orig.tt[t] == 2 and vg_orig._is_content_token(t)]
    matches = 0; ti = 0
    for gt in gen_tids:
        if ti < len(tids) and gt == tids[ti]:
            matches += 1; ti += 1
    total_m3 += matches; total_t3 += nt
print(f'  Original vectors natural match: {total_m3}/{total_t3} ({100*total_m3/max(1,total_t3):.1f}%)')

results = {
    'pairs_analyzed': n_pairs,
    'avg_sim_before': round(total_sim_pre/max(1,n_pairs), 4),
    'avg_sim_after': round(total_sim_post/max(1,n_pairs), 4),
    'delta': round((total_sim_post-total_sim_pre)/max(1,n_pairs), 4),
    'trained_natural_match': f'{total_m2}/{total_t2} ({100*total_m2/max(1,total_t2):.1f}%)',
    'original_natural_match': f'{total_m3}/{total_t3} ({100*total_m3/max(1,total_t3):.1f}%)'
}
rpath = os.path.join(out_dir, 'training_results.json')
with open(rpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {rpath}')
