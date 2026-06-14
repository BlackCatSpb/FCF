"""Evaluate checkpoint: val vPPL, vector space metrics, generation samples.
Usage: python eval_metrics.py [checkpoint_tag]
  checkpoint_tag defaults to 'latest' (auto from checkpoint_state.json)
  Examples: '145k', '21k', 'latest'
"""
import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import os, json, time, glob, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator
from eva.symbolic.fcf_config import FCFConfig

CFG = FCFConfig()
BASE = CFG.data_dir
sp = spm.SentencePieceProcessor(model_file=CFG.bpe_model_path)
TEST_SEEDS = ['князь', 'жизнь', 'человек', 'война', 'развитие']
GEN_MAX_WORDS = 25

def clean_sp(s):
    return s.replace('\u2581', ' ').strip()

# ── Resolve checkpoint ──
tag = sys.argv[1] if len(sys.argv) > 1 else 'latest'
if tag == 'latest':
    state_path = CFG.ckpt_state_path
    if os.path.exists(state_path):
        with open(state_path) as f:
            s = json.load(f)
        tag = f"{s['line'] // 1000}k"
    else:
        # fallback: most recent *k.json
        files = glob.glob(os.path.join(BASE, 'concept_space_*k.json'))
        if not files:
            print("No checkpoints found")
            sys.exit(1)
        tag = re.search(r'_(\d+k)\.json$', max(files, key=os.path.getmtime)).group(1)

CS_PATH = CFG.cs_path.replace('.json', f'_{tag}.json')
LAT_PATH = CFG.lattice_path.replace('.json', f'_{tag}.json')
LAT_NPZ = LAT_PATH.replace('.json', '.lattice.npz')

for p in [CS_PATH, LAT_PATH, LAT_NPZ]:
    if not os.path.exists(p):
        print(f"Missing: {p}")
        sys.exit(1)

import shutil, tempfile
tmpdir = tempfile.mkdtemp(prefix='eval_')
bn = tag.split('_')[-1] if '_' in tag else tag
shutil.copy2(CS_PATH, os.path.join(tmpdir, f'concept_space_{bn}.json'))
shutil.copy2(LAT_PATH, os.path.join(tmpdir, f'syntax_lattice_{bn}.json'))
shutil.copy2(LAT_NPZ, os.path.join(tmpdir, f'syntax_lattice_{bn}.lattice.npz'))
meta_src = LAT_PATH.replace('.json', '.meta.json')
if os.path.exists(meta_src):
    shutil.copy2(meta_src, os.path.join(tmpdir, f'syntax_lattice_{bn}.meta.json'))
CS_PATH = os.path.join(tmpdir, f'concept_space_{bn}.json')
LAT_PATH = os.path.join(tmpdir, f'syntax_lattice_{bn}.json')

print(f"Evaluating checkpoint: {tag}")
print(f"  CS:  {CS_PATH}")
print(f"  Lat: {LAT_PATH}")

# ── Load ──
t0 = time.time()
cs = ConceptSpace.load(CS_PATH)
lattice = SyntaxLattice()
lattice.load(LAT_PATH)
gen = CrystalGenerator(cs, sp, lattice)
load_t = time.time() - t0

n_valid = cs.concept_vectors._valid.sum()
n_total = len(cs.concept_vectors._valid)
print(f"  Load: {load_t:.1f}s | {n_valid}/{n_total} valid vectors | {len(lattice.connections)} connections")

results = {'checkpoint': tag}

# ── 1. Val perplexity ──
print("\n--- Val Perplexity ---")
if os.path.exists(CFG.val_corpus_path):
    t0 = time.time()
    eval_result = gen.evaluate(CFG.val_corpus_path, max_lines=CFG.eval_max_lines)
    eval_t = time.time() - t0
    print(f"  PPL={eval_result['perplexity']:.0f} vPPL={eval_result['vec_perplexity']:.0f}")
    print(f"  acc@1={eval_result['accuracy_top1']:.3f} vacc@1={eval_result['vec_accuracy_top1']:.3f}")
    print(f"  ({eval_t:.1f}s, {eval_result.get('n_tokens', 0)} tokens)")
    for k, v in eval_result.items():
        if isinstance(v, (int, float)):
            results[f'val_{k}'] = v

# ── 2. Vector space metrics ──
print("\n--- Vector Space ---")
t0 = time.time()
valid_cids = [c for c in range(cs.vocab_size) if cs.concept_vectors._valid[c]]
n_valid = len(valid_cids)
print(f"  Valid concepts: {n_valid}/{cs.vocab_size}")

# Sample-based cos distribution (full matrix is 146K^2 → impossible)
sample_k = min(2000, n_valid)
rng = np.random.RandomState(42)
sampled = rng.choice(valid_cids, size=sample_k, replace=False)
vecs = np.array([cs.concept_vectors._data[c] for c in sampled], dtype=np.float32)
sims = vecs @ vecs.T
triu = sims[np.triu_indices(sample_k, k=1)]
cos_mean = float(triu.mean())
cos_std = float(triu.std())
cos_min = float(triu.min())
cos_max = float(triu.max())
cos_gt_01 = float((triu > 0.1).mean())
cos_gt_02 = float((triu > 0.2).mean())
cos_gt_05 = float((triu > 0.5).mean())
vec_t = time.time() - t0

print(f"  cos: mean={cos_mean:.4f} std={cos_std:.4f} [{cos_min:.4f}, {cos_max:.4f}]")
print(f"  frac >0.1: {cos_gt_01*100:.1f}%  >0.2: {cos_gt_02*100:.1f}%  >0.5: {cos_gt_05*100:.1f}%")
print(f"  ({vec_t:.1f}s for {sample_k} samples)")

results['vec_cos_mean'] = cos_mean
results['vec_cos_std'] = cos_std
results['vec_cos_min'] = cos_min
results['vec_cos_max'] = cos_max
results['vec_cos_gt_01'] = cos_gt_01
results['vec_cos_gt_02'] = cos_gt_02
results['vec_cos_gt_05'] = cos_gt_05
results['vec_sample_k'] = sample_k

# ── Vector consistency ──
ok, total = 0, 0
for cid in sampled[:500]:
    v = cs.concept_vector(cid)
    if v is not None:
        total += 1
        if abs(np.linalg.norm(v) - 1.0) < 1e-4:
            ok += 1
consistency = ok / max(total, 1)
results['vec_consistency'] = consistency
print(f"  Consistency: {ok}/{total} ({consistency*100:.1f}%)")

# ── 3. Generation samples ──
print("\n--- Generation ---")
gen_samples = []
for seed in TEST_SEEDS:
    try:
        result = gen.generate(seed_word=seed, max_words=GEN_MAX_WORDS)
        text = result['text'].replace('\n', ' ').strip()
        n_words = result['word_count']
        score = result['score']
        gen_samples.append({'seed': seed, 'text': text, 'words': n_words, 'score': score})
        print(f"  [{seed}]({n_words}w, {score:.2f}): {text[:100]}")
    except Exception as e:
        gen_samples.append({'seed': seed, 'text': f'[ERROR: {e}]', 'words': 0, 'score': 0})
        print(f"  [{seed}]: ERROR {e}")

results['gen_samples'] = gen_samples
results['test_seeds'] = TEST_SEEDS

# ── 4. Top-K analysis for test seeds ──
print("\n--- Top-5 neighbours ---")
for seed in TEST_SEEDS:
    cid = sp.PieceToId(seed)
    if cid < 0:
        continue
    if not cs.concept_vectors._valid[cid]:
        print(f"  [{seed}](CID {cid}): invalid vector")
        continue
    top = cs.topk_similar_concepts(cid, k=5)
    names = [clean_sp(sp.IdToPiece(c)) if c < sp.vocab_size() else f'[ID{c}]' for c, _ in top]
    sims = [f'{s:.4f}' for _, s in top]
    print(f"  [{seed}](CID {cid}): {', '.join(f'{n}({s})' for n, s in zip(names, sims))}")
    results.setdefault('topk', {})[seed] = [(int(c), float(s)) for c, s in top]

# ── Save ──
out_path = os.path.join(BASE, f'eval_{tag}.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_path}")

# ── Print markdown summary ──
print("\n" + "=" * 60)
print("MARKDOWN SUMMARY")
print("=" * 60)
print(f"## Eval {tag}")
print(f"| Metric | Value |")
print(f"|--------|-------|")
print(f"| PPL | {results.get('val_perplexity', 'N/A'):} |")
print(f"| vPPL | {results.get('val_vec_perplexity', 'N/A'):} |")
print(f"| acc@1 | {results.get('val_accuracy_top1', 'N/A'):.4f} |")
print(f"| vacc@1 | {results.get('val_vec_accuracy_top1', 'N/A'):.4f} |")
print(f"| cos mean | {cos_mean:.4f} |")
print(f"| cos std | {cos_std:.4f} |")
print(f"| cos >0.1 | {cos_gt_01*100:.1f}% |")
print(f"| cos >0.2 | {cos_gt_02*100:.1f}% |")
print(f"| Consistency | {consistency*100:.1f}% |")
print(f"| Valid vectors | {n_valid}/{cs.vocab_size} |")
print(f"\n### Generation")
for s in gen_samples:
    print(f"- **{s['seed']}** ({s['words']}w): {s['text'][:120]}")
