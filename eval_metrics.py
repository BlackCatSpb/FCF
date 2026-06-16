"""Evaluate checkpoint: val vPPL, vector space metrics, generation samples.
Usage: python eval_metrics.py [checkpoint_tag]
  checkpoint_tag defaults to 'latest' (auto from checkpoint_state.json)
  Examples: '145k', '21k', 'latest'
"""
import sys; sys.path.insert(0, os.path.dirname(__file__))
import os, json, time, glob, re, shutil, tempfile
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


def main():
    """Run full evaluation on a checkpoint."""
    tag = sys.argv[1] if len(sys.argv) > 1 else 'latest'
    if tag == 'latest':
        state_path = CFG.ckpt_state_path
        if os.path.exists(state_path):
            with open(state_path) as f:
                s = json.load(f)
            tag = f"{s['line'] // 1000}k"
        else:
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

    tmpdir = tempfile.mkdtemp(prefix='eval_')
    bn = tag.split('_')[-1] if '_' in tag else tag
    shutil.copy2(CS_PATH, os.path.join(tmpdir, f'concept_space_{bn}.json'))
    shutil.copy2(LAT_PATH, os.path.join(tmpdir, f'syntax_lattice_{bn}.json'))
    shutil.copy2(LAT_NPZ, os.path.join(tmpdir, f'syntax_lattice_{bn}.lattice.npz'))
    meta_src = LAT_PATH.replace('.json', '.meta.json')
    if os.path.exists(meta_src):
        shutil.copy2(meta_src, os.path.join(tmpdir, f'syntax_lattice_{bn}.meta.json'))
    codes_src = CS_PATH.replace('.json', '.codes.npz')
    if os.path.exists(codes_src):
        shutil.copy2(codes_src, os.path.join(tmpdir, f'concept_space_{bn}.codes.npz'))
    CS_PATH = os.path.join(tmpdir, f'concept_space_{bn}.json')
    LAT_PATH = os.path.join(tmpdir, f'syntax_lattice_{bn}.json')

    print(f"Evaluating checkpoint: {tag}")

    t0 = time.time()
    cs = ConceptSpace.load(CS_PATH)
    lattice = SyntaxLattice()
    lattice.load(LAT_PATH)
    gen = CrystalGenerator(cs, sp, lattice)
    print(f"  Load: {time.time()-t0:.1f}s | {sum(cs.concept_vectors.valid)}/{cs.vocab_size} vectors")

    results = {'checkpoint': tag}

    if os.path.exists(CFG.val_corpus_path):
        print("\n--- Val Perplexity ---")
        t0 = time.time()
        eval_result = gen.evaluate(CFG.val_corpus_path, max_lines=CFG.eval_max_lines)
        print(f"  PPL={eval_result['perplexity']:.0f} vPPL={eval_result['vec_perplexity']:.0f}")
        print(f"  acc@1={eval_result['accuracy_top1']:.3f} vacc@1={eval_result['vec_accuracy_top1']:.3f}")
        print(f"  ({time.time()-t0:.1f}s)")
        for k, v in eval_result.items():
            if isinstance(v, (int, float)):
                results[f'val_{k}'] = v

    valid_candidates = [c for c in range(cs.vocab_size) if cs.concept_vectors.valid[c]]
    k = min(2000, len(valid_candidates))
    rng = np.random.RandomState(42)
    if k == 0:
        sampled = np.array([], dtype=int)
    else:
        sampled = rng.choice(valid_candidates, size=k, replace=False)
    vecs = np.array([cs.concept_vectors.data[c] for c in sampled], dtype=np.float32)
    triu = (vecs @ vecs.T)[np.triu_indices(len(sampled), k=1)]
    cos_mean, cos_std = float(triu.mean()), float(triu.std())
    cos_gt_01 = float((triu > 0.1).mean())
    cos_gt_02 = float((triu > 0.2).mean())

    print(f"\n--- Vector Space ---")
    print(f"  cos: mean={cos_mean:.4f} std={cos_std:.4f}  >0.1: {cos_gt_01*100:.1f}%  >0.2: {cos_gt_02*100:.1f}%")

    results.update(vec_cos_mean=cos_mean, vec_cos_std=cos_std,
                   vec_cos_gt_01=cos_gt_01, vec_cos_gt_02=cos_gt_02)

    ok = sum(1 for c in sampled[:500]
             if cs.concept_vector(c) is not None and abs(np.linalg.norm(cs.concept_vector(c))-1.0) < 1e-4)
    results['vec_consistency'] = ok / 500
    print(f"  Consistency: {ok}/500 ({results['vec_consistency']*100:.1f}%)")

    print("\n--- Generation ---")
    gen_samples = []
    for seed in TEST_SEEDS:
        try:
            r = gen.generate(seed_word=seed, max_words=GEN_MAX_WORDS)
            text = r['text'].replace('\n', ' ').strip()
            gen_samples.append({'seed': seed, 'text': text, 'words': r['word_count'], 'score': r['score']})
            print(f"  [{seed}]({r['word_count']}w): {text[:100]}")
        except Exception as e:
            gen_samples.append({'seed': seed, 'text': f'[ERROR: {e}]', 'words': 0, 'score': 0})
            print(f"  [{seed}]: ERROR {e}")
    results['gen_samples'] = gen_samples

    print("\n--- Top-5 neighbours ---")
    for seed in TEST_SEEDS:
        cid = sp.PieceToId(seed)
        if cid < 0 or not cs.concept_vectors.valid[cid]:
            continue
        top = cs.topk_similar_concepts(cid, k=5)
        names = [f"{clean_sp(sp.IdToPiece(c))}({s:.4f})" for c, s in top]
        print(f"  [{seed}](CID {cid}): {', '.join(names)}")
        results.setdefault('topk', {})[seed] = [(int(c), float(s)) for c, s in top]

    out_path = os.path.join(BASE, f'eval_{tag}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
