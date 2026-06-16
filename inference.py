"""Inference Engine — read-only generation from trained checkpoint.
Usage:
  python inference.py --prompt "князь" --checkpoint latest
  python inference.py --batch                (interactive mode)
  python inference.py --neighbours "война"   (top-10 neighbours)
  python inference.py --eval                 (run full eval, save JSON)
"""
import sys; sys.path.insert(0, os.path.dirname(__file__))
import os, json, time, glob, re, shutil, tempfile, argparse
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


def clean_sp(s):
    return s.replace('\u2581', ' ').strip()


class InferenceEngine:
    """Read-only generation engine. Copies checkpoints to temp to avoid
    file-lock conflicts with live training."""

    def __init__(self, tag='latest', config=None):
        self.tag = self._resolve_checkpoint(tag)
        self.config = config or {}
        self.tmpdir = tempfile.mkdtemp(prefix=f'inf_{self.tag}_')
        self.sp = sp
        print(f"InferenceEngine: checkpoint {self.tag}", flush=True)

        t0 = time.time()
        self._copy_checkpoint()
        self._load()
        self.gen = CrystalGenerator(self.cs, self.sp, self.lattice, config=self.config)
        print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    def _resolve_checkpoint(self, tag):
        if tag != 'latest':
            return tag
        state_path = CFG.ckpt_state_path
        if os.path.exists(state_path):
            with open(state_path) as f:
                s = json.load(f)
            return f"{s['line'] // 1000}k"
        files = glob.glob(os.path.join(BASE, 'concept_space_*k.json'))
        if not files:
            raise FileNotFoundError("No checkpoints found")
        return re.search(r'_(\d+k)\.json$', max(files, key=os.path.getmtime)).group(1)

    def _copy_checkpoint(self):
        cs_src = CFG.cs_path.replace('.json', f'_{self.tag}.json')
        codes_src = cs_src.replace('.json', '.codes.npz')
        lat_src = CFG.lattice_path.replace('.json', f'_{self.tag}.json')
        npz_src = lat_src.replace('.json', '.lattice.npz')
        meta_src = lat_src.replace('.json', '.meta.json')

        files = [(cs_src, f'concept_space_{self.tag}.json'),
                 (codes_src, f'concept_space_{self.tag}.codes.npz'),
                 (lat_src, f'syntax_lattice_{self.tag}.json'),
                 (npz_src, f'syntax_lattice_{self.tag}.lattice.npz')]
        for src, name in files:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(self.tmpdir, name))

        if os.path.exists(meta_src):
            shutil.copy2(meta_src, os.path.join(self.tmpdir,
                         f'syntax_lattice_{self.tag}.meta.json'))

        self.cs_path = os.path.join(self.tmpdir, f'concept_space_{self.tag}.json')
        self.lat_path = os.path.join(self.tmpdir, f'syntax_lattice_{self.tag}.json')

    def _load(self):
        self.cs = ConceptSpace.load(self.cs_path)
        self.lattice = SyntaxLattice()
        self.lattice.load(self.lat_path)

    def generate(self, prompt, max_words=30, beam_width=5, query_words=None):
        t0 = time.time()
        result = self.gen.generate(
            seed_word=prompt, max_words=max_words, beam_width=beam_width,
            query_words=query_words if query_words is not None else prompt.split())
        result['time'] = round(time.time() - t0, 2)
        return result

    def retrieve(self, query, k=10):
        """Query → centroid → top-k nearest concepts (RAG retrieval step)."""
        ids = self.sp.encode(query)
        vecs = [self.cs.concept_vector(cid) for cid in ids
                if self.cs.concept_vector(cid) is not None]
        if not vecs:
            return []
        centroid = np.mean(vecs, axis=0).astype(np.float32)
        n = np.linalg.norm(centroid)
        if n > 1e-10:
            centroid /= n

        valid_idxs = np.where(self.cs.concept_vectors.valid)[0]
        if len(valid_idxs) == 0:
            return []
        data = self.cs.concept_vectors.data[valid_idxs]
        sims = data @ centroid
        top_k = min(k, len(sims))
        idx = np.argpartition(-sims, top_k)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        cids = valid_idxs[idx]

        return [(int(cid), clean_sp(self.sp.IdToPiece(int(cid))) if int(cid) < self.sp.vocab_size() else f'[ID{cid}]',
                 float(sims[idx[i]])) for i, cid in enumerate(cids)]

    def neighbours(self, word, k=10):
        cid = self.sp.PieceToId(word)
        if cid < 0 or not self.cs.concept_vectors.valid[cid]:
            return []
        top = self.cs.topk_similar_concepts(cid, k=k)
        return [(int(c), clean_sp(self.sp.IdToPiece(int(c))) if int(c) < self.sp.vocab_size() else f'[ID{c}]',
                 float(s)) for c, s in top]

    def concept_info(self, word):
        cid = self.sp.PieceToId(word)
        vec = self.cs.concept_vector(cid) if cid >= 0 else None
        freq = self.lattice.concept_freq.get(cid, 0)
        return {'cid': cid, 'has_vector': vec is not None, 'freq': int(freq)}

    def run_eval(self, val_path=None, seeds=None, max_lines=3250):
        def clean_sp(s):
            return s.replace('\u2581', ' ').strip()
        val_path = val_path or CFG.val_corpus_path
        seeds = seeds or ['князь', 'жизнь', 'человек', 'война', 'развитие']

        results = {'checkpoint': self.tag}

        if os.path.exists(val_path):
            t0 = time.time()
            ev = self.gen.evaluate(val_path, max_lines=max_lines)
            for k, v in ev.items():
                if isinstance(v, (int, float)):
                    results[f'val_{k}'] = v
            print(f"  vPPL={ev.get('vec_perplexity',0):.0f}  PPL={ev.get('perplexity',0):.0f}  "
                  f"acc@1={ev.get('accuracy_top1',0):.3f}")

        n_valid = sum(self.cs.concept_vectors.valid)
        rng = np.random.RandomState(42)
        sampled = rng.choice([c for c in range(self.cs.vocab_size)
                              if self.cs.concept_vectors.valid[c]],
                             size=min(2000, n_valid), replace=False)
        vecs = np.array([self.cs.concept_vectors.data[c] for c in sampled], dtype=np.float32)
        triu = (vecs @ vecs.T)[np.triu_indices(len(sampled), k=1)]
        cos_mean, cos_std = float(triu.mean()), float(triu.std())
        results.update(vec_cos_mean=cos_mean, vec_cos_std=cos_std,
                       vec_cos_gt_01=float((triu > 0.1).mean()),
                       vec_cos_gt_02=float((triu > 0.2).mean()))

        gen_samples = []
        for seed in seeds:
            try:
                r = self.gen.generate(seed_word=seed, max_words=25)
                gen_samples.append({'seed': seed, 'text': r['text'].replace('\n',' ').strip(),
                                    'words': r['word_count'], 'score': r['score']})
            except Exception as e:
                gen_samples.append({'seed': seed, 'text': f'[ERROR: {e}]', 'words': 0, 'score': 0})
        results['gen_samples'] = gen_samples

        out_path = os.path.join(BASE, f'eval_{self.tag}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_path}")
        return results

    def cleanup(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()


def main():
    parser = argparse.ArgumentParser(description='FCF Inference Engine')
    parser.add_argument('--checkpoint', default='latest', help='Checkpoint tag (e.g. 21k, latest)')
    parser.add_argument('--prompt', help='Single generation prompt')
    parser.add_argument('--max-words', type=int, default=30)
    parser.add_argument('--beam-width', type=int, default=5)
    parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling threshold (1.0=disable)')
    parser.add_argument('--len-norm-alpha', type=float, default=0.7, help='Length normalization exponent')
    parser.add_argument('--block-ngram', type=int, default=4, help='Block repeating n-grams')
    parser.add_argument('--mmi-lambda', type=float, default=0.2, help='MMI penalty strength (0=disable)')
    parser.add_argument('--neighbours', help='Show top-10 neighbours for word')
    parser.add_argument('--retrieve', help='RAG: retrieve top-10 concepts for query')
    parser.add_argument('--query', help='RAG: query words to steer generation (comma-sep)')
    parser.add_argument('--batch', action='store_true', help='Interactive batch mode')
    parser.add_argument('--eval', action='store_true', help='Run full evaluation')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    config = {k: v for k, v in [
        ('top_p', args.top_p),
        ('len_norm_alpha', args.len_norm_alpha),
        ('block_ngram', args.block_ngram),
        ('mmi_lambda', args.mmi_lambda),
    ] if v is not None}

    with InferenceEngine(args.checkpoint, config=config) as eng:
        if args.eval:
            eng.run_eval()
            return

        if args.prompt:
            qw = args.query.split(',') if args.query else None
            r = eng.generate(args.prompt, max_words=args.max_words,
                             beam_width=args.beam_width, query_words=qw)
            print(r['text'].replace('\n', ' ').strip())
            print(f"  [{r['word_count']}w, score={r['score']:.2f}, "
                  f"time={r['time']}s]", flush=True)

        if args.neighbours:
            top = eng.neighbours(args.neighbours)
            print(f"Top-{len(top)} for '{args.neighbours}':")
            for cid, name, sim in top:
                print(f"  {name} (CID {cid}): {sim:.4f}")

        if args.retrieve:
            top = eng.retrieve(args.retrieve)
            print(f"RAG retrieve top-{len(top)} for '{args.retrieve}':")
            for cid, name, sim in top:
                print(f"  {name} (CID {cid}): {sim:.4f}")

        if args.batch:
            print("Interactive mode. Enter prompts (empty line to quit):")
            while True:
                try:
                    line = input('> ').strip()
                    if not line:
                        break
                    qw = args.query.split(',') if args.query else None
                    r = eng.generate(line, max_words=args.max_words,
                                     beam_width=args.beam_width, query_words=qw)
                    print(r['text'].replace('\n', ' ').strip())
                    print(f"  [{r['word_count']}w, score={r['score']:.2f}]", flush=True)
                except (EOFError, KeyboardInterrupt):
                    break


if __name__ == '__main__':
    main()
