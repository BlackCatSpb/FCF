"""Full training (146K BPE). Process line-by-line. Checkpoints overwrite."""

import os
# Prevent numpy/BLAS thread contention on single-CPU workloads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys; sys.path.insert(0, os.path.dirname(__file__))
import time, json, os, shutil, argparse, glob, random, re, zlib, math
import numpy as np
import sentencepiece as spm
from sklearn.decomposition import PCA

def _quiet(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(f"[WARN] {func.__name__} failed: {e}", file=sys.stderr)
        return None

def _load_morph(path, sp_path):
    """Load or build MorphVocab (builds from corpus if no cached file)."""
    from eva.symbolic.morph_vocab import MorphVocab
    if os.path.exists(path):
        return MorphVocab.load(path, sp_model_path=sp_path)
    print("  MorphVocab not cached — building from corpus...")
    mv = MorphVocab.build(sp_model_path=sp_path)
    mv.save(path)
    return mv
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator
from eva.symbolic.parameter_optimizer import ParameterOptimizer
from eva.symbolic.fcf_config import FCFConfig

# ── Config ──────────────────────────────────────────────────────
CFG = FCFConfig()
os.makedirs(CFG.data_dir, exist_ok=True)
os.makedirs(CFG.vis_dir, exist_ok=True)

# Redirect stdout to UTF-8 log file (terminal cp1251 can't print ▁)
LOG_FILE = CFG.log_path
class TeeOut:
    def __init__(self):
        self._log_fh = open(LOG_FILE, 'a', encoding='utf-8')
    def write(self, s):
        self._log_fh.write(s)
        self._log_fh.flush()
        try:
            sys.__stdout__.write(s)
        except UnicodeEncodeError:
            sys.__stdout__.write(s.encode('ascii', errors='replace').decode('ascii'))
        sys.__stdout__.flush()
    def flush(self):
        self._log_fh.flush()
        sys.__stdout__.flush()
    def __del__(self):
        self.close()
    def close(self):
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

old_stdout = sys.stdout
sys.stdout = TeeOut()

sp = spm.SentencePieceProcessor(model_file=CFG.bpe_model_path)
V = sp.vocab_size()

# ── Helpers ─────────────────────────────────────────────────────

def load_checkpoint_state():
    if os.path.exists(CFG.ckpt_state_path):
        try:
            with open(CFG.ckpt_state_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('line'), data.get('epoch', 1), data.get('global_step', 0)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[WARN] Corrupted checkpoint state ({e}), starting fresh",
                  file=sys.stderr)
            return None, None, 0
    return None, None, 0

# ── Parse args ──────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', type=int, default=1, help='number of epochs (default: 1). Auto-resume wraps corpus for epoch 2+')
parser.add_argument('--resume', '-r', nargs='?', const='', default='',
                    help='resume from checkpoint. Default: auto from checkpoint_state.json. With Nk: load numbered (e.g. 6k)')
parser.add_argument('--fast', '-f', action='store_true', help='fast mode: higher lr + negative sampling, always fresh')
parser.add_argument('--fresh', action='store_true', help='force fresh start even if checkpoint exists')
parser.add_argument('--max-lines', type=int, default=0, help='limit training to N lines (for testing)')
args = parser.parse_args()
RESUME = args.resume
FAST = args.fast
FRESH = args.fresh or FAST
MAX_LINES = args.max_lines

print(f"vocab_size = {V}")

if FRESH:
    RESUME = None
    ckpt = CFG.ckpt_state_path
    if os.path.exists(ckpt):
        os.remove(ckpt)

if FAST:
    print("FAST mode: base_lr=0.15, neg_samples=3, decay_every=250, eval_every=1000")

resume_global_step = 0
if RESUME is not None:
    if RESUME == '':
        rl, r_epoch, r_global_step = load_checkpoint_state()
        if rl is None:
            print("No checkpoint found — starting fresh.")
            RESUME = None
        else:
            resume_line = rl
            resume_epoch = r_epoch
            resume_global_step = r_global_step
            ckpt_k = resume_line // 1000
            resume_tag = f"{ckpt_k}k"
            print(f"\nAuto-resuming from line {resume_line} (checkpoint_state.json) — loading {resume_tag}")
            cs_path = CFG.cs_path.replace('.json', f'_{resume_tag}.json')
            lat_path = CFG.lattice_path.replace('.json', f'_{resume_tag}.json')
            lat_ok = os.path.exists(lat_path) or os.path.exists(lat_path.replace('.json', '.lattice.npz'))
            if not os.path.exists(cs_path) or not lat_ok:
                print(f"Checkpoint {resume_tag} not found: {cs_path} / {lat_path}")
                cs_path = CFG.cs_path
                lat_path = CFG.lattice_path
                if not os.path.exists(cs_path) or not os.path.exists(lat_path.replace('.json', '.lattice.npz')):
                    print(f"Base checkpoint also not found. Giving up.")
                    sys.exit(1)
                resume_tag = 'base'
                print(f"  (fallback to base paths)")
    else:
        resume_tag = RESUME
        r = RESUME.lower().rstrip('k')
        try: resume_line = int(r) * 1000
        except ValueError:
            print(f"Invalid resume value: {RESUME}")
            sys.exit(1)
        resume_epoch = 1
        print(f"Resuming from '{RESUME}' (line {resume_line})")
        cs_path = CFG.cs_path.replace('.json', f'_{RESUME}.json')
        lat_path = CFG.lattice_path.replace('.json', f'_{RESUME}.json')
        if not os.path.exists(cs_path) or not (os.path.exists(lat_path) or os.path.exists(lat_path.replace('.json', '.lattice.npz'))):
            print(f"Checkpoint not found: {cs_path} / {lat_path}")
            sys.exit(1)

if RESUME is not None:
    cs = _quiet(ConceptSpace.load, cs_path)
    if cs is None:
        print(f"Failed to load ConceptSpace from {cs_path}")
        sys.exit(1)
    lattice = SyntaxLattice()
    load_ng = not FAST
    if _quiet(lattice.load, lat_path, load_ngrams=load_ng) is None:
        print(f"Failed to load SyntaxLattice from {lat_path}")
        sys.exit(1)
    ng_info = f" ({[len(v) for v in lattice.ngrams.values()]} prefixes)" if load_ng else " (ngrams skipped)"
    print(f"  Loaded {len(cs.concept_vectors)} vectors, {len(lattice.concept_freq)} concepts{ng_info}")

    if not any(lattice.ngrams.values()):
        print("  Rebuilding lattice n-grams from corpus...")
        try:
            lattice.build(CFG.corpus_path, sp, max_n=CFG.max_n)
        except Exception as e:
            print(f"FATAL: lattice.build failed: {e}", file=sys.stderr)
            sys.exit(1)

    mv = _load_morph(CFG.morph_vocab_path, CFG.bpe_model_path)
    path_overrides = mv.get_path_overrides()
    if not hasattr(cs, 'H') or cs.H is None:
        print("  Rebuilding H matrix + octree fields from loaded lattice...")
        try:
            cs.build_octree_fields(lattice, n_anchors=CFG.n_anchors, min_lcp=CFG.octree_min_lcp,
                                   gamma=CFG.octree_gamma, path_overrides=path_overrides)
        except Exception as e:
            print(f"FATAL: build_octree_fields failed: {e}", file=sys.stderr)
            sys.exit(1)

else:
    print(f"\nInitializing {V} concepts @ {CFG.dim}D...")
    cs = ConceptSpace(vocab_size=V, dim=CFG.dim)
    try:
        cs.init_concepts()
    except Exception as e:
        print(f"FATAL: init_concepts failed: {e}", file=sys.stderr)
        sys.exit(1)
    cs.init_homeostasis()

    mv = _load_morph(CFG.morph_vocab_path, CFG.bpe_model_path)
    path_overrides = mv.get_path_overrides()

    print("Building SyntaxLattice...")
    lattice = SyntaxLattice()
    try:
        lattice.build(CFG.corpus_path, sp, max_n=CFG.max_n)
    except Exception as e:
        print(f"FATAL: lattice.build failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Octree H + fields...")
    try:
        cs.build_octree_fields(lattice, n_anchors=CFG.n_anchors, min_lcp=CFG.octree_min_lcp,
                               gamma=CFG.octree_gamma, path_overrides=path_overrides)
    except Exception as e:
        print(f"FATAL: build_octree_fields failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Diagnostics ────────────────────────────────────────────────
    # (functions defined unconditionally below; baseline run once)

def mean_cosine_sim(cs, sample=2000):
    all_cids = list(cs.concept_vectors.keys())
    if not all_cids:
        return 0.0, 0.0
    rng_state = np.random.RandomState(42)
    cids = rng_state.choice(all_cids, size=min(sample, len(all_cids)), replace=False).tolist()
    vecs_list = [cs.concept_vectors.get(c) for c in cids]
    vecs_list = [v for v in vecs_list if v is not None]
    vecs = np.array(vecs_list, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    vecs /= norms
    n = len(vecs)
    n_pairs = min(5000, n * (n - 1) // 2)
    rng = np.random.RandomState(42)
    sims = np.empty(n_pairs, dtype=np.float32)
    for k in range(n_pairs):
        i = rng.randint(n); j = rng.randint(n)
        while j == i: j = rng.randint(n)
        sims[k] = float(vecs[i] @ vecs[j])
    return float(np.mean(sims)), float(np.std(sims))

def check_consistency(cs, sample=500):
    all_cids = list(cs.concept_vectors.keys())
    if not all_cids:
        return 0, 0
    rng_state = np.random.RandomState(42)
    cids = rng_state.choice(all_cids, size=min(sample, len(all_cids)), replace=False).tolist()
    ok = 0
    for cid in cids:
        v_stored = cs.concept_vectors.get(cid)
        v_code = cs.fractal.compute_vector(cid)
        if v_stored is not None and v_code is not None and abs(float(np.dot(v_stored, v_code)) - 1.0) < 1e-6:
            ok += 1
    return ok, len(cids)

# ── Baseline ────────────────────────────────────────────────────

# ── Metric pairs (from config) ──────────────────────────────────
CFG.build_metric_pairs(morph_vocab=mv, lattice=lattice, sp=sp)

if RESUME is None:
    mean_sim, std_sim = mean_cosine_sim(cs)
    ok, total = check_consistency(cs)
    print(f"  cos={mean_sim:.4f}±{std_sim:.4f} con={ok}/{total}")

# ── Parameter Optimizer (auto-tuning with config-driven rules) ──

opt = ParameterOptimizer(CFG)
if FAST:
    opt.p['full_lr'].set(CFG.fast_lr)
    opt.p['neg_samples'].set(CFG.fast_neg_samples)

# Restore optimizer state from checkpoint (metric buffers + adapted params)
if RESUME is not None:
    opt_path = CFG.cs_path.replace('.json', f'_{resume_tag}.opt.json')
    if not os.path.exists(opt_path):
        opt_path = CFG.cs_path.replace('.json', '.opt.json')
    if not os.path.exists(opt_path):
        # TN-34: search with data_dir prefix (matches CheckpointManager naming)
        opt_path = os.path.join(CFG.data_dir, f'concept_space_{resume_tag}.opt.json')
    if not os.path.exists(opt_path):
        # Broad fallback: any .opt.json in data_dir
        opt_files = sorted(glob.glob(os.path.join(CFG.data_dir, '*.opt.json')))
        if opt_files:
            opt_path = opt_files[-1]
    if os.path.exists(opt_path):
        with open(opt_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        opt.load_state(saved)
        print(f"Loaded optimizer state")

def get_lr(line_idx):
    if line_idx < CFG.lr_warmup_lines:
        return opt.p['full_lr'].current * (line_idx + 1) / CFG.lr_warmup_lines
    # Cosine annealing with warm restarts
    T_0 = max(CFG.lr_cosine_T0, 1)
    T_mult = CFG.lr_cosine_mult
    # Find which restart period we're in
    t = line_idx - CFG.lr_warmup_lines
    cur_T = T_0
    while t >= cur_T:
        t -= cur_T
        cur_T = int(cur_T * T_mult)
    cos_pct = min(t / max(cur_T, 1), 1.0)
    return opt.p['full_lr'].current * max(0.5 * (1.0 + math.cos(math.pi * cos_pct)), 0.05)


# ── Train/val split ──────────────────────────────────────────────

with open(CFG.corpus_path, 'r', encoding='utf-8') as f:
    all_lines = [l.strip() for l in f if l.strip()]

# Shuffle before split so val set isn't biased toward longest lines
rng = random.Random(42)
rng.shuffle(all_lines)
n_val = max(1, int(len(all_lines) * CFG.val_pct))
val_lines = all_lines[:n_val]
train_lines = all_lines[n_val:]

# ── Curriculum: sort train lines by BPE length (short → easy first) ──
train_lens = [len(sp.encode(l)) for l in train_lines]
train_pairs = sorted(zip(train_lens, train_lines), key=lambda x: x[0])
train_lens = [p[0] for p in train_pairs]
train_lines = [p[1] for p in train_pairs]

# Self-paced learning: re-rank by concept_error at each checkpoint
def _rescore_lines(lines, gen):
    if not hasattr(gen, 'concept_error') or not gen.concept_error:
        return lines
    scores = []
    for line in lines:
        ids = gen._encode_input(line)
        if not ids:
            scores.append(0.0)
        else:
            mean_err = sum(gen.concept_error.get(cid, 0.5) for cid in ids) / len(ids)
            scores.append(mean_err)
    return [l for _, l in sorted(zip(scores, lines))]


from eva.symbolic.checkpoint_manager import CheckpointManager

class TrainingPipeline:
    """Encapsulates the main training loop with batch scheduling, checkpoints, and reporting."""
    def __init__(self, gen, sp, cs, lattice, opt, cfg):
        self.gen = gen
        self.sp = sp
        self.cs = cs
        self.lattice = lattice
        self.opt = opt
        self.cfg = cfg
        self.ppl_history = []
        self.vppl_history = []
        self.last_fluct_lines = 0
        self.global_step = 0
        self.total_pairs_since_last_decay = 0
        self.last_stat_time = 0.0
        self.last_cos_time = 0.0
        self.last_cos_sim = (0.0, 0.0)
        self.ngram_last_total = 0
        # TN-4: Early Stopping
        self.best_score = float('inf')
        self.best_ckpt_name = None
        self.patience_counter = 0
        self.patience = 5
        # TN-12: Switched Evaluation cycle counter
        self._eval_count = 0
        # AM-7: Async Checkpoint Manager
        ckpt_keep = getattr(cfg, 'cleanup_keep', 5)
        self.ckpt_mgr = CheckpointManager(
            data_dir=os.path.dirname(cfg.cs_path),
            cleanup_keep=ckpt_keep)
        # TN-32: preserve curriculum progress after rescore
        self._rescore_cp = None

    def _checkpoint(self, epoch, idx, elapsed, epoch_lines, destab_scale, t_start,
                    epoch_train=None, start_line=None, global_step=0):
        gen = self.gen; cs = self.cs; lattice = self.lattice; opt = self.opt
        rate = idx / max(elapsed, 0.1)
        pct = 100 * idx / epoch_lines
        print(f"\n[{pct:5.1f}%] {idx:7d}L | {rate:4.0f} L/s")
        mean_sim, std_sim = mean_cosine_sim(cs)
        ok, total_c = check_consistency(cs)
        ng_total = sum(len(v) for v in lattice.ngrams.values())
        ng_new = ng_total - self.ngram_last_total
        self.ngram_last_total = ng_total
        print(f"  cos={mean_sim:.4f}±{std_sim:.4f} con={ok}/{total_c} ngrams={ng_total}(+{ng_new})")
        ckpt_k = idx // 1000
        ckpt_name = f"{ckpt_k}k"
        # AM-7: Async checkpoint save (non-blocking)
        self.ckpt_mgr.save(ckpt_name, cs, lattice, opt)
        self.ckpt_mgr.cleanup()
        # TN-31: Save checkpoint_state.json at every checkpoint
        try:
            ckpt = {'line': idx, 'epoch': epoch, 'global_step': global_step, 'timestamp': time.time()}
            with open(self.cfg.ckpt_state_path + '.tmp', 'w', encoding='utf-8') as _f:
                json.dump(ckpt, _f)
            os.replace(self.cfg.ckpt_state_path + '.tmp', self.cfg.ckpt_state_path)
        except Exception as e:
            print(f"[WARN] checkpoint_state save failed: {e}", file=sys.stderr)
        n_upd = cs._update_count
        avg_delta = (cs._total_shift / max(n_upd, 1)) * 1e3
        cs._total_shift = 0.0; cs._update_count = 0
        n_code_out, max_code_abs = cs.check_code_range(bound=self.cfg.code_bound)
        vec_ok, vec_total, vec_max_dev = cs.validate_vector_norms()
        if n_code_out > 0 or vec_max_dev > self.cfg.vec_dev_warn:
            print(f"  CODE_DRIFT n_out={n_code_out} max|code|={max_code_abs:.1f} vec_dev={vec_max_dev:.6f}")
        seed = self.cfg.test_seeds[zlib.crc32(str(idx).encode()) % len(self.cfg.test_seeds)]
        if gen.sp is not None:
            result = gen.generate(seed_word=seed, max_words=self.cfg.gen_max_words)
            txt = result.text.replace('\n', ' ').strip()
            print(f"  gen({seed}): {txt}")
        if idx % (CHECKPOINT_EVERY * 5) == 0:
            _quiet(save_3d_vis, cs, self.sp, ckpt_name)
        eval_vppl = eval_acc1 = eval_vacc1 = None
        if idx > 0 and idx % self.cfg.eval_every_fast == 0:
            self._eval_count += 1
            is_full = (self._eval_count * self.cfg.eval_every_fast) % self.cfg.eval_every_full == 0
            max_lines = self.cfg.eval_full_lines if is_full else self.cfg.eval_fast_lines
            eval_result = _quiet(gen.evaluate, self.cfg.val_corpus_path, max_lines=max_lines)
            if eval_result is not None:
                eval_vppl = eval_result.get('vec_perplexity')
                if is_full:
                    ppl = eval_result.get('perplexity')
                    eval_acc1 = eval_result.get('accuracy_top1')
                    eval_vacc1 = eval_result.get('vec_accuracy_top1')
                    self.ppl_history.append((idx, ppl))
                    self.vppl_history.append((idx, eval_vppl))
                    ppl_trend = ''
                    if len(self.ppl_history) >= 2:
                        d = ppl - self.ppl_history[-2][1]
                        ppl_trend = f" {'+' if d > 0 else ''}{d:.0f}"
                    print(f"  PPL={ppl:.0f}{ppl_trend} acc@1={eval_acc1:.3f} vPPL={eval_vppl:.0f} vacc@1={eval_vacc1:.3f}")
                    # TN-4: Early Stopping + Best Checkpoint
                    score = eval_vppl + (1.0 - eval_vacc1 if eval_vacc1 is not None else 0) * 50
                    if score < self.best_score:
                        self.best_score = score
                        self.best_ckpt_name = ckpt_name
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                else:
                    self.vppl_history.append((idx, eval_vppl))
                    print(f"  vPPL={eval_vppl:.0f} (fast)")
        opt.step(mean_cos=mean_sim, std_cos=std_sim, delta=avg_delta, ng_new=ng_new,
                 vec_ppl=eval_vppl, acc1=eval_acc1, vacc1=eval_vacc1)
        # T-B1: Self-paced learning rescore
        if epoch_train is not None and start_line is not None:
            remaining = idx - start_line + 1
            if remaining > 0 and remaining < len(epoch_train):
                self._rescore_cp = _curriculum_p(idx + 1)
                epoch_train = _rescore_lines(epoch_train[idx + 1:], gen)
                idx = -1; start_line = 0
        if self.patience_counter >= self.patience or opt._full_stuck_counter >= 5:
            print(f"  Early stopping: patience={self.patience_counter}/{self.patience}, "
                  f"stuck={opt._full_stuck_counter}")
            self.ckpt_mgr.wait()
            import sys; sys.exit(0)
        return idx, start_line, epoch_train

# Continuous curriculum: ramp max_len, context_window, neg_samples over first fraction
CURICULUM_FRACTION = 0.20
CURICULUM_MIN_LEN = 16
TOTAL_LINES_GLOBAL = len(train_lines)

def _curriculum_p(idx):
    return min(idx / max(TOTAL_LINES_GLOBAL * CURICULUM_FRACTION, 1), 1.0)

def _curriculum_max_len(idx, max_total=10**9):
    p = _curriculum_p(idx)
    return int(CURICULUM_MIN_LEN + (max_total - CURICULUM_MIN_LEN) * p)

def _effective_cp(idx, pipeline):
    cp = _curriculum_p(max(idx, 0))
    if pipeline._rescore_cp is not None:
        if cp >= pipeline._rescore_cp:
            pipeline._rescore_cp = None
        else:
            cp = pipeline._rescore_cp
    return cp

if MAX_LINES > 0:
    train_lines = train_lines[:MAX_LINES]
    print(f"  Limited to {len(train_lines)} lines (--max-lines={MAX_LINES})")

print(f"  {len(train_lines)} train, {len(val_lines)} val")
print(f"  Shortest: {train_lens[0]} BPE tokens, Longest: {train_lens[-1]} BPE tokens")
print(f"  Median: {train_lens[len(train_lens)//2]} BPE tokens")
with open(CFG.val_corpus_path, 'w', encoding='utf-8') as f:
    for l in val_lines:
        f.write(l + '\n')

# ── STDP Training (line by line) ────────────────────────────────

LIVE_REFRESH = 1.0  # seconds between live status updates
COS_REFRESH = 5.0   # seconds between cos/pair recomputation

print("STDP training...")
gen = CrystalGenerator(cs, sp, lattice)
gen.train_lr = opt.p['full_lr'].current
t_start = time.time()

total_lines = len(train_lines)
CHECKPOINT_EVERY = CFG.checkpoint_every
EVAL_EVERY_F = CFG.eval_every_fast if FAST else CFG.eval_every_slow
FLUCTUATE_EVERY = CFG.fluctuate_every
n_trained = 0
ngram_last_total = 0
ppl_history = []
vppl_history = []
last_stat_time = 0.0
last_cos_time = 0.0
last_cos_sim = (0.0, 0.0)
last_fluct_lines = 0

global_step = resume_global_step if RESUME is not None else 0
last_decay_pairs = 0
total_pairs_since_last_decay = 0

def live_status(text):
    """Write one-line status to terminal only — \r updates in place."""
    try:
        sys.__stdout__.write('\r' + text)
    except UnicodeEncodeError:
        safe = text.encode('cp1251', errors='replace').decode('cp1251')
        sys.__stdout__.write('\r' + safe)
    sys.__stdout__.flush()

def save_3d_vis(cs, sp, checkpoint_name):
    vis_dir = CFG.vis_dir

    cids = sorted(cs.concept_vectors.keys())
    if len(cids) == 0:
        return

    # Build matrix 32K×384
    X = np.array([cs.concept_vectors[c] for c in cids], dtype=np.float32)
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    # Randomized SVD for speed
    pca = PCA(n_components=3, random_state=0)
    proj = pca.fit_transform(Xc)  # N×3

    # Rescale to fill [-1, 1] cube
    scale = np.max(np.abs(proj))
    if scale > 0:
        proj = proj / scale

    # Build JSON
    codes = cs.fractal.codes
    tokens = []
    for i, cid in enumerate(cids):
        tok = sp.IdToPiece(cid) if cid < sp.vocab_size() else f'[ID{cid}]'
        f = float(np.linalg.norm(codes.get(cid, np.zeros(384))))
        tokens.append({
            't': tok, 'x': float(proj[i, 0]), 'y': float(proj[i, 1]), 'z': float(proj[i, 2]),
            'f': round(f, 1), 'id': int(cid)
        })

    json_path = os.path.join(vis_dir, f'points_{checkpoint_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False)

    # Latest copy for viewer.html
    latest_path = os.path.join(vis_dir, 'points_latest.json')
    if json_path != latest_path:
        shutil.copy2(json_path, latest_path)

    # Write HTML if not exists
    html_path = os.path.join(vis_dir, 'viewer.html')
    if not os.path.exists(html_path):
        _write_viewer_html(html_path)

    return json_path

def _write_viewer_html(path):
    # Template lives in real_data/viewer_template.html
    with open(os.path.join(CFG.data_dir, 'viewer_template.html'), 'r', encoding='utf-8') as f:
        TEMPLATE = f.read()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(TEMPLATE)

def _final_save(cs, lattice, opt, epoch, total_lines, global_step=0):
    """Save all training state. Each operation protected individually."""
    for _op_name, _op_fn, _op_args in [
        ('cs.save', cs.save, (CFG.cs_path,)),
        ('lattice.save', lattice.save, (CFG.lattice_path,)),
    ]:
        try:
            _op_fn(*_op_args)
        except Exception as e:
            print(f"[WARN] {_op_name} failed: {e}", file=sys.stderr)
    try:
        state = opt.save_state()
        with open(CFG.cs_path.replace('.json', '.opt.json'), 'w', encoding='utf-8') as _of:
            json.dump(state, _of)
    except Exception as e:
        print(f"[WARN] opt.save_state failed: {e}", file=sys.stderr)
    try:
        ckpt = {'epoch': epoch, 'line': total_lines, 'global_step': global_step, 'timestamp': time.time()}
        with open(CFG.ckpt_state_path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(ckpt, f)
        os.replace(CFG.ckpt_state_path + '.tmp', CFG.ckpt_state_path)
    except Exception as e:
        print(f"[WARN] ckpt_state save failed: {e}", file=sys.stderr)
    for f in glob.glob(os.path.join(CFG.data_dir, 'points_*.html')):
        try:
            if os.path.isfile(f):
                os.remove(f)
        except:
            pass

# Determine starting line and epoch
start_line = resume_line if RESUME is not None else 0
total_epochs = args.epochs
current_epoch = resume_epoch if RESUME is not None else 1

if RESUME is not None:
    print(f"Resuming epoch {current_epoch} at line {start_line}")

_ckpt_epoch = current_epoch
idx = start_line
# AM-15/T-B2: Activate TrainingPipeline
pipeline = TrainingPipeline(gen, sp, cs, lattice, opt, CFG)
pipeline.ngram_last_total = sum(len(v) for v in lattice.ngrams.values())
try:
    for epoch in range(current_epoch, total_epochs + 1):
        _ckpt_epoch = epoch
        pipeline.total_pairs_since_last_decay = 0
        if epoch > current_epoch:
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch} / {total_epochs}")
            print(f"{'='*60}")
            # Reset start_line for new epoch
            start_line = 0
            # Re-read corpus with fresh decay
            destab_pct = 0.0  # fresh destab for new epoch

        # Continuous curriculum: max_len ramps from CURICULUM_MIN_LEN → unlimited
        epoch_train = train_lines
        epoch_lines = total_lines

        bs_curve = lambda i: int(CFG.batch_size_start + (CFG.batch_size_end - CFG.batch_size_start) * _curriculum_p(i))
        BATCH_SIZE = bs_curve(idx)
        batch_buffer = []
        batch_lr = None
        batch_destab = 0.0

        idx = start_line
        while idx < len(epoch_train):
            line = epoch_train[idx]
            global_step += 1
            if not line:
                idx += 1
                continue

            # TN-32: effective curriculum (preserved across rescore)
            _eff_cp = _effective_cp(idx, pipeline)
            max_len = int(CURICULUM_MIN_LEN + (10**9 - CURICULUM_MIN_LEN) * _eff_cp)
            if len(sp.encode(line)) > max_len:
                idx += 1
                continue

            # LR warmup
            gen.train_lr = get_lr(idx)

            destab_pct = min(global_step / max(round(opt.p['destab_decay_lines'].current), 1), 1.0)
            destab_from = max(CFG.destab_scale_start, CFG.destab_scale_end)
            destab_to = min(CFG.destab_scale_start, CFG.destab_scale_end)
            destab_scale = destab_from + (destab_to - destab_from) * destab_pct

            batch_buffer.append(line)
            batch_lr = gen.train_lr
            batch_destab = destab_scale
            BATCH_SIZE = int(CFG.batch_size_start + (CFG.batch_size_end - CFG.batch_size_start) * _eff_cp)

            if len(batch_buffer) < BATCH_SIZE and idx < start_line + len(epoch_train) - 1:
                # Check if periodic tasks are due — if so, flush early
                is_fluct_due = (idx + 1 - last_fluct_lines) >= FLUCTUATE_EVERY
                is_decay_due = total_pairs_since_last_decay >= CFG.decay_every_pairs
                if not is_fluct_due and not is_decay_due:
                    continue

            # Flush batch
            _bt = time.time()
            cp = _eff_cp
            cw_target = int(round(opt.p['context_window'].current))
            ns_target = int(round(opt.p['neg_samples'].current))
            pg_target = opt.p['pmi_gate_min'].current
            cw_ramp = max(1, int(round(cw_target * cp)))
            ns_ramp = int(round(ns_target * cp))
            pg_ramp = pg_target * cp
            n_pairs = gen.train_batch(batch_buffer, pmi_strength=opt.p['pmi_strength'].current,
                pmi_gate_min=pg_ramp,
                base_lr=batch_lr,
                neg_samples=ns_ramp,
                context_window=cw_ramp,
                inh_strength=opt.p['inh_strength'].current,
                inh_threshold=opt.p['inh_threshold'].current,
                neg_lr_ratio=CFG.neg_lr_ratio, field_gate=CFG.field_gate,
                use_torch=CFG.use_torch, destab_scale=batch_destab,
                gradient_noise_scale=opt.p['gradient_noise_scale'].current,
                momentum_mu=CFG.momentum_mu)
            total_pairs_since_last_decay += n_pairs
            _batch_ms = (time.time() - _bt) * 1000
            _n = len(batch_buffer)
            if 'batch_log' not in locals() or batch_log is None:
                csv_path = os.path.join(CFG.data_dir, '_batch_timing.csv')
                batch_log = open(csv_path, 'a', encoding='utf-8')
                if os.path.getsize(csv_path) == 0:
                    batch_log.write('idx,lines,batch_ms,speed_lps\n')
            batch_log.write(f'{idx},{_n},{_batch_ms:.0f},{_n/max(_batch_ms,1)*1000:.0f}\n')
            batch_log.flush()
            n_trained += len(batch_buffer)
            batch_buffer = []
            now = time.time()
            elapsed = now - t_start

            # ---- Periodic tasks (line-based) ----
            if idx > 0 and idx - last_fluct_lines >= FLUCTUATE_EVERY:
                cs.fluctuate_fractal(fluctuation_amp=opt.p['fluctuation_amp'].current,
                                     decay=opt.p['decay_rate'].current,
                                     repel_strength=opt.p['repel_strength'].current,
                                     generator=gen)
                last_fluct_lines = idx

            if idx > 0 and total_pairs_since_last_decay >= CFG.decay_every_pairs:
                lattice.decay_all(rare_concept_protect=True, rare_threshold=3)
                lattice.decay_connections()
                cs.decay_usage(decay=0.98, rare_protect=True)
                total_pairs_since_last_decay = 0

            # ---- Live status (every ~1 second on terminal) ----
            if now - last_stat_time >= LIVE_REFRESH:
                rate = idx / max(elapsed, 0.1)
                pct = 100 * idx / epoch_lines
                if rate >= 0.1:
                    eta = (epoch_lines - idx) / rate
                    eta_h, eta_m = int(eta // 3600), int((eta % 3600) // 60)
                    eta_s = f"ETA {eta_h}h{eta_m:02d}m"
                else:
                    eta_s = "ETA ---"

                if now - last_cos_time >= COS_REFRESH:
                    last_cos_sim = mean_cosine_sim(cs)
                    last_cos_time = now

                mean_sim, std_sim = last_cos_sim
                live_status(f"[{pct:4.1f}%] {idx:6d}L | {rate:4.0f} L/s | {eta_s} | "
                            f"{elapsed/60:.0f}min cos={mean_sim:.4f}±{std_sim:.4f}")
                try:
                    with open(CFG.status_path + '.tmp', 'w', encoding='utf-8') as _sf:
                        json.dump({'line': idx, 'total': epoch_lines, 'pct': pct,
                                   'rate': rate, 'eta_s': eta_s, 'elapsed_min': elapsed/60,
                                   'cos_mean': mean_sim, 'cos_std': std_sim}, _sf)
                    os.replace(CFG.status_path + '.tmp', CFG.status_path)
                except Exception:
                    pass
                last_stat_time = now

            # ---- Line-based checkpoint + full report (AM-15/T-B2: via pipeline) ----
            if idx > 0 and idx % CHECKPOINT_EVERY == 0:
                pipeline.global_step = global_step
                result = pipeline._checkpoint(epoch, idx, elapsed, epoch_lines, destab_scale, t_start, epoch_train, start_line, global_step)
                if result is not None:
                    idx, start_line, epoch_train = result
                gen.train_lr = pipeline.opt.p['full_lr'].current
                if pipeline.opt._full_stuck_counter >= 5 and last_fluct_lines != idx:
                    print(f"  FULL_STUCK — forcing fluctuate")
                    cs.fluctuate_fractal(fluctuation_amp=opt.p['fluctuation_amp'].current,
                                         decay=opt.p['decay_rate'].current,
                                         repel_strength=opt.p['repel_strength'].current,
                                         generator=gen)
                    last_fluct_lines = idx
                print()
            idx += 1

except KeyboardInterrupt:
    print("\n\n[EVA] Training interrupted — saving checkpoint...")
    try:
        _final_save(cs, lattice, opt, _ckpt_epoch, idx, global_step)
    except BaseException:
        pass
    print("[EVA] Checkpoint saved. Exiting.")
    sys.exit(0)
finally:
    try:
        if 'batch_log' in locals() and batch_log is not None:
            batch_log.close()
    except Exception:
        pass
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    sys.stdout = old_stdout

# Final save
_final_save(cs, lattice, opt, _ckpt_epoch, idx, global_step)

# ── Final diagnostics ───────────────────────────────────────────

print("--- Final ---")
mean_sim, std_sim = mean_cosine_sim(cs)
ok, total = check_consistency(cs)
print(f"  cos={mean_sim:.4f}±{std_sim:.4f} con={ok}/{total}")

print("--- Generation ---")
for seed in CFG.test_seeds:
    result = gen.generate(seed_word=seed, max_words=CFG.gen_max_words)
    txt = result.text
    print(f"  [{seed}] {txt}  ({result.score:.2f})")

t_total = time.time() - t_start
print(f"  {n_trained} lines in {t_total:.0f}s ({n_trained/t_total:.0f} L/s)")
print("Done.")


