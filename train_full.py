"""Полное обучение: STDP + CollocationMatrix на full_corpus_ru_clean.txt.
С сохранением каждые 10K, онлайн-прогресс, тестовая генерация.
"""
import sys, os, time, json, argparse, pickle
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from eva.symbolic.fcf_config import FCFConfig, EnvironmentResolver
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator
from eva.symbolic.sp_compat import load_piece_model

CORPUS = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru_clean.txt'
CHECKPOINT_DIR = r'C:\Users\black\OneDrive\Desktop\FCF\checkpoints'
LOG_FILE = r'C:\Users\black\OneDrive\Desktop\FCF\train.log'
STATE_FILE = os.path.join(CHECKPOINT_DIR, 'state.pkl')
META_FILE = os.path.join(CHECKPOINT_DIR, 'meta.json')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Параметры ──
parser = argparse.ArgumentParser()
parser.add_argument('--fresh', action='store_true', help='Fresh start (clear checkpoints)')
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
parser.add_argument('--corpus', type=str, default=CORPUS,
                    help='Path to corpus file (default: full_corpus_ru_clean.txt)')
parser.add_argument('--max-lines', type=int, default=0,
                    help='Stop after N valid lines (0 = whole corpus)')
parser.add_argument('--vocab-size', type=int, default=0)
parser.add_argument('--field-bits', type=int, default=512)
parser.add_argument('--learned-fields', action='store_true')
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--neg-samples', type=int, default=3)
parser.add_argument('--context-window', type=int, default=4)
parser.add_argument('--base-lr', type=float, default=0.001)
parser.add_argument('--fluctuation-amp', type=float, default=0.003)
parser.add_argument('--pmi-gate', type=float, default=0.0)
parser.add_argument('--gen-every', type=int, default=10000,
                    help='Generate test text every N lines (0 = disable)')
parser.add_argument('--gen-max-words', type=int, default=100,
                    help='Max tokens for test generation')
parser.add_argument('--qwen-seed', type=str, default='',
                    help='Path to .npy with Qwen-seeded concept vectors')
args = parser.parse_args()
if args.corpus != CORPUS:
    CORPUS = args.corpus

# ── Число строк в корпусе (кэшируем) ──
CORPUS_LINES_FILE = os.path.join(CHECKPOINT_DIR, 'corpus_lines.txt')
if os.path.exists(CORPUS_LINES_FILE):
    TOTAL_LINES = int(open(CORPUS_LINES_FILE).read().strip())
else:
    TOTAL_LINES = sum(1 for _ in open(CORPUS, 'r', encoding='utf-8'))
    with open(CORPUS_LINES_FILE, 'w') as cf:
        cf.write(str(TOTAL_LINES))
if args.max_lines > 0:
    TOTAL_LINES = min(TOTAL_LINES, args.max_lines)

if args.fresh and os.path.exists(CHECKPOINT_DIR):
    for f_name in os.listdir(CHECKPOINT_DIR):
        f_path = os.path.join(CHECKPOINT_DIR, f_name)
        try:
            if os.path.isfile(f_path):
                os.remove(f_path)
        except Exception:
            pass

_log_fh = open(LOG_FILE, 'a' if args.resume else 'w', encoding='utf-8')
def log(msg=''):
    print(msg)
    _log_fh.write(str(msg) + '\n')
    _log_fh.flush()

TEST_SEEDS = ['на', 'человек', 'большой', 'ходить', 'сегодня',
              'мир', 'время', 'жизнь', 'новый', 'говорить']

cfg = FCFConfig()
env = EnvironmentResolver()
sp = load_piece_model(env.bpe_model_path)

resume_lines = 0
resume_pairs = 0

if args.resume and os.path.exists(STATE_FILE) and os.path.exists(META_FILE):
    log(f'Resuming from {STATE_FILE}...')
    with open(META_FILE, 'r') as mf:
        meta = json.load(mf)
    resume_lines = meta.get('lines', 0)
    resume_pairs = meta.get('pairs', 0)
    log(f'  previous progress: {resume_lines:,} lines, {resume_pairs:,} pairs')
    with open(STATE_FILE, 'rb') as sf:
        state = pickle.load(sf)
    cs = state['cs']
    lattice = state['lattice']
    gen = CrystalGenerator(cs, sp, lattice)
    gen._use_colloc = True
    gen.concept_error = state.get('concept_error', gen.concept_error)
    gen.hormones = state.get('hormones', gen.hormones)
    log('  model state restored.')
else:
    V = args.vocab_size if args.vocab_size else sp.vocab_size()
    log(f'Init ConceptSpace (vocab={V})...')
    cs = ConceptSpace(vocab_size=V, dim=256)
    cs.init_concepts()
    cs.init_homeostasis()
    if args.qwen_seed and os.path.exists(args.qwen_seed):
        log(f'Seeding from Qwen: {args.qwen_seed}')
        n = cs.seed_from_qwen(args.qwen_seed)
        log(f'  {n} vectors loaded')
    else:
        cs.seed_alphabet_basis(sp, seed_multi_char=True)
    if args.learned_fields:
        cs.build_learned_fields(n_field_bits=args.field_bits, sp=None)
    cs.build_collocation_matrix()
    cs.build_multi_level_encoder()

    log('Init lattice + generator...')
    lattice = SyntaxLattice()
    lattice.concept_freq = {i: max(10 - i, 1) for i in range(V)}
    for n in range(2, 5):
        lattice.ngrams[n] = {}
    gen = CrystalGenerator(cs, sp, lattice)
    gen._use_colloc = True

V = cs.vocab_size
log(f'Config:')
log(f'  corpus={CORPUS} ({TOTAL_LINES:,} lines)')
log(f'  vocab={V}, batch={args.batch_size}, lr={args.base_lr}')
log(f'  neg_samples={args.neg_samples}, context_window={args.context_window}')
log(f'  field_bits={args.field_bits}, learned_fields={args.learned_fields}')
log(f'  pmi_gate={args.pmi_gate}, fluctuation={args.fluctuation_amp}')
log(f'  gen_every={args.gen_every}, gen_max_words={args.gen_max_words}')
log(f'  qwen_seed={"yes" if args.qwen_seed else "no"}')
if resume_lines:
    log(f'  resuming from line {resume_lines:,}')
log()

# ── Сохранение чекпоинта ──
def save_checkpoint(total_lines, total_pairs):
    state = {
        'cs': cs,
        'lattice': lattice,
        'concept_error': gen.concept_error,
        'hormones': gen.hormones,
    }
    with open(STATE_FILE, 'wb') as sf:
        pickle.dump(state, sf, protocol=pickle.HIGHEST_PROTOCOL)
    with open(META_FILE, 'w') as mf:
        json.dump({'lines': total_lines, 'pairs': total_pairs}, mf)

# ── Тестовая генерация ──
def run_test_generation(elapsed):
    n_seeds = min(3, len(TEST_SEEDS))
    seeds = list(np.random.RandomState(hash(f'{total_lines}_{time.time()}') % 2**31).choice(TEST_SEEDS, n_seeds, replace=False))
    for sw in seeds:
        try:
            res = gen.generate(seed_word=sw, max_words=args.gen_max_words, use_colloc=True)
            txt = res.text if res and res.text else '(empty)'
            log(f'  [{sw}] {txt[:200]}')
        except Exception as e:
            log(f'  [{sw}] ERROR: {e}')

# ── Основной цикл ──
t0 = time.time()
last_status_time = t0
total_lines = resume_lines
total_pairs = resume_pairs
batch_times = []
lines_to_skip = resume_lines
gen_var_counter = 0

try:
    with open(CORPUS, 'r', encoding='utf-8') as f:
        batch = []

        if lines_to_skip > 0:
            skipped = 0
            for line in f:
                if len(line.strip()) < 10:
                    continue
                skipped += 1
                if skipped >= lines_to_skip:
                    break
            log(f'Skipped {skipped} valid lines, resuming from line {resume_lines}')

        for line_no, line in enumerate(f):
            stripped = line.strip()
            if len(stripped) < 10:
                continue
            batch.append(stripped)

            if len(batch) >= args.batch_size:
                tb = time.time()
                field_gate = 0.2 if args.learned_fields else 0.0
                n = gen.train_batch(batch,
                    base_lr=args.base_lr, use_torch=None,
                    pmi_gate_min=args.pmi_gate,
                    neg_samples=args.neg_samples,
                    context_window=args.context_window,
                    field_gate=field_gate,
                    fluctuation_amp=args.fluctuation_amp)
                batch_times.append(time.time() - tb)
                total_pairs += n
                total_lines += len(batch)
                batch = []

                if args.max_lines > 0 and total_lines >= TOTAL_LINES:
                    log(f'Reached --max-lines {TOTAL_LINES:,}, stopping.')
                    break

                elapsed = time.time() - t0
                lines_this_run = total_lines - resume_lines
                remaining = max(TOTAL_LINES - total_lines, 0)
                rate = lines_this_run / max(elapsed, 0.001)
                avg_b = np.mean(batch_times[-100:]) if batch_times else 0
                l2 = cs.colloc._total_cid[2]
                l3 = cs.colloc._total[3]
                eta = (elapsed / max(lines_this_run, 1)) * remaining if lines_this_run > 0 else 0

                # Онлайн-прогресс: каждые 10 строк
                if total_lines % 10 < args.batch_size:
                    sys.stdout.write(f'\r[{elapsed/60:.1f}min] '
                        f'{total_lines:,}/{TOTAL_LINES:,} '
                        f'{rate:.0f}l/s '
                        f'pairs={total_pairs:,} '
                        f'colloc_l2={l2:,} '
                        f'l3={l3} '
                        f'batch={avg_b:.2f}s '
                        f'ETA={eta/3600:.0f}h      ')
                    sys.stdout.flush()

                # Полный лог + чекпоинт + генерация: каждые 10K строк
                if total_lines % args.gen_every < args.batch_size and total_lines > resume_lines and total_lines > 0:
                    print()
                    log(f'[{elapsed/60:.1f}min] '
                        f'lines={total_lines:,}/{TOTAL_LINES:,} '
                        f'pairs={total_pairs:,} '
                        f'colloc_l2={l2:,} l3={l3} '
                        f'rate={rate:.0f}l/s '
                        f'batch={avg_b:.2f}s '
                        f'ETA={eta/3600:.0f}h')
                    log('Saving checkpoint...')
                    save_checkpoint(total_lines, total_pairs)
                    log(f'Checkpoint saved ({os.path.getsize(STATE_FILE)/1024**2:.0f} MB).')
                    log(f'Test generation (max_words={args.gen_max_words}):')
                    run_test_generation(elapsed)

        if batch:
            tb = time.time()
            field_gate = 0.2 if args.learned_fields else 0.0
            n = gen.train_batch(batch,
                base_lr=args.base_lr, use_torch=None,
                pmi_gate_min=args.pmi_gate,
                neg_samples=args.neg_samples,
                context_window=args.context_window,
                field_gate=field_gate,
                fluctuation_amp=args.fluctuation_amp)
            batch_times.append(time.time() - tb)
            total_pairs += n
            total_lines += len(batch)

except KeyboardInterrupt:
    print()
    log('\nKeyboardInterrupt — saving checkpoint...')
    if total_lines > resume_lines:
        save_checkpoint(total_lines, total_pairs)
        log(f'Checkpoint saved at {total_lines:,} lines.')
    log('Interrupted.')

elapsed = time.time() - t0
log(f'\nDone. {total_lines:,}/{TOTAL_LINES:,} lines, {total_pairs:,} pairs')
log(f'Elapsed: {elapsed/60:.1f} min')
log(f'Colloc L2: {cs.colloc._total_cid[2]:,}')
log(f'Colloc L3: {cs.colloc._total[3]:,}')

if total_lines >= TOTAL_LINES:
    save_checkpoint(total_lines, total_pairs)
    log('Final checkpoint saved.')
log('Done.')
