#!/usr/bin/env python3
"""Live visual training on large corpus with real-time metrics + test generation.

Usage:
    python train_live.py                       # uses full_corpus_ru.txt
    python train_live.py --corpus path/to/corpus.txt
    python train_live.py --max-lines 1000      # process only 1000 lines

Metrics shown in real-time:
    [PROGRESS]  lines, speed, elapsed, ETA
    [MEMORY]    connections, role_memory, n-grams
    [GATE]      core confidence, modifier field, top cores
    [TEST GEN]  generated text for fixed seeds (every N lines)
"""
import sys, os, time, json, argparse

# Ensure UTF-8 for console output (Russian text)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tqdm

from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.concept_tokenizer import ConceptTokenizer
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator


# ── Configuration ──────────────────────────────────────────────────────────

CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'real_data', 'full_corpus_ru.txt')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_data')
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

TEST_EVERY = 500          # lines between test generations
SAVE_EVERY = 1000         # lines between model saves (named checkpoints)
SAVE_EVERY_LIVE = 50      # lines between live checkpoint overwrites
MAX_WORDS_PER_LINE = 50   # skip lines longer than this

# Fixed test queries to evaluate generation quality during training
TEST_SEEDS = [
    (['какой', 'человек'], 'человек'),
    (['какое', 'время'], 'время'),
    (['какая', 'работа'], 'работа'),
    (['какая', 'история'], 'история'),
    (['какой', 'космос'], 'космос'),
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def fmt_time(sec):
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    if h: return f'{h}h{m:02d}m{s:02d}s'
    if m: return f'{m}m{s:02d}s'
    return f'{s}s'


def save_checkpoint(cs, lattice, gen, line_count, elapsed, ckpt_dir, tag=''):
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f'_{tag}' if tag else ''
    cs_path = os.path.join(ckpt_dir, f'concept_space{tag}.json')
    lattice_path = os.path.join(ckpt_dir, f'syntax_lattice{tag}.json')
    cs.save(cs_path)
    lattice.save(lattice_path)
    meta = {
        'lines': line_count,
        'elapsed_sec': elapsed,
        'connections': len(lattice.connections),
        'role_memory': len(gen.gate.role_memory),
        'concepts': len(cs.cid_list),
        'resolve_cache': len(gen._resolve_cache),
    }
    meta_path = os.path.join(ckpt_dir, f'meta{tag}.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return cs_path, lattice_path


def run_test_generation(gen, cs, seeds, max_words=8):
    """Run test generation queries, return list of (label, text, core)."""
    results = []
    for q_words, seed in seeds:
        try:
            result = gen.generate(seed_word=seed, query_words=q_words, max_words=max_words)
            text = result.get('text', '')
            core_cid = result.get('core_cid')
            core = cs.concept_info.get(core_cid, {}).get('anchor', '?') if core_cid else '?'
            results.append((seed, text, core))
        except Exception as e:
            results.append((seed, f'[ERR: {e}]', '?'))
    return results


# ── Dashboard ──────────────────────────────────────────────────────────────

class Dashboard:
    """Terminal dashboard — static header drawn once, only live values update."""

    NUM_STATIC_LINES = 10  # lines before the update block starts
    UPDATE_ROW_OFFSET = 4  # first row that changes (0-indexed from static top)

    def __init__(self, total_lines=None):
        self.start_time = time.time()
        self.prev_count = 0
        self.prev_time = self.start_time
        self.total_lines = total_lines
        self._rendered = False
        self._template_lines = []

    def _safe_write(self, text):
        try:
            sys.stdout.write(text)
        except UnicodeEncodeError:
            sys.stdout.write(text.encode('ascii', errors='replace').decode('ascii'))

    def render_static(self):
        """Draw the static frame once."""
        lines = []
        lines.append('=' * 62)
        lines.append('  EVA-Ai LIVE TRAINING - concept learning from corpus')
        lines.append('=' * 62)
        lines.append('')
        lines.append('')
        lines.append('')
        lines.append('')
        lines.append('')
        lines.append('')
        lines.append('')
        lines.append('')
        lines.append('')
        lines.append('')
        lines.append('')
        self._safe_write('\033[2J\033[H' + '\n'.join(lines))
        self._rendered = True

    def update(self, line_count, connections, role_mem, ngram_counts,
               gate_info, test_results=None):
        if not self._rendered:
            self.render_static()

        now = time.time()
        elapsed = now - self.start_time
        rate = max((line_count - self.prev_count) / max(now - self.prev_time, 0.001), 0)
        self.prev_count = line_count
        self.prev_time = now

        pct = ''
        if self.total_lines:
            pct = f' ({line_count/self.total_lines*100:.1f}%)'
        eta = ''
        if rate > 0 and self.total_lines:
            remaining = (self.total_lines - line_count) / rate
            eta = f' ETA {fmt_time(remaining)}'

        bi = ngram_counts.get(2, 0)
        tri = ngram_counts.get(3, 0)
        core_conf = gate_info.get('core_confidence', 0)
        mod_size = gate_info.get('modifier_field_size', 0)
        top_cores = gate_info.get('top_cores', [])
        top_str = ', '.join(top_cores[:5]) if top_cores else '(cold)'
        conn_rate = gate_info.get('connections_per_1k', 0)
        status = 'warming' if conn_rate < 1 else 'steady' if conn_rate < 5 else 'accelerating'

        # Row 4: PROGRESS line
        self._safe_write(f'\033[4;1H  [PROGRESS]  {line_count:,} lines{pct}')
        # Row 5: speed line
        self._safe_write(f'\033[5;1H              speed {rate:.1f} l/s  elapsed {fmt_time(elapsed)}{eta}')
        # Row 7: MEMORY line
        self._safe_write(f'\033[7;1H  [MEMORY]    connections={connections:,}  '
                         f'role_mem={role_mem:,}  bi={bi:,}  tri={tri:,}')
        # Row 9: GATE line
        self._safe_write(f'\033[9;1H  [GATE]      conf={core_conf:.2f}  '
                         f'mods={mod_size}  cores=[{top_str}]')
        # Row 10: connections rate
        self._safe_write(f'\033[10;1H              conn/1K={conn_rate:.1f}  ({status})')
        # Row 12: tips
        tip = 'Cold start: caches filling. Speed will increase.' if conn_rate < 1 else \
              f'Checkpoint every {SAVE_EVERY:,} lines, live overwrite every {SAVE_EVERY_LIVE} lines.'
        self._safe_write(f'\033[12;1H  [{"=" * 60}\033[13;1H  {tip}')
        self._safe_write(f'\033[13;1H  {tip}')

        # Test generation (rows 15+)
        if test_results:
            self._safe_write(f'\033[15;1H  [TEST GEN]  --- last {len(test_results)} queries ---')
            for i, (label, text, core) in enumerate(test_results):
                display = text if len(text) <= 90 else text[:87] + '...'
                self._safe_write(f'\033[{16 + i};1H  [{label:10s}] core={core:10s}  txt={display}')

        # Move cursor below output
        self._safe_write(f'\033[{22};1H')
        sys.stdout.flush()


# ── Main ───────────────────────────────────────────────────────────────────

def run_training(args):
    global TEST_EVERY, SAVE_EVERY

    if args.test_every: TEST_EVERY = args.test_every
    if args.save_every: SAVE_EVERY = args.save_every
    if args.max_lines: max_lines = args.max_lines
    else: max_lines = None

    corpus_path = args.corpus or CORPUS_PATH
    model_dir = args.model_dir or MODEL_DIR
    ckpt_dir = args.checkpoint_dir or CKPT_DIR

    print(f'Loading model from {model_dir}...', flush=True)
    t0 = time.time()

    tok = ConceptTokenizer()
    tok.initialize()
    cs = ConceptSpace.load(os.path.join(model_dir, 'concept_space.json'))
    lattice = SyntaxLattice()
    lattice.load(os.path.join(model_dir, 'syntax_lattice.json'))
    gen = CrystalGenerator(cs, tok, lattice)

    load_time = time.time() - t0
    print(f'  Loaded: {len(cs.concept_info):,} concepts, '
          f'{len(lattice.connections):,} connections'
          f'  ({load_time:.1f}s)', flush=True)
    print(f'Corpus: {corpus_path}', flush=True)
    print(f'  Test gen every {TEST_EVERY:,} lines', flush=True)
    print(f'  Save every {SAVE_EVERY:,} lines', flush=True)
    print(f'  Max words/line: {MAX_WORDS_PER_LINE}', flush=True)
    print()

    # Resume from checkpoint
    start_from = 0
    if args.resume:
        print(f'Resuming from {args.resume}...')
        resume_cs = os.path.join(args.resume, 'concept_space.json')
        resume_lat = os.path.join(args.resume, 'syntax_lattice.json')
        if os.path.exists(resume_cs):
            try:
                cs = ConceptSpace.load(resume_cs)
                gen.cs = cs
            except (json.JSONDecodeError, KeyError) as e:
                print(f'  WARNING: corrupted checkpoint ({e}), starting fresh')
        if os.path.exists(resume_lat):
            try:
                lattice.load(resume_lat)
            except (json.JSONDecodeError, KeyError) as e:
                print(f'  WARNING: corrupted lattice ({e}), starting fresh')
        meta_path = os.path.join(args.resume, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            start_from = meta.get('lines', 0)

    # Count total lines
    print('Counting corpus lines...', end=' ', flush=True)
    total_raw = 0
    with open(corpus_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip() and len(line.strip()) > 10:
                total_raw += 1
    total = min(total_raw, max_lines) if max_lines else total_raw
    print(f'{total:,} lines', flush=True)

    dash = Dashboard(total_lines=total)
    line_count = start_from
    checkpoint_connections = len(lattice.connections)
    total_words_processed = 0

    pbar = tqdm.tqdm(total=total, initial=start_from,
                     desc='Training', unit='line',
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    with open(corpus_path, 'r', encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            if max_lines and line_count >= max_lines:
                break

            # Clean and filter
            clean = gen._clean_text(raw_line.strip())
            if len(clean) < 10:
                pbar.update(0)
                continue

            words = clean.split()
            if len(words) < 3 or len(words) > MAX_WORDS_PER_LINE:
                pbar.update(0)
                continue

            # Skip if already processed (resume mode)
            if line_count < start_from:
                line_count += 1
                pbar.update(1)
                continue

            # Train on this line as a sentence
            gen.train_from_text(clean)
            line_count += 1
            total_words_processed += len(words)

            pbar.update(1)

            # Gate info
            gate_info = {}
            if hasattr(gen, 'gate'):
                rm = gen.gate.role_memory
                gate_info['role_memory_size'] = len(rm)
                gate_info['core_confidence'] = min(len(rm) / 100, 1.0) if len(rm) > 0 else 0
                gate_info['modifier_field_size'] = len(gen._modifier_field) if hasattr(gen, '_modifier_field') else 0
                if rm:
                    sorted_cores = sorted(rm.items(),
                        key=lambda x: x[1].get('core_count', 0) if isinstance(x[1], dict) else 0,
                        reverse=True)
                    gate_info['top_cores'] = [
                        cs.concept_info.get(c, {}).get('anchor', f'cid_{c}')
                        for c, _ in sorted_cores[:10]
                        if isinstance(c, int)
                    ]
                else:
                    gate_info['top_cores'] = []
                dc = len(lattice.connections) - checkpoint_connections
                gate_info['connections_per_1k'] = dc / max((line_count - start_from) / 1000, 1)

            ngram_counts = {n: len(ng) for n, ng in lattice.ngrams.items()}

            # Test generation
            test_results = None
            if line_count % TEST_EVERY < 1 and line_count > start_from:
                test_results = run_test_generation(gen, cs, TEST_SEEDS)

            # Dashboard
            dash.update(
                line_count=line_count,
                connections=len(lattice.connections),
                role_mem=len(gen.gate.role_memory),
                ngram_counts=ngram_counts,
                gate_info=gate_info,
                test_results=test_results,
            )

            # Save checkpoint (named, every SAVE_EVERY)
            if line_count % SAVE_EVERY < 1 and line_count > start_from:
                elapsed = time.time() - t0
                save_checkpoint(cs, lattice, gen, line_count, elapsed, ckpt_dir, tag=str(line_count))
                checkpoint_connections = len(lattice.connections)

            # Live checkpoint (overwrite main files every SAVE_EVERY_LINE)
            if line_count % SAVE_EVERY_LIVE < 1 and line_count > start_from:
                cs.save(os.path.join(ckpt_dir, 'concept_space.json'), include_morph=False)
                lattice.save(os.path.join(ckpt_dir, 'syntax_lattice.json'))
                meta = {
                    'lines': line_count,
                    'connections': len(lattice.connections),
                    'role_memory': len(gen.gate.role_memory),
                    'concepts': len(cs.cid_list),
                    'resolve_cache': len(gen._resolve_cache),
                }
                with open(os.path.join(ckpt_dir, 'meta.json'), 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

    pbar.close()

    elapsed = time.time() - t0
    print(f'\n{"=" * 62}')
    print(f'  TRAINING COMPLETE')
    print(f'  {line_count:,} lines processed in {fmt_time(elapsed)}')
    print(f'  {total_words_processed:,} total words')
    print(f'  Connections: {len(lattice.connections):,}')
    print(f'  Role memory: {len(gen.gate.role_memory):,} words')
    print(f'  Final test generations:')
    for label, text, core in run_test_generation(gen, cs, TEST_SEEDS, max_words=10):
        print(f'  [{label:10s}] core={core:12s}  txt={text[:100]}')

    save_checkpoint(cs, lattice, gen, line_count, elapsed, ckpt_dir, tag='final')
    print(f'Final model saved to {ckpt_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live visual training for EVA-Ai')
    parser.add_argument('--corpus', help='Path to corpus file')
    parser.add_argument('--model-dir', help='Model directory')
    parser.add_argument('--checkpoint-dir', help='Checkpoint directory')
    parser.add_argument('--test-every', type=int, default=500)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--max-lines', type=int, help='Max lines to process')
    parser.add_argument('--resume', help='Resume from checkpoint directory')
    args = parser.parse_args()
    run_training(args)
