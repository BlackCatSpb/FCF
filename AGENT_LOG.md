# FCF — Agent Log

## Session: Architecture overhaul (2026-06-11)

### Context
- Full 145K training completed: PPL=25926, vecPPL=27990, acc@1=0.106, vacc@1=0.000
- 21 bugs from ARCHITECTURE_REVIEW.md fixed and pushed (170c1cd, 0468197, e6bd7a6)
- User wants me to act as lead developer, execute autonomously, think deeply
- **New architecture direction**: Finite anchor matrix + hierarchical field mask
  instead of STDP-only training. Mask = binary field activation + shift, not separate attention.
- QWEN3-4B OpenVINO model analyzed (config.json, OpenVINO XML, tokenizer) for insights:
  - 146K vocab, 36 layers, GQA, RoPE, SwiGLU, RMSNorm
  - Key insight: 36 sequential layers → 3 parallel subspaces (z_c, z_a, z_m)
  - RoPE → z_a rotation for position encoding
  - GQA → anchor field sharing (N_a=1024 anchors)

### Core Problem
Vacc@1=0.000 because vectors remain a "random gas" (cos=0.0003±0.0532).
Root cause: STDP is purely Hebbian (pull together) with no effective repulsion:
- neg_samples=0 (default) — no negative sampling
- inh_threshold=0.35 (default) — at cos_std=0.053, est_frac=0.0004 (0.04% pairs inhibited)
- PMI gate only weights, doesn't filter — noise pairs still contribute

### Phase 1: Architecture fixes (DONE)
- Parameter defaults: neg_samples=0→2, inh_threshold=0.35→0.10, pmi_gate_min=0.05→0.20
- PMI filter: skip pairs with PMI ≤ threshold
- gen_quick scripts for interactive generation

### Phase 2: Technical debt (DONE)
- Lazy n-gram load in SyntaxLattice (25s→<2s for generation)
- reset_fractal.py rewritten, requirements.txt cleaned
- Checkpoint fix: cleanup by mtime, periodic base save every 5K

### Phase 3: Hierarchical FractalField + H matrix (DONE, pushed as 0610f20, adc48cf)
#### Architecture
- **Code split**: z = [z_c(256) | z_a(128) | z_m(128)]
  - z_c: identity (slow, lr_c=0.01)
  - z_a: attention shift (fast, lr_a=1.0)
  - z_m: meta-gate (medium, lr_m=0.1) — modulates lr and inhibition
- **H matrix**: SyntaxLattice.build_anchor_matrix() → sparse CSR 1024×1024 (min_pmi=4.5, 0.41% dense)
  - Top-1024 most frequent concepts as anchors
  - PMI from 2-gram co-occurrence
- **PMI fields**: Each concept's field = self + anchors with PMI > 4.5
  - Field sizes: min=1, max=20, mean=4.8 (out of 1024)
  - Stored as packed uint8[128] per concept
  - Saved/loaded in npz checkpoint (backward compat)
- **Field-aware training**: delta_v → delta_z @ basis.T, subspace learning rates, meta-gate
- **Field mask in generation**: context field bits OR'd, candidates with overlap boosted
- **No BMSSP**: N_a=1024 graph too dense for hierarchical expansion. Direct PMI fields instead.

#### Files changed
- `syntax_lattice.py`: build_anchor_matrix()
- `concept_space.py`: H storage, build_fields_from_lattice(), field_bits API, field serialization
- `crystal_generator.py`: field-aware train_from_text (subspace projection, meta-gate), field mask in _branch
- `train_full.py`: H+fields initialization in fresh + resume paths

#### Performance
- H matrix build: ~1s
- PMI fields for 32K concepts: ~6s (precomputed co-occurrence index)
- Training one sentence: ~0.016s

### Current Training (started 2026-06-11 22:30 ET)
- **Mode**: `python train_full.py --fast`
- **Status**: Running in separate terminal window
- **First checkpoint** (500 lines) shows cos=0.0672 at 1.4% — same as old architecture at this stage
- **Expected**: Vector separation should appear gradually as subspace-specific learning accumulates
- **Run-time**: ~150K lines × 5 L/s ≈ 8 hours (fast mode)
- **Log**: `real_data/train_log.txt`

### Key Decisions
- PMI filter: pairs with PMI ≤ 0 are noise → skip entirely. Only PMI-positive pairs contribute.
- inh_threshold=0.10 targets ~5% of pairs at current cos_std=0.053
- neg_samples=3 (fast mode) at 50% of pull LR
- H matrix built with min_pmi=4.5 (PMI P50 in resulting matrix = 5.28)
- No BMSSP — graph with N_a=1024 too dense. Direct PMI fields more discriminative.
- SP vocab is 16000 (bpe_ru.model) NOT 32000 — train_full.py uses bpe_ru_32k.model for 32K vocab

## Session: PyTorch GPU benchmark + integration (2026-06-12)

### Context
- After octree encoding integration, benchmark showed numpy loop at 588 pairs/s
- Hypothesis: PyTorch GPU batching would give 100x+ speedup

### Results
- **Synthetic benchmark (500 lines, 51K pairs)**: GPU 101,649 pairs/s vs numpy 588 pairs/s (173x!)
- **Real corpus (200 lines, ~60 pairs/line)**: GPU 8.8 lines/s vs numpy 9.8 lines/s (GPU is SLOWER)
- **Root cause**: GPU kernel launch + CPU↔GPU transfer overhead dominates for small batch (60 pairs)
  - 173x was for 51K pairs in one batch, not 60 pairs × 200 times

### Changes made
- `crystal_generator.py`: added `_ensure_torch()`, merged `train_from_text` with `use_torch` param
- `train_full.py`: calls `train_from_text(..., use_torch=True)` 
- Default remains `use_torch=False` — GPU doesn't help per-line

### Key lesson
GPU batching only helps when the batch is large enough to amortize kernel launch + transfer overhead. For `train_from_text` (60 pairs/line, ~20 tokens), the optimal is numpy CPU with per-pair Python loop. GPU would need either:
1. **Cross-line batching**: buffer pairs from N lines, process as one GPU batch (changes learning dynamics)
2. **Full GPU training loop**: keep all vectors on GPU, do STDP updates there (major refactor)
3. **Cython/Numba**: JIT-compile the hot loop for Python overhead elimination

### Next
- [ ] Profile the actual per-line Python overhead (pair collection vs compute)
- [ ] Try Numba JIT for the inner STDP accumulation loop
- [ ] Or: buffer line pairs across a mini-batch (e.g., 50 lines at once) and apply accumulated updates

## Integration into FCF (2026-06-12)
- **New module**: `eva/symbolic/fractal_encoding.py` — digits(), path(), lcp(), H_weighted(), path_index()
- **ConceptSpace.build_octree_fields()** added — replaces PMI pipeline:
  - Selects top 1024 anchors by frequency
  - Precomputes octree paths for all concepts
  - Builds H matrix (CSR) from H_weighted (23.3% dense)
  - Builds field_bits via prefix grouping: O(n) instead of O(n²)
  - Default min_lcp=2 (anchors share at least 2 octree levels)
- **train_full.py** updated to call `build_octree_fields` instead of PMI pipeline

### Performance
- `build_octree_fields`: 0.67s (32K concepts, 1024 anchors)
- `CS.load`: 0.75s (was 204s — fixed NpzFile __getitem__ bottleneck)
- Field density: mean=37.1 bits/concept (vs 3.4 in PMI model)
- Field-gate distribution: 95.7% pairs fw=0.1, 4.3% pairs fw=15.7 mean

### Found bugs fixed
1. `from_dict` QR re-encoding: `codes_mat @ basis.T` → `codes_mat @ basis` (wrong dims, path never triggered due to tight 1e-5 threshold)
2. NpzFile repeated `__getitem__` access 10000x slower than pre-extracted array (Windows/numpy quirk)
3. Relaxed orthogonality check threshold: 1e-5 → 1e-3; batched re-encoding via matmul

### Test results (synthetic)
- **LCP distribution**: 86.2% LCP=0, 11.9% LCP=1, 1.6% LCP=2, 0.2% LCP=3 — pure geometric (1/8)^k
- **H>0**: 13.7% vs theoretical 12.5% (nested) vs 88.2% (old) vs 0.1% (flat PMI)
- **Cosine per LCP**: exactly 1−0.25^{LCP}: LCP=0→0, LCP=1→0.75, LCP=2→0.94, LCP=3→0.98
- **Field-overlap vs LCP**: 100% correlation — LCP<2 → ov=0, LCP≥2 → ov>0

### Files modified
- `eva/symbolic/fractal_encoding.py` — new module
- `eva/symbolic/concept_space.py` — build_octree_fields + load fix + QR fix
- `eva/symbolic/crystal_generator.py` — _ensure_torch + use_torch in train_from_text
- `train_full.py` — switch to build_octree_fields + use_torch
- `AGENT_LOG.md` — this log

### Architecture decisions
- Octree encoding is deterministic (no PMI statistics needed)
- H matrix is derivable from LCP (no storage needed long-term)
- Field_bits compact: 128 bytes/concept for 1024 anchors
- GPU not effective for per-line training (batch too small)

## Session: PMI gate optimization — cached prefix totals (2026-06-12)

### Profile results (500 lines, 57804 pairs)
- **PMI gate: 560.7ms** (9.7us/pair) — **65% of total time** — THE BOTTLENECK
- STDP update: 169.8ms
- Field gate: 70.7ms
- Pair build: 59.8ms (negligible)

### Root cause
`_pmi_weight()` calls `sum(prefix_counter.values())` for EVERY pair. With 32K prefixes and ~2000 avg next-tokens per prefix, this sums 115M values across 58K pairs.

### Fix
- Added `_prefix_total` and `_skip2_total` caches to `SyntaxLattice.__init__`
- `_refresh_prefix_totals()` precomputes `sum(counter.values())` per prefix after build/load
- `_pmi_weight` now uses O(1) dict lookup instead of O(K) sum

### Changes
- `syntax_lattice.py`: `_prefix_total`, `_skip2_total` dicts + `_refresh_prefix_totals()` called at end of `build()` and `load()`
- `crystal_generator.py`: `_pmi_weight` uses `lattice._prefix_total` / `lattice._skip2_total`
- Remove duplicate `train_from_text` method definition (was dead code)

### Speedup
- PMI gate: **560ms → 58ms** (9.7× faster)
- End-to-end 200-line training: **345ms/line → 95ms/line** (3.6× faster)
- Total 145K lines: **~13.9 hours → ~3.8 hours** (at current speed)

### Next bottleneck
STDP update, specifically **lateral inhibition** (~170ms → now ~60% of time). Options:
1. ~~Batch concept vectors into numpy array~~ (done, _vec_array + get_vec/set_vec)
2. Numba JIT for the inner STDP accumulation loop
3. Cross-line mini-batch for GPU efficiency

## Session: Inhibition optimization — array-backed vectors + fast sampling (2026-06-12)

### Profile (200 lines, with PMI cache fix)
Sub-component breakdown of `train_from_text`:
| Component | Time | % |
|-----------|------|---|
| Pair building + gates | 0.5ms/line | 0.5% |
| Delta accumulation | 0.6ms/line | 0.6% |
| apply_code_update | 1.3ms/line | 1.4% |
| **Lateral inhibition** | **88ms/line** | **97.5%** |

**Root cause**: `_lateral_inhibition_fractal()` called `np.random.permutation(32000)` for EVERY gen_cid (6827×), plus `np.array([concept_vectors[c] for c in ...])` list comprehension with 200 dict lookups. Each `_apply_vector_update` call then did 2× matrix-vector multiply (fractal code round-trip).

### Fixes
1. **`_vec_array` + `_cid_to_idx`** (ConceptSpace): O(1) array-backed vector access via `get_vec(cid)` / `set_vec(cid, v)`. `_lateral_inhibition_fractal` now does `_vec_array[sampled_indices]` instead of `np.array([dict[c] for c in ...])`.
2. **Fast random sampling**: replaced `permutation(32000)` with `randint(32000, size=N)` + `unique()` — 30× faster (528μs→17μs per call).
3. **sample_size cap**: `200 * min(len(updates), 5)` → `100` (fixed). 200 random samples are statistically representative.
4. **Removed redundant `compute_vector`** in `_apply_vector_update`: `apply_code_update` already normalizes code → `code @ basis` is unit-normed → no need to recompute.
5. **`sync_vec_array()`** called after `init_concepts()` and `load()`.

### Changes
- `concept_space.py`: `_vec_array`, `_cids`, `_cid_to_idx`; `sync_vec_array()`, `get_vec()`, `set_vec()`; fast sampling in `_lateral_inhibition_fractal`; removed redundant `compute_vector` in `_apply_vector_update`
- `crystal_generator.py`: `cs.get_vec()` / `cs.set_vec()` in STDP loop; sample_size=100
- `syntax_lattice.py`: `_refresh_prefix_totals()` (from previous session)

### Speedup
- **345ms/line → 21.1ms/line** (16× faster overall)
- Full 145K corpus: **~13.9h → ~51 min**
- Inhibition: 88ms/line → 14.5ms/line (6× faster)
- PMI gate: 560ms → 58ms (9.7×, from previous session)

### Current bottleneck (86% of time)
Still lateral inhibition: `_apply_vector_update` per affected concept does `delta_v @ basis.T` + `apply_code_update` (which does `code_new @ basis`). ~48K calls at 60μs each. Next options:
1. Skip inhibition for gen_cids with tiny `total_elr`
2. Batch `delta_code` computation across all affected concepts in one matmul
3. Numba JIT for the `_apply_vector_update` + `apply_code_update` hot path

---

## Cognitive Architecture Roadmap (2026-06-14)

### Current capability
- **STDP**: pairwise token proximity in ±2 window (bigram-level)
- **Centroid pull**: sentence-level bag-of-tokens alignment
- **Generation**: stateless beam continuation (always same output pattern)
- **No query, no retrieval, no composition, no episodic memory**

### What's missing for cognitive meta-structure

#### 1. Query → Retrieval → Generation (separate processes)
- Query-encoder: запрос → centroid vector
- Retrieval: top-K ближайших concept_vectors по cos
- Conditioned beam: найденные концепты как priors для генерации
- *Сложность: ~2 дня*
- *Эффект: разная генерация по разным запросам*

#### 2. Episodic FIFO memory (context between sentences)
- Буфер последних 5-10 предложений
- Каждое новое предложение получает centroid не только своих токенов, но и summary предыдущих
- Связность: «война→мир→Толстой→1869»
- *Сложность: ~1 день*

#### 3. Hierarchical compression (phrase chunking)
- Фразы, клаузы, предложения → сжатые векторы (384→64→384 autoencoder)
- Иерархический ConceptSpace (phrase_space, clause_space)
- Композиция: «чёрный кот» ≠ «чёрный» + «кот»
- *Сложность: ~3-4 дня*

#### 4. Compositional binding (Semantic Pointer Architecture)
- Circular convolution вместо bag-of-vectors
- agent(кот) ⊛ action(ловит) ⊛ patient(мышь) → bound vector
- Unbinding: bound ⊘ agent ≈ кот
- *Сложность: ~2-3 дня*

#### 5. Predictive coding (System 1 → System 2)
- Вектор предсказывает следующее состояние поля
- Ошибка предсказания = сигнал обучения (не co-occurrence)
- Высокая ошибка → System 2 (планирование, поиск)
- *Сложность: ~5 дней*

#### 6. Generation becomes a reaction
```
Запрос → Query-encoder → retrieval (FIFO memory + concept space)
                              ↓
                    Reasoning (composition + binding)
                              ↓
                    Response generation (один из выходов)
```
- Обучение, флуктуация, inhibition работают фоном
- Генерация — не главный процесс, а реакция на внутреннее состояние

### Priority order
1. Query → retrieve → conditioned beam
2. Episodic FIFO (5-10 предложений)
3. Phrase chunking (384→64→384)
4. Compositional binding (circular convolution)
5. Predictive coding
