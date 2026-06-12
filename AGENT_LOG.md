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

## Session: Nested octree encoding (2026-06-11)

### Context
- Previous H matrix: flat PMI (1024 anchors, 0.41% dense)
- Previous field model: each concept = itself + PMI-related anchors
- Bug: `perm_hash` used linear LCG `(val * P + level * Q + R) % M`, making L0-match predict all-level match (100% propagation)

### Fix
- **Nested octree encoding**: each decimal digit of `val` → octant (0..7) at that level
- `field(val)` = tuple of octants (not anchor_ids) = path through octree
- `H[i,j]` = sum γ^l over LCP (longest common prefix), not independent level checks
  - Closed form: H = (1 − γ^{LCP})/(1−γ), with γ=0.5 → H = 2(1 − 0.5^{LCP})
- `perm_hash` → non-linear Python `hash((val, level))` to avoid linear propagation
- Vector encoding: prefix hashing via FNV-1a → adir pool (32K random unit vectors)

### Results (test_fractal_v2.py)
- **LCP distribution**: 86.2% LCP=0, 11.9% LCP=1, 1.6% LCP=2, 0.2% LCP=3 — pure geometric (1/8)^k
- **H>0**: 13.7% — discriminative (vs 84% old, 0.1% flat)
- **Cosine per LCP**: exactly 1−0.25^{LCP}:
  - LCP=0: cos=−0.005±0.050 (zero)
  - LCP=1: cos=0.750±0.015 (theory: 0.75)
  - LCP=2: cos=0.938±0.005 (theory: 0.9375)
  - LCP=3: cos=0.984±0.001 (theory: 0.9844)
- **Field-gate**: within-cluster H up to 17x higher than cross-cluster (real cluster differentiation)

### Integration into FCF (2026-06-12)
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
- `CS.load`: 0.75s (was 204s — fixed NpzFile __getitem__ bottleneck: pre-extract arrays before dict comprehensions)
- Field density: mean=37.1 bits/concept (vs 3.4 in PMI model)
- Field-gate distribution: 95.7% pairs fw=0.1, 4.3% pairs fw=15.7 mean

### Found bugs fixed
1. `from_dict` QR re-encoding: `codes_mat @ basis.T` → `codes_mat @ basis` (wrong dims, path never triggered due to tight 1e-5 threshold)
2. NpzFile repeated `__getitem__` access 10000x slower than pre-extracted array (Windows/numpy quirk)
3. Relaxed orthogonality check threshold: 1e-5 → 1e-3; batched re-encoding via matmul

### Key insight
- Decimal digits → octree levels: any number maps to fractal coordinate
- LCP = level of common ancestry in octree = semantic similarity
- H directly computable from LCP, no PMI statistics needed
- 16-digit paths (16 bytes) encode ALL anchor relationships implicitly

### Next
- [ ] Evaluate with full training run (monitor cos_std trend)
- [ ] Consider PyTorch port: octree paths as int16 tensors, batched LCP on GPU
