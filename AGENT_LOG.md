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

### Next
- Monitor training progress via train_log.txt
- Evaluate cos_std trend after first 5K lines
- If cos_std still flat: adjust subspace learning rates or increase repulsion
