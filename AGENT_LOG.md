# FCF — Agent Log

## Session: Architecture overhaul (2026-06-11)

### Context
- Full 145K training completed: PPL=25926, vecPPL=27990, acc@1=0.106, vacc@1=0.000
- 21 bugs from ARCHITECTURE_REVIEW.md fixed and pushed (170c1cd)
- User wants me to act as lead developer, execute autonomously, think deeply

### Core Problem
Vacc@1=0.000 because vectors remain a "random gas" (cos=0.0003±0.0532).
Root cause: STDP is purely Hebbian (pull together) with no effective repulsion:
- neg_samples=0 (default) — no negative sampling
- inh_threshold=0.35 (default) — at cos_std=0.053, est_frac=0.0004 (0.04% pairs inhibited)
- PMI gate only weights, doesn't filter — noise pairs still contribute

### Phase 1: Architecture fixes (IN PROGRESS)

#### Done
1. **parameter_optimizer.py** — changed defaults:
   - `neg_samples`: 0→1 (enable negative sampling)
   - `inh_threshold`: default 0.35→0.10, min 0.20→0.05, max 0.60→0.30
   - `pmi_gate_min`: default 0.1→0.20 (acts as filter)
   - inh_threshold rule: lowered guard from 0.25→0.06 so optimizer can reduce further

2. **crystal_generator.py** — added PMI filter in `train_from_text`:
   - Skip pair entirely if `pmi_w <= pmi_gate_min` (eliminates noise pairs)
   - Updated defaults: neg_samples=0→1, inh_threshold=0.35→0.10, pmi_gate_min=0.05→0.20

3. **gen_quick.bat + gen_quick.py** — interactive gen script with Russian UTF-8 output

#### Remaining
- Fix reset_fractal.py (dead code, wrong attributes)
- Fix requirements.txt (wrong packages, missing sentencepiece)
- Add lazy n-gram loading to SyntaxLattice (25s→<2s load)
- Update ARCHITECTURE.md or mark as obsolete

### Key Decisions
- PMI filter: pairs with PMI ≤ 0 are noise → skip entirely. Only PMI-positive pairs contribute.
- inh_threshold=0.10 targets ~5% of pairs at current cos_std=0.053
- neg_samples=1 at 10% of pull LR — gentle but persistent repulsion

### Next
- Push Phase 1 changes to git
- Phase 2: Technical debt (reset_fractal.py, requirements.txt, lazy load)
- Phase 3: Train with books data (500K+ rows)
