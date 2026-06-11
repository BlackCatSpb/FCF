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

### Phase 1: Architecture fixes (DONE, pushed in Phase 2 commit)

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

### Phase 2: Technical debt (DONE, pushed)

#### Done
1. **reset_fractal.py** — rewrote entirely:
   - Removed dead code (cid_list, word_to_cid, concept_transitions)
   - Uses `list(cs.concept_vectors.keys())` for cids
   - Added cleanup of checkpoint_state.json, numbered checkpoints, optimizer state
   - Atomic file overwrite via `.tmp_reset` + `os.replace`

2. **requirements.txt** — removed unused packages (transformers, fastapi, uvicorn, pydantic);
   added `sentencepiece>=0.1.99`

3. **syntax_lattice.py** — added `load_ngrams=False` param to `load()`:
   - Skips n-gram array loading (lines 584-601) when False
   - Skips skip2 loading (lines 607-617) when False
   - Still loads connections and concept_freq (needed for generation)
   - Predict() returns [] when n-grams are empty dicts → generator falls back to graph+vector

4. **gen_quick.py** — uses `lattice.load(..., load_ngrams=False)` for fast startup

### Key Decisions
- PMI filter: pairs with PMI ≤ 0 are noise → skip entirely. Only PMI-positive pairs contribute.
- inh_threshold=0.10 targets ~5% of pairs at current cos_std=0.053
- neg_samples=1 at 10% of pull LR — gentle but persistent repulsion
- Lazy n-gram load: connections+concept_freq still needed for generation; n-grams only needed for training

### Next
- Phase 3: Train with books data (500K+ rows) using new params
- Evaluate gen quality, vacc@1 improvement
