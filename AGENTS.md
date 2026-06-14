# FCF — Agent Session Log

## Context
Russian BPE concept navigation model. 146K wordpiece tokens, 384D hypersphere.
Multi-epoch STDP training with centroid pull, fractal fields, contrastive objective.
Checkpoints at real_data/concept_space_{tag}.json + syntax_lattice_{tag}.\*.

## Progress

### [2026-06-14] Vector space fixes (commit 2f75fb8)
- **Centroid pull LR:** 0.1 → 0.3 (crystal_generator.py:848)
- **Theta-gate:** position-relative cap at 5 tokens (lines 603/623/789)
- **Contrastive objective:** hard-negative push-pull via PPMI-filtered 80-candidate sampling
- **GPU path:** _ensure_torch full-V tensors, _invalidate_torch after fluctuate, gated at 500+ pairs
- **checkpoint_state.json:** {line:21000, epoch:2}
- **run_train.bat:** --epochs 3

### [2026-06-14] eval_metrics.py + inference.py (commits f328984, 94229ed)
- `eval_metrics.py [tag]`: vPPL, cos distribution, generation samples, top-k neighbours
- `inference.py`: read-only generation, --prompt/--batch/--neighbours/--eval modes
- Both copy checkpoints to temp dir (no file-lock conflicts with training)
- math.log guard fix in evaluate()

### [2026-06-14] Token diversity (commit 75851c7)
- **Top-p (nucleus) sampling** in `_branch()`: configurable via `top_p` (default 0.9)
- **Length normalization** in `generate()`: beam sort + final normalized by `len^alpha` (default 0.7)
- **N-gram blocking** in `_branch()`: configurable n (default 4), extended from hardcoded trigram
- **MMI re-ranking** in `generate()`: per-token `-lambda * log P(cid)` penalty (default 0.2)
- All exposed via CLI in `inference.py`: `--top-p`, `--len-norm-alpha`, `--block-ngram`, `--mmi-lambda`

### [2026-06-14] Curriculum learning (commit 66c3b53)
- **Length-based curriculum**: train lines sorted by BPE token count (short → long)
- **Epoch-dependent max length**: epoch 1 = 32 tokens, epoch 2 = 128, epoch 3 = unlimited
- **Pre-filtered per epoch**: each epoch only trains on lines ≤ its max length
- Progress percentages and ETAs use epoch-relative line counts

### [2026-06-14] RAG query conditioning (commit pending)
- `inference.py --retrieve "query"`: explicit retrieval step — query → centroid → top-k nearest concepts
- `inference.py --query "term1,term2" --prompt "..."` — query words steer generation via intent centroid bonus
- `inference.py --neighbours "word"` — single-token nearest-neighbours (restored)
- `generate()` in CrystalGenerator already supports `query_words` → centroid → _branch intent bonus — now wired through inference.py CLI

## Current State
- Training live (PID 15496), epoch 2 at ~21000L
- Vector space fixes active: centroid pull LR 0.3, theta-gate cap 5, contrastive push-pull
- Baseline eval at 21k shows cos=0.0124, vac@1=0 — need post-epoch-3 comparison

## Priority Queue
1. ⬜ Visualization: t-SNE concept space
2. ⬜ RAG: centroid + _branch bonus (95% ready)
3. ⬜ GPU acceleration: batched lateral inhibition via CUDA

## Key Decisions
- **Centroid pull LR 0.3** — boost clustering speed (from 0.1)
- **Theta-gate cap 5** — only first 5 tokens decay, rest get full LR
- **Contrastive:** PPMI-filtered 80-cand sampling (O(80·cos) vs O(146K·cos))
- **GPU:** use_torch=True but 0.67× CPU at current batch sizes; future: batched lateral inh
- **Eval:** standalone script with temp copies to avoid file-lock conflicts
- **Inference:** wrapper around CrystalGenerator, not a reimplementation

## Relevant Files
- `C:\Users\black\OneDrive\Desktop\FCF\eva\symbolic\crystal_generator.py`
- `C:\Users\black\OneDrive\Desktop\FCF\eval_metrics.py`
- `C:\Users\black\OneDrive\Desktop\FCF\inference.py`
- `C:\Users\black\OneDrive\Desktop\FCF\train_full.py`
- `C:\Users\black\OneDrive\Desktop\FCF\run_train.bat`
- `C:\Users\black\OneDrive\Desktop\FCF\real_data\checkpoint_state.json`
- `C:\Users\black\OneDrive\Desktop\FCF\FCF_IMPLEMENTATION_ROADMAP.md`
