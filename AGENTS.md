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

### [2026-06-14] GPU acceleration (commit 4be3d93)
- **Vectorized LR computation**: field_weight, theta_gate, composite LR in one shot
  on GPU via tensor math + scatter_add — replaces Python per-pair loop bottleneck
- **GPU batched lateral inhibition**: similarity matrix on device (G × V matmul),
  top-k thresholding, vectorized delta computation
- **No 500-pair threshold**: GPU path fires for any pair count when `use_torch=True`

### [2026-06-14] Visualization (commit pending)
- `viz_tsne.py [tag] [--n 3000] [--perplexity 30]`: standalone t-SNE visualization,
  samples concept vectors stratified by frequency, outputs Three.js interactive 3D viewer
- `train_full.py`: wired PCA `save_3d_vis()` at every 5th checkpoint
- `real_data/vis/`: created, viewer.html, Three.js-based interactive point cloud
- Serve via `python serve_vis.py` → http://127.0.0.1:8080/viewer.html

## Current State
- Training (epoch 1 resumed at ~21138L, Qwen knowledge active)
- Qwen knowledge: 1.53M high-confidence pairs (deduplicated from 15.96M, pruned from 20.6M raw)
- Qwen factor distribution: boosted/reduced/neutral tracked in eval outputs

### [2026-06-22] Qwen distillation, NPZ pruning, eval logging

- **QwenKnowledge integrated:** `real_data/qwen_knowledge.npz` (1.53M deduplicated,
  count≥3, 10 MB), loaded via `FCFConfig.qwen_knowledge_path` in `train_full.py:513`,
  modulates STDP learning rate per pair
- **NPZ pruned & deduplicated** (`prune_qwen_knowledge.py`): 20.6M raw → 15.96M unique →
  **1.53M high-confidence** (count≥3), 92 MB → **10 MB**
- **Low-sim repulsion** (`qwen_knowledge.py:get_factor`): three regimes —
  cos < 0.15 → neutral (1.0), 0.15 ≤ cos < 0.20 → linear repel (0.85→1.0),
  cos ≥ 0.20 → boost (1.0+cos*0.3, capped at 1.5)
- **Eval logging** (`eval_metrics.py`): qwen_factor distribution now reported:
  `qwen_boosted/qwen_reduced/qwen_neutral` percentages + `qwen_factor_mean`/`std`
- **Minesweeper inverted** (`crystal_generator.py:_update_cluster_potential`): high CE → boost (up to 1.2),
  low CE → reduce (down to 0.8). Rare/struggling concepts get MORE learning signal.
- **Antonym repel** (`stdp_trainer.py:_build_pairs` + `_gpu_stdp_core`): Russian antonym dictionary
  (22 pairs from eva_ai/contradiction_miner), decoded via `sp.IdToPiece`, flagged in GPU meta channel.
  Antonym pairs receive inverted pair_delta × 2.0 (active repel).
- **Cluster centroid pull** (`stdp_trainer.py:_cluster_centroid_pull`): pulls concept vectors toward
  octree cluster centroid (via `_cluster_map` anchor), pull_strength=0.05, called every batch.

## Priority Queue
- ✅ Qwen knowledge npz integrated + deduplicated + pruned (1.53M pairs, 10 MB)
- ✅ Minesweeper inversion
- ✅ Antonym repel
- ✅ Cluster centroid pull
- ✅ Low-sim repulsion (cos 0.15–0.20 → repel)
- ✅ Eval logging (qwen_factor distribution)

## Key Decisions
- **NPZ pruning:** count≥3 filter after deduplication (92.7% of raw entries were noise/count<3)
- **Low-sim repulsion:** linear repulse band [0.15, 0.20) to counteract STDP over-similarity
- **Qwen factor regimes:** three-way (neutral/repel/boost) instead of binary (unknown+threshold→boost)
- **Centroid pull LR 0.3** — boost clustering speed (from 0.1)
- **Theta-gate cap 5** — only first 5 tokens decay, rest get full LR
- **Contrastive:** PPMI-filtered 80-cand sampling (O(80·cos) vs O(146K·cos))
- **GPU:** use_torch=True but 0.67× CPU at current batch sizes; future: batched lateral inh
- **Eval:** standalone script with temp copies to avoid file-lock conflicts
- **Inference:** wrapper around CrystalGenerator, not a reimplementation
- **Minesweeper inverted:** rare/high-error concepts boosted (was: punished)
- **Antonym repel:** hardcoded 22-pair Ru dictionary, BPE-token-level detection, GPU path
- **Cluster centroid pull:** octree cluster anchors via `_cluster_map`, not sentence centroids
- **Qwen knowledge:** precomputed offline (Colab, RuadaptQwen3-4B), loaded as read-only LR modulator

## Relevant Files
- `eva/symbolic/crystal_generator.py`
- `eva/symbolic/stdp_trainer.py`
- `eva/symbolic/qwen_knowledge.py`
- `eva/symbolic/fcf_config.py`
- `train_full.py`
- `eval_metrics.py`
- `prune_qwen_knowledge.py`
- `real_data/qwen_knowledge.npz`
- `real_data/checkpoint_state.json`
- `PLAN.md`
