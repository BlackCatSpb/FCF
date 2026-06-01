# EVA — Architecture & Status

## Core Principles
- **Deterministic**: encode/decode without loss, all stats from real data
- **Data-driven heads**: 6 heads read from sparse DB (transitions, morph, syntax, semantic, concept, contradiction)
- **Transformer = weighter**: 33K params, learns to weight 6 heads
- **Hierarchical storage**: corpus -> sentence -> word -> token, **21 MB** (was 10.6 GB)
- **Metadata = semantics**: 97 bits of language structure in CoordinatePacker
- **Multi-text**: text_id (dims 89-96, 8 bits) supports up to 255 texts

## Storage
- `real_data/v5/hierarchical/` — 2 MB: WP-only (sentences, CSR, morph/syntax cache)
- `real_data/v5/conceptnet/` — 3.7 MB: ConceptNet-only (sentences, CSR, caches)
- `real_data/v5/heads_meta.pkl` — 7.7 MB: merged (WP + CN) metadata
- `real_data/v5/heads_meta.pkl` stats:
  - 504,804 sentences (27,061 WP + 477,743 CN)
  - 14,700,456 tokens (2.4M WP + 12.3M CN)
  - 2,280,414 words (319K WP + 1.96M CN)
  - 80,149 unique transitions (40,970 WP + 80,149 CN minus overlap)

## HeadsEnsemble (6 heads)
| Head | Source | Role |
|------|--------|------|
| Morph | morph_logprob[wl][pos] array(V) | P(token \| word_len, pos_in_word) |
| Syntax | syntax_logprob[wn] array(V) | P(token \| word_num) |
| Transition | log_prob_csr[prev] sparse | P(token \| prev_token) |
| Semantic | trans_sim_sparse | cosine sim of transition patterns |
| Concept | concept_scores array(V) | sqrt(count * n_next) / sqrt(max) |
| Contra | contra_penalty sparse(VxV) | penalty for P=0 pairs with sim ≥ 0.4 |

Weighted merge: WP × 2.0, CN × 1.0 (morph, syntax, token_counts). Transition CSR summed.

## WeightTransformer
- embed(8) + 6 scalars → 32 → 6 (Softplus), 33,486 params
- Trained: 15.8% val_acc (+76% vs rule-based)
- Self-trains during autonomous loop on `train_buffer` (10K samples)

## Generation Loop
- deterministic: SENT_OPEN(0) → WORD_OPEN(157) → ... → SENT_CLOSE(159)
- content: 6 heads → weighted sum → mask → select (argmax or temperature, 0.0-1.0)
- reserved dims (97-383) filled per-token: head weights, winning head, concept, attractor
- Position-aware WORD_CLOSE bonus (sigmoid ramp pos=2-6)
- Min 3 words before SENT_CLOSE allowed

## ConceptNet Integration
- Source: `conceptnet.db` (10.25 GB SQLite, 34M assertions)
- Filter: Russian→Russian edges (480K / 34M)
- Templates: form_of→«форма слова», is_a→«это», related_to→«связан с», synonym→«то же», etc.
- Output: 477,743 Russian sentences, 29 MB text
- BPE tokenized with boundary tokens 157-160 (compatible with WP)
- Morph/syntax distributions merged into heads with WP-weight 2×, CN-weight 1×

## Autonomous Loop (`eva/core/`)
Cycles through 4 phases continuously:

1. **THINK** — generates text (~45K tokens in 15s), collects training data (10K buffer)
2. **ANALYZE** — concept clustering (transition-similar tokens), contradiction audit
3. **LEARN** — self-trains WeightTransformer, saves to `models/weight_transformer_best.pt`
4. **OPTIMIZE** — saves concept clusters to DB, updates metadata

### Dashboard
- Real-time web UI at `http://localhost:8383`
- Cards: phase, tokens generated, contradictions, concepts, accuracy, gen rate, uptime, disk
- Bar charts: head weight distribution, accuracy trend (100), generation rate trend (100)
- Scrolling event log with timestamps
- Auto-refresh every 2.5s
- Two API endpoints: `/api/state` (JSON), `/api/log` (JSON)

### Desktop Shortcut
- `EVA.lnk` on desktop → runs `run_eva.ps1` → launches think_loop + dashboard

## Data Compression (unique)
| Component | Before | After | Ratio |
|-----------|--------|-------|-------|
| Trajectory Store | 10.6 GB | deleted | ∞ |
| Heads metadata | raw counts | 7.7 MB | ~1000× |
| Transitions | dense [V×V] | CSR sparse | ~400× |
| Morph/syntax | full arrays | sparse V-dim | ~200× |

Detailed: see [COMPRESSION.md](COMPRESSION.md)

## Key Stats
- Disk: **21 MB** (from 10.6 GB)
- Transitions: 80,149 unique pairs (CSR sparse)
- Contradictions: 9,012 pairs
- Heads: 11.6K calls/sec
- Generation: ~2,600 tokens/sec
- WeightTransformer: 33,486 params (1000× less than LLM embedding layer)
- VRAM: 0 MB (CPU-only, numpy)

## File Structure
```
eva/core/                     <- Autonomous system
  database.py                 Hierarchical storage manager
  dashboard.py                Web dashboard (port 8383)
  think_loop.py               Autonomous loop (4 phases)

eva/symbolic/                 <- Core modules
  heads.py                    HeadsEnsemble (6 heads, 11.6K calls/s)
  weight_transformer.py       WeightTransformer (33K params)
  generation_loop.py          Autoregressive generation
  reserved_dims.py            Reserved dim filler (97-383)
  coordinate_packer.py        384-dim coordinate encoder
  bpe_tokenizer.py            BPE tokenizer (vocab 4096)
  char_vocab.py               Character vocab (legacy)

real_data/v5/
  hierarchical/               WP-only storage (~2 MB)
  conceptnet/                 CN-only storage (~3.7 MB)
  heads_meta.pkl              Merged metadata (7.7 MB)

models/
  weight_transformer_best.pt  Trained transformer checkpoint

Root:
  run_eva.ps1                 Launcher (PowerShell)
  build_conceptnet_text.py    ConceptNet → Russian text
  build_conceptnet_trajectories.py  Tokenize + merge heads
  coordinate_packer.py        CoordinatePacker
```

## Launch
1. Double-click **EVA** on desktop
2. Or: `powershell -File run_eva.ps1` from FCF dir
3. Open `http://localhost:8383` in browser
4. `Ctrl+C` to stop

Direct: `python -X utf8 eva/core/think_loop.py --port 8383`
