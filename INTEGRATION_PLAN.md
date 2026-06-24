# Integration Plan

## P0 — Hardcoded constants → FCFConfig (done)

| Пункт | Файлы | Статус |
|-------|-------|--------|
| hormonal_system → FormulaCoefficients | `hormonal_system.py` | ✓ |
| stdp_trainer STDP → FormulaCoefficients | `stdp_trainer.py` | ✓ |
| adaptive_controller SubspaceConfig → FCFConfig | `adaptive_controller.py` | ✓ |
| crystal_generator constants → FCFConfig | `crystal_generator.py` | ✓ |
| parameter_optimizer thresholds → FCFConfig | `parameter_optimizer.py` | ✓ |
| concept_space init → FCFConfig | `concept_space.py` | ✓ |
| Special token IDs → FCFConfig | `crystal_generator.py` | ✓ |
| Seeds → SeedRegistry | `concept_space.py` | ✓ |
| Param cascade (Param._def) | `parameter_optimizer.py` | ✓ |

## P1 — Remaining hardcoded cleanup

- [ ] concept_space.py: `fields_collapse` thresholds (0.85, 0.15, 10)
- [ ] concept_space.py: `fluctuate` cos modulation thresholds (0.25, 0.05, 0.2)
- [ ] concept_space.py: `_decompose_word` short-word threshold (4, 0.4)
- [ ] concept_space.py: `reinit_rare` default threshold (3)
- [ ] crystal_generator.py: `block_ngram` config
- [ ] concept_space.py: `_lateral_inhibition_fractal` default strength/threshold → ParamDef

## P2 — FormulaCoefficients → FCFConfig

- [ ] Merge `FormulaCoefficients` fields into `FCFConfig.formula_*`
- [ ] Remove `FormulaCoefficients` class
- [ ] Update all imports to use `FCFConfig()`

## P3 — Run-time hot-reload

- [ ] `FCFConfig.from_file(path)` — load from JSON
- [ ] `Param.reload()` — re-read all ParamDef defaults from config
- [ ] Observer pattern: config change → Param cascade

## Architecture

- `FCFConfig` — frozen dataclass, single source of truth
- `FormulaCoefficients` — will be merged into FCFConfig
- `SeedRegistry` — deterministic RNG per component
- `Param._def` — live link to ParamDef for cascade

## Tests

294 passed, 7 skipped (no SentencePiece, no fp16, no cluster_map)
