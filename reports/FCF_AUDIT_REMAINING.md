# Phase 7 Final Audit — Remaining Hardcode (24.06.2026)

## Executive Summary

**Phases 0–6 завершены.** Остаётся **~120 значений хардкода** в ~15 файлах.  
Ключевое открытие: `FormulaCoefficients` содержит все необходимые коэффициенты, но 2 файла (гормоны, STDP) их **не читают** — это P0-баги.

## Приоритеты

### P0 — Конфиг есть, код его игнорирует (~30 значений)

| # | Файл | Что | Сколько | Статус |
|---|------|-----|---------|--------|
| 1 | `hormonal_system.py:30-190` | init baselines + update formulas | ~20 | **Не читает FormulaCoefficients** |
| 2 | `stdp_trainer.py:485-542` | freq_weight, field_weight, hormonal_mod | ~10 | **Не читает FormulaCoefficients** |

Оба файла имеют поля-дубликаты в `FormulaCoefficients` (fcf_config.py:363-430), используют хардкод.

### P1 — Нет полей в FCFConfig (~40 значений)

| # | Файл | Категория | Сколько | Статус |
|---|------|-----------|---------|--------|
| 3 | `crystal_generator.py` | Graph search + branch params | ~15 | Нет в config |
| 4 | `parameter_optimizer.py` | Plateau/metric/maxlen/detector | ~15 | Нет в config |
| 5 | `adaptive_controller.py` | SubspaceConfig + update() const | ~15 | Нет в config |
| 6 | `concept_space.py` | FractalField capacity + init scales | ~7 | Нет в config |

### P2 — Дублирование и скрипты (~50 значений)

| # | Файлы | Что | Сколько | Статус |
|---|-------|-----|---------|--------|
| 7 | `crystal_generator.py:47-48`, `model/*.py` | Special token IDs (0/1/2) | 6 | Дубли в 3 файлах |
| 8 | `concept_space.py:364,472` | Прямые `RandomState()` | 2 | Не через SeedRegistry |
| 9 | `train_full.py`, `inference.py`, `filter_corpus.py`, `eval_checkpoint.py`, `visualize.py` | Seeds, sample_size, thresholds | ~25 | Не централизованы |

## Детальный список всех значений

### P0-A: hormonal_system.py (20 значений)

| Строка | Сейчас | Должен читать |
|--------|--------|---------------|
| 30 | `self.dopamine = 0.5` | `_fc.da_baseline` (есть: 398) |
| 31 | `self.serotonin = 0.5` | `_fc.ht_baseline` (есть: 399) |
| 32 | `self.noradrenaline = 0.3` | `_fc.na_baseline` (есть: 400) |
| 33 | `self.acetylcholine = 0.5` | `_fc.ach_baseline` (есть: 401) |
| 46 | `self.tonic_decay = 0.95` | `_fc.tonic_decay` (есть: 402) |
| 47 | `self.phasic_decay = 0.7` | `_fc.phasic_decay` (есть: 403) |
| 92 | `da_coherence = 0.05` | `_fc.da_coherence_strength` (есть: 404) |
| 102 | `novelty * 0.4` | `* _fc.da_curiosity_strength` (есть: 405) |
| 105 | `max(0, delta_match) * 0.5` | `* _fc.da_mastery_strength` (есть: 406) |
| 113 | `da_coherence -= 0.1` | `- _fc.da_boredom_penalty` (есть: 407) |
| 128 | `surprise * 0.6` | `* _fc.ach_surprise_strength` (есть: 410) |
| 129 | `(1.0 - confidence) * 0.5` | `* _fc.ach_uncertainty_strength` (есть: 411) |
| 131 | `surprise * 0.15` | `* _fc.ach_match_strength` (есть: 412) |
| 134 | `novelty * 0.5` | `* _fc.ach_novelty_scale` (есть: 413) |
| 142 | `0.3 + 0.4 * (1.0 - avg_match)` | `_fc.ht_baseline_part + _fc.ht_match_scale * ...` (есть: 417-418) |
| 143 | `(target - self.serotonin) * 0.1` | `* _fc.ht_adapt_rate` (есть: 419) |
| 148 | `0.2 + 0.5 * surprise + 0.3 * (1.0 - confidence)` | `_fc.na_baseline_part + ...` (есть: 420-422) |
| 149 | `(target - self.na) * 0.3` | `* _fc.na_adapt_rate` (есть: 423) |
| 153 | `0.3 + 0.5 * novelty` | `_fc.ach_novelty_baseline + _fc.ach_novelty_scale_tonic * novelty` (есть: 424-425) |
| 155 | `novelty_target = 0.2` | `_fc.ach_well_known_floor` (есть: 426) |
| 158 | `(target - self.ach) * 0.15` | `* _fc.ach_tonic_drift` (есть: 427) |
| 160 | `self.ach_phasic * 0.1` | `* _fc.ach_phasic_integration` (есть: 416) |
| 165 | `self.da_phasic * 0.1` | `* _fc.da_phasic_to_tonic` (есть: 408) |
| 184 | `max(0.1 + 0.9 * risk, 0.05)` | Через `_fc.da_temperature_min/scale` (есть: 428-429) |
| 190 | `1.0 - self.na * 0.5` | Через `_fc.na_beam_scale` (есть: 430) |

### P0-B: stdp_trainer.py — _build_pairs (10 значений)

| Строка | Сейчас | Должен читать |
|--------|--------|---------------|
| 485 | `* 0.15` | `* _fc.freq_weight_log_scale` |
| 533 | `* 2.0` | `* _fc.field_weight_log_scale` |
| 533 | `min(..., 3.0)` | `min(..., _fc.field_weight_cap)` |
| 534 | `else 0.1` | `else _fc.field_weight_floor` |
| 536 | `max(freq_weight, 0.05)` | `max(freq_weight, _fc.freq_weight_min)` |
| 537 | `(0.5 + ...)` | `(_fc.hormonal_mod_baseline + ...)` |
| 537 | `* 0.5` | `* _fc.hormonal_mod_scale` |
| 539 | `max(theta_gate, 0.1)` | `max(theta_gate, _fc.theta_fast_min)` (есть: 317) |
| 541 | `* 3.0` | `* _fc.theta_tau_slow_mult` (есть: 313) |
| 542 | `max(theta_slow, 0.02) * 0.3` | `* _fc.theta_slow_min * _fc.theta_slow_scale` (есть: 318-319) |

### P1-A: crystal_generator.py — graph search (15 значений)

| Строка | Параметр | Сейчас |
|--------|----------|--------|
| 693 | `_graph_search(B=2.0)` | `2.0` |
| 693 | `max_candidates=30` | `30` |
| 693 | `max_depth=5` | `5` |
| 727 | `connections_of(u, top_k=8)` | `8` |
| 780 | `_graph_search(B=1.2, max_candidates=30)` | `1.2`, `30` |
| 787 | `syn_preds[:80]` | `80` |
| 796 | `hdc_predict(k=30)` | `30` |
| 798 | `hscore > 0.05` | `0.05` |
| 809 | `search_in_sector(depth=1, k=40)` | `40` |
| 809 | `if len(sim_candidates) < 5` | `5` |
| 811 | `focal_refine(target_k=20)` | `20` |
| 813 | `topk_similar_concepts(k=20, sample_size=500)` | `500` |
| 815 | `sim > 0.05` | `0.05` |
| 620 | `seq[-6:]` | `6` |
| 930, 934 | `n_candidates = min(15 + int(15 * theta_temp), ...)` | `15` |

### P1-B: parameter_optimizer.py (15 значений)

| Строка | Параметр | Сейчас |
|--------|----------|--------|
| 48 | `toward_default(rate=0.03)` | `0.03` |
| 65 | `MetricBuffer(maxlen=10)` | `10` |
| 80 | `plateau(patience=3, rel_thresh=0.005)` | `3`, `0.005` |
| 117 | `MetricBuffer(10)` | `10` |
| 118 | `MetricBuffer(10)` | `10` |
| 119-124 | `MetricBuffer(8) × 5`, `MetricBuffer(6)` | `8`, `6` |
| 130 | `_flat_thresh = 0.002` | `0.002` |
| 132 | `_cos_trend_buffer(maxlen=5)` | `5` |
| 140 | `_full_stuck_counter >= 5` | `5` |
| 143 | `plateau(rel_thresh=0.002)` | `0.002` |
| 145 | `plateau(rel_thresh=0.02)` | `0.02` |
| 163,173 | `ctx.get('inh_threshold', 0.1)` | `0.1` |
| 374 | `PlateauDetector(100, 20, 0.5, 0.1, 0.05)` | все args |
| 382 | `ema_alpha = 0.05` | `0.05` |
| 406 | `steps_in_plateau * 0.01` | `0.01` |

### P1-C: adaptive_controller.py (15 значений)

| Строка | Параметр | Сейчас |
|--------|----------|--------|
| 21 | `l_c_ratio = 0.6` | `0.6` |
| 22 | `l_a_ratio = 0.25` | `0.25` |
| 23 | `l_m_ratio = 0.15` | `0.15` |
| 26 | `density_threshold_grow = 0.15` | `0.15` |
| 27 | `density_threshold_prune = 0.01` | `0.01` |
| 28 | `l1_target_density = 0.08` | `0.08` |
| 29 | `growth_factor = 1.5` | `1.5` |
| 32 | `sector_depths = [4, 10, 20]` | `[4, 10, 20]` |
| 89 | `np.abs(z_c) > 1e-4` | `1e-4` |
| 97 | `> 10000` | `10000` |
| 103 | `_n_updates > 10` | `10` |
| 106 | `min(ratio * 1.03, 0.75)` | `1.03`, `0.75` |
| 108 | `max(ratio * 0.97, 0.3)` | `0.97`, `0.3` |
| 113 | `remaining * 0.6` | `0.6` |
| 114 | `remaining * 0.4` | `0.4` |

### P1-D: concept_space.py FractalField (7 значений)

| Строка | Параметр | Сейчас |
|--------|----------|--------|
| 275 | `hdc_memory_max = 20000` | `20000` |
| 370 | `n_active = max(int(l_c * 0.03), 8)` | `0.03`, `8` |
| 376 | `z_a * 0.01` | `0.01` |
| 379 | `z_m * 0.001` | `0.001` |
| 392 | `init_fields(n_anchors=1024)` | `1024` |
| 498 | `new_lambda = current * 1.1` | `1.1` |

## Итого по фазам

| Фаза | Значения | Файлы | Приоритет |
|------|----------|-------|-----------|
| P0-A | 20 | 2 | Critical |
| P0-B | 10 | 2 | Critical |
| P1-A | 15 | 2 | High |
| P1-B | 15 | 2 | High |
| P1-C | 15 | 2 | High |
| P1-D | 7 | 2 | Medium |
| P2-A | 6 | 4 | Medium |
| P2-B | 2 | 1 | Medium |
| P2-C | 25 | 5 | Low |
| **Total** | **~115** | **~15** | |

## Примечания

1. **FormulaCoefficients уже полный** — не нужно добавлять поля, нужно читать. HormonalSystem + STDPTrainer игнорируют существующий конфиг.
2. **SubspaceConfig — кандидат на удаление** — дублирует функционал FCFConfig. После P1-C можно убрать класс, читать напрямую из config.
3. **Special token IDs** — простая централизация, но 3 файла. После P2-A удалить `_BOS_ID`/`_EOS_ID` глобалы.
4. **crystal_generator.py** — `ce_max = min(3 * vocab_size // 4, 100000)` — hardcoded `100000` останется как formula-константа (максимальный размер error tracker).
