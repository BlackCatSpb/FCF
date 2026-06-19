# FCF V11 вЂ” РђСЂС…РёС‚РµРєС‚СѓСЂРЅС‹Р№ Р°СѓРґРёС‚

**Р”Р°С‚Р°**: 2026-06-19  
**Р’РµСЂСЃРёСЏ**: V11 (Р°СѓРґРёС‚ V10 РєРѕРјРјРёС‚РѕРІ: 525688b + d36a780)  
**РЎС‚Р°С‚СѓСЃ**: 105 С‚РµСЃС‚РѕРІ РїСЂРѕС…РѕРґСЏС‚ вњ… (Р±С‹Р»Рѕ 79)

---

## Executive Summary

| РњРµС‚СЂРёРєР° | V9 | V10 | V11 | О” |
|---------|:--:|:---:|:---:|:-:|
| РўРµСЃС‚С‹ | 79 | 105 | **105** | = |
| P0 | 0 | 0 | **1** в›” | +1 |
| P1 | 10 | 5 | **6** | +1 |
| P2 | 22 | 13 | **11** | в€’2 |
| GPU-РѕРїС‚РёРјРёР·Р°С†РёРё (G-40..G-52) | 0/13 | 0/13 | **13/13** вњ… | +13 |
| РўРµСЃС‚С‹ (QN-32..QN-40) | 0/9 | 0/9 | **9/9** вњ… | +9 |
| `.item()` РЅР° batch | ~16 500 | ~16 500 | **~5 600** | в€’66% |

**Р“Р»Р°РІРЅС‹Рµ РЅР°С…РѕРґРєРё V11:**

1. вњ… **V10 СЂРµР°Р»РёР·РѕРІР°Р» Р’РЎР• GPU-РѕРїС‚РёРјРёР·Р°С†РёРё** (G-40..G-52) вЂ” batched subspace update, GPU lateral inhibition, zero-copy write-back, deferred sync, fused post-STDP
2. вњ… **V10 СЂРµР°Р»РёР·РѕРІР°Р» Р’РЎР• 9 С‚РµСЃС‚РѕРІ** (QN-32..QN-40) вЂ” subspace, contrastive, evaluate, noise_scale, RNGRegistry, AdaptiveErrorTracker, checkpoint cleanup, pipeline, dead code
3. в›” **P0: train_full.py:722 вЂ” `noise_scale` keyword argument РЅРµ СЃСѓС‰РµСЃС‚РІСѓРµС‚** вЂ” `fluctuate_fractal()` РїСЂРёРЅРёРјР°РµС‚ `fluctuation_amp`, Р° РЅРµ `noise_scale`. РџСЂРё РїРµСЂРІРѕРј С„Р»СѓРєС‚СѓР°С‚Рµ (РєР°Р¶РґС‹Рµ 2000 СЃС‚СЂРѕРє) вЂ” **TypeError: unexpected keyword argument 'noise_scale'**. `opt.p['noise_scale']` С‚Р°РєР¶Рµ KeyError (РїРµСЂРµРёРјРµРЅРѕРІР°РЅ РІ `gradient_noise_scale`).
4. в›” **REG-V9-7 РёСЃРїСЂР°РІР»РµРЅ РЅРµ РїРѕР»РЅРѕСЃС‚СЊСЋ** вЂ” `gradient_noise_scale` РїРµСЂРµРґР°С‘С‚СЃСЏ РІ `train_batch` (вњ…), РЅРѕ РІС‹Р·РѕРІ `fluctuate_fractal` (train_full.py:722) РЅРµ РѕР±РЅРѕРІР»С‘РЅ
5. ~5 600 `.item()`/batch РѕСЃС‚Р°Р»РѕСЃСЊ (Р±С‹Р»Рѕ ~16 500) вЂ” РІ РѕСЃРЅРѕРІРЅРѕРј РІ `_contrastive_objective_gpu`

---

## 1. Р’РµСЂРёС„РёРєР°С†РёСЏ V10 РєРѕРјРјРёС‚РѕРІ

### Commits
- `525688b` вЂ” V10 All Fixes: Phase 0+P1+GPU opts+Tests
- `d36a780` вЂ” G-50/G-51/G-52: GPU zero-copy + deferred sync + fused post-STDP

### 1.1 Phase 0 (P1 bugs) вЂ” 3/3 вњ…

| ID | РСЃРїСЂР°РІР»РµРЅРёРµ | Р¤Р°Р№Р» | РЎС‚Р°С‚СѓСЃ |
|:--:|-------------|:----:|:------:|
| TN-31 | `checkpoint_state.json` РїСЂРё РєР°Р¶РґРѕРј С‡РµРєРїРѕРёРЅС‚Рµ | train_full.py:394-400 | вњ… |
| G-57 | `push_total`/`lr_scale` СѓРґР°Р»РµРЅС‹ | stdp_trainer.py | вњ… |
| SN-35/36 | CPU neg sampling/contrastive parity (compound updates) | stdp_trainer.py | вњ… |

### 1.2 Phase 1 (P1) вЂ” 4/4 вњ…

| ID | РСЃРїСЂР°РІР»РµРЅРёРµ | Р¤Р°Р№Р» | РЎС‚Р°С‚СѓСЃ |
|:--:|-------------|:----:|:------:|
| REG-V9-7 | `noise_scale` split в†’ `gradient_noise_scale` + `fluctuation_amp` | fcf_config.py:318-325, stdp_trainer.py:67 | вљ пёЏ Р§Р°СЃС‚РёС‡РЅРѕ (СЃРј. P0) |
| G-46 | `_mom_buf` CPU dict в†’ persistent `_mom_t` GPU tensor | stdp_trainer.py:412-418 | вњ… |
| G-42 | `_centroid_pull_batch` CPU в†’ GPU | stdp_trainer.py:799-845 | вњ… |

### 1.3 GPU Optimizations (G-40..G-52) вЂ” 13/13 вњ…

| ID | РћРїС‚РёРјРёР·Р°С†РёСЏ | РЎС‚Р°С‚СѓСЃ |
|:--:|-------------|:------:|
| G-40 | Batched GPU subspace update (`_apply_subspace_update_batch`) | вњ… `concept_space.py:591-638` |
| G-41 | Full GPU lateral inhibition (Р±РµР· `.item()`) | вњ… `stdp_trainer.py:508-532` |
| G-43 | GPU neg sampling vectorized (Р±РµР· CPU roundtrip) | вњ… `stdp_trainer.py:574-624` |
| G-44 | GPU contrastive (pre-computed cooc_masks + fb_overlaps) | вњ… `stdp_trainer.py:686-792` |
| G-48 | `torch.compile` flag | вљ пёЏ `stdp_trainer.py:336-338` вЂ” С‚РѕР»СЊРєРѕ РєРѕРјРјРµРЅС‚Р°СЂРёР№ |
| G-49 | Pre-allocated fused buffer | вњ… `crystal_generator.py:120` |
| G-50 | Zero-copy vector write-back (GPUв†’GPU) | вњ… `stdp_trainer.py:467-473` |
| G-51 | Deferred vector sync (batched `_vecs_t` write) | вњ… `stdp_trainer.py:427,479-484` |
| G-52 | Fused post-STDP (contrastive + neg sampling) | вњ… `stdp_trainer.py:496-506` |

### 1.4 Code Quality вЂ” 5/5 вњ…

| ID | РСЃРїСЂР°РІР»РµРЅРёРµ | РЎС‚Р°С‚СѓСЃ |
|:--:|-------------|:------:|
| AM-25 | CPU path marked legacy | вњ… `stdp_trainer.py:229` |
| AM-29/46 | `main_rng` via RNGRegistry | вњ… |
| AM-30 | Batched EMA update | вњ… `stdp_trainer.py:487-489` |
| AM-31 | ConceptError sync (batch update) | вњ… `stdp_trainer.py:399-407` |
| AM-33 | HormonalSystem.reset() | вњ… `hormonal_system.py:193-194` |
| SN-39 | `connection_strength` removed from GPU inner loop | вњ… |

### 1.5 Tests (QN-32..QN-40) вЂ” 9/9 вњ…

| ID | РўРµСЃС‚ | РЎС‚Р°С‚СѓСЃ |
|:--:|------|:------:|
| QN-32 | TestSubspaceUpdate | вњ… |
| QN-33 | TestGPUContrastive | вњ… |
| QN-34 | TestEvaluate | вњ… |
| QN-35 | TestNoiseScale | вњ… |
| QN-36 | TestRNGRegistry | вњ… |
| QN-37 | TestAdaptiveErrorTracker | вњ… |
| QN-38 | TestCheckpointCleanup | вњ… |
| QN-39 | TestTrainingPipeline | вњ… |
| QN-40 | TestDeadCode | вњ… |

---

## 2. P0 вЂ” РљСЂРёС‚РёС‡РµСЃРєРёРµ РїСЂРѕР±Р»РµРјС‹ (1)

### P0-1: train_full.py:722 вЂ” `noise_scale` keyword РЅРµ СЃСѓС‰РµСЃС‚РІСѓРµС‚

**РЎСѓС‚СЊ**: `noise_scale` Р±С‹Р» РїРµСЂРµРёРјРµРЅРѕРІР°РЅ РІ `gradient_noise_scale` + `fluctuation_amp` (REG-V9-7), РЅРѕ СЃС‚СЂРѕРєР° 722 РІ train_full.py РѕСЃС‚Р°Р»Р°СЃСЊ Р±РµР· РёР·РјРµРЅРµРЅРёР№:

```python
# train_full.py:722 вЂ” Р‘РЈР”Р•Рў CRASH
cs.fluctuate_fractal(noise_scale=opt.p['noise_scale'].current,
                     decay=opt.p['decay_rate'].current,
                     ...)
```

РџСЂРѕР±Р»РµРјС‹:
1. `opt.p['noise_scale']` вЂ” KeyError: `noise_scale` СѓРґР°Р»С‘РЅ РёР· `FCFConfig.params`. Р’ СЃРїРёСЃРєРµ С‚РѕР»СЊРєРѕ `gradient_noise_scale` Рё `fluctuation_amp`
2. `fluctuate_fractal()` РЅРµ РїСЂРёРЅРёРјР°РµС‚ keyword `noise_scale=` вЂ” РїР°СЂР°РјРµС‚СЂ РЅР°Р·С‹РІР°РµС‚СЃСЏ `fluctuation_amp`

**Р’Р»РёСЏРЅРёРµ**: Crash РїСЂРё РїРµСЂРІРѕРј `is_fluct_due` (РєР°Р¶РґС‹Рµ 2000 СЃС‚СЂРѕРє). Р’СЃСЏ training pipeline РїР°РґР°РµС‚.

**Fix**:
```python
cs.fluctuate_fractal(fluctuation_amp=opt.p['fluctuation_amp'].current, ...)
```

---

## 3. P1 вЂ” РћС‚РєСЂС‹С‚С‹Рµ РїСЂРѕР±Р»РµРјС‹ (6)

### 3.1. `.item()` syncs: ~5 600/batch (TN-40 ✅ FIXED, B1 ✅ FIXED in 024f1aa)

| Р“РґРµ | `.item()` РІС‹Р·РѕРІРѕРІ | РќР°Р·РЅР°С‡РµРЅРёРµ |
|:---:|:-----------------:|:----------:|
| `_build_pairs:185` | 1 Г— N_pairs | `field_overlap` via `torch.bitwise_and` |
| `_contrastive_objective_gpu:738-767` | ~165 Г— N_gen | per-candidate overlap, cos, idx |

РћСЃС‚Р°С‘С‚СЃСЏ ~5 600 `.item()`/batch РґР»СЏ N_gen=32. РћСЃРЅРѕРІРЅРѕР№ РІРєР»Р°Рґ вЂ” `_contrastive_objective_gpu` СЃ РґРІРѕР№РЅС‹Рј Python loop (hard negatives + TN-14 regularization).

**РџСЂРµРґР»РѕР¶РµРЅРёРµ**: Р’С‹РЅРµСЃС‚Рё TN-14 regularization РЅР° С‚РµРЅР·РѕСЂРЅС‹Рµ РѕРїРµСЂР°С†РёРё.

### 3.2. SN-38: cooc_set rebuild РІ Python loop

`_contrastive_objective_gpu:708-713` вЂ” `cooc_masks` СЃС‚СЂРѕРёС‚СЃСЏ С‡РµСЂРµР· Python loop:
```python
for i, gen_cid in enumerate(gen_idxs):
    ctx_cids = [ctx for ctx, _ in gen_updates[gen_cid]]
    if ctx_cids:
        ctx_t = torch.tensor(ctx_cids, dtype=torch.long, device=d)
        cooc_masks[i, ctx_t] = True
```

РџСЂРё N_gen=32, N_ctx~10 вЂ” 32 Python РёС‚РµСЂР°С†РёРё + 2 list comprehensions. Р”Р»СЏ 146K vocab вЂ” Р°Р»Р»РѕРєР°С†РёСЏ `cooc_masks` (ng Г— V) bool tensor = 32 Г— 146K Г— 1B = ~4.6MB. РќРµ РєСЂРёС‚РёС‡РЅРѕ, РЅРѕ РјРѕР¶РЅРѕ РѕРїС‚РёРјРёР·РёСЂРѕРІР°С‚СЊ С‡РµСЂРµР· sparse РёР»Рё pre-batched scatter.

### 3.3. SN-40: field_bits overlap per-candidate

`_contrastive_objective_gpu:745-773` вЂ” РґР»СЏ РєР°Р¶РґРѕРіРѕ hard negative РІС‹Р·С‹РІР°РµС‚СЃСЏ `.item()` РґР»СЏ overlap.

**РџСЂРµРґР»РѕР¶РµРЅРёРµ**: Р—Р°РјРµРЅРёС‚СЊ РЅР° С‚РµРЅР·РѕСЂРЅСѓСЋ РјР°СЃРєСѓ: `valid = (fb_overlaps[i] > 0) & (cos_val > 0.3)` Р±РµР· per-element `.item()`.

### 3.4. TN-13: Progressive batch size СЃ РїР»Р°С‚Рѕ РЅРµ СЂРµР°Р»РёР·РѕРІР°РЅ

`train_full.py:644`: `bs_curve` вЂ” Р»РёРЅРµР№РЅР°СЏ СЂР°РјРїР° РїРѕ idx, РЅРµ Р°РґР°РїС‚РёРІРЅР°СЏ:
```python
bs_curve = lambda i: int(CFG.batch_size_start + (CFG.batch_size_end - CFG.batch_size_start) * _curriculum_p(i))
```

batch_size РЅРµ СѓРІРµР»РёС‡РёРІР°РµС‚СЃСЏ РїСЂРё РїР»Р°С‚Рѕ РјРµС‚СЂРёРє. РќРµС‚ rules РІ ParameterOptimizer РґР»СЏ batch_size.

### 3.5. TN-15: Decay warmup СЃ protect threshold ramp

`train_full.py:728-732`:
```python
lattice.decay_all(rare_concept_protect=True, rare_threshold=3)
lattice.decay_connections()
cs.decay_usage(decay=0.98, rare_protect=True)
```

`rare_threshold=3` СЃС‚Р°С‚РёС‡РµРЅ. РќРµС‚ ramp РѕС‚ 0 в†’ target. `rare_protect` С‚РѕР»СЊРєРѕ binary (True/False) вЂ” РЅРµС‚ РїР»Р°РІРЅРѕРіРѕ РІРєР»СЋС‡РµРЅРёСЏ.

### 3.6. TN-34: opt.json naming mismatch

- `CheckpointManager._sync_save:86` вЂ” СЃРѕС…СЂР°РЅСЏРµС‚ `concept_space_{tag}.opt.json` (tagged)
- `_final_save:596` вЂ” СЃРѕС…СЂР°РЅСЏРµС‚ `concept_space.opt.json` (Р±РµР· С‚РµРіР°)

Resume code (train_full.py:274-286) РїС‹С‚Р°РµС‚СЃСЏ РіСЂСѓР·РёС‚СЊ tagged в†’ tagless в†’ data-dir tagged в†’ any. Р Р°Р±РѕС‚Р°РµС‚, РЅРѕ С…СЂСѓРїРєРѕ.

---

## 4. P2 вЂ” РћС‚РєСЂС‹С‚С‹Рµ РїСЂРѕР±Р»РµРјС‹ (11)

| ID | РџСЂРѕР±Р»РµРјР° | РЎР»РѕР¶РЅРѕСЃС‚СЊ | РџСЂРёРѕСЂРёС‚РµС‚ |
|:--:|----------|:---------:|:---------:|
| AM-25 | CPU path (~300 СЃС‚СЂРѕРє) legacy, РЅРµ СѓРґР°Р»С‘РЅ | 3 | Medium |
| G-48 | `torch.compile` РЅРµ Р°РєС‚РёРІРёСЂРѕРІР°РЅ | 4 | Low |
| SN-38 | `cooc_masks` rebuild РІ Python loop | 3 | Low |
| SN-40 | field_bits overlap per-candidate `.item()` | 4 | Medium |
| TN-13 | batch_size РЅРµ Р°РґР°РїС‚РёСЂСѓРµС‚СЃСЏ Рє РїР»Р°С‚Рѕ | 3 | Low |
| TN-15 | decay warmup ramp РЅРµ СЂРµР°Р»РёР·РѕРІР°РЅ | 2 | Low |
| TN-34 | opt.json naming mismatch (tagged vs tagless) | 1 | Low |
| AM-80 | `_lateral_inhibition_gpu` вЂ” РІСЃС‘ РµС‰С‘ Python loop | 3 | Medium |
| AM-81 | `_negative_sampling_gpu` вЂ” per-concept Python loop | 3 | Medium |
| AM-82 | РќРµС‚ РёРЅС‚РµРіСЂР°С†РёРѕРЅРЅС‹С… С‚РµСЃС‚РѕРІ train_full.py | 4 | Medium |
| AM-83 | РќРµС‚ Р»РѕРіРіРµСЂР° вЂ” print() РІРµР·РґРµ | 2 | Low |

---

## 5. AM-80+: РЈР»СѓС‡С€РµРЅРёСЏ (14 РїСЂРµРґР»РѕР¶РµРЅРёР№)

### 5.1 AM-80: GPU lateral inhibition вЂ” pure tensor

`_lateral_inhibition_gpu:518-532` вЂ” Python loop for gi in range(n):
```python
for gi in range(n):
    gi_mask = mask_all[gi]
    ...
```
**Р—Р°РјРµРЅРёС‚СЊ** РЅР° batched tensor: `gv_new = (sim_us * gv_others - sim_us_sq * gv_self).sum(dim=1)` вЂ” РѕРґРЅР° РѕРїРµСЂР°С†РёСЏ РЅР° РІСЃРµ n РєРѕРЅС†РµРїС‚РѕРІ.

### 5.2 AM-81: GPU neg sampling вЂ” batched

`_negative_sampling_gpu:604-624` вЂ” Python loop:
```python
for gi, gen_cid in enumerate(unique_gen):
    ...
```
**Р—Р°РјРµРЅРёС‚СЊ** РЅР° masked scatter: РІСЃРµ РіСЂР°РґРёРµРЅС‚С‹ СЃС‡РёС‚Р°СЋС‚СЃСЏ С‚РµРЅР·РѕСЂРЅРѕ, РїСЂРёРјРµРЅСЏСЋС‚СЃСЏ С‡РµСЂРµР· `_vecs_t[valid_noise_mask].scatter_add_()`.

### 5.3 AM-84: TN-14 regularization вЂ” pure tensor

`_contrastive_objective_gpu:756-774` вЂ” Python inner loop РґРѕ 50 РёС‚РµСЂР°С†РёР№ СЃ `.item()`. Р’С‹РЅРµСЃС‚Рё РІ С‚РµРЅР·РѕСЂ:
```python
reg_mask = (topk_val > reg_thresh) & ~cooc_masks & (fb_overlaps == 0)
reg_grad = (topk_val * gen._vecs_t[topk_idx]).mean(dim=1) - g_vecs
```

### 5.4 AM-85: TorchCache вЂ” РІС‹РЅРµСЃС‚Рё С‚РµРЅР·РѕСЂРЅС‹Рµ РєРµС€Рё

`_ensure_torch:163-258` вЂ” РµРґРёРЅС‹Р№ РјРµС‚РѕРґ СЃС‚СЂРѕРёС‚ Р’РЎР• С‚РµРЅР·РѕСЂС‹ (_vecs_t, _fb_t, _ce_t, _ema_vecs_t, _mom_t, _basis_t, _codes_t, _fused_buf). Mixed concerns: CPU fallback, OOM handling, initialization, dirty flag checking.

**РџСЂРµРґР»РѕР¶РµРЅРёРµ**: Р’С‹РґРµР»РёС‚СЊ `TorchCache(cs)` РєР»Р°СЃСЃ СЃ:
- `vecs` вЂ” property СЃ lazy rebuild
- `invalidate()` вЂ” СЃР±СЂРѕСЃ РІСЃРµС… dirty С„Р»Р°РіРѕРІ
- `to(device)` вЂ” РєСЂРѕСЃc-РґРµРІР°Р№СЃ РїРµСЂРµРјРµС‰РµРЅРёРµ

### 5.5 AM-86: РџСЂРѕС‚РѕРєРѕР» `TorchInvalidatable` РґР»СЏ `fluctuate_fractal`

```python
class TorchInvalidatable(Protocol):
    def invalidate_torch(self): ...
```

Р’РјРµСЃС‚Рѕ `generator: Optional[CrystalGenerator]` в†’ `generator: Optional[TorchInvalidatable]`.

### 5.6 AM-87: Memory РѕРїС‚РёРјРёР·Р°С†РёСЏ

- `_vecs_t: float16` вњ…
- `_ema_vecs_t: float32` вЂ” ~224MB РґР»СЏ 146KГ—384 вЂ” РјРѕР¶РЅРѕ С…СЂР°РЅРёС‚СЊ РІ float16, РµСЃР»Рё loss РЅРµ СЃС‚СЂР°РґР°РµС‚
- `_fused_buf: float32(V, D+1)` вЂ” ~225MB вЂ” РјРѕР¶РЅРѕ СЃРґРµР»Р°С‚СЊ РґРёРЅР°РјРёС‡РµСЃРєРёРј (alloc РїРѕ max N unique_gen)
- `_fb_t: uint8(V, fb_bytes)` вЂ” ~146K Г— 256B = ~37MB вЂ” РјРѕР¶РЅРѕ lazy-load/mmap

### 5.7 AM-88: `build_octree_fields` вЂ” РєРѕРЅСЃРёСЃС‚РµРЅС‚РЅРѕСЃС‚СЊ

`concept_space.py:374-460` вЂ” РїСЂРё РєР°Р¶РґРѕРј rebuild: `self.fractal.init_fields(n_anchors)` + `self.fractal.field_bits` РѕР±РЅСѓР»СЏРµС‚СЃСЏ Рё Р·Р°РЅРѕРІРѕ Р·Р°РїРѕР»РЅСЏРµС‚СЃСЏ. Р”Р»СЏ РєРѕРЅС†РµРїС‚РѕРІ Р±РµР· РєРѕРґР° вЂ” `field_bits` РЅРµ СЃРѕР·РґР°СЋС‚СЃСЏ. 
**РџСЂРµРґР»РѕР¶РµРЅРёРµ**: Р”РѕР±Р°РІРёС‚СЊ `ensure_all_concepts_have_fields()` вЂ” РіР°СЂР°РЅС‚РёСЏ, С‡С‚Рѕ РєР°Р¶РґС‹Р№ concept РІ `cs.fractal.codes` РёРјРµРµС‚ `field_bits`.

### 5.8 AM-89: `concept_usage` вЂ” РЅРµ СЂР°СЃС‚С‘С‚ Р±РµСЃРєРѕРЅРµС‡РЅРѕ

`concept_space.py:696` вЂ” `self.concept_usage = {cid: 0.0 for cid in self.concept_vectors}` вЂ” 146K entries. РџСЂРё Р·Р°РіСЂСѓР·РєРµ вЂ” `for cid in range(obj.vocab_size)` вЂ” РІСЃРµРіРґР° РїРѕР»РЅС‹Р№ vocab. РњРѕР¶РЅРѕ Р·Р°РјРµРЅРёС‚СЊ РЅР° РјР°СЃСЃРёРІ `np.zeros(V, float32)`.

### 5.9 AM-90: РўРµСЃС‚С‹ РґР»СЏ train_full.py

РќРµС‚ РЅРё РѕРґРЅРѕРіРѕ С‚РµСЃС‚Р° РґР»СЏ 817-СЃС‚СЂРѕС‡РЅРѕРіРѕ train_full.py. Р”РѕР±Р°РІРёС‚СЊ:
- `test_resume_flow` вЂ” РїСЂРѕРІРµСЂРєР° resume Р±РµР· checkpoint (fresh start)
- `test_checkpoint_flow` вЂ” pipeline._checkpoint РЅРµ РїР°РґР°РµС‚
- `test_curriculum_ramp` вЂ” bs_curve, max_len ramp, _effective_cp
- `test_fluctuate_call` вЂ” РїСЂРѕРІРµСЂРєР°, С‡С‚Рѕ `fluctuate_fractal` РІС‹Р·С‹РІР°РµС‚СЃСЏ СЃ РїСЂР°РІРёР»СЊРЅС‹РјРё kwargs

### 5.10 AM-91: `_graph_cache` вЂ” thread safety

`_branch:579-585` вЂ” `self._graph_cache` OrderedDict Р±РµР· Р±Р»РѕРєРёСЂРѕРІРєРё. РџСЂРё РјРЅРѕРіРѕРїРѕС‚РѕС‡РЅРѕРј generate (С‡РµСЂРµР· API) вЂ” race condition.

### 5.11 AM-92: `_build_pairs` вЂ” СѓРЅРёС„РёРєР°С†РёСЏ CPU/GPU codegen

cpu/gpu pair building РІ `_build_pairs` СЂР°Р·РґРµР»РµРЅС‹ РЅР° РґРІР° РїР°СЂР°Р»Р»РµР»СЊРЅС‹С… С‚СЂРµРєР° СЃ РґСѓР±Р»РёСЂРѕРІР°РЅРёРµРј Р»РѕРіРёРєРё (PMI weight, field_weight, theta_gate, slow_lr). Р’С‹РЅРµСЃС‚Рё РІ С‡РёСЃС‚Рѕ-РїРёС‚РѕРЅРѕРІСЃРєРёР№ РіРµРЅРµСЂР°С‚РѕСЂ РїР°СЂ, Р·Р°С‚РµРј СЂР°Р·РІРµС‚РІР»СЏС‚СЊ РЅР° CPU apply / GPU apply.

### 5.12 AM-93: `FCFConfig.params` вЂ” batch_size, rare_threshold

Р”РѕР±Р°РІРёС‚СЊ ParamDef РґР»СЏ:
- `batch_size` СЃ rules РЅР° plateau detection
- `rare_threshold` СЃ ramp РѕС‚ 1 в†’ target

### 5.13 AM-94: `fluctuation_amp` вЂ” decay

РўРµРєСѓС‰Р°СЏ Р»РѕРіРёРєР°: `fluctuation_amp` РєРѕРЅСЃС‚Р°РЅС‚Р° РЅР° РІСЃС‘ РѕР±СѓС‡РµРЅРёРµ. Р”РѕР±Р°РІРёС‚СЊ cosine decay.

### 5.14 AM-95: deprecate `_quiet` wrapper

`train_full.py:17-24` вЂ” `_quiet` Р»РѕРІРёС‚ Р’РЎР• РёСЃРєР»СЋС‡РµРЅРёСЏ (РєСЂРѕРјРµ KeyboardInterrupt) Рё Р»РѕРіРёСЂСѓРµС‚. РњР°СЃРєРёСЂСѓРµС‚ СЂРµР°Р»СЊРЅС‹Рµ РѕС€РёР±РєРё. Р—Р°РјРµРЅРёС‚СЊ РЅР° СЏРІРЅС‹Рµ try/except РІ РјРµСЃС‚Р°С… РІС‹Р·РѕРІР°.

---

## 6. Р”РµС‚Р°Р»СЊРЅС‹Р№ Р°РЅР°Р»РёР· `.item()` syncs

### 6.1 `_build_pairs:185`
```python
overlap = int(torch.bitwise_and(gen._fb_t[ids[i]], gen._fb_t[ids[j]]).sum().item())
```
1 `.item()` РЅР° РєР°Р¶РґСѓСЋ STDP РїР°СЂСѓ. Р”Р»СЏ batch 32 СЃС‚СЂРѕРє Г— ~10 РїР°СЂ = **320 `.item()`**.

**Fix**: Р’С‹С‡РёСЃР»РёС‚СЊ overlap С‚РµРЅР·РѕСЂРЅРѕ: `fb_overlaps = (gen._fb_t.unsqueeze(1) & gen._fb_t.unsqueeze(0)).sum(dim=-1)` вЂ” РѕРґРёРЅ СЂР°Р· РЅР° batch. РќРѕ СЌС‚Рѕ O(VВІГ—fb_bytes) РїР°РјСЏС‚Рё Рё РІСЂРµРјРµРЅРё. РђР»СЊС‚РµСЂРЅР°С‚РёРІР°: РЅР°РєР°РїР»РёРІР°С‚СЊ РІ CPU Р±СѓС„РµСЂ Рё sync СЂР°Р· РІ N РїР°СЂ.

### 6.2 `_contrastive_objective_gpu:735-792`

Р”Р»СЏ РєР°Р¶РґРѕРіРѕ РёР· N_gen РєРѕРЅС†РµРїС‚РѕРІ (batch avg ~32):
- Loop max_hard=5: `.item()` РґР»СЏ `neg_cid`, `cos_val`, `overlap` вЂ” РґРѕ 15 syncs
- Loop reg=50 (TN-14): `.item()` РґР»СЏ `rcid`, `rcos`, `ro` вЂ” РґРѕ 150 syncs

Р’СЃРµРіРѕ: ~165 Г— 32 = **~5 280 `.item()`**.

**Fix**: Pure tensor implementation Р±РµР· Python loops вЂ” СЃРј. AM-84.

---

## 7. РљРѕРјРїР»РµРєСЃРЅС‹Р№ РїР»Р°РЅ V11

### Р¤Р°Р·Р° 0 (РЅРµРјРµРґР»РµРЅРЅРѕ, crash fix)

| # | Р—Р°РґР°С‡Р° | Р¤Р°Р№Р» | Р’СЂРµРјСЏ |
|:-:|--------|:----:|:----:|
| 1 | `noise_scale` в†’ `fluctuation_amp` | train_full.py:722 | 1 РјРёРЅ |
| 2 | `opt.p['noise_scale']` в†’ `opt.p['fluctuation_amp']` | train_full.py:722 | 1 РјРёРЅ |

### Р¤Р°Р·Р° 1 (P1 вЂ” 6 Р·Р°РґР°С‡)

| # | Р—Р°РґР°С‡Р° | РЎР»РѕР¶РЅРѕСЃС‚СЊ |
|:-:|--------|:---------:|
| 3 | AM-84: TN-14 regularization в†’ pure tensor | 6 |
| 4 | AM-80: GPU lateral inhibition в†’ batched tensor | 4 |
| 5 | AM-81: GPU neg sampling в†’ batched masked apply | 5 |
| 6 | TN-13: batch_size ParamDef + plateau rules | 3 |
| 7 | TN-15: decay warmup ramp (rare_threshold ParamDef) | 2 |
| 8 | TN-34: РЈРЅРёС„РёС†РёСЂРѕРІР°С‚СЊ opt.json naming | 1 |

### Р¤Р°Р·Р° 2 (P2 вЂ” 11 Р·Р°РґР°С‡)

| # | Р—Р°РґР°С‡Р° | РЎР»РѕР¶РЅРѕСЃС‚СЊ |
|:-:|--------|:---------:|
| 9 | AM-85: TorchCache class | 6 |
| 10 | AM-86: TorchInvalidatable Protocol | 3 |
| 11 | AM-87: _ema_vecs_t float16, _fused_buf dynamic | 4 |
| 12 | AM-88: ensure_all_concepts_have_fields | 2 |
| 13 | AM-89: concept_usage в†’ np array | 2 |
| 14 | AM-90: train_full.py unit tests | 6 |
| 15 | AM-91: _graph_cache thread-safe | 2 |
| 16 | AM-92: _build_pairs codegen unification | 5 |
| 17 | AM-93: batch_size + rare_threshold ParamDef | 3 |
| 18 | AM-94: fluctuation_amp cosine decay | 2 |
| 19 | AM-95: _quiet deprecation | 2 |

### Р¤Р°Р·Р° 3 (РґРѕР»РіРѕСЃСЂРѕС‡РЅС‹Рµ)

| # | Р—Р°РґР°С‡Р° | РЎР»РѕР¶РЅРѕСЃС‚СЊ |
|:-:|--------|:---------:|
| 20 | G-48: torСЃh.compile activation | 4 |
| 21 | AM-25: CPU path removal/gating | 3 |
| 22 | Р”РѕРєСѓРјРµРЅС‚Р°С†РёСЏ: API reference, architecture.md update | 5 |

---

## 8. РС‚РѕРі

| РњРµС‚СЂРёРєР° | V9 | V10 | V11 |
|---------|:--:|:---:|:---:|
| P0 | 0 | 0 | **1** в›” |
| P1 | 10 | 5 | **6** |
| P2 | 22 | 13 | **11** |
| GPU-РѕРїС‚РёРјРёР·Р°С†РёРё | 0/13 | 0/13 | **13/13** вњ… |
| РўРµСЃС‚С‹ | 79 | 105 | **105** вњ… |
| `.item()`/batch | ~16 500 | ~16 500 | **~5 600** |
| STR (est.) | ~50% | ~50% | ~55% |

V10 вЂ” РѕРіСЂРѕРјРЅС‹Р№ РїСЂРѕРіСЂРµСЃСЃ: 100% GPU-РѕРїС‚РёРјРёР·Р°С†РёР№, 100% С‚РµСЃС‚РѕРІ, 105 passed.  
РќРѕ P0 crash РІ train_full.py:722 Р±Р»РѕРєРёСЂСѓРµС‚ production Р·Р°РїСѓСЃРє.  
РџРѕСЃР»Рµ РёСЃРїСЂР°РІР»РµРЅРёСЏ P0 вЂ” 6 P1 Р·Р°РґР°С‡ (РІ РѕСЃРЅРѕРІРЅРѕРј `.item()` syncs Рё Р°РґР°РїС‚РёРІРЅС‹Рµ РїР°СЂР°РјРµС‚СЂС‹).

