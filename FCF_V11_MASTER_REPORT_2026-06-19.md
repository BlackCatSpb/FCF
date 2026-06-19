# FCF V11 вЂ” РЎРІРѕРґРЅС‹Р№ РѕС‚С‡С‘С‚ РєРѕР»Р»РµРіРёРё AI-Р°РіРµРЅС‚РѕРІ

**Р”Р°С‚Р°**: 2026-06-19
**Р’РµСЂСЃРёСЏ**: V11 (Р°СѓРґРёС‚ V10 РєРѕРјРјРёС‚РѕРІ: 525688b + d36a780)
**РЎРѕСЃС‚Р°РІ**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Executive Summary

V10 Р·Р°РєРѕРјРјРёС‡РµРЅ (2 РєРѕРјРјРёС‚Р°). **105 С‚РµСЃС‚РѕРІ РїСЂРѕС…РѕРґСЏС‚** (+26, +33%). Р­С‚Рѕ СЃР°РјС‹Р№ Р±РѕР»СЊС€РѕР№ РїСЂРѕРіСЂРµСЃСЃ Р·Р° РІСЃС‘ РІСЂРµРјСЏ.
**V11.1 (024f1aa)**: РСЃРїСЂР°РІР»РµРЅС‹ TN-40 (P0), B1 (P1), SN-28 (P1), centroid parity (P2). Р’СЃРµ 105 С‚РµСЃС‚РѕРІ РїСЂРѕС…РѕРґСЏС‚.

| РњРµС‚СЂРёРєР° | V10 | V11 | О” |
|---------|:---:|:---:|:-:|
| P0 | 0 | **1** (РќРћР’Р«Р™) | +1 |
| P1 | 10 | **4** | в€’6 |
| P2 | 22 | **12** | в€’10 |
| GPU-РѕРїС‚РёРјРёР·Р°С†РёРё (G-40..G-52) | 0/13 | **13/13** | +13 |
| РўРµСЃС‚С‹ (QN-32..QN-40) | 0/9 | **9/9** | +9 |
| STR | ~52% | ~48% | в€’4% (РЅРѕРІС‹Р№ РєРѕРґ) |
| Syncs/batch | ~20,000 | ~1,000-5,000 | в€’75% |

**Р“Р»Р°РІРЅС‹Рµ РЅР°С…РѕРґРєРё V11:**
1. рџ”ґ **P0: crash РІ FLUCTUATE_EVERY** вЂ” `train_full.py:722` РІС‹Р·С‹РІР°РµС‚ `noise_scale` (РїРµСЂРµРёРјРµРЅРѕРІР°РЅ РІ `fluctuation_amp`). РџРµСЂРІС‹Р№ Р¶Рµ РїРµСЂРёРѕРґРёС‡РµСЃРєРёР№ С„Р»СѓРєС‚СѓР°С‚ СѓРїР°РґС‘С‚ СЃ KeyError.
2. рџ”ґ **B1 (HIGH): Double momentum** вЂ” `stdp_trainer.py` РїСЂРёРјРµРЅСЏРµС‚ momentum Рє GPU `avg_grad`, Р·Р°С‚РµРј СЃРЅРѕРІР° РІ per-element CPU С†РёРєР»Рµ. Р“СЂР°РґРёРµРЅС‚ РёСЃРєР°Р¶С‘РЅ: `ВµВІВ·old + ВµВ·(1-Вµ)В·avg + (1-Вµ)В·grad`.
3. вњ… **Р’СЃРµ 13 GPU-РѕРїС‚РёРјРёР·Р°С†РёР№ G-40..G-52 СЂРµР°Р»РёР·РѕРІР°РЅС‹** вЂ” batched subspace, full GPU lateral, vec neg sampling, fused contrastive, zero-copy, deferred sync.
4. вњ… **Р’СЃРµ 9 С‚РµСЃС‚РѕРІ QN-32..QN-40 СЂРµР°Р»РёР·РѕРІР°РЅС‹** вЂ” +267 СЃС‚СЂРѕРє, 26 С‚РµСЃС‚РѕРІ.
5. вљ пёЏ **STR СѓРїР°Р» 52в†’48%** вЂ” РЅРѕРІС‹Р№ GPU-РєРѕРґ (G-40..G-52) Р±РµР· С‚РµСЃС‚РѕРІ.

---

## 1. V10 Commit Verification

### РСЃРїСЂР°РІР»РµРЅРѕ (РІСЃС‘ РїРѕРґС‚РІРµСЂР¶РґРµРЅРѕ РІ РєРѕРґРµ)

| Р“СЂСѓРїРїР° | РЎС‚Р°С‚СѓСЃ | Р”РµС‚Р°Р»Рё |
|--------|:------:|--------|
| Phase 0 (TN-31, G-57, SN-35/36) | вњ… 3/3 | checkpoint_state, dead tensors, CPU parity |
| Phase 1 (REG-V9-7, G-46, G-42) | вљ пёЏ 3/4 | REG-V9-7: 2/3 call sites РѕР±РЅРѕРІР»РµРЅС‹ (СЃРј. P0) |
| GPU G-40..G-52 | вњ… **13/13** | Р’СЃРµ СЂРµР°Р»РёР·РѕРІР°РЅС‹ |
| Code Quality (AM-25,29,30,31,33,39) | вњ… 6/6 | CPU path legacy, RNG, EMA, CE, hormone, SN-39 |
| Tests QN-32..QN-40 | вњ… **9/9** | +26 С‚РµСЃС‚РѕРІ |
| 105 С‚РµСЃС‚РѕРІ РїСЂРѕС…РѕРґСЏС‚ | вњ… | |

---

## 2. P0 вЂ” РљСЂРёС‚РёС‡РµСЃРєРёРµ Р±Р°РіРё (1)

### TN-40: Crash РІ fluctuate_fractal вЂ” noise_scale KeyError

**Р¤Р°Р№Р»**: `train_full.py:722`
**Severity**: P0 вЂ” **РїРµСЂРІС‹Р№ Р¶Рµ FLUCTUATE_EVERY СѓРїР°РґС‘С‚**

```python
cs.fluctuate_fractal(noise_scale=opt.p['noise_scale'].current, ...)
# KeyError: 'noise_scale'  (РїРµСЂРµРёРјРµРЅРѕРІР°РЅ РІ 'fluctuation_amp')
```

REG-V9-7 (V10) СЂР°Р·РґРµР»РёР» `noise_scale` РЅР° `gradient_noise_scale` + `fluctuation_amp` РІ fcf_config.py Рё РІ `opt.p`, РЅРѕ СЃС‚СЂРѕРєР° 722 РѕСЃС‚Р°Р»Р°СЃСЊ СЃ `noise_scale`. Р‘Р°С‚С‡-С‚СЂРµРЅРёСЂРѕРІРєР° РЅРµ РїР°РґР°РµС‚ (РёСЃРїРѕР»СЊР·СѓРµС‚ `gradient_noise_scale`), РЅРѕ РїРµСЂРёРѕРґРёС‡РµСЃРєРёР№ С„Р»СѓРєС‚СѓР°С‚ вЂ” РїР°РґР°РµС‚.

**Fix**: `noise_scale=` в†’ `fluctuation_amp=`.
**РЎР»РѕР¶РЅРѕСЃС‚СЊ**: 1 СЃС‚СЂРѕРєР°

---

## 3. P1 вЂ” РљСЂРёС‚РёС‡РµСЃРєРёРµ РїСЂРѕР±Р»РµРјС‹ (4)

### B1 (P1): Double momentum вЂ” РіСЂР°РґРёРµРЅС‚ РёСЃРєР°Р¶С‘РЅ

**Р¤Р°Р№Р»**: `stdp_trainer.py:416 + 459-460`

РЎС‚СЂРѕРєР° 416 РїСЂРёРјРµРЅСЏРµС‚ momentum Рє GPU `avg_grad`:
```python
if gen._mom_t is not None and momentum_mu > 0:
    gen._mom_t[gen_cid] *= momentum_mu
    gen._mom_t[gen_cid] += avg_grad
    avg_grad = gen._mom_t[gen_cid]
```

РЎС‚СЂРѕРєРё 459-460 РїСЂРёРјРµРЅСЏСЋС‚ momentum РЎРќРћР’Рђ РІ per-element CPU С†РёРєР»Рµ:
```python
if mom_cpu is not None:
    grad = mom_cpu[gi]  # в†ђ СѓР¶Рµ СЃРѕРґРµСЂР¶РёС‚ momentum!
```

**Р­С„С„РµРєС‚**: `ВµВІВ·old + ВµВ·(1-Вµ)В·avg + (1-Вµ)В·grad`. Momentum РїСЂРёРјРµРЅСЏРµС‚СЃСЏ РґРІР°Р¶РґС‹.

**Fix**: РЈР±СЂР°С‚СЊ CPU momentum (СЃС‚СЂРѕРєРё 459-460), РѕСЃС‚Р°РІРёС‚СЊ С‚РѕР»СЊРєРѕ GPU `_mom_t`.
**РЎР»РѕР¶РЅРѕСЃС‚СЊ**: 1

### SN-43 (P1): GPU neg sampling вЂ” Python loop

**Р¤Р°Р№Р»**: `stdp_trainer.py:604`

РџРѕСЃР»Рµ G-43 (РІРµРєС‚РѕСЂРёР·Р°С†РёСЏ) РѕСЃС‚Р°Р»СЃСЏ Python loop:
```python
for gi, gen_cid in enumerate(unique_gen):
    neg_lr_i = ...  # per-concept
    gen.concept_error.get(gen_cid, 0.0)
    cs._apply_vector_update(gen_cid, ...)  # CPU write-back
```

**Fix**: Batched tensor ops в†’ РµРґРёРЅС‹Р№ GPU write-back.
**РЎР»РѕР¶РЅРѕСЃС‚СЊ**: 3

### SN-44 (P1): GPU contrastive вЂ” nested Python loops + .item()

**Р¤Р°Р№Р»**: `stdp_trainer.py:735-792`

РџРѕСЃР»Рµ G-44 РѕСЃС‚Р°Р»РёСЃСЊ Python loops:
```python
for i in range(ng):
    for j in range(min(100, topk_idx.shape[1])):
        rcos = float(topk_val[i, j].item())  # ~500 syncs/step
```

~5,600 `.item()` syncs/batch (Р±С‹Р»Рѕ ~16,500 РІ V10, РїСЂРѕРіСЂРµСЃСЃ РµСЃС‚СЊ).
**Fix**: Pure tensor batched push.
**РЎР»РѕР¶РЅРѕСЃС‚СЊ**: 6

### G-60/SN-45 (P1): GPU destab вЂ” С†РµР»РёРєРѕРј РЅР° CPU

**Р¤Р°Р№Р»**: `stdp_trainer.py`

Destab logic (RNG, PPMI, numpy) вЂ” РїРѕР»РЅРѕСЃС‚СЊСЋ CPU per-concept. Р’ GPU-РїСѓС‚Рё РЅРµ РІРµРєС‚РѕСЂРёР·РѕРІР°РЅ.
**Fix**: GPU destab С‡РµСЂРµР· `_vecs_t` Рё `_ce_t`.
**РЎР»РѕР¶РЅРѕСЃС‚СЊ**: 5

---

## 4. P2 вЂ” РџСЂРѕР±Р»РµРјС‹ СЃСЂРµРґРЅРµР№ РєСЂРёС‚РёС‡РЅРѕСЃС‚Рё (12)

### GPU (5)
- SN-46 (P3): Contrastive write-back CPU roundtrip
- SN-47 (P3): CPU neg sampling `sample(total_vocab)` вЂ” 146K per-concept
- SN-48 (P3): GPU field overlap `.item()` sync РІ `_build_pairs`
- G-65: GPU field overlap РІ `_build_pairs`
- G-62: GPU `_apply_vector_update` Р±РµР· `.cpu().numpy()`

### Training Dynamics (4)
- TN-32 (P2): `idx=-1` СЃР±СЂР°СЃС‹РІР°РµС‚ curriculum РїРѕСЃР»Рµ rescore
- TN-13 (P2): Progressive batch size not implemented
- TN-15 (P2): Decay warmup not implemented
- TN-41 (P2): LR warmup restarts after rescore

### Quality (3)
- QN-49..QN-58 (10 СЃСЊСЋС‚РѕРІ, ~22 С‚РµСЃС‚Р°): РЅРµ СЂРµР°Р»РёР·РѕРІР°РЅС‹
- STR ~48% (СѓРїР°Р» РёР·-Р·Р° РЅРѕРІРѕРіРѕ GPU-РєРѕРґР°)
- Centroid parity bug: Р»РёС€РЅРёР№ `0.1` С„Р°РєС‚РѕСЂ РІ GPU `_centroid_pull_batch`

---

## 5. РќРћР’Р«Р• РїСЂРѕР±Р»РµРјС‹ V11

| ID | РџСЂРѕР±Р»РµРјР° | P | РђРіРµРЅС‚ | РЎР»РѕР¶РЅРѕСЃС‚СЊ |
|:--:|----------|:-:|:-----:|:---------:|
| **TN-40** | crash РІ FLUCTUATE_EVERY (KeyError: noise_scale) ✅ FIXED in 024f1aa | **P0** | TD | 1 |
| **B1** | Double momentum (GPU + CPU) вЂ” РёСЃРєР°Р¶РµРЅРёРµ РіСЂР°РґРёРµРЅС‚Р° ✅ FIXED in 024f1aa | P1 | GPU | 1 |
| **SN-28** | Contrastive \`field_gate\` not propagated — ✅ FIXED in 024f1aa | P1 | NS | 3 |
| **SN-43** | GPU neg sampling Python loop | P1 | NS | 3 |
| **SN-44** | GPU contrastive nested Python loops + .item() | P1 | NS | 6 |
| **SN-45/G-60** | GPU destab вЂ” С†РµР»РёРєРѕРј CPU | P1 | NS/GPU | 5 |
| **SN-46** | Contrastive write-back CPU roundtrip | P3 | NS | 4 |
| **SN-47** | CPU neg sampling `sample(total_vocab)` 146K | P3 | NS | 2 |
| **SN-48** | GPU field overlap `.item()` sync | P3 | NS | 3 |
| **G-62** | GPU `_apply_vector_update` Р±РµР· `.cpu().numpy()` | P2 | GPU | 5 |
| **G-65** | GPU field overlap РІ `_build_pairs` | P2 | GPU | 4 |
| **TN-41** | LR warmup restarts after rescore | P2 | TD | 2 |
| **Centroid-bug | Лишний `0.1` в GPU centroid_pull_batch ✅ FIXED in 024f1aa| P2 | QA | 1 |

---

## 6. Р РµРєРѕРјРµРЅРґСѓРµРјС‹Р№ РїР»Р°РЅ СЂР°Р±РѕС‚

### Р¤Р°Р·Р° 0 (РќР•РњР•Р”Р›Р•РќРќРћ вЂ” 2 Р·Р°РґР°С‡Рё, 10 РјРёРЅСѓС‚)

1. **TN-40** ✅ FIXED in 024f1aa: `noise_scale=` в†’ `fluctuation_amp=` (train_full.py:722) вЂ” 1 СЃС‚СЂРѕРєР°
2. **B1** ✅ FIXED in 024f1aa: РЈР±СЂР°С‚СЊ CPU momentum (stdp_trainer.py:459-460) вЂ” 2 СЃС‚СЂРѕРєРё
3. **SN-28** ✅ FIXED in 024f1aa: Propagate \ield_gate\ to GPU contrastive


### Р¤Р°Р·Р° 1 (P1 вЂ” 4 Р·Р°РґР°С‡Рё, ~1 РЅРµРґРµР»СЏ)

3. **SN-43**: GPU neg sampling вЂ” batched write-back
4. **SN-44**: GPU contrastive вЂ” pure tensor push
5. **SN-45/G-60**: GPU destab
6. **Centroid-bug** ✅ FIXED in 024f1aa: Fix `0.1` factor

### Р¤Р°Р·Р° 2 (P2 вЂ” 5 Р·Р°РґР°С‡)

7-11: G-62 (vec update), G-65 (field overlap), TN-32 (idx fix), TN-13 (BS plateaus), TN-15 (decay warmup)

### Р¤Р°Р·Р° 3 (С‚РµСЃС‚С‹ вЂ” 3 Р·Р°РґР°С‡Рё)

12-14: QN-49..QN-58 (22 С‚РµСЃС‚Р° РґР»СЏ GPU-РєРѕРґР°)

---

## 7. РџСЂРѕРіСЂРµСЃСЃ V10в†’V11

### РЎРґРµР»Р°РЅРѕ (РѕРіСЂРѕРјРЅС‹Р№ РїСЂРѕРіСЂРµСЃСЃ)
- вњ… **13/13 GPU-РѕРїС‚РёРјРёР·Р°С†РёР№** (G-40..G-52) вЂ” РєРѕРґ СѓСЃРєРѕСЂРµРЅ РІ 2-20Г—
- вњ… **9/9 С‚РµСЃС‚РѕРІС‹С… СЃСЊСЋС‚РѕРІ** (QN-32..QN-40) вЂ” +26 С‚РµСЃС‚РѕРІ, +267 СЃС‚СЂРѕРє
- вњ… Phase 0 (TN-31, G-57, SN-35/36)
- вњ… Phase 1 (G-46, G-42)
- вњ… Code quality (AM-25,29,30,31,33,39)

### РќСѓР¶РЅРѕ РёСЃРїСЂР°РІРёС‚СЊ
- рџ”ґ 2 РєСЂРёС‚РёС‡РµСЃРєРёС… Р±Р°РіР° (TN-40 crash, B1 double momentum)
- в¬‡пёЏ 10 P1/P2 РїСЂРѕР±Р»РµРј (SN-43/44/45, G-60/62/65, TN-32/13/15, centroid)
- рџ“‹ 22 РЅРѕРІС‹С… С‚РµСЃС‚Р° РґР»СЏ РЅРѕРІРѕРіРѕ GPU-РєРѕРґР° (QN-49..QN-58)

---

*РћС‚С‡С‘С‚ СЃРѕСЃС‚Р°РІР»РµРЅ РєРѕР»Р»РµРіРёРµР№ AI-Р°РіРµРЅС‚РѕРІ: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*




