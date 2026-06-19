# FCF Training-Dynamics Report V11 вЂ” V10 Post-Commit Audit

**Р”Р°С‚Р°**: 2026-06-19
**РђРіРµРЅС‚**: Training-Dynamics Agent
**Р’РµСЂСЃРёСЏ**: V11 (Р°СѓРґРёС‚ РєРѕРјРјРёС‚Р° 525688b вЂ” V10 TD fixes)

---

## 1. РџСЂРѕРІРµСЂРєР° V10 TD Fixes (Р·Р°СЏРІР»РµРЅС‹ РІ V10 report, СЂР°Р·РґРµР» 5)

| ID | РЎС‚Р°С‚СѓСЃ РІ V10 report | РЎС‚Р°С‚СѓСЃ РІ РєРѕРґРµ (525688b) | Р”РµС‚Р°Р»Рё |
|:---|:-------------------:|:----------------------:|--------|
| **TN-31** checkpoint_state С‡РµРєРїРѕРёРЅС‚С‹ | рџџЎ РќРћР’Р«Р™ P1 | вњ… **FIXED** | `_checkpoint()` Р°С‚РѕРјР°СЂРЅРѕ СЃРѕС…СЂР°РЅСЏРµС‚ `.tmp` + `os.replace` |
| **TN-33** pipeline.global_step sync | рџџЎ РќРћР’Р«Р™ P2 | вњ… **FIXED** | `pipeline.global_step = global_step` + РїР°СЂР°РјРµС‚СЂ `_checkpoint(..., global_step)` |
| **REG-V9-7** noise_scale split | рџ”ґ P1 | вљ пёЏ **Р§РђРЎРўРР§РќРћ** | ParamDef split вњ…, РЅРѕ **train_full.py:722 РїСЂРѕРїСѓС‰РµРЅ** (СЃРј. TN-40) |
| **REG-V9-8** momentum_mu РёР· `opt.p[]` | вљ пёЏ P2 | вљ пёЏ **Р§РђРЎРўРР§РќРћ** | РР· `CFG.momentum_u` (РЅРµ С…Р°СЂРґРєРѕРґ), РЅРѕ РЅРµ РёР· `opt.p['momentum_mu']` |
| **TN-34** opt.json naming | рџџЎ РќРћР’Р«Р™ P2 | вњ… **MITIGATED** | 4-СѓСЂРѕРІРЅРµРІС‹Р№ fallback вЂ” СЂР°Р±РѕС‡РёР№ |
| **TN-32** rescore idx=-1 curriculum | рџџЎ РќРћР’Р«Р™ P2 | вљ пёЏ **MITIGATED** | `_rescore_cp` СЃРѕС…СЂР°РЅСЏРµС‚ curriculum, РЅРѕ `idx=-1` РѕСЃС‚Р°С‘С‚СЃСЏ |
| **TN-35** _rescore_lines СЂРµРєРѕРјРїСЊСЋС‚ | рџџў РќРћР’Р«Р™ P3 | вќЊ **NOT FIXED** | РќРµ СЂРµР°Р»РёР·РѕРІР°РЅ |
| **TN-13** Progressive BS plateaus | вќЊ P2 | вќЊ **NOT FIXED** | РќРµ СЂРµР°Р»РёР·РѕРІР°РЅ |
| **TN-15** Decay warmup ramp | вќЊ P2 | вќЊ **NOT FIXED** | РќРµ СЂРµР°Р»РёР·РѕРІР°РЅ |

---

## 2. TN-40 (P0): `noise_scale` KeyError вЂ” REGULAR FLUCTUATE COMPLETELY BROKEN — ✅ FIXED in 024f1aa

**Р¤Р°Р№Р»/СЃС‚СЂРѕРєР°**: `train_full.py:722`

**РЎСѓС‚СЊ**: V10 РєРѕРјРјРёС‚ (525688b) split `noise_scale` ParamDef РЅР° `gradient_noise_scale` Рё `fluctuation_amp` РІ `fcf_config.py`. РћР±РЅРѕРІР»РµРЅС‹ 2 РёР· 3 call site РІ `train_full.py`, РЅРѕ **СЂРµРіСѓР»СЏСЂРЅС‹Р№ РїРµСЂРёРѕРґРёС‡РµСЃРєРёР№ fluctuate РїСЂРѕРїСѓС‰РµРЅ**:

```python
# train_full.py:722 вЂ” РќР• РРЎРџР РђР’Р›Р•РќРћ (KeyError РїСЂРё РїРµСЂРІРѕРј FLUCTUATE_EVERY)
cs.fluctuate_fractal(noise_scale=opt.p['noise_scale'].current, ...)
```

**Р§С‚Рѕ РёСЃРїСЂР°РІР»РµРЅРѕ РІ V10:**
- РЎС‚СЂРѕРєР° 703: `noise_scale=opt.p['noise_scale'].current` в†’ `gradient_noise_scale=opt.p['gradient_noise_scale'].current` вњ…
- РЎС‚СЂРѕРєР° 770: `noise_scale=opt.p['noise_scale'].current` в†’ `fluctuation_amp=opt.p['fluctuation_amp'].current` вњ…

**Р”РІРѕР№РЅР°СЏ РїРѕР»РѕРјРєР° СЃС‚СЂРѕРєРё 722:**
1. `opt.p['noise_scale']` вЂ” KeyError: `noise_scale` Р±РѕР»СЊС€Рµ РЅРµ Р·Р°СЂРµРіРёСЃС‚СЂРёСЂРѕРІР°РЅ РІ `ParameterOptimizer.p`
2. `noise_scale=` вЂ” TypeError: `fluctuate_fractal()` РЅРµ РїСЂРёРЅРёРјР°РµС‚ `noise_scale` kwargs (РїР°СЂР°РјРµС‚СЂ РЅР°Р·С‹РІР°РµС‚СЃСЏ `fluctuation_amp`)

**РџРѕСЃР»РµРґСЃС‚РІРёСЏ:**
- РџРµСЂРІС‹Р№ Р¶Рµ `idx % FLUCTUATE_EVERY == 0` (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ 2000) РІС‹Р·С‹РІР°РµС‚ **unhandled crash**
- Р РµРґРєРёР№ СЃС†РµРЅР°СЂРёР№: РµСЃР»Рё `FLUCTUATE_EVERY > CHECKPOINT_EVERY` Рё С‚СЂРµРЅРёСЂРѕРІРєР° РЅРµ РґРѕС…РѕРґРёС‚ РґРѕ 2000 СЃС‚СЂРѕРє, Р±Р°Рі РЅРµ РїСЂРѕСЏРІР»СЏРµС‚СЃСЏ
- Force-fluctuate (full stuck) СЂР°Р±РѕС‚Р°РµС‚ вњ… вЂ” СЌС‚РѕС‚ call site РёСЃРїСЂР°РІР»РµРЅ

**Fix:**
```python
# train_full.py:722
cs.fluctuate_fractal(fluctuation_amp=opt.p['fluctuation_amp'].current,
                     decay=opt.p['decay_rate'].current,
                     repel_strength=opt.p['repel_strength'].current,
                     generator=gen)
```

---

## 3. TN-41 (P2): `get_lr(idx)` СЃР±СЂР°СЃС‹РІР°РµС‚СЃСЏ РЅР° warmup РїРѕСЃР»Рµ rescore

**Р¤Р°Р№Р»/СЃС‚СЂРѕРєР°**: `train_full.py:666`

**РЎСѓС‚СЊ**: `_rescore_cp` РјРµС…Р°РЅРёР·Рј (TN-32) РєРѕСЂСЂРµРєС‚РЅРѕ СЃРѕС…СЂР°РЅСЏРµС‚ `batch_size` Рё `max_len` РїРѕСЃР»Рµ `idx=-1`, РЅРѕ `gen.train_lr = get_lr(idx)` (СЃС‚СЂРѕРєР° 666) РёСЃРїРѕР»СЊР·СѓРµС‚ **СЃС‹СЂРѕР№ idx**, РЅРµ `_effective_cp`:

```python
# idx=0 РїРѕСЃР»Рµ rescore
gen.train_lr = get_lr(0)  # в†’ lr_warmup_lines: (0+1) / lr_warmup_lines в†’ РјРёРЅРёРјР°Р»СЊРЅС‹Р№ warmup
```

**Р­С„С„РµРєС‚**: РџРѕСЃР»Рµ РєР°Р¶РґРѕРіРѕ rescore LR РїР°РґР°РµС‚ РґРѕ СѓСЂРѕРІРЅСЏ warmup (в‰€ `base_lr / 1000`), С…РѕС‚СЏ `_rescore_cp` СѓРґРµСЂР¶РёРІР°РµС‚ batch_size/max_len РЅР° СѓСЂРѕРІРЅРµ pre-rescore.

**Р¦РµРїРѕС‡РєР°:**
1. Checkpoint РЅР° idx=5000: `_rescore_cp = _curriculum_p(5001) = 0.171`
2. `idx = -1` в†’ `idx = 0` в†’ `get_lr(0)` = `opt.p['full_lr'].current * 1/1000`
3. `_effective_cp(0)` = 0.171 (СЃРѕС…СЂР°РЅС‘РЅ), batch_size=12, max_len=~170K
4. LR СЂР°СЃС‚С‘С‚ С‚РѕР»СЊРєРѕ РµСЃС‚РµСЃС‚РІРµРЅРЅРѕ С‡РµСЂРµР· cosine annealing СЃ restart

**Fix:**
```python
gen.train_lr = get_lr(idx) * _eff_cp + gen.train_lr * (1 - _eff_cp)
# РР»Рё: РїРµСЂРµРґР°РІР°С‚СЊ idx С‡РµСЂРµР· _eff_cp СЃ РїРµСЂРµСЃС‡С‘С‚РѕРј
```

---

## 4. TN-42 (P3): `last_fluct_lines` desync РїРѕСЃР»Рµ rescore

**Р¤Р°Р№Р»/СЃС‚СЂРѕРєР°**: `train_full.py:726`

**РЎСѓС‚СЊ**: РџРѕСЃР»Рµ `idx=-1` (rescore) в†’ `idx=0`, `last_fluct_lines` РѕСЃС‚Р°С‘С‚СЃСЏ РЅР° pre-rescore Р·РЅР°С‡РµРЅРёРё (РЅР°РїСЂРёРјРµСЂ, 5000). РЎР»РµРґСѓСЋС‰РёР№ `idx - last_fluct_lines` РѕС‚СЂРёС†Р°С‚РµР»РµРЅ, `if idx > 0` РЅРµ СЃСЂР°Р±Р°С‚С‹РІР°РµС‚, РЅРѕ fluctuate СЃРґРІРёРіР°РµС‚СЃСЏ:

```
pre-rescore: last_fluct_lines = 5000
post-rescore: idx = 1 в†’ idx - 5000 = -4999 < FLUCTUATE_EVERY в†’ skip
             idx = 7000 в†’ idx - 5000 = 2000 >= FLUCTUATE_EVERY в†’ trigger
```

**Р’Р»РёСЏРЅРёРµ**: fluctuate РЅР° Р±РѕР»СЊС€РёС… РёРЅС‚РµСЂРІР°Р»Р°С… РЅРµ СЂР°РІРЅРѕРјРµСЂРµРЅ вЂ” РїРѕСЃР»Рµ rescore СЃР»РµРґСѓСЋС‰РёР№ fluctuate Р·Р°РґРµСЂР¶РёРІР°РµС‚СЃСЏ. РќРµРєСЂРёС‚РёС‡РЅРѕ, РЅРѕ РЅР°СЂСѓС€Р°РµС‚ РїСЂРµРґРїРѕР»РѕР¶РµРЅРёСЏ `fluctuation_amp` Р°РґР°РїС‚Р°С†РёРё.

**Fix:**
```python
last_fluct_lines = min(last_fluct_lines, idx)  # РїРѕСЃР»Рµ rescore СЃР±СЂРѕСЃРёС‚СЊ
```

---

## 5. TN-43 (P2): `momentum_mu` РЅРµ Р°РґР°РїС‚РёРІРµРЅ

**Р¤Р°Р№Р»/СЃС‚СЂРѕРєР°**: `train_full.py:704`, `fcf_config.py:425`

**РЎСѓС‚СЊ**: `momentum_mu` РїРµСЂРµРґР°С‘С‚СЃСЏ РєР°Рє `CFG.momentum_mu = 0.9` (СЃС‚Р°С‚РёС‡РµСЃРєР°СЏ РєРѕРЅСЃС‚Р°РЅС‚Р°). V9 report С‚СЂРµР±РѕРІР°Р» `opt.p['momentum_mu'].current`. V10 РёСЃРїСЂР°РІРёР» С‚РѕР»СЊРєРѕ `CFG` РІРјРµСЃС‚Рѕ С…Р°СЂРґРєРѕРґР°.

**Р—Р°С‡РµРј Р°РґР°РїС‚РёРІРЅС‹Р№ momentum_mu:**
- РќР° СЂР°РЅРЅРёС… СЌС‚Р°РїР°С…: РЅРёР·РєРёР№ momentum (0.5) вЂ” Р±С‹СЃС‚СЂР°СЏ Р°РґР°РїС‚Р°С†РёСЏ
- РќР° РїРѕР·РґРЅРёС… СЌС‚Р°РїР°С…: РІС‹СЃРѕРєРёР№ momentum (0.95) вЂ” СЃС‚Р°Р±РёР»СЊРЅРѕСЃС‚СЊ
- `ParameterOptimizer` СѓР¶Рµ РёРјРµРµС‚ РјРµС…Р°РЅРёР·Рј rule-based adaptation

**Fix:**
```python
# fcf_config.py РґРѕР±Р°РІРёС‚СЊ:
ParamDef('momentum_mu',     0.5,    0.99,   0.9,   0.05, rules=[
    AdaptRule('cos_flat >= 5', 'momentum_mu', 'shift', 0.02),
    AdaptRule('cos_trend > 0.001 and mean_cos > 0.01', 'momentum_mu', 'shift', 0.01),
]),

# train_full.py:704
momentum_mu=opt.p['momentum_mu'].current,
```

---

## 6. РњР°С‚СЂРёС†Р° СЃС‚Р°С‚СѓСЃР° V11

| РћР±Р»Р°СЃС‚СЊ | РЎС‚Р°С‚СѓСЃ | РџСЂРёРѕСЂРёС‚РµС‚ |
|:--------|:------:|:---------:|
| **TN-31** checkpoint_state РїСЂРё С‡РµРєРїРѕРёРЅС‚Р°С… | вњ… FIXED | вЂ” |
| **TN-33** pipeline.global_step sync | вњ… FIXED | вЂ” |
| **REG-V9-7** noise_scale split (fcf_config) | вњ… DONE | вЂ” |
| **REG-V9-7** noise_scale split (train_full:703) | вњ… FIXED | вЂ” |
| **REG-V9-7** noise_scale split (train_full:770) | вњ… FIXED | вЂ” |
| **TN-34** opt.json naming | вњ… MITIGATED | вЂ” |
| **REG-V9-8** momentum_mu РёР· CFG | вњ… DONE | вЂ” |
| **TN-32** rescore idx=-1 (_rescore_cp) | вљ пёЏ MITIGATED | P2 |
| **TN-40** `noise_scale` KeyError (train_full:722) | рџ”ґ **РќРћР’Р«Р™** | **P0** |
| **TN-41** get_lr(idx) warmup after rescore | рџџЎ **РќРћР’Р«Р™** | P2 |
| **TN-42** last_fluct_lines desync | рџџў **РќРћР’Р«Р™** | P3 |
| **TN-43** momentum_mu РЅРµ Р°РґР°РїС‚РёРІРµРЅ | рџџў **РќРћР’Р«Р™** | P2 |
| **TN-13** Progressive BS plateaus | вќЊ РќРµ СЂРµР°Р»РёР·РѕРІР°РЅ | P2 |
| **TN-15** Decay warmup ramp | вќЊ РќРµ СЂРµР°Р»РёР·РѕРІР°РЅ | P2 |
| **TN-35** _rescore_lines СЂРµРєРѕРјРїСЊСЋС‚ | вќЊ РќРµ СЂРµР°Р»РёР·РѕРІР°РЅ | P3 |

---

## 7. РџСЂРµРґР»РѕР¶РµРЅРёСЏ TN-40+

### TN-40 (P0): РСЃРїСЂР°РІРёС‚СЊ `noise_scale` в†’ `fluctuation_amp` РІ train_full.py:722
```python
cs.fluctuate_fractal(fluctuation_amp=opt.p['fluctuation_amp'].current, ...)
```

### TN-41 (P2): `get_lr` РЅРµ РґРѕР»Р¶РµРЅ СЃР±СЂР°СЃС‹РІР°С‚СЊСЃСЏ РЅР° warmup РїРѕСЃР»Рµ rescore
Р›РёР±Рѕ:
- РЎРѕС…СЂР°РЅСЏС‚СЊ `train_lr` РІ `_rescore_cp` Рё РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ `pipeline._rescore_lr`
- Р›РёР±Рѕ РЅРµ СЃР±СЂР°СЃС‹РІР°С‚СЊ `idx` РґРѕ 0, Р° РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ `idx = start_line - 1`

### TN-42 (P3): РЎРёРЅС…СЂРѕРЅРёР·РёСЂРѕРІР°С‚СЊ `last_fluct_lines` РїРѕСЃР»Рµ rescore
```python
last_fluct_lines = min(last_fluct_lines, idx)
```

### TN-43 (P2): `momentum_mu` С‡РµСЂРµР· `opt.p` (Р°РґР°РїС‚РёРІРЅС‹Р№)
Р”РѕР±Р°РІРёС‚СЊ `ParamDef('momentum_mu', ...)` РІ fcf_config.py Рё РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ `opt.p['momentum_mu'].current`.

### TN-44 (P2): РЈРґР°Р»РёС‚СЊ `idx=-1` вЂ” Р·Р°РјРµРЅРёС‚СЊ РЅР° `idx = idx` (no-op)
Rescore РґРѕР»Р¶РµРЅ СЃРѕСЂС‚РёСЂРѕРІР°С‚СЊ _remaining_ lines, РЅРµ РїРµСЂРµР·Р°РїСѓСЃРєР°СЏ С†РёРєР»:
```python
epoch_train = epoch_train[:idx+1] + _rescore_lines(epoch_train[idx+1:], gen)
# idx РЅРµ РјРµРЅСЏРµС‚СЃСЏ
```
РўРµРєСѓС‰РёР№ `_rescore_cp` workaround вЂ” РєРѕСЃС‚С‹Р»СЊ. РџСЂРёС‡РёРЅР° (`idx=-1`) РЅРµ РёСЃРїСЂР°РІР»РµРЅР°.

### TN-45 (P3): `pipeline.last_fluct_lines` вЂ” РјС‘СЂС‚РІС‹Р№ РєРѕРґ
`TrainingPipeline.__init__` СѓСЃС‚Р°РЅР°РІР»РёРІР°РµС‚ `self.last_fluct_lines = 0`, РЅРѕ СЌС‚Рѕ РїРѕР»Рµ РЅРёРіРґРµ РЅРµ С‡РёС‚Р°РµС‚СЃСЏ. РЈРґР°Р»РёС‚СЊ.

---

## 8. Р’С‹РІРѕРґ

**V10 РєРѕРјРјРёС‚ (525688b) РёСЃРїСЂР°РІРёР» 3 РёР· 6 Р·Р°СЏРІР»РµРЅРЅС‹С… TD-РїСЂРѕР±Р»РµРј:**

| Р—Р°СЏРІР»РµРЅРѕ | РСЃРїСЂР°РІР»РµРЅРѕ |
|:---------|:----------:|
| TN-31 | вњ… |
| TN-33 | вњ… |
| REG-V9-7 noise_scale split | вљ пёЏ **2/3 call sites** |
| REG-V9-8 momentum_mu | вљ пёЏ Р§Р°СЃС‚РёС‡РЅРѕ |
| TN-34 | вњ… Mitigated |
| TN-32 idx=-1 | вљ пёЏ Mitigated |

**РљСЂРёС‚РёС‡РµСЃРєРёР№ РЅРѕРІС‹Р№ Р±Р°Рі TN-40 (P0)**: `train_full.py:722` СЃСЃС‹Р»Р°РµС‚СЃСЏ РЅР° `opt.p['noise_scale']`, РєРѕС‚РѕСЂРѕРіРѕ Р±РѕР»СЊС€Рµ РЅРµС‚. Р РµРіСѓР»СЏСЂРЅС‹Р№ РїРµСЂРёРѕРґРёС‡РµСЃРєРёР№ fluctuate СѓРїР°РґС‘С‚ СЃ KeyError РїСЂРё РїРµСЂРІРѕРј Р¶Рµ `FLUCTUATE_EVERY`.

**Р’С‚РѕСЂРёС‡РЅР°СЏ РїСЂРѕР±Р»РµРјР° TN-41 (P2)**: РџРѕСЃР»Рµ rescore (TN-32) `get_lr(0)` РґР°С‘С‚ warmup LR, СЃРІРѕРґСЏ РЅР° РЅРµС‚ `_rescore_cp` СЃРѕС…СЂР°РЅРµРЅРёРµ curriculum.

### Р РµРєРѕРјРµРЅРґСѓРµРјС‹Р№ РїРѕСЂСЏРґРѕРє СЃР»РµРґСѓСЋС‰РµР№ РёС‚РµСЂР°С†РёРё:
1. **TN-40** (P0) ✅ FIXED in 024f1aa вЂ” РёСЃРїСЂР°РІРёС‚СЊ `noise_scale` в†’ `fluctuation_amp` РІ train_full.py:722
2. **TN-41** (P2) вЂ” РЅРµ СЃР±СЂР°СЃС‹РІР°С‚СЊ LR РЅР° warmup РїРѕСЃР»Рµ rescore
3. **TN-44** (P2) вЂ” Р·Р°РјРµРЅРёС‚СЊ `idx=-1` РЅР° `epoch_train = epoch_train[:idx+1] + _rescore_lines(...)`
4. **TN-43** (P2) вЂ” momentum_mu РІ opt.p
5. **TN-45** (P3) вЂ” СѓРґР°Р»РёС‚СЊ РјС‘СЂС‚РІС‹Р№ `self.last_fluct_lines`
6. **TN-42** (P3) вЂ” last_fluct_lines sync РїРѕСЃР»Рµ rescore
7. **TN-13** (P2) вЂ” Progressive BS plateaus
8. **TN-15** (P2) вЂ” Decay warmup ramp



