# FCF V12 Архитектурный аудит — 2026-06-19

**Проверяющий**: Architect-AI
**Базовые коммиты**: V11.2 (a705223), V11.3 (d00e979), hotfixes (ac27cce, 56f69a0)
**Текущее состояние**: 122 passed, 4 failed, 3 skipped

---

## 1. V11.2 Fixes — проверка регрессий

### G-60/SN-45: GPU destabilization
- **Файл**: `stdp_trainer.py:492-510`
- Статус: ✅ **Fixed без регрессий**
- RNG на GPU через `torch.rand`, PPMI через `_vecs_t[rand_idx]`, ни одного `.item()` или `.numpy()` в hot path
- Destab маска: `torch.rand(ng) < destab_p` — векторно
- Mix GPU: `torch.where(destab_mask[:, None], acc * (1-mix) + destab_update, acc)`

### SN-43: GPU neg sampling batched
- **Файл**: `stdp_trainer.py:653-714`
- Статус: ✅ **Fixed без регрессий**
- Reuse `gen._gpu_elr_avg` через `_gpu_stdp_apply` (zero-cost, уже есть в регистрах GPU)
- Batched write-back: `gen._vecs_t[cids_batch] = vecs_batch` → один D2H
- CPU roundtrip только через `cs._apply_vector_update` (вынужденно — sync CPU dict)

### SN-44: GPU contrastive pure-tensor
- **Файл**: `stdp_trainer.py:777-909`
- Статус: ✅ **Fixed без регрессий**
- Pre-computed `cooc_masks` как `torch.bool` тензор (G-44)
- Pre-computed `fb_overlaps` как `torch.long` тензор (G-65 chunked по i)
- Цикл `for i in range(ng)` — **единственный оставшийся Python loop**
- Все `.item()` удалены, маски — `gather(1, best_idx)` на GPU

---

## 2. VRAM оптимизации V11.3 — проверка

| Оптимизация | Коммит | Статус | Экономия |
|------------|--------|:------:|:--------:|
| `_fused_buf` dynamic growth | 3150d5e | ✅ V×(D+1)=225MB → <1MB | ~224MB |
| `_ema_vecs_t` fp16 | 4030b54 | ✅ fp32→fp16 | ~112MB |
| `_mom_t` fp16 | 4030b54 | ✅ fp32→fp16 | ~112MB |
| `_torch_dirty` removal | 1768f27 | ✅ +160s/batch fix | ~640MB PCIe xfer |

**Потенциальная проблема AM-96.1**: fp16 для `_mom_t` + `_ema_vecs_t` → риск underflow для концепций с редкими обновлениями. При `lr=0.003` и `momentum_mu=0.9` минимальная представимая дельта в fp16 = 6e-8. Рекомендуется `torch.bfloat16` если архитектура поддерживает, или эскалация до fp32 для `|grad| < 1e-4`.

---

## 3. QwenKnowledge модуль — аудит интеграции

### Файлы:
- `eva/symbolic/qwen_knowledge.py` — 109 строк (новый)
- `precompute_qwen_knowledge.py` — 355 строк (новый)

### Архитектура:
- `QwenKnowledge` class: загрузка `.npz` → HashMap (packed int64 key → cos_sim)
- `get_factor(cid_a, cid_b)` → `lr *= 1.0 + cos_sim * factor_strength`, clipped
- `inject_qwen_knowledge()` — заглушка-документация

### Интеграция:
```
train_full.py:512-513  → QwenKnowledge(path) → gen qwen_knowledge=
crystal_generator.py:70  → self.qwen_knowledge = qwen_knowledge
stdp_trainer.py:246-247  → lr *= gen.qwen_knowledge.get_factor(ids[i], ids[j])
```

**Проблемы интеграции AM-96.2:**
1. **Не тестируется** — ни одного теста на QwenKnowledge
2. **CPU-only** — `get_factor()` работает через dict lookup. При 146K концепций × 8M пар → O(1) но Python overhead
3. **Нет graceful degradation** — `.npz` файл может отсутствовать → `factor=1.0`, но нет warn
4. **inject_qwen_knowledge()** — функция-заглушка с `pass`, декларирует usage но не вызывается. Код встроен напрямую в `_build_pairs`

**Рекомендация AM-96.3**: GPU кэш через `torch.frombuffer` для packed triples → O(1) GPU lookup без Python dict.

---

## 4. Новые изменения в коде (uncommitted diff)

| Файл | Изменение | Риск |
|------|----------|:----:|
| `checkpoint_manager.py` | `cleanup()` → `_cleanup_old()`, добавлен `ckpt_state` | 🔴 **4 теста упали** |
| `concept_space.py` | `batch_dot()` метод | 🟡 Не тестируется |
| `crystal_generator.py` | `_cf_t`, `_pt2_t`, `_skip2_t` GPU тензоры | 🟢 OK |
| `stdp_trainer.py` | On-GPU PMI, avg_elr reuse, gpu freq path | 🟢 OK |
| `syntax_lattice.py` | Incremental `_prefix_total`/`_skip2_total` | 🟢 OK |
| `train_full.py` | QwenKnowledge, .tmp recovery, rescore fix | 🟢 OK |
| `fcf_config.py` | `checkpoint_every: 500→5000` | 🟡 влияет на частоту |

---

## 5. 4 failed теста — CheckpointManager API

Все 4 failures — `AttributeError: 'CheckpointManager' object has no attribute 'cleanup'`:

```
test_mgr_cleanup:            722  mgr.cleanup()
test_cleanup_removes_old:    750  mgr.cleanup()
test_cleanup_keep:          1026  mgr.cleanup(keep=3)
test_cleanup_below_keep:    1036  mgr.cleanup()
```

**Причина**: `cleanup()` был удалён, `_cleanup_old()` — приватный, вызывается автоматически в `_sync_save()`. Новый API: cleanup происходит внутри `save()` (после успешного сохранения чекпоинта).

**Чинить**: 
- `test_cleanup_keep`: сохранить 5+ чекпойтов, `mgr.wait()`, проверить `_saved_tags`
- `test_mgr_cleanup`: сохранить, `mgr.wait()`, проверить что осталось `cleanup_keep` файлов
- `test_cleanup_removes_old`: аналогично
- `test_cleanup_below_keep`: сохранить < keep, проверить что все остались

---

## 6. AM-96: Проблемы и улучшения

### Критические (P1):
| ID | Проблема | Файл | Fix |
|:--:|----------|:----:|-----|
| **AM-96.1** | fp16 underflow для редких концепций | crystal_generator.py:279,316 | `bfloat16` или fp32 для `|grad|<1e-4` |
| **AM-96.2** | QwenKnowledge не тестируется | qwen_knowledge.py | Добавить тесты load/get_factor/integration |

### Средние (P2):
| ID | Проблема | Файл | Fix |
|:--:|----------|:----:|-----|
| **AM-96.3** | On-GPU PMI `.total_freq_t.item()` — CPU sync | stdp_trainer.py:170 | `_total_freq_t` как float32 тензор, gather на GPU |
| **AM-96.4** | `_contrastive_objective_gpu` Python loop над ng | stdp_trainer.py:868-901 | Вынести в `torch.vmap` или compile |
| **AM-96.5** | `_build_pairs` GPU path → `.cpu().numpy()` каждый batch | stdp_trainer.py:164-167 | `torch.index_select` + `__getindex__` через `torch.compile` |

### Низкие (P3):
| ID | Проблема | Файл | Fix |
|:--:|----------|:----:|-----|
| **AM-96.6** | `inject_qwen_knowledge()` — dead code | qwen_knowledge.py:98-109 | Удалить или сделать декоратором |
| **AM-96.7** | `_cached_update` closure per instance — GC pressure | crystal_generator.py:132-138 | `__init__` → bound method |
| **AM-96.8** | `checkpoint_every: 500→5000` без перенастройки decay | fcf_config.py:377 | Синхронизировать с `decay_every_pairs` |
| **AM-96.9** | EMA fp16 → `_vecs_t` copy каждый eval | crystal_generator.py:344 | EMA в fp32, cast при копировании |
| **AM-96.10** | precompute_qwen_knowledge.py: `gcd()` not imported | precompute_qwen_knowledge.py:26 | Добавить `import gc` (str → int?) |

---

## 7. Остаток V11 P2/P3 (неисправленное)

| ID | Priority | Проблема | Текущий статус |
|:--:|:--------:|----------|:--------------:|
| SN-46 | P3 | Contrastive CPU roundtrip `_apply_vector_update` | Still `.cpu().numpy()` at 907-909 |
| SN-47 | P3 | CPU neg sampling `sample(total_vocab)` 146K | CPU fallback only |
| SN-48 | P3 | GPU field overlap `.item()` sync | `_overlap_lookup` → CPU numpy int |
| G-62 | P2 | GPU vec update w/o `.cpu().numpy()` | Все `_apply_vector_update` через numpy |
| G-65 | P2 | GPU field overlap in `_build_pairs` | Pre-computed but CPU overlap matrix |
| TN-13 | P2 | Progressive batch size | Только plateau doubling (нет ramp) |
| TN-15 | P2 | Decay warmup | Done partially (train_full.py:738-740) |
| TN-32 | P2 | Rescore idx=-1 curriculum reset | Частично (rescore_line gate) |

---

## 8. Общий вывод

| Метрика | V11 | V12 | Δ |
|---------|:---:|:---:|:-:|
| Tests passed | 122 | 122 | 0 |
| Tests failed | 0 | **4** | +4 🔴 |
| Tests skipped | 3 | 3 | 0 |
| P0 (crash) | 0 | 0 | 0 |
| GPU destab | CPU | **GPU** ✅ | +1 |
| GPU neg sampling | loop | **batched** ✅ | +1 |
| GPU contrastive | .item() loop | **pure tensor** ✅ | +1 |
| Qwen integration | none | **added** ✅ | +1 |
| VRAM (est) | ~2048MB | ~1824MB | −224MB |

**Verdict**: Архитектурный прогресс есть (GPU destab, neg sampling, contrastive — все векторные). 4 теста упали из-за рефакторинга CheckpointManager API (cleanup → авто-клининг внутри save). QwenKnowledge интеграция чистая, но не тестируется.

**Рекомендуемый порядок действий:**
1. AM-96.1: починить 4 теста (CheckpointManager) — 10 мин
2. AM-96.2: тесты для QwenKnowledge — 20 мин
3. AM-96.4: `torch.vmap` для contrastive GPU loop — 30 мин
4. AM-96.3: GPU `_total_freq_t` без `.item()` — 15 мин
5. AM-96.6: Удалить dead code `inject_qwen_knowledge` — 2 мин
