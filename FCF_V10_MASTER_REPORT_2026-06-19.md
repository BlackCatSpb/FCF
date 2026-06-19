# FCF V10 — Сводный отчёт коллегии AI-агентов

**Дата**: 2026-06-19
**Версия**: V10 (аудит V9 коммитов: a0fe15b + cccc392 + 21ee6ca)
**Состав**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Executive Summary

V9 закоммичен (3 коммита). Все 10 P1/P2 V9 проблем исправлены. **79 тестов проходят** (2 skipped).

| Метрика | V9 | V10 | Δ |
|---------|:--:|:---:|:-:|
| P0 | 0 | **0** | = |
| P1 открыто | 15 | **10** | −5 |
| P2 открыто | 25 | **22** | −3 |
| GPU-оптимизации (G-40..G-49) | 0/10 | **0/10** | = |
| Тесты (QN-32..QN-40) | 0/9 | **0/9** | = |
| STR | ~52% | ~50% | −2% |

**Главные находки V10:**

1. ✅ **V9 коммиты не внесли регрессий** — все корректны
2. ❌ **CPU neg sampling**: `v_gen` stale — **last-update-wins** (только последний valid neg применяется), GPU `.sum(dim=0)` применяет все. **P1 parity bug при neg_samples>1.**
3. ❌ **CPU contrastive**: hard negatives перезаписывают друг друга, GPU `.mean(dim=0)`. **P1 parity bug.**
4. ❌ **~16,500 `.item()` syncs/batch** не исправлен — главное узкое место
5. ❌ **`_apply_subspace_update`** — всё ещё 100% CPU/numpy
6. ❌ **`checkpoint_state.json`** не обновляется при чекпоинтах — **P1 потеря прогресса**
7. ❌ **0/10 GPU-оптимизаций**, **0/9 тестов** реализованы

---

## 1. V9 Commit Verification (a0fe15b + cccc392 + 21ee6ca)

### Исправлено — корректно (11 из 11)

| ID | Fix | Файл | Верификация |
|:--:|-----|:----:|:-----------:|
| REG-V9-1 | `self.l_c` → `self.fractal.l_c` | concept_space.py:564 | ✅ |
| REG-V9-2 | `code_new /= np.linalg.norm(code_new)` | concept_space.py:574 | ✅ |
| REG-V9-8 | `momentum_mu` config field | fcf_config.py:421 | ✅ |
| SN-22.1 | GPU neg sampling `mean()` → `sum()` | stdp_trainer.py:574-579 | ✅ |
| SN-22.2 | field_gate guard in CE reweighting | stdp_trainer.py:583-586 | ✅ |
| SN-24 | Momentum blend: `grad = mu*mom + (1-mu)*grad` | stdp_trainer.py:459-461 | ✅ |
| SN-25 | Slow STDP pairs in gpu_meta_l | stdp_trainer.py:216-224 | ✅ |
| SN-31 | Dead `_subspace_update` removed | stdp_trainer.py | ✅ |
| AM-32 | `_graph_cache` LRU (maxlen=5000) | crystal_generator.py:72-73 | ✅ |
| G-45 | Persistent CUDA events | stdp_trainer.py:29-32 | ✅ |
| G-47 | `lerp_` for EMA | stdp_trainer.py:463-464 | ✅ |

---

## 2. P1 — Открытые проблемы (10)

### 2.1 CPU/GPU Parity Bugs (НОВЫЕ от V10 — SN-35, SN-36)

| ID | Проблема | Зона | Сложность |
|:--:|----------|:----:|:---------:|
| **SN-35** | CPU neg sampling: `v_gen` stale — **last-update-wins** при neg_samples>1. GPU `.sum(dim=0)` применяет все. | NS | 2 |
| **SN-36** | CPU contrastive: hard negatives **перезаписывают** друг друга (последний побеждает). GPU `.mean(dim=0)` смешивает все. | NS | 2 |

### 2.2 GPU Performance (старые — без прогресса)

| ID | Проблема | Агент | Сложность |
|:--:|----------|:-----:|:---------:|
| G-40 | `_apply_subspace_update` — 100% CPU/numpy в GPU-пути | GPU | 6 |
| G-43/AM-43 | GPU neg sampling — per-concept loop с CPU sync | GPU | 5 |
| G-44/AM-42 | GPU Contrastive — ~16,500 `.item()` syncs/batch | GPU/NS | 7 |
| G-46 | `_mom_buf` — CPU dict, per-element roundtrip | GPU | 4 |
| G-42 | `_centroid_pull_batch` — полностью CPU | GPU | 3 |

### 2.3 Training Pipeline (НОВЫЕ от V10)

| ID | Проблема | Агент | Сложность |
|:--:|----------|:-----:|:---------:|
| **TN-31** | `checkpoint_state.json` не обновляется при чекпоинтах — потеря прогресса при сбое | TD | 1 |
| REG-V9-7 | `noise_scale` управляет gradient noise + fractal fluctuation (2 механики) | Arch | 2 |

### 2.4 Dead/Misnamed Code

| ID | Проблема | Агент | Сложность |
|:--:|----------|:-----:|:---------:|
| AM-25 | CPU path (~300 строк) не удалён | Arch | 3 |
| **G-57** | `push_total` / `lr_scale` аллоцированы, НЕ ИСПОЛЬЗУЮТСЯ в GPU contrastive | GPU | 1 |

---

## 3. P2 — Открытые проблемы (22)

### 3.1 GPU Optimization

| ID | Оптимизация | Сложность | Ускорение |
|:--:|-------------|:---------:|:---------:|
| G-41 | Full GPU lateral inhibition (без `.item()`) | 4 | 5-10× |
| G-48 | `torch.compile` на `_gpu_stdp_apply` | 4 | 2-3× |
| G-49 | Pre-allocate fused buffers | 3 | 1.5× |
| G-50 | Zero-copy vector write-back (GPU→GPU) | 5 | 2-5× |
| G-51 | Deferred vector sync (batched `_vecs_t` write) | 3 | 1.5-3× |
| G-52 | Fused contrastive + neg sampling | 4 | 1.2-2× |

### 3.2 Training Dynamics

| ID | Проблема | Сложность |
|:--:|----------|:---------:|
| TN-13 | Progressive batch size with plateaus | 2 |
| TN-15 | Decay warmup with protect threshold ramp | 2 |
| **TN-32** | `idx = -1` сбрасывает curriculum (batch_size→8, max_len→16) после rescore | 1 |
| **TN-33** | `pipeline.global_step` — всегда 0 (мёртвая переменная) | 1 |
| **TN-34** | opt.json naming mismatch — tagged vs tagless | 2 |

### 3.3 Arch/Code Quality

| ID | Проблема | Сложность |
|:--:|----------|:---------:|
| AM-29/AM-46 | RNG consolidation (7+ RNG, RNGRegistry не используется) | 3 |
| REG-V9-9 | Monkey-patch на lattice.update/decay_all | 3 |
| AM-30 | EMA batch update (per-concept loop) | 2 |
| AM-31 | ConceptError batch sync | 2 |
| AM-33 | HormonalSystem.reset() | 1 |

### 3.4 Testing (0/9 реализованы)

| ID | Тесты | Сложность |
|:--:|-------|:---------:|
| QN-32 | subspace update tests | 3 |
| QN-33 | GPU contrastive tests | 4 |
| QN-34 | evaluate tests | 3 |
| QN-35 | noise_scale tests | 2 |
| QN-36 | RNGRegistry tests | 3 |
| QN-37 | AdaptiveErrorTracker tests | 3 |
| QN-38 | Checkpoint cleanup tests | 2 |
| QN-39 | TrainingPipeline tests | 3 |
| QN-40 | Dead code tests | 1 |

---

## 4. НОВЫЕ проблемы V10 (SN-35..SN-42)

| ID | Проблема | Зона | P | Сложность |
|:--:|----------|:----:|:-:|:---------:|
| SN-35 | CPU neg sampling: **last-update-wins** (только последний valid neg). GPU `.sum(dim=0)` применяет все градиенты | NS | P1 | 2 |
| SN-36 | CPU contrastive: hard negatives **перезаписывают** друг друга. GPU `.mean(dim=0)` смешивает | NS | P1 | 2 |
| SN-38 | `cooc_set` перестраивается в inner loop GPU contrastive | NS | P3 | 2 |
| SN-39 | `connection_strength` Python call в inner loop GPU contrastive | NS | P2 | 2 |
| SN-40 | `field_bits` overlap per-candidate в inner loop | NS | P3 | 3 |
| SN-41 | `_ema_steps` монотонный рост без сброса | NS | P3 | 1 |
| SN-42 | `push_total`/`lr_scale` — аллоцированы, не используются | NS | P2 | 1 |
| TN-31 | `checkpoint_state.json` не обновляется при чекпоинтах | TD | P1 | 1 |
| TN-32 | `idx = -1` сбрасывает curriculum после rescore | TD | P2 | 1 |
| TN-33 | `pipeline.global_step` всегда 0 | TD | P2 | 1 |
| TN-34 | opt.json naming mismatch | TD | P2 | 2 |
| G-57 | `push_total`/`lr_scale` — мёртвые тензоры | GPU | P1 | 1 |

---

## 5. Рекомендуемый план работ

### Фаза 0 (немедленно — 3 задачи, ~1 час)

1. **TN-31**: Сохранять `checkpoint_state.json` при каждом чекпоинте (3 строки)
2. **G-57**: Убрать мёртвые `push_total`/`lr_scale` + добавить assert
3. **SN-35/SN-36**: CPU neg sampling/contrastive parity — аккумулировать градиенты, а не перезаписывать

### Фаза 1 (P1 — 4 задачи, ~1 неделя)

4. **REG-V9-7**: Разделить `noise_scale` → `gradient_noise_scale` + `fluctuation_amp`
5. **G-46**: Persistent `_mom_t` tensor (замена CPU dict)
6. **G-42**: GPU `_centroid_pull_batch`
7. **QN-39 + QN-34**: TrainingPipeline + evaluate тесты

### Фаза 2 (GPU Performance — 3 задачи, ~2 недели)

8. **G-40**: Batched GPU subspace update (через `_basis_t`, убрать CPU numpy)
9. **G-43/AM-43**: Vectorized GPU negative sampling
10. **G-44/AM-42**: Batched GPU contrastive + TN-14 (~16,500 syncs→0)

### Фаза 3 (P2 — 8 задач)

11-18: G-41, G-48, G-49, G-51, TN-13, TN-15, TN-32, TN-33, TN-34, QN-32..QN-38

### Фаза 4 (P3 — остальное)

19+: G-50, G-52, AM-29/46, AM-25, AM-30/31/33, SN-38..SN-42, QN-40

---

## 6. Что СДЕЛАНО (V9→V10 прогресс)

### Исправлено (11 P1/P2)
- ✅ REG-V9-1/2 (l_c, norm), REG-V9-8 (momentum_mu config)
- ✅ SN-22.1/22.2 (GPU parity), SN-24 (momentum blend), SN-25 (slow STDP), SN-31 (dead code)
- ✅ AM-32 (graph_cache LRU)
- ✅ G-45 (CUDA events), G-47 (lerp_ EMA)

### Не исправлено (без прогресса)
- ❌ GPU оптимизации: **0/10** (G-40..G-49)
- ❌ Тесты: **0/9** (QN-32..QN-40)
- ❌ GPU Contrastive векторизация (AM-42/SN-19/G-44)
- ❌ GPU neg sampling векторизация (AM-43/G-43)
- ❌ CPU path не удалён (AM-25)
- ❌ RNG не консолидирован (AM-29/46)
- ❌ noise_scale не разделён (REG-V9-7)
- ❌ CPU vs GPU parity: 2 новых P1 (SN-35/36)

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
