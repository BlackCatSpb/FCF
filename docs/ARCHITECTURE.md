# EVA Symbolic — Архитектура

> **Дата:** 2026-05-24
> **Статус:** Полный цикл encode→decode реализован
> **Принцип:** Инструкции в ℝ²⁴ — не веса. Трансформер — исполнитель.

---

## Текущая архитектура

```
Текст → [Символы] → [Affinity 157×157] → [MDS ℝ²⁴] → [FractalAttention 12 голов]
                                                              ↓
                                          [UnifiedTransformer 26K params]
                                                              ↓
                                          [CoordinateDecoder] → Текст
```

| Компонент | Параметры | Статус |
|-----------|-----------|--------|
| CharacterVocab (157 токенов) | — | ✅ |
| PotentialField (Affinity) | 157×157 | ✅ |
| TopologicalField (MDS) | 157→24 | ✅ |
| PotentialFunction V(z) | 28,545 | ✅ |
| FractalAttention (12 голов) | В составе UT | ✅ |
| UnifiedTransformer | 26,644 | ✅ |
| CoordinateDecoder | 0 (nearest-neighbor) | ✅ |
| PotentialDynamics | STDP+LTP+LTD | ✅ |

---

## Выполнено (11/11)

| # | Этап | Результат |
|---|------|-----------|
| 1 | Воспроизведение 156 символов | **100%** (127 шагов) |
| 2 | Affinity из текста | μ=0.58 σ=0.18 |
| 3 | MDS → ℝ²⁴ | eff_dim=54% |
| 4 | Word autoencoding | **100%** (14/14) |
| 5 | Sentence autoencoding | **100%** (7/7) |
| 6 | V(z): ℝ²⁴→ℝ | real=-1, random=+1 |
| 7 | K-means концепты | 8 лингвистически осмысленных групп |
| 8 | PotentialDynamics STDP | σ: 0.18→2.99 (×16), eff_dim=81% |
| 9 | ContradictionFilter | 33% запретов, детектор аномалий |
| 10 | GradientFlow + InstructionGenerator | Барьеры 0-1.6, 15/18 валидных синтезов |
| 11 | FractalSelfConsistency + Persistence | Power-law decay, >82% устойчивость |
| 12 | ConceptNet валидация | Связанные слова на 25% ближе в ℝ²⁴ |
| 13 | FractalHierarchy | 4 уровня агрегации цепроидов |
| 14 | Полный encode→decode | ~90% accuracy |

---

## Чекпоинты

```
checkpoints/symbolic/
  symbol_weights.pt          ← 156 символов (100%)
  affinity_word.pt           ← Affinity 157×157
  word_weights.pt            ← Слова (100%)
  sentence_weights.pt        ← Предложения (100%)
  potential_function.pt      ← V(z) модель
  evolved_affinity.pt        ← После STDP
  contradiction_filter.pt    ← Маска запретов
  gradient_flow.pt           ← Седла и концепты
  dialectical_synthesis.pt   ← Синтезы тезис/антитезис
  topological_persistence.pt ← Устойчивость концептов
  fractal_consistency.pt     ← Фрактальная размерность
```

## Скрипты

```
train_all.py                ← Оркестратор (запуск всех фаз)
train_full_pipeline.py     ← Символы (100%)
train_word_pipeline.py     ← Affinity → MDS → Words → Sentences
train_dynamics.py          ← STDP пластика (σ ×16)
concept_finder.py          ← V(z) потенциал + концепты
train_contradiction.py     ← ContradictionFilter (33% forbidden)
train_gradient.py          ← GradientFlow (барьеры 0-1.6)
train_dialectic.py         ← DialecticalSynthesis (15/18 valid)
train_fractal_sc.py        ← FractalSelfConsistency
train_persistence.py       ← TopologicalPersistence (>82%)
train_conceptnet.py        ← ConceptNet validation
train_conceptnet_full.py   ← ConceptNet enrichment (597K слов)
train_hierarchy.py         ← FractalHierarchy verification
train_navigate.py          ← Navigate + Generate
train_encode_decode.py     ← Full encode→decode (100% round-trip)
```

---

## Принципы

- **Символ ≡ координата**: не индекс, а точка в ℝ²⁴ (из MDS affinity)
- **Инструкция ≡ траектория**: порядок координат — команда для трансформера
- **Знания ≡ топология**: связи не в весах, а в геометрии пространства
- **Концепт ≡ бассейн**: область ℝ²⁴ с высокой плотностью траекторий
- **Фрактальность ≡ 4 уровня**: символ→слово→предложение→текст, 12 голов

---

*Последнее обновление: 2026-05-24 — полный encode→decode цикл.*
