# EVA Symbolic — Архитектура

> **Дата:** 2026-05-24
> **Статус:** Активная разработка
> **Принцип:** Инструкции в ℝ²⁴ — не веса. Трансформер — исполнитель, не предсказатель.

---

## 1. Текущая архитектура

```
Текст "привет мир"
    │
    ▼
[Символы] п, р, и, в, е, т, ␣, м, и, р
    │
    ▼
[Affinity Matrix 157×157] ← PotentialDynamics (STDP+LTP+LTD)
    │
    ▼
[MDS → ℝ²⁴] 157 координат по 24 измерения
    │
    ▼
[UnifiedMultidimensionalTransformer] ← FractalAttention
    │  вход: координаты символов (траектория)
    │  выход: символы (реконструкция = исполнение инструкции)
    │
    ▼
[CoordinateDecoder] ℝ²⁴ → ближайший символ
    │
    ▼
"привет мир"
```

| Компонент | Файл | Параметры | Статус |
|-----------|------|-----------|--------|
| CharacterVocab | `symbolic/char_vocab.py` | 157 токенов | ✅ |
| PotentialField | `symbolic/potential_field.py` | 157×157 | ✅ |
| TopologicalField | `symbolic/topological_field.py` | MDS 157→24 | ✅ |
| FractalAttentionV2 | `symbolic/fractal_v2.py` | 12 голов, 4 уровня | ✅ |
| UnifiedTransformer | `symbolic/unified_transformer.py` | 26,644 | ✅ |
| PotentialFunction | `concept_finder.py` | 28,545 | ✅ |
| PotentialDynamics | `train_dynamics.py` | STDP+LTP+LTD | ✅ |
| ConceptNet | `real_data/conceptnet/` | Русские концепты | 📦 скачан |

---

## 2. Выполнено

| # | Этап | Результат | Файл |
|---|------|-----------|------|
| 1 | Воспроизведение всех 156 символов | **100%** (127 шагов) | `train_full_pipeline.py` |
| 2 | Affinity matrix из текста | μ=0.58, σ=0.18 | `train_word_pipeline.py` |
| 3 | MDS → ℝ²⁴ координаты | eff_dim=54.6% | `train_word_pipeline.py` |
| 4 | Word autoencoding | **100%** (14/14 слов) | `train_word_pipeline.py` |
| 5 | Sentence autoencoding | **100%** (7/7 предложений) | `train_word_pipeline.py` |
| 6 | PotentialFunction V(z): ℝ²⁴→ℝ | реальное=-1, случайное=+1 | `concept_finder.py` |
| 7 | K-means концепты из координат | 8 лингвистически осмысленных групп | `concept_finder.py` |
| 8 | PotentialDynamics: STDP+LTP+LTD | σ: 0.18 → 2.99 (×16) | `train_dynamics.py` |
| 9 | MDS на эволюционировавшей аффинности | **eff_dim=80.6%** (>60%) | `train_dynamics.py` |
| 10 | Воспроизведение слов после эволюции | **100%** (6/6) | `train_dynamics.py` |
| 11 | Удаление версионных меток | v1-v8 → EVA | Все файлы |

---

## 3. Предстоит

| # | Задача | Компонент | Наследие |
|---|--------|-----------|----------|
| 1 | **ContradictionFilter** | Иммунная система: 5 типов запретов | `symbolic/contradiction_filter.py` |
| 2 | **GradientFlow V(z)** | Навигация: седловые точки → новые инструкции | `legacy/state_grammar_ext.py` |
| 3 | **DialecticalSynthesis** | Тезис/антитезис → синтез | `legacy/state_grammar_final.py` |
| 4 | **InstructionGenerator** | Создание новых инструкций из существующих | Новый |
| 5 | **FractalSelfConsistency** | Масштабная инвариантность | `legacy/state_grammar_final.py` |
| 6 | **TopologicalPersistence** | Устойчивость при возмущениях | `legacy/state_grammar_ext.py` |
| 7 | **ConceptNet валидация** | Сверка с русским ConceptNet | `real_data/conceptnet/` |
| 8 | **Encode→Decode цикл** | Текст → метаданные → инструкция → текст | Новый |
| 9 | **FractalHierarchy** | Иерархия: символ→слово→предложение→текст | `fractal_hierarchy.py` |

---

## 4. Файловая карта

```
Активные:
  train_full_pipeline.py      ← Символы (156, 100%)
  train_word_pipeline.py       ← Слова + предложения (фазы 1-6)
  train_dynamics.py            ← STDP/LTP/LTD пластика
  concept_finder.py            ← Потенциал V(z) + концепты
  verify_symbols.py            ← Проверка воспроизведения

Ядро:
  eva/symbolic/char_vocab.py          ← 157 токенов
  eva/symbolic/potential_field.py     ← Affinity 157×157
  eva/symbolic/topological_field.py   ← MDS ℝ²⁴
  eva/symbolic/fractal_v2.py          ← FractalAttention (12 голов)
  eva/symbolic/unified_transformer.py ← UnifiedTransformer
  eva/symbolic/concept_miner.py       ← ConceptMiner (из legacy)
  eva/symbolic/contradiction_filter.py ← ContradictionFilter (из legacy)
  eva/symbolic/potential_dynamics.py  ← PotentialDynamics (из legacy)

Legacy (архив методов):
  eva/legacy/state_grammar.py         ← V2 классы (InheritanceGraph etc.)
  eva/legacy/state_grammar_ext.py     ← GradientFlow, TopologicalPersistence
  eva/legacy/state_grammar_final.py   ← DialecticalSynthesis, FractalSelfConsistency
  eva/legacy/state_grammar_deep.py    ← creative_potential()
  eva/legacy/instruction_trainer.py   ← InstructionTrainer
  eva/legacy/fcf_system.py            ← FCF интеграция

Чекпоинты:
  checkpoints/symbolic/symbol_weights.pt      ← Символы (100%)
  checkpoints/symbolic/symbol_100pct.pt        ← Милестоун
  checkpoints/symbolic/affinity_word.pt        ← Affinity 157×157
  checkpoints/symbolic/word_weights.pt         ← Слова (100%)
  checkpoints/symbolic/sentence_weights.pt     ← Предложения (100%)
  checkpoints/symbolic/potential_function.pt   ← V(z) модель
  checkpoints/symbolic/evolved_affinity.pt     ← После STDP

Документация:
  docs/ARCHITECTURE.md      ← Этот файл
  docs/EVA_PRINCIPLES.md    ← Принципы системы
  docs/ROADMAP.md           ← Дорожная карта
```

---

## 5. Ключевые принципы

### Символ ≡ координата
Символ — не индекс в таблице, а точка в ℝ²⁴. Позиция вычисляется из статистики языка (MDS из affinity). Символы без связей — базовый уровень.

### Инструкция ≡ траектория
Текст ≡ траектория в ℝ²⁴. Порядок координат — инструкция для трансформера. Трансформер не предсказывает — он исполняет инструкцию.

### Знания ≡ топология
Связи между символами — не в весах модели, а в геометрии пространства. Affinity матрица хранит паттерны. MDS превращает их в координаты. PotentialDynamics делает матрицу живой.

### Концепт ≡ бассейн притяжения
Концепт — область в ℝ²⁴, куда сходятся многие траектории. Не задан вручную — возникает из данных.

### Фрактальность ≡ уровни инструкций
24 измерения = 4 уровня × 6 базовых осей. Символ → слово → предложение → текст. FractalAttention работает на всех масштабах одновременно.

---

*Документ обновляется при каждом значимом изменении архитектуры.*
