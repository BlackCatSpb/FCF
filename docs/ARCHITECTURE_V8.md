# EVA Symbolic v8 — Архитектурный дизайн

> **Дата:** 2026-05-22
> **Статус:** Проектирование
> **Цель:** UnifiedMultidimensionalTransformer — единое ℝ¹² пространство, фрактальные головы, вектор↔символ

---

## 1. Текущее состояние (аудит)

| Что | Статус | Проблема |
|-----|--------|----------|
| Символьный affinity | ✅ Работает | float32 overflow → fixed |
| Трансформер | ❌ Заморожен | `torch.no_grad()` — веса случайные |
| FractalAttention | ❌ Dead code | 400 строк, ни разу не вызван |
| Многомерное пространство | ❌ Не реализовано | 256-dim эмбеддинг, 64-dim MDS — разные системы |
| Вектор↔символ | ❌ Нет | Символы = индексы, не векторы |
| Иерархия уровней | ❌ Post-hoc | Слова/предложения вне цикла обучения |
| Генератор | ⚠️ Affinity only | Трансформер не участвует |
| Кросс-модальность | ❌ Stub | В legacy/ |

**Корень всех проблем:** архитектура спроектирована как Hebbian co-occurrence accumulator, а не как unified transformer. FractalAttention есть, но не подключён.

---

## 2. Целевая архитектура

### 2.1 Единое пространство ℝ¹²

```
12 измерений = 4 базовых × 3 масштаба
  │              │            │
  │              │            └─ символ/слово/предложение/домен
  │              └────────────── 3 оси на уровень
  └───────────────────────────── x,y,z = геометрия внутри уровня
```

**Почему 12, а не 256:**
- 160 символов → 3D достаточно (6³ = 216 > 160)
- 200K слов → 4D достаточно (22⁴ ≈ 234K)
- Избыточность 256-dim не нужна — фрактальная вложенность покрывает

### 2.2 Символ ≡ Вектор

```
Сейчас:   символ 'И' → индекс 96 → Embedding[96] → случайный вектор
Будет:    символ 'И' → координата P_И ∈ ℝ¹²
          P_И стабильна и НЕ случайна — выводится из affinity
          encoding = P_И (без lookup)
          decoding = nearest_neighbor(P, all_known_P)
```

### 2.3 FractalAttention как единственный attention

```
Заменить CausalSelfAttention на FractalAttention:
  │
  ├─ 12 голов = 4 уровня × 3 масштаба-на-уровень
  │
  ├─ Уровень 0 (головы 0-2): символ→символ, масштабы 1,2,4
  ├─ Уровень 1 (головы 3-5): слово→слово, масштабы 1,2,4
  ├─ Уровень 2 (головы 6-8): предложение→предложение
  └─ Уровень 3 (головы 9-11): домен→домен
```

### 2.4 Обучение — единый loss

```
Loss = L_coord + L_word + L_sentence + L_domain

L_coord:     MSE(predicted_vector, target_symbol_position)
L_word:      MSE(centroid_predicted_words, centroid_correct_words)
L_sentence:   MSE(centroid_predicted_sentences, centroid_correct)
L_domain:    кластеризационный (intra-cluster distance)

ВСЕ уровни тренируются одновременно — потому что это ОДНО пространство.
```

---

## 3. План реализации

### Этап A: UnifiedMultidimensionalTransformer
- [ ] Создать `eva/symbolic/unified_transformer.py`
- [ ] Заменить d_model=256 на coord_dim=12
- [ ] Убрать nn.Embedding — символы = координаты из TopologicalField
- [ ] Встроить FractalAttention вместо CausalSelfAttention
- [ ] Добавить символьный энкодер/декодер (координата↔индекс)

### Этап B: Подключить к обучению
- [ ] Заменить `train_to_convergence.py` → `train_unified.py`
- [ ] Loss = MSE(вектор, позиция) вместо count-based
- [ ] Включить gradient-based обновление весов
- [ ] FractalAttention маски на всех уровнях

### Этап C: Валидация
- [ ] Реконструкция текста: encode → decode → сравнить
- [ ] Безошибочное воспроизведение коротких текстов
- [ ] Масштабирование на длинные

### Этап D: Модальность
- [ ] TextEncoder → ℝ¹²
- [ ] Stub ImageEncoder → ℝ¹²
- [ ] Stub AudioEncoder → ℝ¹²
- [ ] Универсальный декодер ℝ¹² → модальность

---

## 4. Самопроверка: что упущено?

### Q1: Не сломает ли 12-dim точность для 160 символов?
A: Нет. 3D достаточно для разделения 160 точек. 12D даёт огромный запас для фрактальной вложенности.

### Q2: Как символы получат начальные координаты?
A: MDS из affinity-матрицы (уже работает в TopologicalField). 880K батчей обучения дают качественный affinity → MDS даст осмысленные координаты.

### Q3: FractalAttention требует word-boundaries. Откуда они?
A: WordBoundaryDetector (transition probability minima). Запустить ДО обучения, сохранить границы как часть датасета.

### Q4: Сколько новых параметров?
A: ~500K (12-dim вместо 256-dim даёт ~16× меньше параметров в attention). Итого модель станет ещё легче.

### Q5: Сохранится ли обратная совместимость?
A: Нет. Это новый модуль `unified_transformer.py`. Старые `potential_field.py`, `symbolic_generator.py` останутся для сравнения.

### Q6: Что с FractalAttn из DataSphere (139 KB)?
A: Архитектура та же, но размерность не та (256 vs 12). Нужно переобучить с нуля.

### Q7: Нужен ли отдельный affinity-слой?
A: Да — как инициализация координат. Affinity → MDS → начальные позиции символов.

### Q8: Как проверять что модель не «сходит с ума»?
A: LogicGuard + consistency check. Уже есть в contemplation.py.

### Q9: Что делать с 30 существующими symbolic-модулями?
A: Оставить. Они — инфраструктура: affinity, grammar, knowledge base, library. UnifiedTransformer использует их как источники данных.

---

## 5. Файлы для реализации

```
НОВЫЕ:
  eva/symbolic/unified_transformer.py   — UnifiedMultidimensionalTransformer
  train_unified.py                       — новый цикл обучения

ИЗМЕНИТЬ:
  eva/symbolic/__init__.py              — добавить импорт
  eva.bat                                — train_unified.py вместо train_to_convergence

ИСПОЛЬЗОВАТЬ (без изменений):
  eva/symbolic/potential_field.py        — affinity → MDS → координаты
  eva/symbolic/topological_field.py      — MDS проекция
  eva/symbolic/fractal_attention.py      — FractalAttention (переиспользовать)
  eva/symbolic/word_level.py             — WordBoundaryDetector
  eva/symbolic/contradiction_filter.py   — защита от бреда
  eva/symbolic/contemplation.py          — фоновое мышление
  eva/transformer.py                     — базовые блоки (RMSNorm, SwiGLU)
```

---

*Документ создан для самопроверки перед реализацией. Все вопросы должны быть отвечены до начала кодирования.*
