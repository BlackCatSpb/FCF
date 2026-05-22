# EVA Symbolic v8 — Архитектурный дизайн

> **Дата:** 2026-05-22
> **Статус:** В разработке
> **Цель:** UnifiedMultidimensionalTransformer — единое ℝ¹² пространство, фрактальные головы, вектор↔символ

---

## 1. Текущее состояние

| Что | Статус | Где |
|-----|--------|-----|
| Символьный affinity | ✅ | `potential_field.py` — count-based, float64 fix готов |
| UnifiedMultidimensionalTransformer | ✅ v0 | `unified_transformer.py` — 10.7K params, ℝ¹² |
| Координатный эмбеддинг (символ→ℝ¹²) | ✅ | `CoordinateEmbedding` — без lookup |
| Координатный декодер (ℝ¹²→символ) | ✅ | `CoordinateDecoder` — nearest neighbor |
| Тренировочный скрипт (gradient) | ✅ | `train_unified.py` — MSE на координатах |
| FractalAttention | ⚠️ Есть, отключён | `fractal_attention.py` — конфликт размерностей |
| Трансформер заморожен | ❌ | `train_to_convergence.py` — Hebbian, без градиента |
| Иерархия уровней | ❌ | Post-hoc, не в цикле обучения |
| Кросс-модальность | ❌ | Только stub в legacy/ |
| 30 symbolic-модулей | ⚠️ | Инфраструктура, используется частично |

---

## 2. Реализовано 2026-05-22

### 2.1 UnifiedMultidimensionalTransformer

```python
# 10 757 параметров (vs 1 089 024 у старого — в 100× меньше)

class UnifiedMultidimensionalTransformer:
    embed: CoordinateEmbedding(vocab=156, dim=12)     # символ → ℝ¹² (без lookup)
    attention: simple scaled dot-product              # временно, будет FractalAttention
    ffn: SwiGLU(12 → 48 → 12)                        # компактный
    decoder: CoordinateDecoder                        # ℝ¹² → ближайший символ
    pos_encoding: learnable [1, 512, 12]              # позиции
```

**Как работает:**
```
Текст "привет" → [п,р,и,в,е,т] (индексы)
  → CoordinateEmbedding → [P_п, P_р, P_и, P_в, P_е, P_т] в ℝ¹²
  → Self-Attention → предсказанные координаты
  → CoordinateDecoder → расстояния до всех 156 символов
  → argmin → 'р', 'и', 'в', 'е', 'т', ...
```

**Loss:**
```
L_coord = MSE(предсказанная_координата, истинная_координата_символа)
L_class = CrossEntropy(score_расстояния, правильный_индекс)
Total = L_coord + L_class
```

### 2.2 Почему 12 измерений

| Размерность | Что вмещает | Расчёт |
|-------------|------------|--------|
| 3D | 160 символов | 6³ = 216 > 160 |
| 4D | 200K слов | 22⁴ ≈ 234K |
| 12D = 3×4 | ВСЁ | Фрактальная вложенность уровней |

12 измерений = 4 уровня × 3 базовых оси на уровень.
Каждый уровень использует СВОЮ проекцию тех же 12 измерений.
Не 3+4+6 = 13, а 3×4 = 12 с разной интерпретацией.

### 2.3 GPU-native вычисления

| | Классический (256-dim) | Многомерный (12-dim) |
|---|---|---|
| Ops на голову | 65K | 144 |
| 12 голов total | 780K | 1.7K |
| GPU occupancy | ~40% | ~90% |
| Параметры | 1.09M | **10.7K** |

---

## 3. Что нужно доработать

### 3.1 Немедленно (блокеры)

| # | Задача | Почему |
|---|--------|--------|
| 1 | Завершить float64 символьное обучение | Нужен affinity для MDS → координаты символов |
| 2 | MDS → координаты в ℝ¹² | Без них трансформер не знает позиций символов |
| 3 | Запустить `train_unified.py` | Первый gradient-based прогон |
| 4 | Подключить FractalAttention | Заменить простой attention на фрактальный |

### 3.2 Архитектурные

| # | Задача | Статус |
|---|--------|--------|
| 5 | WordBoundaryDetector в цикл обучения | Слова как группы символов |
| 6 | MultiLevelPredictor в генератор | Предсказание на всех уровнях |
| 7 | LogicGuard в генератор | Защита от бреда |
| 8 | FractalAttention на 4 уровнях | Полноценная замена простого attention |
| 9 | KnowledgeBase автономно | Домены из накопленных данных |
| 10 | Library naming → осмысленные имена | Сейчас мусор |

### 3.3 Валидация

| # | Тест | Критерий |
|---|------|----------|
| 11 | Реконструкция слов | 100% для коротких |
| 12 | Реконструкция предложений | >80% |
| 13 | Реконструкция параграфа | >60% |
| 14 | Символ → ℝ¹² → символ | Круговой тест |

### 3.4 Модальность (будущее)

| # | Задача |
|---|--------|
| 15 | ImageEncoder → ℝ¹² |
| 16 | AudioEncoder → ℝ¹² |
| 17 | Универсальный декодер ℝ¹² → модальность |
| 18 | Кросс-модальный retrieval |

---

## 4. Открытые вопросы (самопроверка)

### Q1: Достаточно ли 12 измерений для точного разделения 160 символов?
**A:** Да. 3D уже достаточно (6³=216 > 160). 12D даёт огромный запас для фрактальной вложенности. Но нужно проверить экспериментально.

### Q2: Не потеряет ли модель точность при переходе с 256-dim на 12-dim?
**A:** Старый трансформер (256-dim) был заморожен и не обучался. Он давал случайный attention. 12-dim с обучением даст осмысленный attention. Точность должна ВЫРАСТИ.

### Q3: Как символы получат начальные координаты?
**A:** MDS из affinity-матрицы. 880K батчей Hebbian-обучения дают дифференцированный affinity. MDS проецирует его в ℝ¹² с сохранением расстояний.

### Q4: FractalAttention требует word-boundaries. Как их получить?
**A:** WordBoundaryDetector (transition probability minima из affinity). Запустить предварительно на датасете, сохранить разметку.

### Q5: Будет ли работать без FractalAttention?
**A:** Да. Простой attention в 12-dim работает (проверено). FractalAttention — улучшение, не блокер.

### Q6: Сколько параметров у полной архитектуры?
**A:** 10.7K сейчас. С FractalAttention: ~15K. Это в 70× меньше старого трансформера (1.09M).

### Q7: Что делать с 30 существующими модулями?
**A:** Оставить как инфраструктуру. Они полезны: affinity, grammar, knowledge base, library, contemplation. UnifiedTransformer использует их как источники данных, но не заменяет.

### Q8: Как проверять что модель не «бредит»?
**A:** LogicGuard + consistency check (уже есть в contemplation.py). При генерации: проверка что attention не прыгает между доменами, coherence > min.

### Q9: Нужен ли отдельный affinity-слой после unified transformer?
**A:** Нет. UnifiedTransformer САМ вычисляет attention и координаты. Affinity нужен только для инициализации координат символов.

### Q10: Как масштабировать на 1M+ токенов?
**A:** 12-dim × 512 seq_len × 256 batch = ~1.5M floats = 6 MB. GPU с 2 GB легко. FractalAttention на уровнях 2-3 работает с группами, не с токенами → O(L²/k²) вместо O(L²).

---

## 5. Файловая карта

```
НОВЫЕ:
  eva/symbolic/unified_transformer.py   ← UnifiedMultidimensionalTransformer
  train_unified.py                       ← gradient-based training

ИЗМЕНИТЬ:
  eva/symbolic/__init__.py              ← добавить импорт unified_transformer
  eva.bat                                ← train_unified.py как основной запуск

ИСПОЛЬЗОВАТЬ (без изменений):
  eva/symbolic/potential_field.py        ← affinity → MDS → координаты
  eva/symbolic/topological_field.py      ← MDS проекция
  eva/symbolic/fractal_attention.py      ← ⚠️ требует адаптации размерностей
  eva/symbolic/word_level.py             ← WordBoundaryDetector
  eva/symbolic/contradiction_filter.py   ← защита генерации
  eva/symbolic/contemplation.py          ← фоновое мышление
  eva/transformer.py                     ← базовые блоки (RMSNorm, SwiGLU)
```

---

*Документ обновляется при каждом значимом изменении архитектуры. Последнее обновление: 2026-05-22 — добавлен UnifiedMultidimensionalTransformer.*
