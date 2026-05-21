# EVA  — План адаптации (на основе EVA NEW.txt)

## Что уже реализовано в EVA v1

| Компонент | Статус | Совместимость с  |
|-----------|--------|---------------------|
| TransformerBlock + SwiGLU | ✅ | База для SVD |
| FAISS StateStorage | ✅ | Нужен HNSW + PQ |
| SRG (3 компоненты) | ✅ | Добавить Anomaly Detection, Meta-SRG |
| KCA (5 итераций, демпфирование) | ✅ | Добавить монотонный регуляризатор |
| EthicsFilter (5 аксиом) | ✅ | Без изменений |
| CuriosityLoop | ✅ | Добавить Intrinsic Curiosity |
| LoRA + DomainRegistry | ✅ | Заменить на Streaming GMM |
| GrowthController | ✅ | Без изменений |
| Sleep Mode | ✅ | Добавить Dream Mode, Forgetfulness Gate |
| State Algebra | ✅ | Добавить Cross-Domain Translation |
| LanguageTrainer | ✅ | Заменить на Progressive Bootstrapping |

---

## Фаза 1: Атомарный базис (SVD) — критично

**Что:** разложить обученную модель через SVD, сохранить K атомов на матрицу.
**Зачем:** все модификации весов — через коэффициенты c_i, а не прямые обновления.
**Строк кода:** ~150 (новый модуль `fcf/atomic_basis.py`)
**Влияние:** фундаментальное — меняет способ хранения и применения знаний.

```
1. После языкового обучения → SVD(W_Q), SVD(W_K), ... для всех матриц
2. Сохранить K атомов (ошибка реконструкции ≤ 10⁻³)
3. ΔW = Σ c_i · σ_i · u_i · v_i^T
4. Латентный код z → Decoder(z) → коэффициенты c_i
```

---

## Фаза 2: Фрактальная иерархия инструкций

**Что:** 4 уровня кодов: символ → слово → предложение → текст.
**Зачем:** знания организованы иерархически, ссылки между уровнями.
**Строк кода:** ~300 (`fcf/fractal_hierarchy.py`)
**Зависит от:** Фазы 1 (SVD)

---

## Фаза 3: Streaming GMM (замена DomainRegistry)

**Что:** динамическое создание/слияние/удаление доменов через GMM.
**Зачем:** домены не предопределены, рождаются и умирают по метрикам.
**Строк кода:** ~200 (`fcf/streaming_gmm.py`)
**Замена:** DomainRegistry → StreamingGMM

---

## Фаза 4: HNSW + Product Quantization

**Что:** трёхуровневый HNSW с PQ для сжатия векторов.
**Зачем:** масштабирование до миллионов кодов без деградации поиска.
**Строк кода:** ~150 (доработка `fcf/hnsw_index.py`)

---

## Фаза 5: Улучшенный Sleep Mode

**Что:** + Dream Mode, Forgetfulness Gate, Adversarial Validation.
**Строк кода:** ~200 (доработка `fcf/sleep_mode.py`)

---

## Фаза 6: Расширения (не блокируют ядро)

- Anomaly Detection for SRG
- Meta-SRG (тренд уверенности)
- Intrinsic Curiosity
- Cross-Domain Translation в State Algebra
- Self-Descriptive Codes
- Code Provenance Tracking

---

## Порядок реализации

```
Неделя 1: Фаза 1 (SVD) — самый важный, меняет всё
Неделя 2: Фаза 2 (иерархия) + Фаза 3 (Streaming GMM)
Неделя 3: Фаза 4 (HNSW+PQ) + Фаза 5 (Sleep )
Неделя 4: Фаза 6 (расширения) + интеграция
```

## Что НЕ трогаем

- Tokenizer (50K слов) — работает
- Embedding + LM Head (weight tied) — работает
- TransformerBlock (RoPE, SwiGLU, RMSNorm) — база для SVD
- run.py / CLI — минорные правки
- Trainers — заменятся на Progressive Bootstrapping
