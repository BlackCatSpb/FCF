# ПЛАН ДЕЙСТВИЙ v3 — трекинг выполнения

## Фундаментальное изменение архитектуры: EntityField (рекурсивное семантическое поле)

Замена изолированных словарей (char_envelope, harmonizer.morphemes, concept_vectors, sent_cache)
на единое EntityField, где КАЖДАЯ сущность (символ, морфема, слово, предложение, абзац)
имеет один вектор-представление, и все уровни связаны взаимными VSA bind/unbind.

**Принцип:** V(entity) += bind(V(context), LEVEL_ROLE) * lr — симметричная проекция
контекста на каждом уровне. Unbind(entity, LEVEL_ROLE) → superposition контекстов.
2048D достаточно для ~1000 уникальных сущностей на уровне (noise ≈ 1/√1000).

**Cross-level roles:** CHAR (char↔word), MORPH (morph↔word), WORD (word↔sent),
SENT (sent↔para). Один role на пару уровней — bind/unbind симметричны.

## Фаза 0: Диагностика (без изменений кода)
- [x] 0.1 Полнота MorphVocab — phase0_diagnostic.py запущен
- [x] 0.2 Размерность MorphemeField — ~2426 уникальных корней
- [ ] 0.3 VSA accuracy (compose→decompose→compose, 100 итераций, drift < 0.01)

## Фаза 1: Harmonizer (класс в concept_space.py)
- [x] 1.1 Role-векторы (Gram-Schmidt над randn)
- [x] 1.2 compose_word(ctx_vec) — контекстно-зависимый корень
- [x] 1.3 decompose_word
- [x] 1.4 harmonize (error, max 5 iter, damping, dirty-clear)
- [x] 1.5 Dirty-flag tracking
- [x] 1.6 Обратный индекс morph_to_words
- [ ] 1.7 Unit-test: compose↔decompose 100 итераций, drift < 0.01

## Фаза 2: Морфемное поле (MorphemeField)
- [ ] 2.1 GPU focus: morphemes на CPU, только focus-морфемы на GPU
- [ ] 2.2 Gram-Schmidt identity-векторы (отложено — randn норм. квази-ортогонален при 2048D)
- [x] 2.3 build_learned_fields с compose (confidence gate через _decompose_word)
- [x] 2.4 --no-morpheme-field (очистка harmonizer после load + gating в build_learned_fields)
- [ ] 2.5 Sector index для MorphemeField (отложено на phase 2+)
- [x] 2.6 morph_to_words при инициализации

## Фаза 3: Фокусированная гармонизация в STDP
- [x] 3.1 Вызов harmonize() после каждого batch (STDPTrainer._harmonize_batch)
- [x] 3.2 Slow-start: harm_lr * 0.1 на первом чекпоинте, ramp за 5
- [x] 3.3 Полный pass гармонизации всех seen dirty-слов на чекпоинте
- [ ] 3.4 Lateral inhibition: field_bits_overlap > 0.8 && cos > 0.7 → push
- [x] 3.5 Top-down контекст: hdc_ngram_repr per-sentence как sent_vec
- [ ] 3.6 Target ≥ 2 L/s; если < 1 → harmonize каждый 2-й batch

## Фаза 4: Semantic envelope → EntityField (переписано)
- [x] 4.1 EntityField class — единый dict entity_type → ndarray[2048] с VSA bind/unbind и cross-level role vectors (CHAR, MORPH, WORD, SENT, PARA) + type→role mapping
- [x] 4.2 Симметричная проекция: ef.bind(etype, eid, ctx_type, ctx_id, lr) — для каждой пары уровней
- [x] 4.3 В _harmonize_batch: полный цикл char→word→sent для каждого batch
- [x] 4.4 Удаление старого char_envelope, sent_cache — заменено EntityField

## Фаза 5: Sentence-level → EntityField
- [x] 5.1 EntityField c entity type 's' вместо sent_cache
- [x] 5.2 В harmonize: sent_vec = EntityField.get(('s', sent_key))
- [x] 5.3 Preharm checkpoint сохранён

## Фаза 6: Интеграция, тесты, метрики
- [x] 6.1 Save/load Harmonizer в .codes.npz
- [x] 6.2 Полный конфиг: envelope_decay, harm_slow_start_epochs + wiring
- [x] 6.3 Метрики на чекпоинте: morph_drift, envelope_coverage, harm_convergence
- [ ] 6.4 Тест на 5000 строк: signal/noise > 2, drift < 0.3, perplexity не выросла
- [ ] 6.5 Generation E2E: 10 seed-слов × 20 генераций
- [ ] 6.6 Perplexity-based rollback: если >10% → --no-harmonize
