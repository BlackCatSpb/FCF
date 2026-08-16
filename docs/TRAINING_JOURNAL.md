# Журнал обучения FCF (полный конвейер, 2026-08)

Первый полный прогон STDP + CollocationMatrix на подвыборке корпуса.
Токенизатор: **ByteLevel BPE из WideBind (65 536 токенов)** через
`eva/symbolic/sp_compat.py` (SPCompatTokenizer — SentencePiece-совместимая обёртка,
`load_piece_model` выбирает backend по расширению: `.model` → SP, `tokenizer.json` → HF).

## Конфигурация

- **V=65536, dim=256**, learned-fields (field_bits=512, секторы [4,10,20]), neg_samples=3,
  context_window=4, pmi_gate=0.0, base_lr=0.001
- корпус: `real_data/corpus_1m.txt` — подвыборка 1 000 000 валидных строк из
  full_corpus_ru_clean.txt (9 348 348 строк, 1.5 ГБ)
- чекпойнт каждые 10 000 строк (803–833 МБ), gen-тест каждые 10 000
- ETA (по замеру ~1 строка/с): ~220 ч на 1M строк — вопрос производительности открыт
  (первые 20K шли 8 строк/с, затем замедление в ~8 раз — разбирается)

## Прогресс

| lines | pairs | примечание |
|---|---|---|
| 10 000 | 1.55M | первый чекпойнт |
| 20 000 | 3.08M | замедление 8→1 строка/с |
| 40 000 | 6.15M | чекпойнт, затем перезапуск из-за pickle-багов |
| 70 000 | 10.7M | актуально (meta.json 16.08) |

colloc: L2 растёт (~1.3M+), L3 ~700+. Генерация работает, текст пока бессвязный —
ожидаемо при малом корпусе.

## Исправления по ходу (16.08.2026)

1. **`--learned-fields` был мёртвым флагом** (регрессия при упрощении train_full):
   `cs.build_learned_fields(...)` не вызывался, field_bits пустые, field_gate=0.2 лишь
   давил lr на 18%. Возвращён вызов: `cs.build_learned_fields(n_field_bits=args.field_bits, sp=None)`.
2. **Чекпойнт не пикалился**: `threading.Lock` (Python 3.12 — фабрика, не тип) в
   FractalField/EntityField; лямбда-фабрики `defaultdict` в SyntaxLattice; локальные
   обёртки `lattice.update`/`decay_all` из CrystalGenerator; `_after_update_hook` тянул
   генератор в pickle. Добавлены `__getstate__/__setstate__` в concept_space.py и
   syntax_lattice.py (hook и обёртки исключаются, lock'и пересоздаются).
3. **Resume падал**: `f.tell()` после `break` из файлового итератора — `OSError`;
   убран из лога. Добавлены `--corpus` и `--max-lines`.
4. `train.bat`: убран `--vocab-size 256000` — V = sp.vocab_size() = 65536.

## Запуск и контроль

- обучение идёт в отдельном окне терминала (не привязано к сессии):
  `py -3.12 train_full.py --resume --corpus real_data\corpus_1m.txt --learned-fields
  --field-bits 512 --neg-samples 3 --context-window 4 --pmi-gate 0.0 --gen-every 10000`
- Ctrl+C в окне = мягкий стоп с сохранением чекпойнта (KeyboardInterrupt-обработчик)
- resume-контроль: `checkpoints/meta.json` (lines/pairs), `train.log`

## Тесты

376 passed, 7 skipped (включая tests/test_sp_compat.py — 15 тестов адаптера).