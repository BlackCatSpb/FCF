"""
OPTIMIZATION AUDIT v5 — что нужно изменить/оптимизировать.

Результат полного аудита FCF (70 файлов, 19 modules в eva/symbolic/).
"""

# ═══════════════════════════════════════════════════════════════
# A. ХРАНЕНИЕ ДАННЫХ — главная проблема
# ═══════════════════════════════════════════════════════════════

DATA_STORAGE = """
Текущее (проблема):
  trajectory_store_v5.pkl  = 7.3 GB   — все 27K треков [L, 384]
  heads_db.pkl             = 142 MB   — trans_prob [4101,4101]=67MB, morph/syntax, contra
  warpeace_trajectories.pkl= 3.7 GB   — дубликат треков

Проблемы:
  1. 7.3 GB для текста 5.2 MB — Factor 1400× — недопустимо
  2. trans_prob [4101,4101] float32 = 67 MB — плотная матрица, 99.76% нули
  3. Нет иерархической структуры: книг 4, глав ~360, абзацев ~тысячи
  4. Нет быстрых запросов: "найти все слова в книге 2"

Решение:
  ┌─────────────────────────────────────────────────────────┐
  │  level     │ storage                 │ size estimate   │
  ├─────────────────────────────────────────────────────────┤
  │  corpus    │ book_id, metadata       │ < 1 KB          │
  │  book      │ [4] текстов, stats      │ < 10 KB         │
  │  chapter   │ [~360] границы          │ < 100 KB        │
  │  sentence  │ [27K] token seq + centroids  │ 27K × [avg 20] int16 = ~1 MB  │
  │  word      │ [319K] start,end, centroids │ 319K × (2 int16 + 384 float16) = ~250 MB │
  │  token     │ [2.4M] pos_in_word,len,flags  │ 2.4M × 4 int16 = ~20 MB │
  └─────────────────────────────────────────────────────────┘
  
  Итого: ~300 MB вместо 11 GB.
  
  Ключевая оптимизация: не хранить полные [L, 384] треки.
  Все метаданные уже в CoordinatePacker — их можно восстановить
  из token_id + pos_in_word + word_len + word_num + sent_len.
  
  Хранить только:
    - token_id (int16)
    - для content token: pos_in_word, word_len, word_num (uint8)
    - sent_id, word_id (int32)
    - attractor potential (float16) — если нужно
"""

# ═══════════════════════════════════════════════════════════════
# B. TRANSITION MATRIX — sparse вместо dense
# ═══════════════════════════════════════════════════════════════

TRANSITION_MATRIX = """
Сейчас:  trans_prob [4101, 4101] float32 = 67 MB, плотность 0.24%
Нужно:   sparse (CSR) формат: 40,970 ненулевых × (2×int32 + float32) ≈ 0.5 MB
          + log_prob прекомпьют для быстрого lookup = 0.5 MB
          → экономия 66 MB, ускорение загрузки 100x
  
  Дополнительно:
    - transition[L2] = sparse [V, V] int32  (counts)
    - transition_log = sparse [V, V] float32 (log-prob, вычисляется при загрузке)
    - transition_L2 = sparse norm2 (для контрадикции)
    
  И всё — никаких плотных numpy матриц.
"""

# ═══════════════════════════════════════════════════════════════
# C. HEADS — что нужно расширить
# ═══════════════════════════════════════════════════════════════

HEADS_OPTIMIZATION = """
Текущий heads_db.pkl (142 MB):
  morph_dist    = dict: word_len → pos → {tid:count}
  syntax_dist   = dict: word_num → {tid:count}
  trans_prob    = array [4101,4101] float32 = 67 MB
  trans_sim     = dict: tid → [(neighbor, sim)] (sparse)
  contra_pairs  = list: (ta,tb,sim) = 9242 entries
  concept_scores = array [2000] float32
  token_counts  = dict: tid → count

Проблемы:
  1. FULL DENSE trans_prob: 67 MB (67% of database) — 99.76% zeros
  2. concept_scores = 2000 entries, all near 1.0 — бесполезно
  3. morph_dist: только word_len 2-19, но packer до 255
  4. syntax_dist: word_num 0-275, но разрежен на >50
  5. contra_pairs: 9242 — слишком много для генерации (замедляет)
  6. Нет: sent_start_dist, ngram_dist, word_pattern_dist

Оптимизация:
  1. trans_prob → sparse CSR (0.5 MB)
  2. concept_scores → recompute с правильной метрикой (cosine similarity, не RBF)
  3. morph_dist → дополнить до word_len=32 (common max)
  4. contra_pairs → top-1000 только (не 9242)
  5. Добавить отрицательный словарь: P=0 для частых (>10) пар
  6. Добавить ngram_dist: P(token | prev_2_tokens, prev_1_token)
  Итого: ~10 MB вместо 142 MB
"""

# ═══════════════════════════════════════════════════════════════
# D. ЧЕГО НЕ ХВАТАЕТ — новые компоненты
# ═══════════════════════════════════════════════════════════════

MISSING_COMPONENTS = """
1. WEIGHT TRANSFORMER (lightweight, 1-2 слоя, 384→64→6 weights)
   Вход:  h[t] из контекста + unpacked metadata
   Выход: 6 весов w_* для heads
   Параметры: ~50K (vs v3 unified_transformer ~4.2M)
   Обучение: predict weights that make score_all select correct next token
   
2. GENERATION LOOP (generation_v5.py)
   Пока:  heads_v5.score_all() — просто скор для всех токенов
   Надо:  полный цикл:
     context → unpack → predict weights → score tokens → 
     mask invalid → select token → pack → append → repeat
   Специальные токены: SENT_OPEN→WORD_OPEN→...→WORD_CLOSE→...→SENT_CLOSE
   (определяются жёстко, не через heads)
   
3. RESERVED DIM FILLER (reserved_dims.py)
   287 измерений (dims 97-383) заполняются во время генерации:
     dim 97-104: token_probability_bucket (8 бит: какая голова победила)
     dim 105-112: concept_gap_score
     dim 113-120: contradiction_flag
     dim 121-128: attractor_potential (quantized)
     ...
   Это те самые "семантические метки", которые transformer учится писать.
   
4. NEGATIVE DICTIONARY
   Для частых токенов (count > 10): P=0 транзишены → contradiction
   Сейчас: 9242 contra пар, включая !→( и другие punctuation
   Надо: только BPE content tokens, trans_sim > 0.9, count > 10
   Ожидание: ~200-500 пар, все семантически осмысленные

5. HIERARCHICAL QUERIES
   query(word, pos) → найти все вхождения с такой позицией
   query(sentence_id) → получить полный трек
   query(token_id, context_hash) → найти похожий контекст
   Нужен индекс: inverted index по token_id + tuple из координат
"""

# ═══════════════════════════════════════════════════════════════
# E. V3 LEGACY — что удалить/заархивировать
# ═══════════════════════════════════════════════════════════════

V3_LEGACY = """
Файлы для АРХИВИРОВАНИЯ (больше не нужны, всё заменено heads_v5.py):
  eva/symbolic/unified_transformer.py    — 829 строк, v3 transformer
  eva/symbolic/phase1_model.py           — 512 строк, 12-layer 384-dim
  eva/symbolic/heads.py                  — 422 строк, все neural heads
  eva/symbolic/potential_fields.py       — 1223 строк, TPF/WVF/HAF/SRG/KCA
  eva/symbolic/thought_loop.py           — v3 thought loop
  eva/symbolic/train_v3.py               — v3 training
  eva/symbolic/continuous_runtime.py     — v3 continuous generation
  eva/symbolic/h2k_pipeline.py           — v3 hard-to-knowledge
  
Файлы для СОХРАНЕНИЯ (используются v5):
  eva/symbolic/trajectory_store.py       — TrajectoryStore (нужен)
  eva/symbolic/bpe_tokenizer.py          — BPE vocab (нужен)
  eva/symbolic/char_vocab.py             — Character vocabs
  
Архивировано в _archive/: train_phase*.py, hybrid_train, etc.
"""

# ═══════════════════════════════════════════════════════════════
# F. ЗАКЛЮЧЕНИЕ
# ═══════════════════════════════════════════════════════════════

CONCLUSION = """
ТРИ ГЛАВНЫХ ОПТИМИЗАЦИИ:

1. ИЕРАРХИЧЕСКОЕ ХРАНЕНИЕ (11 GB → 300 MB)
   - Не хранить полные треки [L, 384]
   - Хранить только token_id + метаданные (int16/uint8)
   - Sentence→tokens можно восстановить через CoordinatePacker
   - Книги, главы, абзацы → отдельные уровни

2. SPARSE TRANSITION MATRIX (67 MB → 0.5 MB)
   - CSR format для 40,970 ненулевых (0.24%)
   - log_prob прекомпьют
   - numpy плотные матрицы → scipy.sparse

3. WEIGHT TRANSFORMER (50K параметров, вместо 4.2M)
   - 1-2 слоя, 384→64→6
   - Учится предсказывать веса heads
   - Не учится предсказывать токены (heads делают это)
   - Заполняет reserved dims
"""

print("A. DATA STORAGE:")
print(DATA_STORAGE)
print("B. TRANSITION MATRIX:")
print(TRANSITION_MATRIX)
print("C. HEADS OPTIMIZATION:")
print(HEADS_OPTIMIZATION)
print("D. MISSING COMPONENTS:")
print(MISSING_COMPONENTS)
print("E. V3 LEGACY:")
print(V3_LEGACY)
print("F. CONCLUSION:")
print(CONCLUSION)
