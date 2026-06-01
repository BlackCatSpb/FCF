"""
ГЛУБОКИЙ АНАЛИЗ: Concept, Contradiction и семантика в структурированных координатах v5.

Ключевой вопрос: что такое concept и contradiction на каждом уровне иерархии?

Концепция:
  ┌─────────────────────────────────────────────────────┐
  │  Уровень       │ Concept                      │ Contradiction                │
  ├─────────────────┼──────────────────────────────┼─────────────────────────────┤
  │  TOKEN (0)      │ частый n-gram               │ P=0 для частого n-gram      │
  │  MORPH (1)      │ pattern "префикс+корень"    │ невалидная буква в слове    │
  │  SYNTAX (2)     │ pattern "Subj+Verb+Obj"     │ P=0 для грамматически возм. │
  │  SEMANTIC (3)   │ кластер смыслов             │ смысловая несовместимость   │
  │  CORPUS (4)     │ тема / сюжет                │ нарушение причинности       │
  └─────────────────────────────────────────────────────┘

Семантика — не context_vectors (это просто среднее по координатам).
Настоящая семантика = отношения между уровнями:
  - Token A семантически связан с B если они встречаются в ПОХОЖИХ контекстах
    (похожие pos_in_word, word_len, word_num, окружение flags)
  - Concept = плотный кластер в attractor + однородные метаданные
  - Contradiction = ожидаемая связь (из семантики) но P=0 в transition matrix
"""
import sys, os, pickle, math, time
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
from collections import defaultdict
from coordinate_packer import CoordinatePacker
from eva.symbolic.trajectory_store import TrajectoryStore
from eva.symbolic.bpe_tokenizer import BPEVocab

packer = CoordinatePacker()
cv = BPEVocab()
V = packer.V

# Load data
store = TrajectoryStore()
store.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\trajectory_store_v5.pkl')
with open(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\potential_db.pkl', 'rb') as f:
    db = pickle.load(f)

trans_count = db['transition_count']
trans_prob = db['transition_prob']
delta_sparse = db['delta_mean_sparse']
context_vectors = db['context_vectors']
context_counts = db['context_counts']

print("="*70)
print("АНАЛИЗ СЕМАНТИКИ: context_vectors vs. multi-level semantics")
print("="*70)

# ─── 1. Проблема context_vectors: это среднее по координатам, а не семантика ───
# context_vectors[tid] = mean of all h[t] where token_id == tid
# Проблема: h[t] содержит много метаданных (pos_in_word, word_num, etc)
# Поэтому context_vectors[tid] — это среднее ПОЗИЦИЙ, где встречается токен
# Два токена могут иметь похожие context_vectors просто потому что
# они встречаются на похожих позициях (синтаксис), а не из-за смысла

print(f"\n  context_vectors[tid] = mean(h[t] for all t where token == tid)")
print(f"  Проблема: h[t] = [token_id|pos|word_len|...|flags|...|reserved]")
print(f"  context_vectors смешивает: позицию + метаданные + 0")
print(f"  Два токена похожи по context_vectors = они на ПОХОЖИХ ПОЗИЦИЯХ")
print(f"  Это more syntax than semantics!")

# Solution: compute SEMANTIC vectors that EXCLUDE positional metadata
# Only use: token_id + reserved dims (once they're filled)
# Or better: use transition patterns as semantic signature

# ─── 2. Настоящая семантика: transition patterns ───
# Два токена семантически похожи, если у них ПОХОЖИЕ ПАТТЕРНЫ ПЕРЕХОДОВ
#   P(src→*): distribution of destinations
#   P(*→dst): distribution of sources

print(f"\n  Настоящая семантика = transition pattern similarity:")
print(f"  sim(tok_A, tok_B) = cosine(P(A→*), P(B→*)) + cosine(P(*→A), P(*→B))")

# Compute row-normalized transition vectors for tokens with data
row_sum = trans_count.sum(axis=1, keepdims=True)
row_sum = np.maximum(row_sum, 1)
row_norm = (trans_count / row_sum).astype(np.float32)  # [V, V]

col_sum = trans_count.sum(axis=0, keepdims=True)
col_sum = np.maximum(col_sum, 1)
col_norm = (trans_count / col_sum).astype(np.float32)  # [V, V]

# Tokens with at least 10 transitions total
tokens_with_trans = (trans_count.sum(axis=1) + trans_count.sum(axis=0)) > 10
print(f"  Tokens with transition data: {tokens_with_trans.sum()}/{V}")

# Compute transition-pattern similarity for a sample
valid_tokens = np.where(tokens_with_trans)[0]
n_valid = len(valid_tokens)
print(f"\n  Computing transition-pattern similarity for {n_valid} tokens...")

# For each valid token, find top-5 similar tokens by transition patterns
# Use: sim = 0.5 * cos(row_norm[A], row_norm[B]) + 0.5 * cos(col_norm[A], col_norm[B])
n_sample = min(200, n_valid)  # compute for first 200
sample_tokens = valid_tokens[:n_sample]
sample_row = row_norm[sample_tokens]  # [n_sample, V]
sample_col = col_norm[sample_tokens]

# Norm for cosine
sample_row_n = sample_row / np.linalg.norm(sample_row, axis=1, keepdims=True)
sample_col_n = sample_col / np.linalg.norm(sample_col, axis=1, keepdims=True)

row_sim = sample_row_n @ sample_row_n.T  # [n_sample, n_sample]
col_sim = sample_col_n @ sample_col_n.T
trans_sim = 0.5 * row_sim + 0.5 * col_sim  # [n_sample, n_sample]

# Also compute context-vector similarity for comparison
sample_vec = context_vectors[sample_tokens]  # [n_sample, 384]
# Only use dims 0-96 (metadata) for fair comparison, or all 384
sample_vec_n = sample_vec / (np.linalg.norm(sample_vec, axis=1, keepdims=True) + 1e-10)
vec_sim = sample_vec_n @ sample_vec_n.T

print(f"\n  Top-5 token pairs: transition-sim vs context-vector-sim")
print(f"  {'Token A':<10} {'Token B':<10} {'trans_sim':<10} {'vec_sim':<10}  Type")
print(f"  {'-'*60}")

# Get top pairs from both similarity matrices
triu_idx = np.triu_indices(n_sample, k=1)
trans_pairs = list(zip(triu_idx[0], triu_idx[1], trans_sim[triu_idx]))
trans_pairs.sort(key=lambda x: -x[2])

for ia, ib, ts in trans_pairs[:15]:
    ta = sample_tokens[ia]
    tb = sample_tokens[ib]
    vs = vec_sim[ia, ib]
    a_text = cv.decode([ta])
    b_text = cv.decode([tb])
    
    # Determine type: same meta_type?
    a_meta = ta if ta < 161 else (2 if ta >= 161 else 7)  # rough
    b_meta = tb if tb < 161 else 2
    
    print(f"  {a_text:<10} {b_text:<10} {ts:.4f}     {vs:.4f}  "
          f"{'both special' if ta<161 and tb<161 else 'mixed' if ta<161 or tb<161 else 'both BPE'}")


# ─── 3. Concept detection: attractor field density gaps ───
print(f"\n{'='*70}")
print("CONCEPT DETECTION: attractor field density gaps")
print("="*70)

# Load attractor field
print(f"\n  Loading attractor field...")
import torch
try:
    haf = torch.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\attractor_field_v5.pt',
                     map_location='cpu', weights_only=False)
    if hasattr(haf, 'attractors') and hasattr(haf.attractors, 'centers'):
        centers = haf.attractors.centers[:haf.attractors.n_attractors].numpy()
        counts = haf.attractors.counts[:haf.attractors.n_attractors].numpy()
        valid = haf.attractors.valid_mask[:haf.attractors.n_attractors].numpy()
        centers = centers[valid]
        counts = counts[valid]
        print(f"  Attractors: {len(centers)} valid centers")
    elif hasattr(haf, 'centers'):
        centers = haf.centers.numpy()
        counts = getattr(haf, 'counts', np.ones(len(centers))).numpy()
        print(f"  Centers: {len(centers)}")
    else:
        # Try loading v5 attractor field directly
        af_path = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\attractor_field_v5.pt'
        af = torch.load(af_path, map_location='cpu', weights_only=False)
        # Explore structure
        print(f"  Type: {type(af)}")
        if hasattr(af, 'state_dict'):
            sd = af.state_dict()
            for k, v in sd.items():
                print(f"    {k}: {v.shape}")
        centers = None
except Exception as e:
    print(f"  Load error: {e}")
    centers = None

# Concept detection idea:
# 1. Find all dense attractors (count > threshold)
# 2. Compute pairwise distances between dense attractors
# 3. A CONCEPT = dense attractor that is ISOLATED from other dense attractors
#    i.e., its nearest neighbor among other dense attractors is FAR
# 4. A BOUNDARY between concepts = region of LOW density between two HIGH density regions
#
# Alternative: use the attractor field to compute potential at each trajectory point,
# then find points with:
#   - LOW potential (sparse region)
#   - SURROUNDED by HIGH potential (dense regions)  
#   = semantic gap = concept boundary

if centers is not None:
    print(f"\n  Computing attractor pairwise distances...")
    n_att = min(5000, len(centers))
    
    # Sample attractors (dense ones)
    dense_idx = np.argsort(-counts)[:n_att]
    dense_centers = centers[dense_idx]
    dense_counts = counts[dense_idx]
    
    # Compute nearest-neighbor distance among dense attractors
    from scipy.spatial.distance import cdist
    
    dist_matrix = cdist(dense_centers, dense_centers, metric='euclidean')
    np.fill_diagonal(dist_matrix, np.inf)  # exclude self
    nn_dist = dist_matrix.min(axis=1)  # nearest neighbor distance per attractor
    
    print(f"\n  Attractor nearest-neighbor distances:")
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(nn_dist, p)
        print(f"    P{p}: {val:.3f}")
    
    # CONCEPT attractors: those with LARGE nn_dist (isolated dense clusters)
    concept_threshold = np.percentile(nn_dist, 90)
    concept_attractors = dense_idx[nn_dist > concept_threshold]
    print(f"\n  CONCEPT attractors (isolated, nn_dist > P90={concept_threshold:.3f}):")
    print(f"    {len(concept_attractors)}/{n_att}")
    
    # For each concept attractor, show what's near it
    for idx in concept_attractors[:5]:
        c = centers[idx]
        c_count = counts[idx]
        
        # Find nearest tokens to this attractor center
        # Use context_vectors as representative of token positions
        vecs = context_vectors  # [V, 384]
        vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
        c_norm = c / (np.linalg.norm(c) + 1e-10)
        sim_to_c = vecs_norm @ c_norm  # [V]
        nearest_tokens = np.argsort(-sim_to_c)[:5]
        token_texts = [f"'{cv.decode([t])}' ({sim_to_c[t]:.3f})" for t in nearest_tokens]
        
        print(f"    Attractor count={c_count:.0f}, nn_dist={np.sort(nn_dist[dense_idx==idx])[0] if idx in dense_idx else 'N/A':.3f}")
        print(f"      Nearest tokens: {', '.join(token_texts)}")


# ─── 4. Contradiction detection: multi-level ───
print(f"\n{'='*70}")
print("CONTRADICTION DETECTION: multi-level")
print("="*70)

# Level 1: Token-level contradiction
#   Two tokens that SHOULD be adjacent (based on transition patterns) but NEVER are
#   Detected: P(src→dst) = 0 AND P(src→other) > threshold AND trans_sim(src, dst) > threshold

print(f"\n  Level 1: TOKEN-LEVEL contradictions")
print(f"  P(A→B)=0 but transition patterns of A and B are similar")
print(f"  i.e. 'A should transition to B based on A's pattern, but it never does'")

# Use the transition similarity we computed above
contra_token_count = 0
print(f"\n  Contradiction candidates (trans_sim > 0.8, P=0):")
for ia, ib, ts in trans_pairs:
    ta, tb = sample_tokens[ia], sample_tokens[ib]
    if trans_count[ta, tb] == 0 and trans_count[tb, ta] == 0:
        if ts > 0.8 and contra_token_count < 20:
            a_text = cv.decode([ta])
            b_text = cv.decode([tb])
            # Check grammar: would these ever be adjacent in a real sentence?
            a_special = ta in (156, 157, 158, 159, 160) or ta < 4 or ta == 156
            b_special = tb in (156, 157, 158, 159, 160) or tb < 4 or tb == 156
            if not a_special and not b_special:
                print(f"    '{a_text}' ({ta:4d}) <-> '{b_text}' ({tb:4d}): "
                      f"trans_sim={ts:.4f}, P=0, both content tokens")


# Level 2: Word-level contradiction
#   Two word TYPES that should never be adjacent in a sentence
#   e.g. "он он" (he he) — grammatically unusual
#   Detected: high freq of each, but P(word_N→word_{N+1})=0 for that pair

print(f"\n\n  Level 2: WORD-LEVEL contradictions")
print(f"  Two word-TYPES that are frequent but never adjacent")
print(f"  Analyzed from sentence-level word sequences")

# Build word sequences: for each sentence, get WORD_START tokens
word_bigrams = defaultdict(int)  # (first_word_token, second_word_token) → count
word_types = defaultdict(int)    # first_word_token → total count

for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    tokens = htraj.ids
    traj = htraj.symbol_trajectory
    
    in_word = False
    word_start_token = -1
    word_tokens_seq = []
    
    for t, tid in enumerate(tokens):
        info = packer.unpack_token(traj[t])
        flags = info['flags']
        
        is_start = (flags >> packer.F_WORD_START) & 1
        is_end = (flags >> packer.F_WORD_END) & 1
        is_special = (flags >> packer.F_SPECIAL) & 1
        
        if is_start and not is_special:
            word_start_token = tid
            word_tokens_seq.append(tid)
        elif is_end and not is_special and word_start_token >= 0:
            pass  # word complete
    
    # Record word-level bigrams
    for i in range(len(word_tokens_seq) - 1):
        w1 = word_tokens_seq[i]
        w2 = word_tokens_seq[i + 1]
        if (w1 >= 4 and w1 < 157) or (w1 > 160):  # content token
            word_bigrams[(w1, w2)] += 1
            word_types[w1] += 1

# Find "contradictions": frequent word types that never form a pair
print(f"\n  Word types: {len(word_types)}, word bigrams: {len(word_bigrams)}")

# For frequent word types, which pairs are missing?
frequent_words = [w for w, c in sorted(word_types.items(), key=lambda x: -x[1])[:50]]
freq_word_set = set(frequent_words)

missing_pairs = []
for w1 in frequent_words[:20]:
    for w2 in frequent_words[:20]:
        if w1 != w2 and (w1, w2) not in word_bigrams and (w2, w1) not in word_bigrams:
            # Both are frequent individually but never adjacent
            missing_pairs.append((w1, w2, word_types[w1], word_types[w2]))

missing_pairs.sort(key=lambda x: -(x[2] + x[3]))
print(f"\n  Missing word bigrams (both frequent, never adjacent):")
for w1, w2, c1, c2 in missing_pairs[:15]:
    w1_text = cv.decode([w1])
    w2_text = cv.decode([w2])
    # Check if they're grammatically compatible
    print(f"    '{w1_text}' ({w1:4d}, {c1}) <-> '{w2_text}' ({w2:4d}, {c2}) — never adjacent")


# Level 3: Morphological contradiction
#   A subword/token that violates the expected structure of a word
#   e.g. a consonant that appears at position 0 where only vowels are expected

print(f"\n\n  Level 3: MORPHOLOGICAL contradictions")
print(f"  Token at position X in word-length Y where it SHOULD not appear")

# Build the morph distribution (reuse from analyze_heads.py concept)
morph = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
rng_word_lens = []

for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    traj = htraj.symbol_trajectory
    tokens = htraj.ids
    
    in_word = False
    word_tokens = []
    for t, tid in enumerate(tokens):
        info = packer.unpack_token(traj[t])
        flags = info['flags']
        is_start = (flags >> packer.F_WORD_START) & 1
        is_end = (flags >> packer.F_WORD_END) & 1
        is_special = (flags >> packer.F_SPECIAL) & 1
        
        if is_start and not is_special:
            in_word = True
            word_tokens = [tid]
        elif in_word:
            if not is_special:
                word_tokens.append(tid)
            if is_end:
                wl = len(word_tokens)
                rng_word_lens.append(wl)
                for pi, wt in enumerate(word_tokens):
                    morph[wl][pi][wt] += 1
                in_word = False
                word_tokens = []

# For each (word_len, position), compute entropy and find "contradictions"
# A contradiction = token with ZERO count at this position, but NON-ZERO at other positions
# of the same word length
# Also: tokens with very low probability (< 1%) that should fit

print(f"\n  Token-position contradictions:")
for wl in [3, 4, 5]:
    if wl not in morph:
        continue
    
    # Get all tokens used at ANY position for this word length
    all_tokens_for_len = set()
    for pos in morph[wl]:
        all_tokens_for_len.update(morph[wl][pos].keys())
    
    # For each position, which tokens are MISSING?
    for pos in range(wl):
        if pos not in morph[wl]:
            continue
        tokens_here = set(morph[wl][pos].keys())
        missing = all_tokens_for_len - tokens_here
        total = sum(morph[wl][pos].values())
        
        if len(missing) > 0 and total > 100:
            # Show top-5 missing tokens (by total frequency across other positions)
            missing_freqs = [(t, sum(morph[wl][p].get(t, 0) for p in range(wl) if p != pos))
                             for t in missing]
            missing_freqs.sort(key=lambda x: -x[1])
            top_missing = missing_freqs[:5]
            missing_str = ', '.join([f"'{cv.decode([t])}' ({c})" for t, c in top_missing if c > 10])
            if missing_str:
                print(f"    len={wl}, pos={pos}: missing {missing_str}")


# ─── 5. Итоговый дизайн heads ───
print(f"\n{'='*70}")
print("ИТОГОВЫЙ ДИЗАЙН HEADS с учётом семантики")
print("="*70)
print("""
ОСНОВНОЕ ОТКРЫТИЕ:
  context_vectors = mean(h[t]) — это НЕ семантика, а средняя ПОЗИЦИЯ токена.
  Два токена могут быть похожи по context_vectors просто потому что оба —
  предлоги (стоят на word_num=1...5, никогда на word_num=0).
  Это syntax, а не semantics.

НАСТОЯЩАЯ СЕМАНТИКА:
  sim(A, B) = 0.5·cos(P(A→*), P(B→*)) + 0.5·cos(P(*→A), P(*→B))
  Два токена семантически похожи, если они имеют ПОХОЖИЕ ПАТТЕРНЫ ПЕРЕХОДОВ:
  какие токены после них, какие перед ними.

АРХИТЕКТУРА HEADS (data-driven, no neural network):

┌─────────────────────────────────────────────────────────────────────┐
│  Head            │ Уровень  │ Вход                     │ Выход      │
├─────────────────────────────────────────────────────────────────────┤
│  MorphHead       │ TOKEN    │ pos_in_word, word_len    │ P(token|pos)│
│  SyntaxHead      │ WORD     │ word_num, sent_len,      │ P(word_tok)│
│  TransitionHead  │ PAIR     │ token_id, context_ids    │ P(next|ctx)│
│  SemanticHead    │ PATTERN  │ transition similarity    │ sim_matrix │
│  ConceptHead     │ DENSITY  │ attractor potential      │ gap_score  │
│  ContraHead      │ MISSING  │ expected but P=0         │ penalty    │
└─────────────────────────────────────────────────────────────────────┘

ATTENTION MASKS:
  MorphHead:    attends to tokens within same WORD_OPEN...WORD_CLOSE
  SyntaxHead:   attends to word_START tokens only (one per word)
  TransitionHead: attends to previous N tokens (N-gram context)
  SemanticHead:  attends to all tokens with similar transition patterns
  ConceptHead:   attends to regions in attractor field (not tokens)
  ContraHead:    attends to pairs with high trans_sim but P=0

КАЖДАЯ ГОЛОВА:
  - Не учится (no parameters)
  - Читает из потенциальной базы (pre-computed statistics)
  - Применяет attention mask → правильный уровень иерархии
  - Возвращает score или penalty

СКОРИНГ (итоговая оценка):
  score(next_token|context) = 
    w_morph · P_morph(next) + 
    w_syntax · P_syntax(next) + 
    w_trans · P_trans(next|prev) + 
    w_sem · sim_semantic(next, context) +
    w_concept · concept_gap(next) -
    w_contra · contradiction_penalty(next)

  Веса w_* предсказываются transformer-ом из контекста:
  - На word_start: w_syntax ↑ (ожидаем начало слова)
  - В середине слова: w_morph ↑ (структура слова)
  - На редких токенах: w_sem ↑ (семантика вместо статистики)
""")
