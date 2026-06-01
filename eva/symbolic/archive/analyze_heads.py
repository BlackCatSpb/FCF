"""
АНАЛИЗ: что модель должна извлечь из структурированных метаданных.

У нас есть:
- 27,061 полных треков предложений [L x 384]
- Каждый h[t] содержит 97 бит структурированной перфокарты
- 2.4M токенов с полным контекстом
- attractor field с потенциалами
- potential_db с 40K transitions, delta_mean, context_vectors

Задача: спроектировать Heads так, чтобы они извлекали
морфологию, синтаксис и семантику — каждый на своём уровне иерархии.
"""
import sys, os, pickle, math
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

# ═══════════════════════════════════════════════════════════════
# 1. МОРФОЛОГИЯ — структура слова из pos_in_word + word_len
# ═══════════════════════════════════════════════════════════════

print("="*70)
print("1. МОРФОЛОГИЯ: структура слова")
print("="*70)

# Collect: for each word length, what's the distribution of tokens at each position?
# e.g. word_len=4: pos_0=[т,с,в,п,н...], pos_1=[о,е,а...], pos_2=[р,л,н...], pos_3=[й,а,е...]

morph = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
# morph[word_len][position_id][token_id] = count

rng_word_lens = []  # distribution of word lengths

for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    traj = htraj.symbol_trajectory
    tokens = htraj.ids
    
    # Find word boundaries from trajectory flags
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

print(f"  Words analyzed: {len(rng_word_lens):,}")
print(f"  Avg word length: {np.mean(rng_word_lens):.2f} tokens")
print(f"  Max word length: {max(rng_word_lens)} tokens")

# Top tokens at each position for short words (length 2-5)
for wl in [2, 3, 4, 5]:
    if wl not in morph:
        continue
    print(f"\n  Word length = {wl} tokens (freq={sum(morph[wl][0].values()):,}):")
    for pos in range(wl):
        if pos not in morph[wl]:
            continue
        top = sorted(morph[wl][pos].items(), key=lambda x: -x[1])[:5]
        total = sum(morph[wl][pos].values())
        tokens_str = ', '.join([f"'{cv.decode([t])}' ({100*c/total:.0f}%)" for t, c in top])
        print(f"    pos[{pos}]: {tokens_str}")

# ─── Key insight: entropy per position in word ───
print(f"\n  Entropy by position in word (avg over all lengths):")
pos_entropy = defaultdict(list)
for wl in morph:
    for pos in morph[wl]:
        counts = np.array(list(morph[wl][pos].values()))
        probs = counts / counts.sum()
        ent = -np.sum(probs * np.log2(probs + 1e-10))
        pos_entropy[wl].append((pos, ent))

# Show for a few word lengths
for wl in [2, 4, 6]:
    if wl in pos_entropy:
        entries = pos_entropy[wl]
        print(f"    len={wl}: " + " ".join([f"p{p}={e:.1f}" for p, e in entries]))


# ═══════════════════════════════════════════════════════════════
# 2. СИНТАКСИС — структура предложения из word_num + sent patterns
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("2. СИНТАКСИС: структура предложения")
print("="*70)

# For each word position in sentence, what's the distribution of:
# - first token of the word (token that starts each word)
# - last token of the word
# - length of the word

syntax_word_start = defaultdict(list)    # word_num -> [token_ids that start word N]
syntax_word_end = defaultdict(list)      # word_num -> [token_ids that end word N]
syntax_word_len = defaultdict(list)      # word_num -> [length of word N]

for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    traj = htraj.symbol_trajectory
    tokens = htraj.ids
    
    in_word = False
    wn = -1
    wtokens = []
    
    for t, tid in enumerate(tokens):
        info = packer.unpack_token(traj[t])
        flags = info['flags']
        is_start = (flags >> packer.F_WORD_START) & 1
        is_end = (flags >> packer.F_WORD_END) & 1
        is_special = (flags >> packer.F_SPECIAL) & 1
        
        if is_start and not is_special:
            wn += 1
            in_word = True
            wtokens = [tid]
        elif in_word:
            if not is_special:
                wtokens.append(tid)
            if is_end:
                syntax_word_start[wn].append(wtokens[0])
                syntax_word_end[wn].append(wtokens[-1])
                syntax_word_len[wn].append(len(wtokens))
                in_word = False

# Show patterns for first 5 word positions
print(f"  Total words analyzed: {sum(len(v) for v in syntax_word_start.values()):,}")
print(f"  Max word position: {max(syntax_word_start.keys())}")
print()

for wn in range(min(6, max(syntax_word_start.keys()))):
    if wn not in syntax_word_start:
        continue
    # Top starting tokens
    starts = syntax_word_start[wn]
    start_counts = defaultdict(int)
    for s in starts:
        start_counts[s] += 1
    top_starts = sorted(start_counts.items(), key=lambda x: -x[1])[:7]
    
    # Top ending tokens
    ends = syntax_word_end[wn]
    end_counts = defaultdict(int)
    for e in ends:
        end_counts[e] += 1
    top_ends = sorted(end_counts.items(), key=lambda x: -x[1])[:5]
    
    # Avg length
    avg_len = np.mean(syntax_word_len[wn])
    
    n = len(starts)
    start_str = ', '.join([f"'{cv.decode([t])}' ({100*c/n:.0f}%)" for t, c in top_starts])
    end_str = ', '.join([f"'{cv.decode([t])}' ({100*c/n:.0f}%)" for t, c in top_ends])
    print(f"  Word #{wn} ({n:,} occurrences, avg_len={avg_len:.1f}):")
    print(f"    starts with: {start_str}")
    print(f"    ends with:   {end_str}")


# ═══════════════════════════════════════════════════════════════
# 3. СЕМАНТИКА — attractor field density + context vectors
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("3. СЕМАНТИКА: контекстные вектора и плотность attractor field")
print("="*70)

# For each token, we have context_vectors (avg h[t] from all occurrences)
# Two tokens are semantically similar if their context vectors are close
# Two tokens are syntactically similar if they appear in similar positions

context = db['context_vectors']
context_counts = db['context_counts']

# Find semantically close token pairs
print(f"  Computing token similarity matrix...")
# Only compute for tokens with enough data
min_occurrences = 10
valid_tokens = [tid for tid in range(V) if context_counts[tid] >= min_occurrences]
valid_vecs = np.array([context[tid] for tid in valid_tokens])
norm_vecs = valid_vecs / np.linalg.norm(valid_vecs, axis=1, keepdims=True)
sim_matrix = norm_vecs @ norm_vecs.T  # [n_valid, n_valid]

# For each token, find its nearest neighbors
print(f"  Token neighborhoods (for {len(valid_tokens)} tokens with >=10 occurrences):")
for tid in [267, 268, 270, 271, 274, 275, 284, 315, 334, 436]:
    if tid not in valid_tokens:
        continue
    idx = valid_tokens.index(tid)
    sims = sim_matrix[idx]
    neighbors = np.argsort(-sims)[1:8]  # skip self (index 0)
    neighbor_str = ', '.join([f"'{cv.decode([valid_tokens[n]])}' ({sims[n]:.3f})" for n in neighbors])
    text = cv.decode([tid])
    print(f"    '{text}' ({tid}): [{neighbor_str}]")


# ═══════════════════════════════════════════════════════════════
# 4. CONCEPT & CONTRADICTION — из potential_db
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("4. CONCEPT & CONTRADICTION")
print("="*70)

# CONCEPT: dense attractor region = well-known concept
# From storage, we already have attractor field with ~2.4M points
# High-density regions = frequent patterns = concepts

# CONTRADICTION: P(src→dst) = 0 when P(src→*) > threshold
# A transition that never occurs but should be possible

# Count destinations per source
src_dest_count = (trans_count > 0).sum(axis=1)  # [4101]
src_total = trans_count.sum(axis=1)              # [4101]
src_freq = src_total > 0                         # tokens that have ANY transitions

# Find "suspicious zeros": tokens with many destinations but missing some expected ones
# Expected if cos(context_vec[src], context_vec[dst]) is high
# i.e. two semantically related tokens never appear next to each other

print(f"  Tokens with transitions: {src_freq.sum()}/{V}")
print(f"  Avg destinations/src: {src_dest_count[src_freq].mean():.1f}")

# Build contradiction candidates: high-similarity context vectors with zero transitions
print(f"\n  Finding contradiction candidates (similar context, zero transitions)...")
n_contra = 0
contra_candidates = []
# Only check tokens with enough data
for src in valid_tokens[:100]:  # first 100 valid tokens
    src_idx = valid_tokens.index(src)
    # Find most similar tokens
    sims = sim_matrix[src_idx]
    top_sim = np.argsort(-sims)[1:20]  # top 20 similar (excluding self)
    for dst_idx in top_sim:
        dst = valid_tokens[dst_idx]
        if dst <= src:  # skip symmetric
            continue
        # If transition never occurs in either direction
        if trans_count[src, dst] == 0 and trans_count[dst, src] == 0:
            if n_contra < 20:
                s_text = cv.decode([src])
                d_text = cv.decode([dst])
                print(f"    '{s_text}' ({src}) <-> '{d_text}' ({dst}): "
                      f"sim={sims[dst_idx]:.3f}, never adjacent")
                contra_candidates.append((src, dst, sims[dst_idx]))
            n_contra += 1

print(f"  Total contradiction candidates: {n_contra}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY: архитектура Heads
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("ИТОГ: архитектура Heads для v5")
print("="*70)
print("""
MORPHOLOGY HEAD (внимание внутри слова)
  Вход:  h[t] для всех t внутри одного WORD_OPEN...WORD_CLOSE
  Смотрит: pos_in_word, word_len, какие токены на каких позициях
  Выход: "на позиции X слово длиной Y обычно стоит токен Z"
  Тренировка: из morph[word_len][position] — реальные распределения

SYNTAX HEAD (внимание между словами)
  Вход: sequence of word centroids [wn, 384]
  Смотрит: word_num, sent_len, P(word_N | word_{N-1})
  Выход: "на позиции N в предложении обычно слово типа X"
  Тренировка: из syntax_word_start — реальные распределения слово-позиция

SEMANTICS HEAD (внимание через весь контекст)
  Вход: context_vectors [V, 384] + attractor potentials
  Смотрит: плотность attractor field, similarity между токенами
  Выход: "токен A семантически близок к B"
  Тренировка: из attractor+context similarity — кластеризация

CONCEPT HEAD (потенциальные концепты)
  Вход: attractor potential density + reserved dims
  Смотрит: gaps в поле потенциалов
  Выход: концепт обнаружен в регионе X
  Метод: semantic gap detection (как EVA-Ai ConceptMiner)

CONTRADICTION HEAD (невозможное)
  Вход: transition matrix + context vectors  
  Смотрит: P=0 для семантически близких пар
  Выход: противоречие между X и Y
  Метод: high sim + zero count = contradiction candidate
""")
