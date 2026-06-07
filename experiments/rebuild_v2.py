"""
Rebuild from corpus — new architecture (ConceptNet + concept navigation).

Pipeline:
  1. Clean old data (all files)
  2. Train character-level BPE (Whitespace pre-tokenizer, no ByteLevel)
  3. Build ConceptNet skeleton (from conceptnet_ru.txt)
  4. Build ConceptSpace (SVD on concept transitions from corpus)
  5. Test generation
"""
import sys, os, json, time
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')

DATA_DIR = r'C:\Users\black\OneDrive\Desktop\FCF\real_data'
CORPUS_PATH = os.path.join(DATA_DIR, 'full_corpus_ru.txt')
BPE_PATH = os.path.join(DATA_DIR, 'bpe_tokenizer.json')
SKELETON_PATH = os.path.join(DATA_DIR, 'concept_skeleton.json')
SPACE_PATH = os.path.join(DATA_DIR, 'concept_space.json')

t_start = time.time()

# ── Step 1: Clean old data ──
print("=" * 60)
print("STEP 1: Cleaning old databases")
print("=" * 60)
if os.path.exists(DATA_DIR):
    for f in os.listdir(DATA_DIR):
        if f in ('full_corpus_ru.txt', 'conceptnet') or f.startswith('concept_'):
            continue
        fp = os.path.join(DATA_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)
            print(f"  deleted {f}")
    # Remove the archived version directories too
    for d in ['v5', 'v6', 'v7', 'v8', 'v9']:
        dp = os.path.join(DATA_DIR, d)
        if os.path.isdir(dp):
            import shutil
            shutil.rmtree(dp)
            print(f"  removed {d}")

# ── Step 2: Train character-level BPE ──
print("\n" + "=" * 60)
print("STEP 2: Training character-level BPE (vocab_size=8192)")
print("=" * 60)
from eva.symbolic.concept_tokenizer import train_character_bpe
train_character_bpe(CORPUS_PATH, vocab_size=8192, save_path=BPE_PATH)

# ── Step 3: Build ConceptNet skeleton ──
print("\n" + "=" * 60)
print("STEP 3: Building ConceptNet concept skeleton")
print("=" * 60)
from eva.symbolic.concept_net import ConceptSkeleton
sk = ConceptSkeleton()
sk.build()
sk.save(SKELETON_PATH)
print(f"  {sk.n_concepts} concepts, {len(sk.relations)} relations")

# ── Step 4: Initialize tokenizer ──
print("\n" + "=" * 60)
print("STEP 4: Initializing ConceptTokenizer")
print("=" * 60)
from eva.symbolic.concept_tokenizer import ConceptTokenizer
tok = ConceptTokenizer(bpe_path=BPE_PATH, skeleton_path=SKELETON_PATH)
tok.initialize()
print(f"  Vocab: {len(tok)}, BPE: {tok.bpe_vocab_size}, Concepts: {tok.skeleton.n_concepts}")

# ── Step 5: Build ConceptSpace ──
print("\n" + "=" * 60)
print("STEP 5: Building ConceptSpace")
print("=" * 60)
from eva.symbolic.concept_space import ConceptSpace
cs = ConceptSpace(sk, dim=128)
cs.build(corpus_path=CORPUS_PATH, tok=tok)
cs.save(SPACE_PATH)
print(f"  {len(cs.cid_list)} concepts @ {cs.dim}D")

# ── Step 6: Test generation ──
print("\n" + "=" * 60)
print("STEP 6: Test generation")
print("=" * 60)
from eva.symbolic.concept_generator import ConceptGenerator
gen = ConceptGenerator(cs, tok, {
    'temperature': 0.5,
    'concept_temp': 0.3,
    'word_temp': 0.2,
    'max_words': 15,
    'min_words': 3,
})

for seed in ['князь', 'война', 'сказал', 'человек', 'собака']:
    result = gen.generate(seed_word=seed)
    print(f"  [{seed}] {result['text']}")

elapsed = time.time() - t_start
print(f"\nTotal time: {elapsed:.1f}s")
print("Done!")
