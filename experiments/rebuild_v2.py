"""
Full rebuild with Wikipedia corpus — new architecture.
Pipeline:
  1. Clean old data
  2. Train character-level BPE on Wikipedia
  3. Build ConceptNet skeleton (filters long anchors >30 chars)
  4. Build ConceptSpace (concept transitions from Wikipedia)
  5. Build SyntaxLattice (n-grams from Wikipedia)
  6. Test generation
"""
import sys, os, json, time
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')

DATA_DIR = r'C:\Users\black\OneDrive\Desktop\FCF\real_data'
CORPUS_PATH = os.path.join(DATA_DIR, 'full_corpus_ru.txt')
BPE_PATH = os.path.join(DATA_DIR, 'bpe_tokenizer.json')
SKELETON_PATH = os.path.join(DATA_DIR, 'concept_skeleton.json')
SPACE_PATH = os.path.join(DATA_DIR, 'concept_space.json')
LATTICE_PATH = os.path.join(DATA_DIR, 'syntax_lattice.json')

t_start = time.time()

# ── Step 1: Clean old data (keep corpus, skeleton, conceptnet) ──
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

# ── Step 2: Train character-level BPE ──
print("\n" + "=" * 60)
print("STEP 2: Training character-level BPE (vocab_size=8192)")
print("=" * 60)
import sentencepiece as spm
t = time.time()
_sp = spm.SentencePieceProcessor()
_sp.load(os.path.join(DATA_DIR, 'bpe_ru.model'))
print(f"  Loaded SentencePiece model (vocab={_sp.get_piece_size()}) ({time.time()-t:.1f}s)")

# ── Step 3: Build ConceptNet skeleton ──
print("\n" + "=" * 60)
print("STEP 3: Building ConceptNet concept skeleton (filter >30 chars)")
print("=" * 60)
t = time.time()
class _SkeletonStub:
    n_concepts = 0
    relations = {}
    def save(self, path):
        pass
sk = _SkeletonStub()
print(f"  ConceptSkeleton skipped (using SentencePiece directly) ({time.time()-t:.1f}s)")

# ── Step 4: Initialize tokenizer ──
print("\n" + "=" * 60)
print("STEP 4: Initializing ConceptTokenizer")
print("=" * 60)
class _SPTokenizer:
    def __init__(self, sp):
        self.sp = sp
    def initialize(self):
        pass
    def encode(self, text):
        return self.sp.encode(text)
    def decode(self, ids):
        return self.sp.decode(ids)
    def word_to_cid(self, word):
        return self.sp.encode(word)[0]
    def __len__(self):
        return self.sp.get_piece_size()
    @property
    def bpe_vocab_size(self):
        return self.sp.get_piece_size()
    @property
    def skeleton(self):
        return _SkeletonStub()
tok = _SPTokenizer(_sp)
print(f"  Vocab: {len(tok)}, BPE: {tok.bpe_vocab_size}")

# ── Step 5: Build ConceptSpace ──
print("\n" + "=" * 60)
print("STEP 5: Building ConceptSpace (concept transitions from Wikipedia)")
print("=" * 60)
from eva.symbolic.concept_space import ConceptSpace
t = time.time()
cs = ConceptSpace(sk, dim=128)
cs.build(corpus_path=CORPUS_PATH, tok=tok)
cs.save(SPACE_PATH)
print(f"  {len(cs.cid_list)} concepts @ {cs.dim}D ({time.time()-t:.1f}s)")

# ── Step 6: Build SyntaxLattice ──
print("\n" + "=" * 60)
print("STEP 6: Building SyntaxLattice")
print("=" * 60)
from eva.symbolic.syntax_lattice import SyntaxLattice
t = time.time()
lattice = SyntaxLattice()
lattice.build(corpus_path=CORPUS_PATH, tok=tok)
lattice.save(LATTICE_PATH)
print(f"  {[len(lattice.bigram_prefixes), len(lattice.trigram_prefixes), len(lattice.fourgram_prefixes)]} prefixes ({time.time()-t:.1f}s)")

# ── Step 7: Quick test ──
print("\n" + "=" * 60)
print("STEP 7: Quick generation test")
print("=" * 60)
from eva.symbolic.crystal_generator import CrystalGenerator
from eva.symbolic.hormonal_system import HormonalSystem

hormones = HormonalSystem()
gen = CrystalGenerator(cs, tok, lattice, hormones, {
    'max_words': 12,
    'min_words': 3,
    'temperature': 0.5,
})

for seed in ['князь', 'война', 'человек', 'Россия', 'Москва']:
    result = gen.generate(seed_word=seed)
    print(f"  [{seed}] {result['text']}")

# Check anchors
print("\n  Anchor samples:")
for w in ['князь', 'война', 'Пьер', 'xrqjz']:
    cid, conf = gen.resolve_anchor(w)
    anchor = cs.concept_info.get(cid, {}).get('anchor', '?')
    print(f"    {w} → cid={cid}, conf={conf:.3f}, anchor='{anchor}'")

elapsed = time.time() - t_start
print(f"\nTotal time: {elapsed:.1f}s")
print("Done!")
