"""Generation test from checkpoint — shows REAL text output."""
import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
sys.stdout.reconfigure(encoding='utf-8')
import time
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

BPE_MODEL = r'real_data/bpe_ru_32k.model'
CS_PATH = r'real_data/concept_space.json'
LATTICE_PATH = r'real_data/syntax_lattice.json'

print("Loading SentencePiece...")
sp = spm.SentencePieceProcessor(model_file=BPE_MODEL)

print("Loading ConceptSpace (785MB)...")
t0 = time.time()
cs = ConceptSpace.load(CS_PATH)
print(f"  {time.time()-t0:.1f}s — {len(cs.concept_vectors)} concepts")

print("Loading SyntaxLattice (534MB)...")
t0 = time.time()
lattice = SyntaxLattice()
lattice.load(LATTICE_PATH)
print(f"  {time.time()-t0:.1f}s — {sum(len(v) for v in lattice.ngrams.values())} n-grams, {len(lattice.connections)} connections")

print("Initializing generator...")
gen = CrystalGenerator(cs, sp, lattice)

# ── Diagnostics ──
import numpy as np
vecs = np.array(list(cs.concept_vectors.values()), dtype=np.float32)
rng = np.random.RandomState(42)
pair_sims = [float(vecs[rng.randint(32000)] @ vecs[rng.randint(32000)])
             for _ in range(2000)]
print(f"\nVector field: cos={np.mean(pair_sims):.4f} ± {np.std(pair_sims):.4f}")

for a, b in [('▁соба', 'ка'), ('▁человек', '▁война'), ('▁князь', '▁Андрей'), ('▁любовь', '▁смерть')]:
    ia, ib = sp.PieceToId(a), sp.PieceToId(b)
    if ia >= 0 and ib >= 0:
        va, vb = cs.concept_vector(ia), cs.concept_vector(ib)
        print(f"  sim({a.replace('▁',''):8s},{b.replace('▁',''):8s}) = {float(va @ vb):+.4f} (CID {ia:5d},{ib:5d})")

# Test connections for князь
print(f"\nTop-5 connections for князь (CID 6244):")
for cid, info in lattice.connections_of(6244, top_k=5):
    token = sp.IdToPiece(cid).replace('▁', '_')
    print(f"  {token:20s} (CID {cid:5d}) strength={info['strength']:.3f} type={info['type']}")

# ── Generation ──
seeds = [
    ('князь', 'война'),
    ('человек', 'любовь'),
    ('смерть', None),
    ('собака', None),
]

print(f"\n{'='*60}")
print(f"GENERATION — 10K lines trained")
print(f"{'='*60}")

for seed, query in seeds:
    kwargs = {'seed_word': seed, 'max_words': 15}
    if query:
        kwargs['query_words'] = [query]

    result = gen.generate(**kwargs)
    chain = result['concept_path']
    tokens = [sp.IdToPiece(c) for c in chain[:12]]
    text = result['text']

    print(f"\n[{seed}]" + (f" (query={query})" if query else ""))
    print(f"  tokens: {' '.join(tokens)}{'...' if len(chain) > 12 else ''}")
    print(f"  text:   {text.strip()[:120]}")
    print(f"  score:  {result['score']:.1f}  words: {result['word_count']}")
