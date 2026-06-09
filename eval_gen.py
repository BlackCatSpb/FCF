"""Quick generation evaluation on trained vectors."""
import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import random, math
import numpy as np
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

BPE_MODEL = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\bpe_ru_32k.model'
CORPUS = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru.txt'

sp = spm.SentencePieceProcessor(model_file=BPE_MODEL)
V = sp.vocab_size()
print(f"vocab_size = {V}")

print("\nInitializing...")
cs = ConceptSpace(vocab_size=V, dim=384)
cs.init_concepts()
cs.init_homeostasis()

lattice = SyntaxLattice()
gen = CrystalGenerator(cs, sp, lattice)

# Train on first N lines for quick evaluation
N = 5000
print(f"\nTraining on {N} lines...")
with open(CORPUS, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= N: break
        gen.train_from_text(line.strip())
print(f"  done ({N} lines)")

# Diagnostics
vecs = np.array(list(cs.concept_vectors.values()), dtype=np.float32)
sims = vecs @ vecs.T
n = len(sims)
rng = np.random.RandomState(42)
pair_sims = [sims[rng.randint(n), rng.randint(n)] for _ in range(5000)]
print(f"\n  cos = {np.mean(pair_sims):.4f} ± {np.std(pair_sims):.4f}")

# Check some key pairs
for a, b in [('▁соба', 'ка'), ('▁человек', '▁война'), ('▁князь', '▁Андрей'), ('▁любовь', '▁смерть')]:
    id_a, id_b = sp.PieceToId(a), sp.PieceToId(b)
    if id_a >= 0 and id_b >= 0:
        va, vb = cs.concept_vector(id_a), cs.concept_vector(id_b)
        if va is not None and vb is not None:
            print(f"  sim({a:12s},{b:12s}) = {float(va @ vb):+.4f}")

# Generation
SEEDS = ['князь', 'человек', 'война', 'любовь', 'жизнь', 'смерть']
print(f"\n{'─'*60}")
print(f"{'Generation results (5K lines trained)':^60}")
print(f"{'─'*60}")

for seed in SEEDS:
    result = gen.generate(seed_word=seed, max_words=15)
    text = result['text']
    score = result['score']
    wc = result['word_count']
    chain = result['concept_path']

    # Show token-level path
    tokens = [sp.IdToPiece(c) for c in chain[:10]]
    path_str = ' → '.join(t.replace('▁', '_') for t in tokens)
    
    print(f"\n  [{seed}] (w={wc}, s={score:.1f})")
    print(f"    tokens: {path_str}{'...' if len(chain) > 10 else ''}")
    print(f"    text:   {text[:100]}")

print(f"\n{'─'*60}")
print(f"Cold start (no seed):", end=" ")
result = gen.generate(max_words=15)
print(result['text'][:100])
