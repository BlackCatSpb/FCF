"""Quick generation eval — 500 line training."""
import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

sp = spm.SentencePieceProcessor(
    model_file=r'real_data/bpe_ru_32k.model')
cs = ConceptSpace(vocab_size=sp.vocab_size(), dim=384)
cs.init_concepts()
cs.init_homeostasis()

lattice = SyntaxLattice()
gen = CrystalGenerator(cs, sp, lattice)
gen.train_lr = 0.01

N = 500
with open(r'real_data/full_corpus_ru.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= N:
            break
        gen.train_from_text(line.strip())

# Diagnostics
vecs = np.array(list(cs.concept_vectors.values()), dtype=np.float32)
rng = np.random.RandomState(42)
pair_sims = [float(vecs[rng.randint(32000)] @ vecs[rng.randint(32000)])
             for _ in range(2000)]
print(f'cos = {np.mean(pair_sims):.4f} +/- {np.std(pair_sims):.4f}')

# Tracked pairs
pairs = [
    ('\u2581\u0441\u043e\u0431\u0430', '\u043a\u0430'),
    ('\u2581\u0447\u0435\u043b\u043e\u0432\u0435\u043a', '\u2581\u0432\u043e\u0439\u043d\u0430'),
    ('\u2581\u043a\u043d\u044f\u0437\u044c', '\u2581\u0410\u043d\u0434\u0440\u0435\u0439'),
]
for a, b in pairs:
    ia, ib = sp.PieceToId(a), sp.PieceToId(b)
    if ia >= 0 and ib >= 0:
        s = float(cs.concept_vector(ia) @ cs.concept_vector(ib))
        label_a = a.replace('\u2581', '')
        label_b = b.replace('\u2581', '')
        print(f'  sim({label_a:10s},{label_b:10s}) = {s:+.4f}  cids=({ia},{ib})')
    else:
        print(f'  MISS: {repr(a)}->{ia}  {repr(b)}->{ib}')

# Generation
seeds = ['князь', 'человек', 'война', 'любовь']
for seed in seeds:
    result = gen.generate(seed_word=seed, max_words=12)
    chain = result['concept_path']
    tokens = [sp.IdToPiece(c).replace('\u2581', '_') for c in chain]
    print(f'\n[{seed}] score={result["score"]:.1f}')
    print(f'  tokens: {" ".join(tokens)}')
    print(f'  text:   {result["text"][:80]}')
