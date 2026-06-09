"""Quick gen eval with CID-token debugging."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import sentencepiece as spm
import numpy as np
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

sp = spm.SentencePieceProcessor(model_file=r'real_data/bpe_ru_32k.model')
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

# Show CID->token mapping for seeds
for seed_word in ['князь', 'человек', 'война', 'любовь']:
    ids = gen._encode_input(seed_word)
    tokens = [sp.IdToPiece(c) for c in ids]
    print(f'{seed_word}: cids={ids} tokens={tokens}')

# Generate and dump full paths
for seed in ['князь', 'человек', 'война', 'любовь']:
    result = gen.generate(seed_word=seed, max_words=10)
    chain = result['concept_path']
    tokens = [sp.IdToPiece(c) for c in chain]
    text = sp.decode(chain)
    print(f'\n[{seed}] score={result["score"]:.1f}')
    print(f'  CIDs: {chain}')
    print(f'  text: {repr(text[:100])}')
