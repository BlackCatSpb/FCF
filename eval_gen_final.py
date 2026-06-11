import sys, os; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import time; t0 = time.time()
import sentencepiece as spm
import numpy as np
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

LOG = r'C:\Users\black\OneDrive\Desktop\FCF\gen_output_utf8.txt'
buffer = []

BPE_MODEL = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\bpe_ru_32k.model'
CS_PATH = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space.json'
LAT_PATH = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\syntax_lattice.json'

sp = spm.SentencePieceProcessor(model_file=BPE_MODEL)

buffer.append(f"[{time.time()-t0:.1f}s] Loading ConceptSpace...")
cs = ConceptSpace.load(CS_PATH)

buffer.append(f"[{time.time()-t0:.1f}s] Loading SyntaxLattice...")
lattice = SyntaxLattice()
lattice.load(LAT_PATH)
buffer.append(f"[{time.time()-t0:.1f}s] Lattice loaded ({len(lattice.concept_freq)} concepts)")

gen = CrystalGenerator(cs, sp, lattice)
buffer.append(f"[{time.time()-t0:.1f}s] Generator ready")

from eva.symbolic.vector_health import check_antonym_collapse
ant = check_antonym_collapse(cs, sp)
ants = [f"{k}={v:.2f}" for k, v in ant.items()]
buffer.append(f"  Antonym cosines: {' '.join(ants)}")

SEEDS = [
    'князь', 'человек', 'война', 'любовь', 'жизнь', 'смерть',
    'дом', 'народ', 'время', 'мир', 'душа', 'бог', 'огонь', 'вода',
]
buffer.append(f"\n{'='*70}")
buffer.append("GENERATION FROM FINAL CHECKPOINT (145K lines)")
buffer.append(f"{'='*70}")

for seed in SEEDS:
    result = gen.generate(seed_word=seed, max_words=30)
    txt = result['text'].replace('\n', ' ').strip()
    buffer.append(f"\n  [{seed}] (score={result['score']:.1f}, w={result['word_count']})")
    buffer.append(f"    >> {txt}")

buffer.append(f"\n{'='*70}")
buffer.append("COLD START (no seed)")
result = gen.generate(max_words=30)
buffer.append(f"  >> {result['text'].replace(chr(10), ' ').strip()}")

buffer.append(f"\nTotal: {time.time()-t0:.1f}s")

with open(LOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buffer))
print(f"Written to {LOG}")
