import sys, os; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import time; t0 = time.time()
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

BPE_MODEL = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\bpe_ru_32k.model'
CS_PATH = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space.json'
LAT_PATH = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\syntax_lattice.json'

print(f"[{time.time()-t0:.1f}s] Loading CS...", flush=True)
cs = ConceptSpace.load(CS_PATH)
print(f"[{time.time()-t0:.1f}s] CS loaded: {len(cs.concept_vectors)} vectors", flush=True)

print(f"[{time.time()-t0:.1f}s] Loading lattice...", flush=True)
lattice = SyntaxLattice()
lattice.load(LAT_PATH)
print(f"[{time.time()-t0:.1f}s] Lattice loaded", flush=True)

sp = spm.SentencePieceProcessor(model_file=BPE_MODEL)
gen = CrystalGenerator(cs, sp, lattice)
print(f"[{time.time()-t0:.1f}s] Generator ready", flush=True)

for seed in ['князь', 'человек', 'любовь', 'жизнь']:
    result = gen.generate(seed_word=seed, max_words=25)
    txt = result['text'].replace('\n', ' ').strip()
    print(f"\n[{seed}] {txt}", flush=True)
