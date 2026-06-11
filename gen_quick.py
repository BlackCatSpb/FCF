"""Interactive generation: type a phrase, get a continuation."""
import sys, time; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

t0 = time.time()
print("Loading model...", flush=True)
cs = ConceptSpace.load(r'real_data\concept_space.json')
sp = spm.SentencePieceProcessor(model_file=r'real_data\bpe_ru_32k.model')
lattice = SyntaxLattice()
lattice.load(r'real_data\syntax_lattice.json', load_ngrams=False)
gen = CrystalGenerator(cs, sp, lattice)
print(f"Loaded in {time.time()-t0:.1f}s\n", flush=True)

while True:
    phrase = input("> ").strip()
    if not phrase or phrase.lower() in ('exit', 'quit', 'q'):
        break
    t1 = time.time()
    r = gen.generate(seed_word=phrase, max_words=30)
    txt = r['text'].replace('\n', ' ').strip()
    print(f"[{time.time()-t1:.1f}s] >> {txt}\n")
