"""Interactive generation: type a phrase, get a continuation."""
import sys, time, json, re, glob; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator
from eva.symbolic.morph_vocab import MorphVocab

DATA = r'C:\Users\black\OneDrive\Desktop\FCF\real_data'

def latest_ckpt(pattern):
    files = glob.glob(f'{DATA}\\{pattern}')
    ks = [(int(m.group(1)), f) for f in files for m in [re.search(r'_(\d+)k\.', f)] if m]
    return max(ks, key=lambda x: x[0])[1] if ks else None

t0 = time.time()
print("Loading model...", flush=True)
cs_path = latest_ckpt('concept_space_*k.json')
lat_path = latest_ckpt('syntax_lattice_*k.json')
print(f"  Checkpoint: {cs_path}", flush=True)
cs = ConceptSpace.load(cs_path)
sp = spm.SentencePieceProcessor(model_file=f'{DATA}\\bpe_ru_146k.model')

print("Loading MorphVocab...", flush=True)
mv = MorphVocab(sp, f'{DATA}\\morph_vocab.json', f'{DATA}\\bpe_ru_146k.model')
print(f"  loaded: {len(mv.id_to_word)} words, {len(mv.path_overrides)} overrides")

lat = SyntaxLattice()
lat.load(lat_path, load_ngrams=False)
gen = CrystalGenerator(cs, sp, lat, morph_vocab=mv)
print(f"Loaded in {time.time()-t0:.1f}s\n", flush=True)

while True:
    phrase = input("> ").strip()
    if not phrase or phrase.lower() in ('exit', 'quit', 'q'):
        break
    t1 = time.time()
    r = gen.generate(seed_word=phrase, max_words=30)
    txt = r['text'].replace('\n', ' ').strip()
    print(f"[{time.time()-t1:.1f}s] [{r['score']:.2f}] >> {txt}\n")
