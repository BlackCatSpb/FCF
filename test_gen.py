"""Test generation from latest checkpoint."""
import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

LOG = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\gen_test.txt'

def safe_print(s):
    """Print safely to cp1251 console, log full UTF-8 to file."""
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(s + '\n')
    try:
        sys.stdout.write(s + '\n')
    except UnicodeEncodeError:
        sys.stdout.write(s.encode('cp1251', errors='replace').decode('cp1251') + '\n')
    sys.stdout.flush()

CKPT = '4k'
CS_PATH = fr'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space_{CKPT}.json'
LAT_PATH = fr'C:\Users\black\OneDrive\Desktop\FCF\real_data\syntax_lattice_{CKPT}.json'
BPE_MODEL = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\bpe_ru_32k.model'

safe_print(f"Loading checkpoint {CKPT}...")
cs = ConceptSpace.load(CS_PATH)
sp = spm.SentencePieceProcessor(model_file=BPE_MODEL)
lattice = SyntaxLattice()
lattice.load(LAT_PATH)
gen = CrystalGenerator(cs, sp, lattice)

SEEDS = [
    'зачем', 'почему', 'однако', 'впрочем',
    'следовательно', 'никогда', 'внезапно',
]

for seed in SEEDS:
    result = gen.generate(seed_word=seed, max_words=30)
    txt = result['text'].replace('\n', ' ')
    safe_print(f"\n[{seed}] ({result['score']:.2f})")
    safe_print(f"  {txt}")

