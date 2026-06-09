"""Compare STDP strategies on synthetic data with and without repulsion."""
import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
sys.stdout.reconfigure(encoding='utf-8')
import math, numpy as np
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

sp = spm.SentencePieceProcessor(model_file=r'real_data/bpe_ru_32k.model')

sentences = """
Князь любит войну.
Князь ненавидит мир.
Князь Андрей любит войну.
Князь Андрей ненавидит мир.
Человек любит жизнь.
Человек ненавидит смерть.
Человек должен быть свободен.
Собака любит кость.
Собака грызёт кость.
Собака бежит быстро.
Кошка любит молоко.
Кошка ловит мышь.
Война убивает людей.
Мир приносит радость.
Жизнь прекрасна.
Смерть неизбежна.
Андрей любит княжну Марью.
Марья любит Андрея.
Князь говорит тихо.
Человек говорит громко.
""".strip().split('\n')

tokens_of_interest = [
    'князь', 'человек', 'собака', 'кошка',
    'любит', 'ненавидит', 'говорит', 'бежит', 'грызёт', 'ловит', 'убивает', 'приносит',
    'войну', 'мир', 'жизнь', 'смерть', 'кость', 'молоко', 'мышь', 'людей', 'радость',
]

groups = {
    'subjects': ['князь', 'человек', 'собака', 'кошка'],
    'verbs': ['любит', 'ненавидит', 'говорит', 'бежит'],
    'objects': ['войну', 'мир', 'жизнь', 'смерть', 'кость', 'молоко', 'мышь'],
}

tid = {}
for w in tokens_of_interest:
    ids = sp.encode(w)
    if ids:
        tid[w] = ids[0]

def s(w1, w2, cs_obj):
    i1, i2 = tid.get(w1), tid.get(w2)
    if i1 is not None and i2 is not None:
        v1, v2 = cs_obj.concept_vector(i1), cs_obj.concept_vector(i2)
        if v1 is not None and v2 is not None:
            return float(v1 @ v2)
    return None

configs = [
    ('adjacent baseline', 1, False, False),
    ('context baseline', 2, False, False),
    ('context + PMI', 2, False, True),
    ('context + PMI + repel', 2, True, True),
]

for label, cw, use_repel, use_pmi in configs:
    print(f'\n─── {label} (window={cw}) ───', flush=True)
    cs = ConceptSpace(vocab_size=sp.vocab_size(), dim=384)
    cs.init_concepts()
    cs.init_homeostasis()
    gen = CrystalGenerator(cs, sp, SyntaxLattice())
    gen.train_lr = 0.02

    for epoch in range(100):
        for sent in sentences:
            gen.train_from_text(sent.strip(), context_window=cw,
                                pmi_gate=use_pmi)
        if use_repel and epoch > 0 and epoch % 10 == 0:
            cs._repel_centroid(strength=0.08)
        if use_repel and epoch > 0 and epoch % 50 == 0:
            cs.fluctuate_fractal(noise_scale=0.001, decay=0.999)

    for gname, words in groups.items():
        sims = [s(words[i], words[j], cs) for i in range(len(words)) for j in range(i+1, len(words))]
        sims = [x for x in sims if x is not None]
        if sims:
            print(f'  {gname:10s}: within={np.mean(sims):+.4f}  (std={np.std(sims):.4f})')

    for w1, w2 in [('князь','любит'), ('собака','бежит'), ('любит','войну'),
                   ('князь','человек'), ('любит','ненавидит')]:
        print(f'  sim({w1:10s},{w2:10s}) = {s(w1,w2,cs):+.4f}')
