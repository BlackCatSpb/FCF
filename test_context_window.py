"""Test context-window STDP on synthetic Russian sentences.
Check if vectors encode syntactic roles (subject/predicate/modifier)
and form semantic potential fields."""
import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
sys.stdout.reconfigure(encoding='utf-8')
import math, random
import numpy as np
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

sp = spm.SentencePieceProcessor(model_file=r'real_data/bpe_ru_32k.model')

# ── Synthetic corpus: controlled syntactic patterns ──
# Templates: [SUBJECT] [VERB] [OBJECT] [MODIFIER?]
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
Собака говорит громко.
""".strip().split('\n')

print(f"Synthetic corpus: {len(sentences)} sentences")
all_tokens = set()
for s in sentences:
    all_tokens.update(sp.encode(s))
print(f"Unique tokens: {len(all_tokens)}")

# ── Train with TWO strategies and compare ──
results = {}

for mode, cw in [('adjacent (window=1)', 1), ('context (window=2)', 2)]:
    print(f"\n{'='*60}")
    print(f"Training mode: {mode}")
    print(f"{'='*60}")

    cs = ConceptSpace(vocab_size=sp.vocab_size(), dim=384)
    cs.init_concepts()
    cs.init_homeostasis()
    lattice = SyntaxLattice()
    gen = CrystalGenerator(cs, sp, lattice)
    gen.train_lr = 0.01

    for sent in sentences:
        gen.train_from_text(sent.strip(), context_window=cw)

    # ── Diagnostics ──
    vecs = np.array(list(cs.concept_vectors.values()), dtype=np.float32)
    rng = np.random.RandomState(42)
    pair_sims = [float(vecs[rng.randint(32000)] @ vecs[rng.randint(32000)])
                 for _ in range(2000)]
    print(f"  Global: cos={np.mean(pair_sims):.4f} ± {np.std(pair_sims):.4f}")

    # Check known BPE token IDs for our test words
    tokens_of_interest = [
        'князь', 'человек', 'собака', 'кошка',  # subjects
        'любит', 'ненавидит', 'говорит', 'бежит', 'грызёт', 'ловит', 'убивает', 'приносит',  # verbs
        'войну', 'мир', 'жизнь', 'смерть', 'кость', 'молоко', 'мышь', 'людей', 'радость',  # objects
        'Андрей', 'Марья', 'Марью', 'Андрея',  # names
        'тихо', 'громко', 'быстро', 'должен', 'прекрасна', 'неизбежна',  # modifiers
    ]

    token_ids = {}
    for word in tokens_of_interest:
        ids = sp.encode(word)
        if ids:
            token_ids[word] = ids[0]  # first BPE token

    # Cross-group similarities
    groups = {
        'subjects': ['князь', 'человек', 'собака', 'кошка'],
        'verbs': ['любит', 'ненавидит', 'говорит', 'бежит'],
        'objects': ['войну', 'мир', 'жизнь', 'кость'],
        'names': ['Андрей', 'Марья'],
        'modifiers': ['тихо', 'громко', 'быстро'],
    }

    def sim_between(w1, w2, cs_obj):
        i1, i2 = token_ids.get(w1), token_ids.get(w2)
        if i1 is not None and i2 is not None:
            v1, v2 = cs_obj.concept_vector(i1), cs_obj.concept_vector(i2)
            if v1 is not None and v2 is not None:
                return float(v1 @ v2)
        return None

    def group_mean(group_name, words, cs_obj):
        valid = [w for w in words if w in token_ids]
        if len(valid) < 2:
            return None
        sims = []
        for i in range(len(valid)):
            for j in range(i+1, len(valid)):
                s = sim_between(valid[i], valid[j], cs_obj)
                if s is not None:
                    sims.append(s)
        return np.mean(sims) if sims else None

    # Within-group similarity (should be higher if role-encoding works):
    print(f"\n  Within-group similarities (should be HIGH if role-encoding works):")
    for group_name, words in groups.items():
        m = group_mean(group_name, words, cs)
        if m is not None:
            print(f"    {group_name:12s}: mean={m:+.4f}")

    # Between-group similarity (subject→verb, verb→object — should grow)
    print(f"\n  Cross-group similarities (subject→verb, verb→object):")
    for w1, w2 in [('князь', 'любит'), ('человек', 'говорит'), ('собака', 'бежит'),
                   ('любит', 'войну'), ('ненавидит', 'мир'), ('ловит', 'мышь'),
                   ('князь', 'человек'), ('собака', 'кошка'),
                   ('любит', 'ненавидит'), ('войну', 'мир')]:
        s = sim_between(w1, w2, cs)
        if s is not None:
            print(f"    sim({w1:12s},{w2:12s}) = {s:+.4f}")

    # Top-5 nearest tokens for a key concept
    print(f"\n  Top-5 nearest to 'князь':")
    top = cs.topk_similar_concepts(token_ids['князь'], k=6)
    for cid, s in top:
        token = sp.IdToPiece(cid).replace('▁', '')
        print(f"    {token:15s} (CID {cid:5d}) sim={s:.4f}")

    results[mode] = (cs, gen, sim_between, group_mean)

print(f"\n{'='*60}")
print("SUMMARY: Role-encoding comparison")
print(f"{'='*60}")
# Compare adjacent vs context window
for mode, cw in [('adjacent', 1), ('context', 2)]:
    label = f"{mode} (window={cw})"
    cs, gen, sim_fn, grp_fn = results[f"{mode} (window={cw})"]
    knyaz_lubit = sim_fn('князь', 'любит', cs)
    subj_means = []
    for group_name, words in groups.items():
        m = grp_fn(group_name, words, cs)
        if m is not None: subj_means.append(m)
    print(f"  {label:25s}: within-group={np.mean(subj_means):+.4f} князь→любит={knyaz_lubit:+.4f}" if subj_means else f"  {label}")
