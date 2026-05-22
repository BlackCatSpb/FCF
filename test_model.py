"""Load latest checkpoint and test what the model learned"""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from eva.symbolic import *
from eva.primordial_layer import PrimordialLayer
from eva.config import FCFConfig

# Load from last checkpoint
ckpt_dir = "checkpoints/symbolic/final"
pf_path = os.path.join(ckpt_dir, "potential_field.pt")
weights_path = os.path.join(ckpt_dir, "weights.pt")

config = FCFConfig()
config.d_model = 256; config.vocab_size = 156; config.num_heads = 8; config.max_seq_len = 256

layer = PrimordialLayer(config)
if os.path.exists(weights_path):
    layer.load_state_dict(torch.load(weights_path, map_location='cpu'))
    print(f"Loaded weights: {os.path.getsize(weights_path)//1024} KB")
if torch.cuda.is_available():
    layer = layer.cuda()

char_vocab = CharacterVocab()

# Load potential field
pf = PotentialField(vocab_size=156, embed_dim=256)
if os.path.exists(pf_path):
    pf.load_state_dict(torch.load(pf_path, map_location='cpu'))
    print(f"Loaded potential field: {os.path.getsize(pf_path)//1024} KB")

avg_pot = float(pf.affinity.mean())
max_pot = float(pf.affinity.max())
print(f"Avg affinity: {avg_pot:.4f}")
print(f"Max affinity: {max_pot:.4f}")

# === TESTS ===
print("\n" + "=" * 60)
print("ORDER OF CHARACTERS IN KNOWN WORDS")
print("=" * 60)
test_words = ["привет", "человек", "математика", "компьютер", "история", "вселенная", "природа", "животное"]
for word in test_words:
    chars = char_vocab.encode(word)[1:-1]
    ranks = []
    for i in range(len(chars)-1):
        cont = pf.get_continuation_potential(chars[i]).cpu().numpy()
        rank = np.sum(cont > cont[chars[i+1]])
        ranks.append(rank)
    avg_rank = np.mean(ranks)
    print(f"  '{word}': avg rank {avg_rank:.0f}/{len(cont)} (top {avg_rank/len(cont)*100:.1f}%)")

print("\n" + "=" * 60)
print("WORD COMPLETION")
print("=" * 60)
prefixes = ["мам", "пап", "доч", "сын", "книг", "стол", "челов", "космо", "земл", "вод", "рук", "ног", "голов", "сердц", "машин", "компьют", "телефо", "учител", "студен", "програм"]
for prefix in prefixes:
    syms = char_vocab.encode(prefix)[1:-1]
    cont = pf.get_continuation_potential(syms[-1]).cpu().numpy()
    top5_idx = np.argsort(cont)[-5:][::-1]
    top5 = [char_vocab.idx_to_char(int(i)) for i in top5_idx if int(i) < 156]
    # Show top5 as clean chars
    clean = [c for c in top5 if c.isalpha() or c in ' '][:5]
    if not clean: clean = top5[:5]
    print(f"  '{prefix}...' -> {', '.join(clean)}")

print("\n" + "=" * 60)
print("TOP DIGRAMS BY AFFINITY")
print("=" * 60)
aff = pf.affinity.cpu().numpy()
all_pairs = []
for i in range(156):
    for j in range(156):
        if i != j:
            all_pairs.append((aff[i,j], i, j))
all_pairs.sort(reverse=True)
for score, i, j in all_pairs[:20]:
    ci = char_vocab.idx_to_char(i)
    cj = char_vocab.idx_to_char(j)
    print(f"  '{ci}'+'{cj}' = {score:.4f}")

print("\n" + "=" * 60)
print("GENERATION (SYMBOLIC GENERATOR)")
print("=" * 60)
from eva.symbolic.symbolic_generator import SymbolicGenerator
gen = SymbolicGenerator(layer=layer, char_vocab=char_vocab, potential_field=pf,
                         contradiction_filter=SymbolicContradictionFilter(pf, TopologicalField(pf)),
                         grammar=AssemblyGrammar(pf, 156, 256),
                         concept_miner=SymbolicConceptMiner(pf, TopologicalField(pf), SymbolicContradictionFilter(pf, TopologicalField(pf)), AssemblyGrammar(pf, 156, 256), LogicBridge(pf, 156), GeodesicNavigator(pf, TopologicalField(pf), TangentSpace(pf, TopologicalField(pf)))),
                         topological_field=TopologicalField(pf))

for prompt in ["при", "чело", "космо", "зем", "мам", "сто", "комп", "про"]:
    syms = char_vocab.encode(prompt)[1:-1]
    gen_out = gen.generate(syms, max_new_symbols=40, temperature=0.5)
    text = char_vocab.decode(gen_out)
    print(f"  '{prompt}...' -> '{text}'")
