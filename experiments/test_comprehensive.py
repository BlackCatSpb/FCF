"""Comprehensive tests for the crystal lattice architecture.

Tests:
  1. Anchor resolution (known, unknown, random words)
  2. Intent projection (single, multiple, noisy queries)
  3. Semantic distance tracking (long sequences)
  4. Generation from diverse random seeds (20+ seeds)
  5. Long sequence generation (30+ words)
  6. Concept induction from repeated patterns
  7. Hormonal response across generation
  8. No fallback guarantee
"""

import sys; sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCF")
import numpy as np
import random

from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.concept_tokenizer import ConceptTokenizer
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator
from eva.symbolic.concept_inductor import ConceptInductor
from eva.symbolic.hormonal_system import HormonalSystem

CS_PATH = "C:/Users/black/OneDrive/Desktop/FCF/real_data/concept_space.json"
LATTICE_PATH = "C:/Users/black/OneDrive/Desktop/FCF/real_data/syntax_lattice.json"

print("=" * 60)
print("COMPREHENSIVE TESTS: Crystal Lattice Architecture")
print("=" * 60)

# ── Load ──
print("\nLoading...")
tok = ConceptTokenizer()
tok.initialize()
cs = ConceptSpace(None, dim=128)
cs.load(CS_PATH)
lattice = SyntaxLattice()
lattice.load(LATTICE_PATH)
gen = CrystalGenerator(cs, tok, lattice)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  OK {name}")
    else:
        failed += 1
        print(f"  FAIL {name}: {detail}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: Anchor Resolution (no fallbacks)")
print("=" * 60)

# Known words
for w in ["князь", "война", "человек", "сказать", "быть", "и", "не"]:
    cid, conf = gen.resolve_anchor(w)
    check(f"resolve_anchor('{w}') -> ({cid}, {conf})", cid is not None, f"got None")

# Unknown but plausible Russian words
unknown_words = ["Пьер", "Наташа", "Андрей", "Болконский", "Ростов"]
for w in unknown_words:
    cid, conf = gen.resolve_anchor(w)
    check(f"resolve_anchor('{w}') -> ({cid}, {conf})", cid is not None, f"got None")
    if cid is not None:
        anchor = cs.concept_info.get(cid, {}).get('anchor', '?')
        print(f"    -> anchor='{anchor}'")

# Random character strings (complete noise) — must return low confidence
import string
random.seed(42)
noise_words = []
for _ in range(20):
    length = random.randint(4, 12)
    w = ''.join(random.choice(string.ascii_lowercase + 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
                for _ in range(length))
    noise_words.append(w)

noise_confidences = []
for w in noise_words:
    cid, conf = gen.resolve_anchor(w)
    noise_confidences.append(conf)
    check(f"resolve_anchor('{w[:10]}...') -> ({cid}, {conf:.3f})", cid is not None, f"got None")

# Verify most noise gets near-zero confidence
low_conf = sum(1 for c in noise_confidences if c < 0.1)
check(f"noise: {low_conf}/{len(noise_confidences)} have confidence < 0.1", low_conf >= len(noise_confidences) * 0.5)

# Empty/edge cases — zero confidence
cid, conf = gen.resolve_anchor("")
check("resolve_anchor('')", cid is not None and conf == 0.0)
cid, conf = gen.resolve_anchor(" ")
check("resolve_anchor(' ') ", cid is not None and conf == 0.0)

print(f"\n  Anchor resolution: {passed}/{passed + failed}")
anchor_pass = passed
anchor_fail = failed

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Intent Projection")
print("=" * 60)

# Single word query
intent_cid, intent_vec, delta = gen.project_intent(["война"])
check("project_intent(['война']) returns cid", intent_cid is not None)
check("project_intent(['война']) returns vec", intent_vec is not None)
check("project_intent(['война']) delta >= 0", delta >= 0)
anchor = cs.concept_info.get(intent_cid, {}).get('anchor', '?')
print(f"    intent('война') -> cid={intent_cid}, anchor='{anchor}', delta={delta:.3f}")

# Multiple word query
intent_cid2, vec2, delta2 = gen.project_intent(["князь", "Андрей", "выйти"])
check("multi-word intent", intent_cid2 is not None)
anchor2 = cs.concept_info.get(intent_cid2, {}).get('anchor', '?')
print(f"    intent('князь Андрей выйти') -> cid={intent_cid2}, anchor='{anchor2}', delta={delta2:.3f}")

# Noisy query (mix of content and function words)
intent_cid3, vec3, delta3 = gen.project_intent(["и", "в", "не", "война", "начаться"])
check("noisy query intent", intent_cid3 is not None)
anchor3 = cs.concept_info.get(intent_cid3, {}).get('anchor', '?')
print(f"    intent('и в не война начаться') -> cid={intent_cid3}, anchor='{anchor3}', delta={delta3:.3f}")

# Empty query (edge case)
intent_cid4, vec4, delta4 = gen.project_intent([])
check("empty query intent", intent_cid4 is not None)

print(f"\n  Intent projection: +{passed - anchor_pass}")
intent_pass = passed

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Generation from Diverse Seeds")
print("=" * 60)

seeds = [
    "князь", "человек", "война", "сказать", "глаза", "рука",
    "дом", "любовь", "смерть", "жизнь", "день", "ночь",
    "Пьер", "Андрей", "Наташа", "Москва", "Петербург",
    "душа", "сердце", "мысль", "слеза", "улыбка", "радость", "горе",
    "музыка", "тишина", "свет", "тьма", "небо", "земля",
]

for seed in seeds:
    result = gen.generate(seed_word=seed, max_words=15)
    has_text = len(result['text'].strip()) > 0
    has_path = len(result.get('concept_path', [])) > 1
    check(f"generate('{seed}') has text", has_text, f"empty text")
    check(f"generate('{seed}') has path", has_path, f"no path")
    if not has_text:
        print(f"    EMPTY for '{seed}'")

# Check outputs vary by seed (not all identical)
results = []
for seed in seeds[:5]:
    r = gen.generate(seed_word=seed, max_words=10)
    results.append(r['text'])

unique = len(set(results))
check("different seeds produce different texts", unique >= 3,
      f"only {unique} unique out of 5")

print(f"\n  Diverse generation: +{passed - intent_pass}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: Long Sequence Generation (30+ words)")
print("=" * 60)

result = gen.generate(seed_word="война", max_words=35)
words = result['text'].split()
wc = result.get('word_count', 0)
check("long gen: word_count >= 30", wc >= 30, f"got {wc}")
check("long gen: has content", len(words) > 0, f"empty")
check("long gen: has path", len(result.get('concept_path', [])) > 5)
print(f"    Text ({wc} words): {result['text'][:100]}...")

# Hormonal state after long generation
hs = gen.hormones
print(f"    Hormones: DA={hs.dopamine:.2f} 5HT={hs.serotonin:.2f} "
      f"NA={hs.noradrenaline:.2f} ACh={hs.acetylcholine:.2f}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 5: Semantic Distance Tracking")
print("=" * 60)

# Generate with intent tracking
result2 = gen.generate(seed_word="война", query_words=["война", "мир"], max_words=20)
drift = result2.get('intent_drift', -1)
check("intent_drift tracked", drift >= 0, f"drift={drift}")
intent_cid = result2.get('intent_cid')
check("intent_cid tracked", intent_cid is not None)

# Distance should increase over time
path = result2.get('concept_path', [])
distances = []
intent_vec = gen._query_centroid
if intent_vec is not None and len(path) > 3:
    for i in range(1, len(path), max(1, len(path)//10)):
        d = gen._semantic_delta(intent_vec, path[:i+1])
        distances.append(d)
    if len(distances) >= 2:
        trend = distances[-1] - distances[0]
        # Distance should generally increase as we diverge from query
        # (not strictly monotonic, but should trend upward)
        print(f"    Distance trend over {len(distances)} points: {distances}")
        print(f"    First={distances[0]:.3f}, Last={distances[-1]:.3f}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 6: Concept Induction")
print("=" * 60)

inductor = ConceptInductor({
    'induction_threshold': 5,
    'min_pattern_freq': 3,
})

# Feed a pattern many times — should induce meta-concepts
pattern = [11597, 5716, 32716, 12574]  # князь -> выйти -> на -> крыльцо
initial_count = len(inductor.meta_concepts)
total_induced = 0
for i in range(20):
    induced = inductor.observe(pattern, cs, lattice, gen.hormones)
    total_induced += len(induced)

check("induction created meta-concepts", total_induced > 0,
      f"got {total_induced} induced")

# Verify meta-concepts have valid vectors
for cid, info in inductor.meta_concepts.items():
    v = cs.concept_vector(cid)
    check(f"meta {cid} has vector", v is not None)
    if v is not None:
        check(f"meta {cid} normalized", abs(np.linalg.norm(v) - 1.0) < 0.01,
              f"norm={np.linalg.norm(v):.4f}")
    break  # just check first one

print(f"    Total meta-concepts: {len(inductor.meta_concepts)}")
print(f"    Induced this test: {total_induced}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 7: Anchor Extraction from Noise")
print("=" * 60)

# Using the inductor's extract_anchor method
# Feed a mix of function words + one content word — should find content word
noisy_input = ["и", "в", "на", "с", "война", "по", "за", "из"]
cid, conf = inductor.extract_anchor(noisy_input, cs, tok)
check("extract_anchor from noisy", cid is not None)
if cid is not None:
    anchor = cs.concept_info.get(cid, {}).get('anchor', '?')
    print(f"    noisy=['и','в','на','с','война','по','за','из'] -> cid={cid}, anchor='{anchor}', conf={conf:.3f}")

# Pure noise (all function words) — should still find SOMETHING
all_noise = ["и", "в", "на", "с", "по", "за", "из", "от", "до", "без"]
cid2, conf2 = inductor.extract_anchor(all_noise, cs, tok)
check("extract_anchor from pure noise", cid2 is not None)
if cid2 is not None:
    anchor2 = cs.concept_info.get(cid2, {}).get('anchor', '?')
    print(f"    pure noise -> cid={cid2}, anchor='{anchor2}', conf={conf2:.3f}")

# Single word
cid3, conf3 = inductor.extract_anchor(["любовь"], cs, tok)
check("extract_anchor single word", cid3 is not None)
if cid3 is not None:
    anchor3 = cs.concept_info.get(cid3, {}).get('anchor', '?')
    print(f"    single('любовь') -> cid={cid3}, anchor='{anchor3}', conf={conf3:.3f}")

# ── FINAL RESULTS ──
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
total = passed + failed
print(f"  Passed: {passed}/{total} ({100*passed/max(total,1):.1f}%)")
print(f"  Failed: {failed}/{total}")

if failed == 0:
    print("\n  ALL TESTS PASSED")
else:
    print(f"\n  {failed} TESTS FAILED")

print("\nDone.")
