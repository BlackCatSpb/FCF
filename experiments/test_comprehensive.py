"""Black-box integration tests for FCF generation.
Tests are pure request→response: no concept IDs, no internal state inspection."""

import sys; sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCF")
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.concept_tokenizer import ConceptTokenizer
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

CS_PATH = "C:/Users/black/OneDrive/Desktop/FCF/real_data/concept_space.json"
LATTICE_PATH = "C:/Users/black/OneDrive/Desktop/FCF/real_data/syntax_lattice.json"

print("=" * 60)
print("BLACK-BOX TESTS: FCF Generation")
print("=" * 60)

print("\nLoading...")
tok = ConceptTokenizer()
tok.initialize()
cs = ConceptSpace.load(CS_PATH)
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
print("TEST 1: Basic Generation")
print("=" * 60)

# 1a. Single seed word
r = gen.generate(seed_word="развитие", max_words=10)
check("seed word produces text", len(r['text'].strip()) > 0, f"empty: {r['text']!r}")
check("seed word meets min count", r.get('word_count', 0) >= 3, f"got {r.get('word_count', 0)}")

# 1b. Seed + query
r = gen.generate(seed_word="развитие", query_words=["экономика", "прогресс"], max_words=10)
check("seed+query produces text", len(r['text'].strip()) > 0, "empty")
check("seed+query has word_count", r.get('word_count', 0) >= 3)

# 1c. Different seed, still works
r2 = gen.generate(seed_word="энергия", max_words=10)
check("different seed produces text", len(r2['text'].strip()) > 0, "empty")
check("seeds differ", r['text'] != r2['text'], "identical across seeds")

print(f"  Basic: {passed}/{passed + failed}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Diverse Seeds")
print("=" * 60)

seeds = [
    "человек", "время", "дело", "жизнь", "работа", "место", "вопрос",
    "система", "процесс", "развитие", "результат", "исследование",
    "наука", "технология", "экономика", "общество", "природа",
    "искусство", "космос", "энергия", "информация", "знание",
    "метод", "анализ", "модель", "структура", "функция", "элемент",
    "основание", "источник",
]

texts = []
for seed in seeds:
    r = gen.generate(seed_word=seed, max_words=10)
    has_text = len(r['text'].strip()) > 0
    check(f"generate('{seed}') has text", has_text, "empty")
    texts.append(r['text'])

unique = len(set(texts))
check(f"different seeds: {unique}/{len(seeds)} unique", unique >= len(seeds) * 0.7,
      f"only {unique} unique out of {len(seeds)}")

print(f"  Diverse: {passed}/{passed + failed}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Long Generation")
print("=" * 60)

r = gen.generate(seed_word="развитие", max_words=35)
wc = r.get('word_count', 0)
check("long gen: word_count >= 30", wc >= 30, f"got {wc}")
check("long gen: has text", len(r['text'].strip()) > 0, "empty")
print(f"  Text ({wc} words): {r['text'][:120]}...")

hs = gen.hormones
print(f"  Hormones: DA={hs.dopamine:.2f} 5HT={hs.serotonin:.2f} "
      f"NA={hs.noradrenaline:.2f} ACh={hs.acetylcholine:.2f}")
check("DA after long gen > 0", hs.dopamine > 0, f"DA={hs.dopamine:.4f}")
check("NA after long gen > 0", hs.noradrenaline > 0, f"NA={hs.noradrenaline:.4f}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: Query Influence")
print("=" * 60)

# Generate with and without a query — outputs should differ
r_no_query = gen.generate(seed_word="космос", max_words=10)
r_with_query = gen.generate(seed_word="космос", query_words=["космос", "наука"], max_words=10)
check("no query has text", len(r_no_query['text'].strip()) > 0, "empty")
check("with query has text", len(r_with_query['text'].strip()) > 0, "empty")
check("query vs no-query differ", r_no_query['text'] != r_with_query['text'],
      "identical output")

# Repeat with different seed
r_no_query2 = gen.generate(seed_word="искусство", max_words=10)
r_with_query2 = gen.generate(seed_word="искусство", query_words=["искусство", "культура"], max_words=10)
check("no query 2 has text", len(r_no_query2['text'].strip()) > 0, "empty")
check("with query 2 has text", len(r_with_query2['text'].strip()) > 0, "empty")
check("query2 vs no-query2 differ", r_no_query2['text'] != r_with_query2['text'],
      "identical output")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 5: Noise Robustness")
print("=" * 60)

import string, random
random.seed(42)

# Pure noise seed
noise = ''.join(random.choice(string.ascii_lowercase + 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
                for _ in range(8))
r_noise = gen.generate(seed_word=noise, max_words=10)
check("noise seed produces text", len(r_noise['text'].strip()) > 0,
      f"empty from noise {noise!r}")
check("noise seed has word_count", r_noise.get('word_count', 0) >= 3,
      f"wc={r_noise.get('word_count', 0)}")

# Mix noise with real word — query should help
r_mixed = gen.generate(seed_word=noise, query_words=["наука", "человек"], max_words=10)
check("noise+query produces text", len(r_mixed['text'].strip()) > 0, "empty")

print(f"  Noise robustness: {passed}/{passed + failed}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 6: Generation Consistency")
print("=" * 60)

# Same settings should produce same result within same session
r1 = gen.generate(seed_word="энергия", max_words=8)
r2 = gen.generate(seed_word="энергия", max_words=8)
check("repeat with same seed differs (stateful)", r1['text'] != r2['text'],
      f"identical: {r1['text']!r}")

# ═══════════════════════════════════════════════════════════════
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
