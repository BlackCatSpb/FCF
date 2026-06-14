"""Synthetic test: prove vectors move under STDP + PPMI destabilisation.

No data saved, purely in-memory. Creates mini ConceptSpace + lattice from
a subset of the corpus, runs accelerated STDP, measures cos before/after.
"""

import sys, os, time, math, tempfile
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
import sentencepiece as spm

V = 5000
DIM = 384
TRAIN_LINES = 2000
NEG_SAMPLES = 3
CONTEXT_WINDOW = 2
LR = 0.15

print("=" * 60)
print("TEST: Vector movement under STDP + PPMI destabilisation")
print("=" * 60)

sp = spm.SentencePieceProcessor(
    model_file=r'C:\Users\black\OneDrive\Desktop\FCF\real_data\bpe_ru_146k.model'
)
print(f"\n1. BPE vocab: {sp.vocab_size()} tokens")

from eva.symbolic.concept_space import ConceptSpace
cs = ConceptSpace(vocab_size=V, dim=DIM)
cs.init_concepts()
cs.init_homeostasis()
print(f"2. ConceptSpace: {len(cs.concept_vectors)} vectors")

from eva.symbolic.syntax_lattice import SyntaxLattice
corpus_path = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru.txt'
print(f"3. Building mini SyntaxLattice ({TRAIN_LINES} lines)...")
tmp = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt')
with open(corpus_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= TRAIN_LINES:
            break
        tmp.write(line)
tmp.close()
lattice = SyntaxLattice()
lattice.build(tmp.name, sp, max_n=3)
os.unlink(tmp.name)

from eva.symbolic.crystal_generator import CrystalGenerator
gen = CrystalGenerator(cs, sp, lattice)
gen.train_lr = LR

def measure_cos(cs, n_sample=2000):
    all_cids = list(cs.concept_vectors.keys())
    rng = np.random.RandomState(42)
    cids = rng.choice(all_cids, size=min(n_sample, len(all_cids)), replace=False).tolist()
    vecs = np.array([cs.concept_vectors[c] for c in cids], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    vecs /= norms
    n = len(vecs)
    n_pairs = min(5000, n * (n - 1) // 2)
    sims = np.empty(n_pairs, dtype=np.float32)
    for k in range(n_pairs):
        i = rng.randint(n); j = rng.randint(n)
        while j == i: j = rng.randint(n)
        sims[k] = float(vecs[i] @ vecs[j])
    return float(np.mean(sims)), float(np.std(sims))

def check_con(cs, n_sample=500):
    all_cids = list(cs.concept_vectors.keys())
    rng = np.random.RandomState(42)
    cids = rng.choice(all_cids, size=min(n_sample, len(all_cids)), replace=False).tolist()
    ok = 0
    for cid in cids:
        v_stored = cs.concept_vectors[cid]
        v_code = cs.fractal.compute_vector(cid)
        if v_code is not None and abs(float(np.dot(v_stored, v_code)) - 1.0) < 1e-6:
            ok += 1
    return ok, len(cids)

# Snapshot accumulators before
freq_before = sum(lattice.concept_freq.values())
total_shift_before = cs._total_shift
update_count_before = cs._update_count

cos_before, std_before = measure_cos(cs)
con_before, _ = check_con(cs)
print(f"\n   BEFORE: cos={cos_before:.6f}+-{std_before:.6f} con={con_before}/{min(500, V)}")
print(f"   State: total_shift={cs._total_shift:.4f} updates={cs._update_count} freq_sum={freq_before:.0f}")

print(f"\n4. Running STDP ({TRAIN_LINES} lines, lr={LR}, neg={NEG_SAMPLES})...")
with open(corpus_path, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()][:TRAIN_LINES]

t0 = time.time()
for i, line in enumerate(lines):
    destab_pct = min(i / 2000, 1.0)
    destab_scale = 0.6 + (0.02 - 0.6) * destab_pct
    gen.train_from_text(
        line, pmi_gate=False, neg_samples=NEG_SAMPLES,
        context_window=CONTEXT_WINDOW,
        inh_strength=0.05, inh_threshold=0.10,
        neg_lr_ratio=0.5, field_gate=True, use_torch=False,
        destab_scale=destab_scale
    )
    if (i + 1) % 500 == 0:
        cos_mid, _ = measure_cos(cs)
        print(f"   {i+1:5d} lines - cos={cos_mid:.6f}")

dt = time.time() - t0
print(f"   done in {dt:.1f}s ({TRAIN_LINES/dt:.0f} L/s)")

# Snapshot accumulators after
freq_after = sum(lattice.concept_freq.values())
freq_growth = freq_after - freq_before
shift_growth = cs._total_shift - total_shift_before
update_growth = cs._update_count - update_count_before

cos_after, std_after = measure_cos(cs)
con_after, _ = check_con(cs)
delta_cos = abs(cos_after) - abs(cos_before)

print(f"\n5. RESULTS")
print(f"   BEFORE: cos={cos_before:.6f}+-{std_before:.6f} con={con_before}/{min(500, V)}")
print(f"   AFTER:  cos={cos_after:.6f}+-{std_after:.6f} con={con_after}/{min(500, V)}")
print(f"   Delta|cos| = {delta_cos:.6f}")
print(f"   State: total_shift={cs._total_shift:.4f} (+{shift_growth:.4f}) updates={cs._update_count} (+{update_growth}) freq_sum={freq_after:.0f} (+{freq_growth:.0f})")

print(f"\n6. Top-5 most shifted vectors:")
shifts = []
for cid in list(cs.concept_vectors.keys())[:V]:
    v = cs.concept_vectors[cid]
    code = cs.fractal.compute_vector(cid)
    if code is not None:
        shift = float(np.linalg.norm(v - code))
        shifts.append((cid, shift))
shifts.sort(key=lambda x: -x[1])
for cid, s in shifts[:5]:
    tok = sp.IdToPiece(cid) if cid < sp.vocab_size() else f'CID{cid}'
    tok = tok.replace('\u2581', '_')
    print(f"   CID {cid:5d} ({tok:15s}) shift={s:.6f}")

print(f"\n7. VERDICT")
if abs(cos_after) > abs(cos_before) * 1.5 or abs(cos_after) < abs(cos_before) * 0.5:
    print(f"   [OK] VECTORS MOVED: |cos| {abs(cos_before):.4f} -> {abs(cos_after):.4f}")
    print(f"   Destabilisation + STDP working correctly.")
elif abs(delta_cos) > 1e-6:
    print(f"   [WARN] Small movement: |cos|={delta_cos:.6f} - may need more lines or stronger destab")
else:
    print(f"   [FAIL] NO MOVEMENT detected - something is wrong")

print(f"\n{'='*60}")
