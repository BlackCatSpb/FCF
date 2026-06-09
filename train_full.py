"""Full training (32K BPE). Process line-by-line. Checkpoints overwrite."""

import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import time, json
import numpy as np
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

CORPUS = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru.txt'
BPE_MODEL = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\bpe_ru_32k.model'
CS_PATH = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space.json'
LATTICE_PATH = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\syntax_lattice.json'

sp = spm.SentencePieceProcessor(model_file=BPE_MODEL)
V = sp.vocab_size()
print(f"vocab_size = {V}")

print("\nInitializing ConceptSpace (32K fractal vectors @ 384D)...")
cs = ConceptSpace(vocab_size=V, dim=384)
cs.init_concepts()
cs.init_homeostasis()

print("\nBuilding SyntaxLattice from full corpus...")
lattice = SyntaxLattice()
t0 = time.time()
lattice.build(CORPUS, sp, max_n=4)
t1 = time.time()
print(f"  done in {t1-t0:.1f}s")
print(f"  n-gram prefixes: {[len(v) for v in lattice.ngrams.values()]}")
print(f"  unique concepts: {len(lattice.concept_freq)}")

# ── Diagnostics ────────────────────────────────────────────────

def mean_cosine_sim(cs, sample=2000):
    all_cids = list(cs.concept_vectors.keys())
    rng_state = np.random.RandomState(42)
    cids = rng_state.choice(all_cids, size=min(sample, len(all_cids)), replace=False).tolist()
    vecs = np.array([cs.concept_vectors[c] for c in cids], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    vecs /= norms
    n = len(vecs)
    n_pairs = min(5000, n * (n - 1) // 2)
    rng = np.random.RandomState(42)
    sims = np.empty(n_pairs, dtype=np.float32)
    for k in range(n_pairs):
        i = rng.randint(n); j = rng.randint(n)
        while j == i: j = rng.randint(n)
        sims[k] = float(vecs[i] @ vecs[j])
    return float(np.mean(sims)), float(np.std(sims))

def check_consistency(cs, sample=500):
    all_cids = list(cs.concept_vectors.keys())
    rng_state = np.random.RandomState(42)
    cids = rng_state.choice(all_cids, size=min(sample, len(all_cids)), replace=False).tolist()
    ok = 0
    for cid in cids:
        v_stored = cs.concept_vectors[cid]
        v_code = cs.fractal.compute_vector(cid)
        if v_code is not None and abs(float(np.dot(v_stored, v_code)) - 1.0) < 1e-6:
            ok += 1
    return ok, len(cids)

def pair_sim(cs, sp, a, b):
    id_a = sp.PieceToId(a); id_b = sp.PieceToId(b)
    if id_a < 0 or id_b < 0: return None
    va = cs.concept_vector(id_a); vb = cs.concept_vector(id_b)
    if va is None or vb is None: return None
    return float(va @ vb)

# ── Baseline ────────────────────────────────────────────────────

print("\n--- Baseline diagnostics ---")
mean_sim, std_sim = mean_cosine_sim(cs)
print(f"  mean cosine sim: {mean_sim:.4f} ± {std_sim:.4f}")
ok, total = check_consistency(cs)
print(f"  code-vector consistency: {ok}/{total}")
pairs_to_track = [('▁соба', 'ка'), ('▁ко', 'шка'), ('▁человек', 'а'),
                  ('▁человек', '▁война'), ('▁князь', '▁Андрей')]
for a, b in pairs_to_track:
    s = pair_sim(cs, sp, a, b)
    if s is not None: print(f"  sim({a:12s}, {b:12s}) = {s:.4f}")

# ── Adaptive parameter control ────────────────────────────────

TARGET_STD = 1.0 / np.sqrt(384)  # 0.051 — random uniform on sphere
prev_mean_cos = 0.0

repel_strength = 0.08
base_lr = 0.01
noise_scale = 0.001
decay_rate = 0.9998

def adapt_params(mean_cos, std_cos):
    global repel_strength, base_lr, noise_scale, prev_mean_cos
    changes = []

    # ── Repel strength: maintain mean_cos ≈ 0 ──
    if mean_cos > 0.01:
        repel_strength = min(repel_strength * 1.10, 0.20)
        changes.append(f"repel={repel_strength:.3f}")
    elif mean_cos < -0.005:
        repel_strength = max(repel_strength * 0.90, 0.01)
        changes.append(f"repel={repel_strength:.3f}")

    # ── Noise scale: maintain std_cos ≈ TARGET_STD ──
    if std_cos < TARGET_STD * 0.80:
        noise_scale = min(noise_scale * 1.15, 0.01)
        changes.append(f"noise={noise_scale:.4f}")
    elif std_cos > TARGET_STD * 1.30:
        noise_scale = max(noise_scale * 0.90, 0.0002)
        changes.append(f"noise={noise_scale:.4f}")

    # ── Base LR: trend-following ──
    cos_trend = mean_cos - prev_mean_cos
    if cos_trend > 0.001 and mean_cos > 0.005:
        base_lr = max(base_lr * 0.95, 0.003)
        changes.append(f"lr={base_lr:.4f}")
    elif cos_trend < -0.001 and mean_cos < -0.005:
        base_lr = min(base_lr * 1.05, 0.02)
        changes.append(f"lr={base_lr:.4f}")

    gen.train_lr = base_lr
    prev_mean_cos = mean_cos
    return changes


# ── STDP Training (line by line) ────────────────────────────────

print("\n--- STDP training (fractal self-organisation) ---")
gen = CrystalGenerator(cs, sp, lattice)
gen.train_lr = base_lr  # adaptive LR
t_start = time.time()

with open(CORPUS, 'r', encoding='utf-8') as f:
    lines = f.readlines()

total_lines = len(lines)
REPORT_EVERY = 2000  # lines per progress report
FLUCTUATE_EVERY = 2000  # lines per fluctuation + centroid repel

n_trained = 0

for idx, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue
    gen.train_from_text(line)
    n_trained += 1

    if idx > 0 and idx % FLUCTUATE_EVERY == 0:
        cs.fluctuate_fractal(noise_scale=noise_scale, decay=decay_rate,
                             repel_strength=repel_strength)

    if idx > 0 and idx % REPORT_EVERY == 0:
        pct = 100 * idx / total_lines
        elapsed = time.time() - t_start
        rate = idx / max(elapsed, 1)
        eta = (total_lines - idx) / max(rate, 1)
        mean_sim, std_sim = mean_cosine_sim(cs)
        ok, total_c = check_consistency(cs)
        # Track all diagnostic pairs live
        live_pairs = [('▁соба', 'ка'), ('▁человек', '▁война'),
                      ('▁князь', '▁Андрей'), ('▁любовь', '▁смерть')]
        pair_strs = []
        for a, b in live_pairs:
            s = pair_sim(cs, sp, a, b)
            if s is not None:
                pair_strs.append(f"{a.split('▁')[-1][:4]}/{b.split('▁')[-1][:4]}={s:.2f}")
        sc = ' ' + ' '.join(pair_strs) if pair_strs else ''
        ng = sum(len(v) for v in lattice.ngrams.values())

        # Auto-adapt parameters based on measured invariants
        param_changes = adapt_params(mean_sim, std_sim)
        param_str = f" | repel={repel_strength:.2f} lr={base_lr:.3f}" if not param_changes else f" | {' '.join(param_changes)}"

        # Periodic checkpoint
        if idx > 0 and idx % 10000 == 0:
            cs.save(CS_PATH)
            lattice.save(LATTICE_PATH)

        print(f"  [{pct:5.1f}%] {idx:7d} lines | {rate:5.0f} l/s | "
              f"cos={mean_sim:.4f}±{std_sim:.4f} | con={ok}/{total_c} | "
              f"ng={ng}{sc}{param_str}")

# Final fluctuation (adaptive)
cs.fluctuate_fractal(noise_scale=min(noise_scale * 2, 0.005),
                     decay=0.999, repel_strength=min(repel_strength * 1.25, 0.25))

# ── Final diagnostics ───────────────────────────────────────────

print("\n--- Final diagnostics ---")
mean_sim, std_sim = mean_cosine_sim(cs)
print(f"  mean cosine sim: {mean_sim:.4f} ± {std_sim:.4f}")
ok, total = check_consistency(cs)
print(f"  code-vector consistency: {ok}/{total}")
for a, b in pairs_to_track:
    s = pair_sim(cs, sp, a, b)
    if s is not None: print(f"  sim({a:12s}, {b:12s}) = {s:.4f}")

print("\n--- Generation tests ---")
for seed in ['князь', 'человек', 'война', 'любовь']:
    result = gen.generate(seed_word=seed, max_words=10)
    txt = result['text']
    print(f"  [{seed}] {txt[:60]}  (score={result['score']:.2f})")

t_total = time.time() - t_start
print(f"\nTotal: {n_trained} lines in {t_total:.0f}s ({n_trained/t_total:.0f} l/s)")
print("Saving...")
cs.save(CS_PATH)
lattice.save(LATTICE_PATH)
print("Done.")
