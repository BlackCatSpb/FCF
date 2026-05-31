"""
EVA — Pipeline Orchestrator.

Запускает все фазы по порядку, пропуская уже выполненные.
"""

import subprocess, sys, os, time
sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "symbolic")

PHASES = [
    ("train_full_pipeline.py", "symbol_weights.pt", "Symbols"),
    ("train_word_pipeline.py", "sentence_weights.pt", "Affinity→MDS→Words→Sentences"),
    ("train_dynamics.py", "evolved_affinity.pt", "STDP plasticity"),
    ("concept_finder.py", "potential_function.pt", "V(z) potential"),
    ("train_contradiction.py", "contradiction_filter.pt", "ContradictionFilter"),
    ("train_gradient.py", "gradient_flow.pt", "GradientFlow"),
    ("train_dialectic.py", "dialectical_synthesis.pt", "DialecticalSynthesis"),
    ("train_fractal_sc.py", "fractal_consistency.pt", "FractalSC"),
    ("train_persistence.py", "topological_persistence.pt", "Persistence"),
    ("train_conceptnet_full.py", "conceptnet_weights.pt", "ConceptNet"),
]

print("=" * 60)
print("EVA — Pipeline Orchestrator")
print("=" * 60)

for i, (script, ckpt, desc) in enumerate(PHASES):
    n = i + 1
    sp = os.path.join(SCRIPT_DIR, script)
    cp = os.path.join(CKPT_DIR, ckpt)
    
    if not os.path.exists(sp):
        print(f"[{n}/{len(PHASES)}] SKIP: {script} missing")
        continue
    
    if os.path.exists(cp):
        print(f"[{n}/{len(PHASES)}] SKIP: {desc} ✓")
        continue
    
    print(f"[{n}/{len(PHASES)}] RUN: {desc}...", flush=True)
    r = subprocess.run([sys.executable, sp], cwd=SCRIPT_DIR)
    status = "OK" if r.returncode == 0 else f"ERR({r.returncode})"
    print(f"  {status}")

print("Done.")
