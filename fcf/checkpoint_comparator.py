"""
Checkpoint Comparator — сравнивает качество генерации на разных шагах обучения.
Показывает прогрессию от шага 1000 до последнего.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

def compare_checkpoints(prompts=None, checkpoint_pattern="checkpoints/language/step_*"):
    from fcf.primordial_layer import PrimordialLayer
    from fcf.tokenizer_utils import load_tokenizer
    from fcf.utils import load_primordial_layer

    if prompts is None:
        prompts = [
            "История это наука которая изучает",
            "Математика помогает человечеству решать",
            "Природа Земли удивительна потому что",
        ]

    tokenizer = load_tokenizer("tokenizer.json")

    dirs = sorted(glob.glob(checkpoint_pattern))
    if not dirs:
        dirs = sorted(glob.glob("checkpoints/language/step_0*"))

    if len(dirs) > 5:
        step = len(dirs) // 5
        dirs = dirs[::step] + [dirs[-1]]

    print("=" * 60)
    print(f"FCF Checkpoint Comparator — {len(dirs)} точек")
    print("=" * 60)

    results = []
    for d in dirs:
        step_name = os.path.basename(d)
        try:
            layer = load_primordial_layer(d, PrimordialLayer)
        except Exception:
            continue

        snapshots = len(layer.state_storage)
        avg_conf = layer.meta.average_confidence()

        step_results = []
        for prompt in prompts:
            try:
                r = layer.process_query(prompt, tokenizer, max_new_tokens=40, temperature=0.7)
                step_results.append({
                    "Q": prompt,
                    "A": r["response"][:150],
                    "conf": round(r["confidence"], 3),
                })
            except Exception as e:
                step_results.append({"Q": prompt, "A": f"ERROR: {e}", "conf": 0})

        results.append({
            "step": step_name,
            "snapshots": snapshots,
            "avg_confidence": round(avg_conf, 3),
            "generations": step_results,
        })

    for i, r in enumerate(results):
        print(f"\n--- {r['step']} (snap={r['snapshots']}, conf={r['avg_confidence']}) ---")
        for g in r["generations"][:2]:
            print(f"  Q: {g['Q']}")
            print(f"  A: {g['A'][:100]}")
            print(f"     conf={g['conf']}")

    path = "logs/checkpoint_comparison.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {path}")

    return results


if __name__ == "__main__":
    compare_checkpoints()
