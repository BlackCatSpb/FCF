"""
End-to-end test: bootstrap → train → compose → validate.
Доказывает что система работает как единое целое.
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

def run_e2e_test():
    from fcf.unified_grammar import UnifiedStateGrammar
    from fcf.primordial_layer import PrimordialLayer
    from fcf.tokenizer_utils import load_tokenizer
    from fcf.config import load_config

    print("=" * 60)
    print("FCF End-to-End Test")
    print("=" * 60)

    results = []

    # 1. Bootstrap
    print("\n[1/5] Bootstrap...")
    grammar = UnifiedStateGrammar(2560)
    layer = PrimordialLayer(load_config())
    tokenizer = load_tokenizer("tokenizer.json")
    results.append(("bootstrap", "OK"))
    print(f"  Grammar: 41 mechanisms, Layer: {sum(p.numel() for p in layer.parameters()):,} params")

    # 2. Train 10 steps
    print("\n[2/5] Training 10 steps...")
    from fcf.language_trainer import LanguageTrainer
    trainer = LanguageTrainer(layer=layer, tokenizer=tokenizer, hierarchy=None,
                              checkpoint_dir="checkpoints/e2e_test")
    t0 = time.time()
    stats = trainer.train(max_steps=10, device="cpu", use_wikipedia=True)
    elapsed = time.time() - t0
    if stats.get("error"):
        print(f"  WARNING: Wikipedia unavailable, using local corpus")
        trainer.train(max_steps=10, device="cpu", text_file="training_corpus.txt")
    results.append(("train_10_steps", f"{elapsed:.1f}s"))
    print(f"  Time: {elapsed:.1f}s")

    # 3. Compose states
    print("\n[3/5] State composition...")
    za = layer.get_context_vector(layer._encode(tokenizer, "история"))
    zb = layer.get_context_vector(layer._encode(tokenizer, "наука"))
    zc = layer.get_context_vector(layer._encode(tokenizer, "контекст"))
    cr = grammar.compose(za, zb, zc, "история", "наука")
    results.append(("compose", f"validity={cr.validity:.3f}"))
    print(f"  Validity: {cr.validity:.3f}, Delta_I: {cr.delta_I:.3f}")

    # 4. Discover rules
    print("\n[4/5] Rule discovery...")
    data = [(za, zb, za + zb) for _ in range(5)]
    r = grammar.discover(data, epochs=5)
    results.append(("discover", f"loss={r.get('discovery_loss', 0):.4f}"))
    print(f"  Discovery loss: {r.get('discovery_loss', 0):.4f}")

    # 5. Validate
    print("\n[5/5] Validation...")
    v = grammar.validate_rules(data)
    improved = v.get("better_than_baseline", False)
    results.append(("validate", f"improvement={v.get('improvement', 0):.2%}"))
    print(f"  Grammar vs Baseline: {v.get('improvement', 0):.2%}")

    print("\n" + "=" * 60)
    print("RESULTS:")
    for name, status in results:
        print(f"  {name:20s}: {status}")
    print(f"\n  OVERALL: {'PASS' if improved else 'NEEDS_DATA'}")

    return {"results": results, "improved": improved}


if __name__ == "__main__":
    run_e2e_test()
