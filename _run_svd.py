import sys
sys.path.insert(0, '.')
from fcf.primordial_layer import PrimordialLayer
from fcf.utils import load_primordial_layer
from fcf.atomic_basis import AtomicBasis

print("Loading model...")
layer = load_primordial_layer('checkpoints/language/step_023000', PrimordialLayer)

for threshold in [1e-3, 1e-2, 5e-2]:
    print(f"\n=== Threshold = {threshold} ===")
    basis = AtomicBasis(error_threshold=threshold)
    al = basis.decompose(layer)
    print(basis.summary())
    ratios = basis.get_compression_ratio()
    for name, r in ratios.items():
        print(f"  {name}: compression {r:.0f}x (K={al.k_values[name]})")

