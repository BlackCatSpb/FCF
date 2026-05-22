import shutil, os
src = "checkpoints/symbolic/step_800000"
dst = "checkpoints/symbolic/final"
os.makedirs(dst, exist_ok=True)
for f in ['potential_field.pt', 'weights.pt', 'status.json']:
    s, d = f"{src}/{f}", f"{dst}/{f}"
    if os.path.exists(s):
        shutil.copy(s, d)
        print(f"Copied {f}")
print("Done")
