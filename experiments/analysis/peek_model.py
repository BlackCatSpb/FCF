"""Quick peek into qwen_layer_model.pt structure"""
import zipfile, pickle, io

f = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\models\qwen_layer_model.pt"
with zipfile.ZipFile(f) as z:
    print("Files in archive:")
    for info in z.infolist():
        size_mb = info.file_size / 1_000_000
        print(f"  {info.filename}  ({size_mb:.1f} MB)")

    # Read data.pkl
    data = z.read("qwen_layer_model/data.pkl")
    # Try to extract storage keys from the pickle
    d = pickle.loads(data)
    print("\nType:", type(d))
    if isinstance(d, dict):
        print("Keys:", list(d.keys()))
        for k, v in d.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, dict):
                print(f"  {k}: dict with {len(v)} keys, sample:", list(v.keys())[:5])
            else:
                print(f"  {k}: {type(v).__name__} = {str(v)[:60]}")
    elif isinstance(d, list):
        print(f"  List, len={len(d)}")
        for i, item in enumerate(d[:5]):
            if hasattr(item, 'shape'):
                print(f"    [{i}]: shape={item.shape}")
            else:
                print(f"    [{i}]: {type(item).__name__}")
