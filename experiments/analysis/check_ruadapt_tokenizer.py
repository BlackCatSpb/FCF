"""Check ruadapt Qwen3 tokenizer structure"""
import json, sys
from tokenizers import Tokenizer

# Load vocab.json
with open(r"C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\vocab.json", encoding="utf-8") as f:
    v = json.load(f)
print("Vocab size:", len(v))

# Show structural stats
items = list(v.items())
print("First 5 IDs:", [items[i][0] for i in range(5)])
print("Last 5 IDs:", [items[-1-i][0] for i in range(5)])

# Load tokenizer
t = Tokenizer.from_file(r"C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json")
print("Tokenizer vocab_size:", t.get_vocab_size())

# Test encode Russian
txt = "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"  # Privet mir
ids = t.encode(txt).ids
print("Encode 'Privet mir':", ids[:20])
print("Total tokens:", len(ids))
for tid in ids[:10]:
    tok = t.id_to_token(tid)
    print(f"  {tid}: {repr(tok)}")

# Test with different Russian texts
tests = [
    "\u0417\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439\u0442\u0435",  # Zdravstvuyte
    "\u042d\u0442\u043e \u0442\u0435\u0441\u0442\u043e\u0432\u0430\u044f \u0441\u0442\u0440\u043e\u043a\u0430",  # Test string
    "\u0412 \u043d\u0430\u0447\u0430\u043b\u0435 \u0431\u044b\u043b\u043e \u0441\u043b\u043e\u0432\u043e",  # V nachale bylo slovo
]
print()
for txt in tests:
    ids = t.encode(txt).ids
    print(f"  {repr(txt)[:30]}: {len(ids)} tokens -> {ids[:10]}")

# How does vocab 146K relate to Qwen3 standard (151,678)?
# Check if this is the full Qwen3 vocab or a Russian subset
print()
print("Russian tokens check:")
rus_start = sum(1 for k in v if ord(k[0]) > 0x400 and ord(k[0]) < 0x500) if v else 0
print(f"  Tokens starting with Cyrillic: {rus_start}")
# Check for common patterns
byte_fallback = sum(1 for k in v if len(k) == 1 and ord(k[0]) < 256)
print(f"  Single-byte tokens: {byte_fallback}")

# How does tokenization compare to our V=4101 BPE?
# Map common Russian subwords
print()
print("=== Mapping Qwen tokens to readable samples ===")
for i in [10, 100, 500, 1000, 2000, 50000, 100000, 140000]:
    if i < len(items):
        tok = items[i][0]
        print(f"  {i}: {repr(tok)}")
