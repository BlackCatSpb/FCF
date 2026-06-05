"""Check byte-to-token mapping in current BPE tokenizer"""
from tokenizers import Tokenizer

t = Tokenizer.from_file(r"C:\Users\black\OneDrive\Desktop\FCF\real_data\bpe_tokenizer.json")

# Encode various test strings
tests = [
    ("A", "Latin A"),
    ("a", "Latin a"),
    (" ", "space"),
    ("\n", "newline"),
    ("\t", "tab"),
    ("\u0000", "null byte"),
    ("\u00ff", "byte 255"),
    ("\u041f", "Cyrillic Pe"),  # П
]
for txt, desc in tests:
    enc = t.encode(txt)
    print(f"{desc}: {repr(txt)} -> ids={enc.ids}")
    for tid in enc.ids:
        print(f"  id {tid}: {repr(t.id_to_token(tid))}")

# Check: are bytes 0-255 directly mappable?
print("\nByte mapping:")
# First, what are tokens 6-261?
for i in range(6, 262):
    tok = t.id_to_token(i)
    # Check if this token decodes back to its byte
    decoded = t.decode([i])
    if len(decoded) == 1:
        b = ord(decoded)
        print(f"  id {i}: byte {b} ({repr(decoded)})")
        if i >= 20:
            break
