"""
Prepare War and Peace for training: split by sentences, encode to int32.
No cut words, no partial sentences. EOS after each sentence.
"""
import sys, os, re, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from eva.symbolic.char_vocab import CharacterVocab

cv = CharacterVocab()
SRC = r"C:\Users\black\OneDrive\Desktop\Толстой Лев. Война и мир. Книга 1 - royallib.ru.txt"
OUT = os.path.join(os.path.dirname(__file__), "real_data", "war_and_peace.npy")

print(f"Reading: {SRC}")

# Read raw text
with open(SRC, 'r', encoding='utf-8') as f:
    raw = f.read()

print(f"  Size: {len(raw):,} chars")

# Clean: remove excessive whitespace, normalize line endings
raw = re.sub(r'\r\n|\r', '\n', raw)
raw = re.sub(r'\n{3,}', '\n\n', raw)  # max 2 newlines

# Split into sentences (split at .!? followed by space/newline/capital)
sentences = re.split(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z])', raw)
sentences = [s.strip() for s in sentences if len(s.strip()) > 2]

print(f"  Sentences: {len(sentences):,}")

# Encode each sentence with BOS/EOS
all_ids = []
for sent in sentences:
    ids = cv.encode(sent)
    if len(ids) >= 4:  # BOS + content + EOS
        all_ids.extend(ids)

total = len(all_ids)
print(f"  Tokens: {total:,}")

arr = np.array(all_ids, dtype=np.int32)
np.save(OUT, arr)
print(f"Saved: {OUT} ({total/1e6:.3f}M tokens, {os.path.getsize(OUT)/1e6:.1f} MB)")
print("Done.")
