"""
Encode full_corpus_ru.txt with EVA CharacterVocab → full_corpus_encoded.npy
Sentence boundaries only (no word boundaries) for speed.
"""
import sys, os, re, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()

SRC = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ru.txt")
DST = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_encoded.npy")

print(f"Reading {SRC}...")
with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()
print(f"  {len(text):,} chars")

# Split into sentences
sents = re.split(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z])', text)
print(f"  {len(sents):,} sentences")

# Encode with boundaries
all_ids = []
t0 = time.time()
for i, s in enumerate(sents):
    s = s.strip()
    if len(s) < 2:
        continue
    ids = cv.encode_with_boundaries(s)
    if len(ids) >= 5:
        all_ids.extend(ids)
    if (i + 1) % 50000 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        print(f"  [{i+1:>7,}] {len(all_ids)/1e6:.2f}M tokens, {rate:.0f} sents/s")

data = np.array(all_ids, dtype=np.int32)
np.save(DST, data)
elapsed = time.time() - t0
print(f"\nDone: {len(data)/1e6:.2f}M tokens in {elapsed:.0f}s ({len(data)/elapsed:.0f} tok/s)")
print(f"  Min={data.min()} Max={data.max()} Unique={len(np.unique(data))}")
