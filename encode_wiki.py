"""Encode wiki_ru_large.txt to int32 npy for training."""
import sys, os, numpy as np, time
sys.path.insert(0, os.path.dirname(__file__))
from eva.symbolic.char_vocab import CharacterVocab

cv = CharacterVocab()
txt_path = os.path.join(os.path.dirname(__file__), "real_data", "wiki_ru_large.txt")
out_path = os.path.join(os.path.dirname(__file__), "real_data", "wiki_ru.npy")

print(f"Reading: {txt_path} ({os.path.getsize(txt_path)/1e9:.1f} GB)")

all_ids = []
t0 = time.time()
with open(txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            ids = cv.encode(line)
            all_ids.extend(ids)
        if len(all_ids) % 10_000_000 < 100:
            print(f"\r  {len(all_ids)/1e6:.1f}M tokens, {time.time()-t0:.0f}s", end='', flush=True)

arr = np.array(all_ids, dtype=np.int32)
np.save(out_path, arr)
print(f"\nSaved: {out_path} ({len(arr)/1e6:.1f}M tokens, {os.path.getsize(out_path)/1e9:.1f} GB)")
