"""
Prepare War and Peace for training: split by sentences, encode to int32.
No cut words, no partial sentences. EOS after each sentence.
"""
import sys, os, re, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from eva.symbolic.char_vocab import CharacterVocab

cv = CharacterVocab()
SRC = [
    r"C:\Users\black\OneDrive\Desktop\Толстой Лев. Война и мир. Книга 1 - royallib.ru.txt",
    r"C:\Users\black\OneDrive\Desktop\Толстой Лев. Война и мир. Книга 2 - royallib.ru.txt",
]
OUT = os.path.join(os.path.dirname(__file__), "real_data", "war_and_peace.npy")

all_ids = []

for src_path in SRC:
    print(f"Reading: {src_path}")
    
    raw = None
    for enc in ['windows-1251', 'utf-8', 'cp1251', 'koi8-r']:
        try:
            with open(src_path, 'r', encoding=enc) as f:
                raw = f.read()
            print(f"  Encoding: {enc}, {len(raw):,} chars")
            break
        except:
            continue
    
    if raw is None:
        print(f"  SKIP: can't decode")
        continue
    
    raw = re.sub(r'\r\n|\r', '\n', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    
    sentences = re.split(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z])', raw)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    
    for sent in sentences:
        ids = cv.encode(sent)
        if len(ids) >= 4:
            all_ids.extend(ids)
    
    print(f"  Sentences: {len(sentences):,} | Total tokens: {len(all_ids):,}")

total = len(all_ids)
arr = np.array(all_ids, dtype=np.int32)
np.save(OUT, arr)
print(f"\nSaved: {OUT} ({total/1e6:.3f}M tokens, {os.path.getsize(OUT)/1e6:.1f} MB)")
print("Done.")
