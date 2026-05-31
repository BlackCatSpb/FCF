"""Download Wikipedia RU dump, extract text, encode to CharacterVocab format."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

print("Downloading Wikipedia RU...")
import urllib.request, bz2, xml.etree.ElementTree as ET

URL = "https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2"
OUT = os.path.join(os.path.dirname(__file__), "real_data", "wiki_ru.xml.bz2")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

if not os.path.exists(OUT):
    print(f"Downloading to {OUT}...")
    urllib.request.urlretrieve(URL, OUT)
    print("Downloaded.")

# Parse XML and extract text
print("Extracting text...")
from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()

all_ids = []
count = 0
with bz2.open(OUT, 'rt', encoding='utf-8') as f:
    for event, elem in ET.iterparse(f, events=('end',)):
        if elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}text':
            text = elem.text or ''
            text = text.strip()
            if len(text) > 50:
                ids = cv.encode(text)
                all_ids.extend(ids)
                count += 1
            elem.clear()
        
        if count % 10000 == 0 and all_ids:
            print(f"\r  {count} articles, {len(all_ids)/1e6:.1f}M tokens", end='', flush=True)

# Save
out_npy = os.path.join(os.path.dirname(__file__), "real_data", "wiki_ru.npy")
arr = np.array(all_ids, dtype=np.int32)
np.save(out_npy, arr)
print(f"\nSaved: {out_npy} ({len(arr)/1e6:.1f}M tokens)")
