"""Download Wikipedia RU via HuggingFace datasets (works on local PC)."""
import os, sys
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
from eva.data_manager import DataManager

DATA_DIR = r'C:\Users\black\OneDrive\Desktop\FCF\real_data'
os.makedirs(DATA_DIR, exist_ok=True)

print("[1/2] Wikipedia RU via HuggingFace datasets...")
wiki = DataManager.load_wikipedia(streaming=True)
articles = []
for i, item in enumerate(wiki):
    text = item.get('text', '')
    if len(text) > 500:
        articles.append(text.strip())
    if len(articles) >= 1000:
        break
    if i % 200 == 0:
        print(f"  {len(articles)}/1000 articles...")

wiki_path = os.path.join(DATA_DIR, 'wiki_1000.txt')
with open(wiki_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(articles))
print(f"  OK: {len(articles)} articles, {sum(len(t) for t in articles):,} chars")

# 2. Local corpus as supplement  
print("[2/2] Local corpus supplement...")
local_path = r'C:\Users\black\OneDrive\Desktop\FCF\training_corpus.txt'
if os.path.exists(local_path):
    with open(local_path, encoding='utf-8') as f:
        base = f.read()
    big = (base + '\n') * 500
    with open(os.path.join(DATA_DIR, 'supplement.txt'), 'w', encoding='utf-8') as f:
        f.write(big)
    print(f"  OK: {len(big):,} chars")

# Combine
final_path = os.path.join(DATA_DIR, 'real_corpus.txt')
with open(final_path, 'w', encoding='utf-8') as out:
    for fn in ['wiki_1000.txt', 'supplement.txt']:
        fp = os.path.join(DATA_DIR, fn)
        if os.path.exists(fp):
            with open(fp, encoding='utf-8') as f:
                out.write(f.read() + '\n')

size_mb = os.path.getsize(final_path) / 1e6
print(f"\nDONE: {final_path} ({size_mb:.1f} MB)")
