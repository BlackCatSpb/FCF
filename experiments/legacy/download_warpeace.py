"""Download War and Peace in Russian from GitHub."""
import sys, os, re, time
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import urllib.request

SENT_OUT = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\warpeace_sentences.txt'

# Try multiple sources for Russian War and Peace text
import urllib.parse

sources = [
    'https://raw.githubusercontent.com/nicedoc/sample-data/master/war-and-peace-ru.txt',
    'https://gist.githubusercontent.com/nicedoc/1b1f7b5a26a596e28875e1b4edf8cd6d/raw/war-and-peace-ru.txt',
    'https://raw.githubusercontent.com/bedlate/data-samples/master/war_and_peace_russian.txt',
]
# Also try encoded URL for GitHub
ru_url = 'https://raw.githubusercontent.com/nsu-ai/russian_literature/master/' + urllib.parse.quote('Война_и_мир.txt')
sources.append(ru_url)
# Wikisource extraction fallback
sources.append('https://ru.wikisource.org/w/index.php?title=%D0%92%D0%BE%D0%B9%D0%BD%D0%B0_%D0%B8_%D0%BC%D0%B8%D1%80_(%D0%A2%D0%BE%D0%BB%D1%81%D1%82%D0%BE%D0%B9)/%D0%A2%D0%BE%D0%BC_1&printable=yes')

text = None
for url in sources:
    try:
        print(f'Trying: {url}')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            try:
                text = raw.decode('utf-8')
            except:
                text = raw.decode('cp1251')
        print(f'  Success: {len(text):,} chars')
        break
    except Exception as e:
        print(f'  Failed: {e}')

if text is None:
    print('All sources failed, trying alternative approach...')
    sys.exit(1)

# Clean text
text = re.sub(r'\r\n', '\n', text)
text = re.sub(r'\n{3,}', '\n\n', text)

# Remove Gutenberg headers/footers
text = re.sub(r'^.*?\*\*\* START OF THIS PROJECT GUTENBERG', '', text, flags=re.DOTALL | re.IGNORECASE)
text = re.sub(r'^.*?\*\*\* START OF THE PROJECT GUTENBERG', '', text, flags=re.DOTALL | re.IGNORECASE)
text = re.sub(r'\*\*\* END OF THIS PROJECT GUTENBERG.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
text = re.sub(r'\*\*\* END OF THE PROJECT GUTENBERG.*$', '', text, flags=re.DOTALL | re.IGNORECASE)

# Remove underscores used for italics
text = text.replace('_', '')

# Save raw clean text
RAW_OUT = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\warpeace_raw.txt'
with open(RAW_OUT, 'w', encoding='utf-8') as f:
    f.write(text)
print(f'Raw text: {len(text):,} chars -> {RAW_OUT}')

# Split into sentences with spaCy
print('Loading spaCy for sentence split...')
import spacy
nlp = spacy.load('ru_core_news_sm', disable=['lemmatizer', 'ner', 'tagger'])

print('Splitting sentences...')
t0 = time.time()
# Split by paragraphs first, then sentences
paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]
all_sents = []

for i, para in enumerate(paragraphs):
    para = re.sub(r'\s+', ' ', para).strip()
    if len(para) < 20:
        continue
    doc = nlp(para[:100000])  # cap per paragraph
    for sent in doc.sents:
        s = sent.text.strip()
        s = re.sub(r'^[\s,;:!?\-—\u2014\u2013]+', '', s)
        s = re.sub(r'[\s,;:!?\-—\u2014\u2013]+$', '', s)
        if len(s) >= 15 and s[-1] in '.!?' and re.search(r'[а-яА-ЯёЁ]', s):
            all_sents.append(s)
    if (i + 1) % 100 == 0:
        print(f'  para {i+1}/{len(paragraphs)}, sents={len(all_sents):,} ({time.time()-t0:.1f}s)')

with open(SENT_OUT, 'w', encoding='utf-8') as f:
    for s in all_sents:
        f.write(s + '\n')

print(f'\nTotal: {len(all_sents):,} sentences')
print(f'Saved: {SENT_OUT}')
