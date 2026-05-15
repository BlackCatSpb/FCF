"""Download Russian pre-1917 literature for EVA training."""
import requests, re, os, time

DATA_DIR = r'C:\Users\black\OneDrive\Desktop\FCF\real_data'
os.makedirs(DATA_DIR, exist_ok=True)

BOOKS = {
    # Project Gutenberg IDs for Russian classics (all pre-1917, public domain)
    "tolstoy_war_and_peace": 2600,
    "tolstoy_anna_karenina": 1399,
    "tolstoy_resurrection": 1938,
    "dostoevsky_crime_and_punishment": 2554,
    "dostoevsky_brothers_karamazov": 28054,
    "dostoevsky_idiot": 2638,
    "dostoevsky_demons": 600,
    "dostoevsky_gambler": 2197,
    "pushkin_captain_daughter": 6009,
    "pushkin_eugene_onegin": 23997,
    "pushkin_boris_godunov": 6006,
    "chekhov_stories": 57333,
    "chekhov_cherry_orchard": 1754,
    "gogol_dead_souls": 1081,
    "gogol_taras_bulba": 6000,
    "gogol_evenings_farm": 57445,
    "turgenev_fathers_and_sons": 19121,
    "turgenev_first_love": 57441,
    "lermontov_hero_of_our_time": 913,
    "goncharov_oblomov": 54700,
    "gorky_childhood": 57443,
    "gorky_chelkash": 57442,
    "bunin_stories": 57444,
    "kuprin_duel": 57440,
    "korolenko_blind_musician": 57439,
}

all_text = []

for name, gutenberg_id in BOOKS.items():
    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
    print(f"[{len(all_text)+1}/{len(BOOKS)}] {name}...", end=" ")
    
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}")
            continue
        
        r.encoding = 'utf-8'
        text = r.text
        
        start = text.find("*** START")
        if start == -1:
            start = text.find("START OF")
        if start != -1:
            start = text.find("\n", start) + 1
        else:
            start = 0
        
        end = text.find("*** END")
        if end == -1:
            end = text.find("End of the Project Gutenberg")
        if end == -1:
            end = len(text)
        
        text = text[start:end]
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[^\S\n]+', ' ', text)
        
        header_patterns = ['***', 'START OF', 'Project Gutenberg', 'This eBook', 'Title:', 'Author:', 'Release Date:', 'Language:', 'Character set']
        lines = [l.strip() for l in text.split('\n') if l.strip() and not any(p in l for p in header_patterns)]
        text = '\n'.join(lines)
        
        if len(text) > 1000:
            all_text.append(text)
            word_count = len(text.split())
            print(f"OK ({word_count:,} words)")
        else:
            print(f"Too short ({len(text)} chars)")
        
        time.sleep(0.5)
    
    except Exception as e:
        print(f"ERROR: {e}")

combined_path = os.path.join(DATA_DIR, 'russian_literature.txt')
total_words = 0
with open(combined_path, 'w', encoding='utf-8') as f:
    for text in all_text:
        f.write(text + '\n\n')
        total_words += len(text.split())

size_mb = os.path.getsize(combined_path) / 1e6
print(f"\n{'='*60}")
print(f"Downloaded: {len(all_text)}/{len(BOOKS)} books")
print(f"Total: {total_words:,} words, {size_mb:.1f} MB")
print(f"Saved: {combined_path}")
