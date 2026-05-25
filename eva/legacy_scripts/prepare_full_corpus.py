"""
Download ConceptNet Russian triples + combine with wiki corpus.
Output: full_corpus_ru.txt — полный корпус для символьного обучения.
"""
import os, sys, re, time, json
from urllib.request import urlopen, Request
from urllib.parse import quote_plus

OUT_DIR = os.path.join(os.path.dirname(__file__), "real_data")
os.makedirs(OUT_DIR, exist_ok=True)

def download_conceptnet_ru(limit=500000):
    """Download Russian ConceptNet triples via API."""
    print("[1/3] ConceptNet RU...")
    
    concepts_path = os.path.join(OUT_DIR, "conceptnet_ru.txt")
    concepts = []
    
    params = {
        "language": "ru",
        "limit": min(limit, 1000),
        "format": "json",
    }
    base_url = "https://api.conceptnet.io/c/ru"
    
    seen = set()
    total = 0
    
    try:
        while len(concepts) < limit:
            url = base_url + "?" + "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
            req = Request(url, headers={"User-Agent": "EVA-Symbolic/1.0"})
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            
            edges = data.get("edges", [])
            if not edges:
                break
            
            for edge in edges:
                start = edge.get("start", {}).get("label", "")
                rel = edge.get("rel", {}).get("label", "")
                end = edge.get("end", {}).get("label", "")
                
                if start and end:
                    start = re.sub(r'^/c/ru/', '', start)
                    end = re.sub(r'^/c/ru/', '', end)
                    rel_name = rel.replace("/r/", "")
                    
                    text = f"{start} {rel_name} {end}"
                    key = hash(text)
                    
                    if key not in seen:
                        seen.add(key)
                        concepts.append(text)
            
            total += len(edges)
            print(f"  Downloaded: {len(concepts)} triples ({total} edges)")
            
            if len(concepts) >= limit:
                break
            
            # Pagination
            next_page = data.get("view", {}).get("nextPage", "")
            if not next_page:
                break
            params["offset"] = params.get("offset", 0) + len(edges)
            
    except Exception as e:
        print(f"  API error: {e}")
    
    with open(concepts_path, 'w', encoding='utf-8') as f:
        for c in concepts:
            f.write(c + '\n')
    
    size_mb = os.path.getsize(concepts_path) / 1024 / 1024
    print(f"  Saved: {len(concepts)} triples, {size_mb:.1f} MB")
    return concepts_path


def combine_all():
    """Combine wiki + conceptnet into one massive corpus."""
    print("\n[2/3] Combining corpus...")
    
    clean_ru = os.path.join(OUT_DIR, "clean_ru.txt")
    conceptnet = os.path.join(OUT_DIR, "conceptnet_ru.txt")
    full_corpus = os.path.join(OUT_DIR, "full_corpus_ru.txt")
    
    total_lines = 0
    total_size = 0
    
    with open(full_corpus, 'w', encoding='utf-8') as out:
        # Source 1: Clean wiki text (162 MB)
        if os.path.exists(clean_ru):
            with open(clean_ru, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 5:
                        out.write(line + '\n')
                        total_lines += 1
            size = os.path.getsize(clean_ru) / 1024 / 1024
            print(f"  Wiki: {size:.1f} MB")
        
        # Source 2: ConceptNet triples    
        if os.path.exists(conceptnet):
            with open(conceptnet, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 5:
                        out.write(line + '\n')
                        total_lines += 1
            size = os.path.getsize(conceptnet) / 1024 / 1024
            print(f"  ConceptNet: {size:.1f} MB")
    
    total_size = os.path.getsize(full_corpus) / 1024 / 1024
    print(f"  FULL: {total_lines} lines, {total_size:.1f} MB")
    return full_corpus


def pre_tokenize_full():
    """Pre-tokenize the full corpus."""
    print("\n[3/3] Pre-tokenizing...")
    
    full_corpus = os.path.join(OUT_DIR, "full_corpus_ru.txt")
    npy_out = os.path.join(OUT_DIR, "full_corpus_ids.npy")
    
    if not os.path.exists(full_corpus):
        print("Full corpus not found!")
        return None
    
    sys.path.insert(0, os.path.dirname(__file__))
    from eva.symbolic import CharacterVocab
    import numpy as np
    
    vocab = CharacterVocab()
    all_ids = []
    count = 0
    
    with open(full_corpus, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) < 3:
                continue
            ids = vocab.encode(line)
            ids.append(0)  # separator
            all_ids.extend(ids)
            count += 1
            if count % 200000 == 0:
                print(f"  {count} lines, {len(all_ids):,} tokens")
    
    arr = np.array(all_ids, dtype=np.int32)
    np.save(npy_out, arr)
    
    size_mb = os.path.getsize(npy_out) / 1024 / 1024
    print(f"  Done: {len(all_ids):,} tokens from {count} lines, {size_mb:.1f} MB")
    return npy_out


if __name__ == "__main__":
    download_conceptnet_ru(limit=200000)
    combine_all()
    npy_path = pre_tokenize_full()
    print(f"\n{'='*60}")
    print(f"FULL CORPUS READY: {npy_path}")
    print(f"{'='*60}")
