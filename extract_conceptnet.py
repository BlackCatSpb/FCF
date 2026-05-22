"""Extract Russian triples from EVA-Ai conceptnet.db"""
import os, sys, re

DB_PATH = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\conceptnet.db"
OUT_DIR = os.path.join(os.path.dirname(__file__), "real_data")
OUT_PATH = os.path.join(OUT_DIR, "conceptnet_ru.txt")

print(f"Extracting from: {DB_PATH}")

import sqlite3
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check schema
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"Tables: {[t[0] for t in tables]}")

# Try to find edges with Russian
# ConceptNet SQLite has: edges(source_uri, target_uri, relation, metadata)
# We want entries where source or target starts with /c/ru/
try:
    cursor.execute("SELECT DISTINCT relation_id FROM edges LIMIT 5")
    cols = [d[0] for d in cursor.description]
    print(f"Edge columns: {cols}")
except:
    pass

# Try the conceptnet_lite approach
try:
    from conceptnet_lite import connect
    connect(DB_PATH)
    from conceptnet_lite.db import Edge
    
    count = 0
    ru_count = 0
    
    with open(OUT_PATH, 'w', encoding='utf-8') as out:
        edges = Edge.select().iterator()
        for edge in edges:
            count += 1
            try:
                start = edge.start.uri
                end = edge.end.uri
                
                if '/c/ru/' not in start and '/c/ru/' not in end:
                    continue
                
                rel = edge.relation.name
                weight = edge.etc.get('weight', 1.0) if edge.etc else 1.0
                if weight < 0.5:
                    continue
                
                rel_name = rel.replace("/r/", "")
                start_name = start.split("/")[-1].replace("_", " ")
                end_name = end.split("/")[-1].replace("_", " ")
                
                text = f"{start_name} {rel_name} {end_name}"
                out.write(text + '\n')
                ru_count += 1
                
            except:
                continue
            
            if count % 1000000 == 0:
                print(f"  Scanned: {count:,} edges, Russian: {ru_count:,}")
            
            if ru_count >= 500000:
                break
    
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"Done: {ru_count:,} Russian triples ({count:,} total scanned), {size_mb:.1f} MB")

except ImportError:
    print("conceptnet-lite not installed. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "conceptnet-lite", "-q"])
    print("Installed. Run again.")

conn.close()
