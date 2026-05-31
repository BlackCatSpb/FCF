"""Extract Russian ConceptNet triples - quick"""
import sqlite3, os

DB = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\conceptnet.db"
OUT = os.path.join(os.path.dirname(__file__), "real_data", "conceptnet_ru.txt")

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("PRAGMA table_info(language)")
print(f"Language: {[(r[1], r[2]) for r in cur.fetchall()]}")
cur.execute("PRAGMA table_info(label)")
print(f"Label: {[(r[1], r[2]) for r in cur.fetchall()]}")
cur.execute("PRAGMA table_info(concept)")
print(f"Concept: {[(r[1], r[2]) for r in cur.fetchall()]}")
cur.execute("PRAGMA table_info(relation)")
print(f"Relation: {[(r[1], r[2]) for r in cur.fetchall()]}")
cur.execute("PRAGMA table_info(edge)")
print(f"Edge: {[(r[1], r[2]) for r in cur.fetchall()]}")

# Get Russian language ID
cur.execute("SELECT id, name FROM language WHERE name='ru'")
ru_lang = cur.fetchone()
ru_lang_id = ru_lang[0] if ru_lang else None
print(f"Russian language ID: {ru_lang_id}")

if ru_lang_id:
    # Get Russian concept labels
    cur.execute("SELECT id, text FROM label WHERE language_id = ?", (ru_lang_id,))
    labels = {r[0]: r[1] for r in cur.fetchall()}
    print(f"Russian labels: {len(labels)}")

    # Extract edges with Russian concepts
    cur.execute("SELECT start_id, end_id, relation_id FROM edge LIMIT 5")
    sample_edges = cur.fetchall()
    print(f"Sample edges: {sample_edges}")

    # Get relation names
    cur.execute("SELECT id, name FROM relation")
    relations = {r[0]: r[1] for r in cur.fetchall()}

    # Fast extraction: find concepts with Russian labels, then get their edges
    ru_labels_id_list = list(labels.keys())[:100000]  # Limit to 100K concepts
    
    count = 0
    with open(OUT, 'w', encoding='utf-8') as f:
        for chunk_start in range(0, len(ru_labels_id_list), 5000):
            chunk = ru_labels_id_list[chunk_start:chunk_start+5000]
            placeholders = ','.join('?' * len(chunk))
            
            # Get concept IDs for these labels
            cur.execute(f"SELECT id, label_id FROM concept WHERE label_id IN ({placeholders})", chunk)
            concept_map = {r[0]: r[1] for r in cur.fetchall()}
            
            # Get edges where start or end is one of these concepts
            if concept_map:
                cids = list(concept_map.keys())
                cid_placeholders = ','.join('?' * len(cids))
                cur.execute(f"""
                    SELECT e.start_id, e.end_id, e.relation_id
                    FROM edge e
                    WHERE e.start_id IN ({cid_placeholders}) OR e.end_id IN ({cid_placeholders})
                    LIMIT 50000
                """, cids * 2)
                
                for row in cur.fetchall():
                    start_id, end_id, rel_id = row
                    start_label = labels.get(concept_map.get(start_id, -1), "?")
                    end_label = labels.get(concept_map.get(end_id, -1), "?")
                    rel_name = relations.get(rel_id, "RelatedTo").replace("/r/", "")
                    
                    if start_label != "?" and end_label != "?":
                        f.write(f"{start_label} {rel_name} {end_label}\n")
                        count += 1
                
                if count % 10000 == 0 and count > 0:
                    print(f"  {count:,} triples")
                
                if count >= 500000:
                    break

conn.close()
size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"Done: {count:,} triples, {size_mb:.1f} MB")
