"""
build_conceptnet_text.py — извлечь ConceptNet русские рёбра → текст.

Читает conceptnet.db (SQLite, 10 GB), фильтрует Russian→Russian edges,
конвертирует в естественные русские предложения по шаблонам отношений.
Сохраняет как plain text для подачи в пайплайн (BPE + trajectory).
"""
import sys, os, time, sqlite3, re
sys.path.insert(0, os.path.dirname(__file__))

DB_PATH = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\conceptnet.db'
OUTPUT = os.path.join(os.path.dirname(__file__), 'real_data', 'conceptnet', 'conceptnet_ru.txt')

# ─── Templates: relation → Russian sentence template ───
# {s} = start concept label, {e} = end concept label
# NOTE: relation names from DB are WITHOUT /r/ prefix
TEMPLATES = {
    'is_a':                      '{s} — это {e}.',
    'form_of':                   '{s} — форма слова {e}.',
    'related_to':                '{s} связан с {e}.',
    'derived_from':              '{s} происходит от {e}.',
    'etymologically_derived_from': '{s} происходит от {e}.',
    'etymologically_related_to': '{s} этимологически связан с {e}.',
    'synonym':                   '{s} — то же, что {e}.',
    'antonym':                   '{s} противоположно {e}.',
    'distinct_from':             '{s} отличается от {e}.',
    'similar_to':                '{s} похоже на {e}.',
    'part_of':                   '{s} — часть {e}.',
    'manner_of':                 '{s} является способом {e}.',
    'has_property':              '{s} — {e}.',
    'used_for':                  '{s} используется для {e}.',
    'located_at':                '{s} находится в {e}.',
    'at_location':               '{s} находится в {e}.',
    'capable_of':                '{s} может {e}.',
    'desires':                   '{s} хочет {e}.',
    'has_a':                     '{s} имеет {e}.',
    'made_of':                   '{s} сделан из {e}.',
    'created_by':                '{s} создан с помощью {e}.',
    'causes':                    '{s} вызывает {e}.',
    'causes_effect':             '{s} вызывает {e}.',
    'causes_desire':             '{s} вызывает желание {e}.',
    'receives_action':           '{s} можно {e}.',
    'has_subevent':              '{s} включает {e}.',
    'motivated_by_goal':         '{s} мотивирован {e}.',
    'obstructed_by':             '{s} затрудняется {e}.',
    'symbol_of':                 '{s} символизирует {e}.',
    'not_desires':               '{s} не хочет {e}.',
    'not_has_property':          '{s} не {e}.',
}

# Fallback for unknown relations
FALLBACK = '{s} относится к {e}.'

# Relations to skip (too vague or non-semantic)
SKIP_RELATIONS = set()


def clean_label(text: str) -> str:
    """Clean ConceptNet label: remove underscores, normalize spaces."""
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    print(f'Connecting to ConceptNet DB ({DB_PATH})...')
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Query all Russian→Russian edges with labels and relation names
    print('Querying Russian→Russian edges...')
    c.execute("""
        SELECT
            ls.text AS start_label,
            le.text AS end_label,
            r.name AS relation,
            e.etc
        FROM edge e
        JOIN relation r ON e.relation_id = r.id
        JOIN concept cs ON e.start_id = cs.id
        JOIN concept ce ON e.end_id = ce.id
        JOIN label ls ON cs.label_id = ls.id
        JOIN label le ON ce.label_id = le.id
        WHERE ls.language_id = 145
          AND le.language_id = 145
    """)

    n_total = 0
    n_written = 0
    n_skipped_rel = 0
    n_low_weight = 0
    batch = []

    print(f'Query ready, writing to {OUTPUT}...')
    write_t0 = time.time()

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for row in c:
            n_total += 1
            rel = row['relation']
            start = clean_label(row['start_label'])
            end = clean_label(row['end_label'])

            # Skip empty labels
            if not start or not end:
                continue

            # Skip specific relations
            if rel in SKIP_RELATIONS:
                n_skipped_rel += 1
                continue

            # Check weight if available
            etc = row['etc']
            if etc:
                try:
                    import json
                    meta = json.loads(etc) if isinstance(etc, str) else etc
                    weight = float(meta.get('weight', 1.0))
                    if weight < 0.5:
                        n_low_weight += 1
                        continue
                except:
                    pass

            # Get template
            template = TEMPLATES.get(rel, FALLBACK)
            sentence = template.format(s=start, e=end)

            # Capitalize first letter
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]

            f.write(sentence + '\n')
            n_written += 1

            if n_total % 100000 == 0:
                elapsed = time.time() - write_t0
                rate = n_total / elapsed if elapsed > 0 else 0
                print(f'  {n_total:,} scanned, {n_written:,} written, {rate:,.0f} rows/s')

    elapsed = time.time() - t0
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024

    print(f'\n{"="*50}')
    print('CONCEPTNET EXTRACTION DONE')
    print(f'{"="*50}')
    print(f'  Edges scanned:   {n_total:,}')
    print(f'  Sentences written: {n_written:,}')
    print(f'  Skipped (relation): {n_skipped_rel:,}')
    print(f'  Skipped (weight): {n_low_weight:,}')
    print(f'  Output size:     {size_mb:.1f} MB')
    print(f'  Time:            {elapsed:.1f}s')
    print(f'  Output:          {OUTPUT}')

    conn.close()


if __name__ == '__main__':
    main()
