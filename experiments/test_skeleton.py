"""Test ConceptSkeleton build — uses _add_orphan for words without form_of."""
import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
from eva.symbolic.concept_net import ConceptSkeleton

sk = ConceptSkeleton()
sk.build()

print(f"\nTotal concepts: {sk.n_concepts}")
print(f"Top 30 concepts:")
for cid, anchor, size in sk.top_concepts(30):
    print(f"  [{cid:4d}] {anchor:30s} ({size:3d} words)")
print(f"\nSingleton concepts (size=1): {sum(1 for c in sk.concepts.values() if c['size']==1)}")
print(f"\nConcept relations: {len(sk.relations)}")
if sk.parents:
    print(f"Hierarchy nodes: {len(sk.parents)}")

for test_word in ['собака', 'собаки', 'собакой', 'армия', 'война', 'человек', 'сказал', 'князь', 'большой', 'любить']:
    cid = sk.concept_of(test_word)
    if cid is not None:
        c = sk.concepts[cid]
        print(f"  {test_word:12s} -> concept [{cid:4d}] anchor={c['anchor']!r:20s}, "
              f"satellites={c['satellites'][:8]}")
    else:
        print(f"  {test_word:12s} -> NOT IN CONCEPTNET")

sk.build_meta_concepts()
if sk.cid_to_metas:
    print(f"\nMeta-concepts: {len(sk.cid_to_metas)}")
    for mid, cids in sorted(sk.cid_to_metas.items())[:5]:
        anchors = [sk.concepts[c]['anchor'] for c in cids[:8]]
        print(f"  M{mid}: {anchors}")
