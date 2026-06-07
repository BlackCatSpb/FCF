"""Verify ConceptSkeleton — check specific concepts."""
import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')

out = open(r'C:\Users\black\OneDrive\Desktop\FCF\experiments\skeleton_verify.txt', 'w', encoding='utf-8')

from eva.symbolic.concept_net import ConceptSkeleton

sk = ConceptSkeleton()
sk.build()

out.write(f"Total concepts: {sk.n_concepts}\n")
out.write(f"Relations: {len(sk.relations)}\n\n")

# Test specific words
test_words = ['собака', 'собаки', 'собакой', 'собачку', 'армия', 'война', 'человек',
              'сказал', 'говорить', 'говорил', 'князь', 'большой', 'любить',
              'слабый', 'слабо', 'ослабеть']

for w in test_words:
    cid = sk.concept_of(w)
    if cid is not None:
        c = sk.concepts[cid]
        out.write(f"'{w}' -> concept[{cid:4d}] anchor='{c['anchor']}'\n")
        out.write(f"  satellites ({c['size']}): {c['satellites'][:12]}\n")
        # Get related concepts
        rels = sk.related_concepts(cid)
        if rels:
            rel_data = [(sk.concepts[r]['anchor'], 
                        [rr for (ci,cj), rrl in sk.relations.items()
                         if (ci==cid and cj==r or ci==r and cj==cid) 
                         for rr in rrl]) 
                       for r in rels[:5]]
            out.write(f"  related: {[(a,r) for a,r in rel_data]}\n")
    else:
        out.write(f"'{w}' -> NOT FOUND\n")

# Check concept 0 (largest)
largest_cid = max(sk.concepts.keys(), key=lambda c: sk.concepts[c]['size'])
largest = sk.concepts[largest_cid]
out.write(f"\nLargest concept: [{largest_cid}] anchor='{largest['anchor']}' ({largest['size']} words)\n")
out.write(f"  satellites: {largest['satellites'][:20]}\n")

# Save skeleton for tokenizer
sk.save(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_skeleton.json')
out.write("\nSkeleton saved to concept_skeleton.json\n")
out.close()
