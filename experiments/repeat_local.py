import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.association_graph import AssociationGraph
from eva.symbolic.vector_space import VectorGenerator
from eva.symbolic.text_hierarchy import TextHierarchy

hv = HierarchicalVocab()
heads = HeadsEnsemble('real_data/v8/heads_meta_merged.pkl', 'real_data/v8')
ag = AssociationGraph(n_clusters=48, n_metas=12)
ag.build(heads.log_prob_csr, hv.token_type, decode_fn=hv.decode)
vg = VectorGenerator(heads, ag, hv)
vg.load_refined_vectors('real_data/v8/vectors_refined.pkl')
vg.load_gates('real_data/v8/gates')

th = TextHierarchy('real_data/full_corpus_ru.txt', hv)
sents = th.parse()

results = []

for i, sent in enumerate(sents[:30]):
    text = sent.text.strip()
    if not text or text.isspace():
        results.append({'idx': i, 'text': text, 'skipped': True})
        continue
    enc = hv.encode(' ' + text)
    tids = [t for t in enc if t < 4096 and vg.tt[t] == 2 and vg._is_content_token(t)]
    if len(tids) < 2:
        results.append({'idx': i, 'text': text, 'skipped': True})
        continue

    fw = hv.decode([tids[0]]).strip()
    n_target = len(tids)

    sentence_results = []
    for rep in range(10):
        r = vg.generate(seed_word=fw, target_text=text, max_tokens=80, temperature=0.3)
        matched = vg._target_matches
        match_pct = 100.0 * matched / max(1, n_target)
        gen_text = r['text']
        sentence_results.append({
            'rep': rep+1,
            'matched': matched,
            'target': n_target,
            'pct': round(match_pct, 1),
            'gen': gen_text[:120]
        })
        if matched >= n_target:
            break

    results.append({'idx': i, 'text': text[:60], 'n_target': n_target,
                     'reps': len(sentence_results),
                     'first_pct': sentence_results[0]['pct'],
                     'last_pct': sentence_results[-1]['pct'],
                     'first_match': sentence_results[0]['matched'],
                     'last_match': sentence_results[-1]['matched'],
                     'detail': sentence_results})
    print(f'Sent {i}: {n_target} words, {sentence_results[0]["pct"]}% -> {sentence_results[-1]["pct"]}% ({len(sentence_results)} reps)')

first_avg = sum(r['first_pct'] for r in results if not r.get('skipped')) / max(1, len([r for r in results if not r.get('skipped')]))
last_avg = sum(r['last_pct'] for r in results if not r.get('skipped')) / max(1, len([r for r in results if not r.get('skipped')]))
total_first = sum(r['first_match'] for r in results if not r.get('skipped'))
total_last = sum(r['last_match'] for r in results if not r.get('skipped'))
total_target = sum(r['n_target'] for r in results if not r.get('skipped'))
print(f'\nTotal: {total_first}/{total_target} ({100*total_first/max(1,total_target):.1f}%) -> {total_last}/{total_target} ({100*total_last/max(1,total_target):.1f}%)')
print(f'Avg per sentence: {first_avg:.1f}% -> {last_avg:.1f}%')

out = os.path.join(os.path.dirname(__file__), 'repeat_local_results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'\nSaved to {out}')
