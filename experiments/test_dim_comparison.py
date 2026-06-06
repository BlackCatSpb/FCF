"""Dim comparison: run training at 256, 512, 1024 and compare eval."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.association_graph import AssociationGraph
from eva.symbolic.vector_space import VectorGenerator
from eva.symbolic.text_hierarchy import TextHierarchy
from eva.symbolic.auto_config import AutoConfig

hv = HierarchicalVocab()
results = {}

for dim in [256, 512]:
    print(f'\n{"="*60}')
    print(f'  Testing svd_dim = {dim}')
    print(f'{"="*60}')
    config = AutoConfig()
    config.svd_dim = dim
    config.svd_epochs = 3
    config.context_window = 25

    heads = HeadsEnsemble('real_data/v8/heads_meta_merged.pkl', 'real_data/v8', config=config)
    ag = AssociationGraph(config=config)
    ag.build(heads.log_prob_csr, hv.token_type, decode_fn=hv.decode)

    vg = VectorGenerator(heads, ag, hv, config=config)
    vg.load_refined_vectors('real_data/v8/vectors_refined.pkl')

    th = TextHierarchy('real_data/full_corpus_ru.txt', hv)
    sents = th.parse()

    train_sents = []
    for sent in sents[:5]:
        text = sent.text.strip()
        if not text or text.isspace():
            continue
        enc = hv.encode(' ' + text)
        tids = [t for t in enc if t < config.bpe_limit and vg.tt[t] == 2 and vg._is_content_token(t)]
        if len(tids) < 2:
            continue
        train_sents.append({'text': text, 'tids': tids, 'fw': hv.decode([tids[0]]).strip()})

    print(f'  Training on {len(train_sents)} sentences, {dim}d vectors')

    original_boost = config.target_boost
    final_eval = None
    for epoch in range(config.svd_epochs):
        vg.set_epoch(epoch)
        vg.reset_momentum()
        for i, sent in enumerate(train_sents):
            for rep in range(3):
                r = vg.generate(seed_word=sent['fw'], target_text=sent['text'],
                                max_tokens=config.context_window, temperature=0.3,
                                training_mode=True)
        config.target_boost = 0.0
        total = 0
        hits = 0
        for i, sent in enumerate(train_sents):
            r = vg.generate(seed_word=sent['fw'], max_tokens=config.context_window,
                            temperature=0.0, training_mode=False)
            gen_tids = [t for t in hv.encode(' ' + r['text'])
                        if t < config.bpe_limit and vg.tt[t] == 2 and vg._is_content_token(t)]
            matches = sum(1 for t in gen_tids if t in sent['tids'])
            hits += matches
            total += max(1, len(sent['tids']))
        pct = 100.0 * hits / total if total else 0
        print(f'  Epoch {epoch}: eval {hits}/{total} ({pct:.1f}%)')
        final_eval = pct
        config.target_boost = original_boost

    results[dim] = final_eval
    print(f'  >>> Final eval at {dim}d: {final_eval:.1f}%')

print(f'\n{"="*60}')
print('  SUMMARY')
print(f'{"="*60}')
for dim, pct in sorted(results.items()):
    print(f'  svd_dim={dim}: {pct:.1f}%')
