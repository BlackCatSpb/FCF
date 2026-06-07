import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.association_graph import AssociationGraph
from eva.symbolic.vector_space import VectorGenerator
from eva.symbolic.text_hierarchy import TextHierarchy
from eva.symbolic.auto_config import AutoConfig

hv = HierarchicalVocab()
config = AutoConfig.load('real_data/calibrated_config.pkl')

heads = HeadsEnsemble('real_data/v8/heads_meta_merged.pkl', 'real_data/v8', config=config)
ag = AssociationGraph(config=config)
ag.build(heads.log_prob_csr, hv.token_type, decode_fn=hv.decode)

vg = VectorGenerator(heads, ag, hv, config=config)
vg.load_refined_vectors('real_data/v8/vectors_refined.pkl')

th = TextHierarchy('real_data/full_corpus_ru.txt', hv)
sents = th.parse()

train_sents = []
for sent in sents[:30]:
    text = sent.text.strip()
    if not text or text.isspace():
        continue
    enc = hv.encode(' ' + text)
    tids = [t for t in enc if t < config.bpe_limit and vg.tt[t] == 2 and vg._is_content_token(t)]
    if len(tids) < 2:
        continue
    train_sents.append({'text': text, 'tids': tids, 'fw': hv.decode([tids[0]]).strip()})

print(f'Training on {len(train_sents)} sentences')

original_target_boost = config.target_boost
n_epochs = config.svd_epochs

all_epoch_results = []

for epoch in range(n_epochs):
    vg.set_epoch(epoch)
    vg.reset_momentum()
    subsample_threshold = 0.9

    # ==== TRAINING ====
    skipped = 0
    trained = 0
    for i, sent in enumerate(train_sents):
        match_ratios = []
        for rep in range(10):
            r = vg.generate(seed_word=sent['fw'], target_text=sent['text'],
                            max_tokens=config.context_window, temperature=0.3,
                            training_mode=True)
            match_ratio = r['target_matches'] / max(1, r['target_total'])
            match_ratios.append(match_ratio)
            if match_ratio >= 1.0:
                break
        avg_match = sum(match_ratios) / len(match_ratios)
        if avg_match >= subsample_threshold:
            skipped += 1
        else:
            trained += 1
        print(f'  E{epoch} S{i}: avg_match={avg_match:.2f} reps={len(match_ratios)}' +
              (' SKIP' if avg_match >= subsample_threshold else ''))

    # ==== EVALUATION ====
    config.target_boost = 0.0
    eval_results = []
    for i, sent in enumerate(train_sents):
        r = vg.generate(seed_word=sent['fw'], max_tokens=config.context_window,
                        temperature=0.0, training_mode=False)
        gen_text = r['text']
        gen_tids = [t for t in hv.encode(' ' + gen_text)
                    if t < config.bpe_limit and vg.tt[t] == 2 and vg._is_content_token(t)]
        matches = sum(1 for t in gen_tids if t in sent['tids'])
        match_pct = 100.0 * matches / max(1, len(sent['tids']))
        eval_results.append({
            'idx': i, 'n_target': len(sent['tids']),
            'matched_eval': matches, 'eval_pct': round(match_pct, 1),
        })

    total_matches = sum(r['matched_eval'] for r in eval_results)
    total_target = sum(r['n_target'] for r in eval_results)
    eval_pct = 100.0 * total_matches / max(1, total_target)
    print(f'Epoch {epoch}: eval={total_matches}/{total_target} ({eval_pct:.1f}%)' +
          f' trained={trained} skipped={skipped}')

    all_epoch_results.append({
        'epoch': epoch,
        'svd_lr': vg._svd_lr,
        'eval_total': f'{total_matches}/{total_target}',
        'eval_pct': round(eval_pct, 1),
        'trained': trained,
        'skipped': skipped,
    })

config.target_boost = original_target_boost

# Summary
best_epoch = max(all_epoch_results, key=lambda x: x['eval_pct'])
print(f'\n=== BEST: Epoch {best_epoch["epoch"]} at {best_epoch["eval_pct"]}% ===')
print(f'=== Trained vectors: {len(vg._trained_vectors)} tokens ===')

out = os.path.join(os.path.dirname(__file__), 'train_eval_results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump({'n_epochs': n_epochs,
               'svd_lr': config.svd_lr,
               'svd_lr_decay': config.svd_lr_decay,
               'svd_momentum_beta': config.svd_momentum_beta,
               'svd_neg_feedback_scale': config.svd_neg_feedback_scale,
               'best_epoch': best_epoch['epoch'],
               'best_eval_pct': best_epoch['eval_pct'],
               'epochs': all_epoch_results}, f, ensure_ascii=False, indent=2)
print(f'Saved to {out}')
