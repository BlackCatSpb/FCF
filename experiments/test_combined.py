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
for sent in sents[:10]:
    text = sent.text.strip()
    if not text or text.isspace():
        continue
    enc = hv.encode(' ' + text)
    tids = [t for t in enc if t < config.bpe_limit and vg.tt[t] == 2 and vg._is_content_token(t)]
    if len(tids) < 2:
        continue
    train_sents.append({'text': text, 'tids': tids, 'fw': hv.decode([tids[0]]).strip()})

config.svd_lr = 0.1  # override stale saved config
print(f'Training on {len(train_sents)} sentences')
print(f'Config: lr={config.svd_lr}, decay={config.svd_lr_decay}, momentum={config.svd_momentum_beta}, neg_samples={config.svd_neg_samples}')
print()

original_target_boost = config.target_boost
n_epochs = config.svd_epochs

for epoch in range(n_epochs):
    vg.set_epoch(epoch)
    vg.reset_momentum()

    # ==== TRAINING ====
    for i, sent in enumerate(train_sents):
        for rep in range(5):
            r = vg.generate(seed_word=sent['fw'], target_text=sent['text'],
                            max_tokens=config.context_window, temperature=0.3,
                            training_mode=True)
            match_pct = r['target_matches'] / max(1, r['target_total']) * 100
            if rep == 0 or rep == 4 or match_pct == 100:
                print(f'  E{epoch} S{i} R{rep}: match={r["target_matches"]}/{r["target_total"]} ({match_pct:.0f}%)')
            if match_pct >= 100:
                break

    # ==== EVALUATION ====
    config.target_boost = 0.0
    total_matches = 0
    total_target = 0
    for i, sent in enumerate(train_sents):
        r = vg.generate(seed_word=sent['fw'], max_tokens=config.context_window,
                        temperature=0.0, training_mode=False)
        gen_text = r['text']
        gen_tids = [t for t in hv.encode(' ' + gen_text)
                    if t < config.bpe_limit and vg.tt[t] == 2 and vg._is_content_token(t)]
        matches = sum(1 for t in gen_tids if t in sent['tids'])
        total_matches += matches
        total_target += max(1, len(sent['tids']))
        match_pct = 100.0 * matches / max(1, len(sent['tids']))
        print(f'  EVAL E{epoch} S{i}: {matches}/{len(sent["tids"])} ({match_pct:.1f}%) gen="{gen_text[:60]}"')

    eval_pct = 100.0 * total_matches / total_target
    print(f'>>> Epoch {epoch}: eval {total_matches}/{total_target} ({eval_pct:.1f}%) lr={vg._svd_lr:.6f}')
    print()

    config.target_boost = original_target_boost  # restore before next training epoch
print(f'Done. Trained vectors: {len(vg._trained_vectors)}, momentum buffer: {len(vg._token_momentum)}')
