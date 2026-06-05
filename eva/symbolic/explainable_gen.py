"""
Explainable Generation — каждый выбор токена объясняется правилами.
"""
import sys, math
import numpy as np
from collections import defaultdict

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.generation_loop import GenerationLoop, apply_masks, select_token, V, BOS, EOS


class ExplainableGenerator(GenerationLoop):
    """
    Генератор с объяснением каждого шага.
    
    Каждый выбранный токен сопровождается:
    - Какое правило его выбрало
    - Какая концептуальная ассоциация сработала
    - Какой head дал наибольший вклад
    """
    
    def __init__(self, heads_obj, rule_engine, assoc_graph=None,
                 transformer=None, max_tokens=200, device=None):
        super().__init__(heads_obj, transformer, None, max_tokens, device)
        self.rule_engine = rule_engine
        self.ag = assoc_graph
        self.explanations = []  # история объяснений
    
    def generate(self, temperature=0.0, seed=None, return_compact=False,
                 explain=False):
        if seed is not None:
            np.random.seed(seed)
        
        self.explanations = []
        tokens = [BOS]
        
        while len(tokens) < self.max_tokens:
            meta = self.vocab.metadata_from_ids(tokens)
            ctx = dict(meta[-1])
            ctx['context_tokens'] = tokens[-4:-1] if len(tokens) > 1 else tokens
            
            # --- Concept activation from prev token ---
            concept_act = None
            if self.ag and len(tokens) > 1:
                prev = tokens[-1]
                if prev < 4096:
                    concept_act = self.ag.activate(prev, max_depth=2, decay=0.5)
            
            # --- Compute scores with rules ---
            scores, rule_explanations = self.rule_engine.compute_scores(ctx, concept_act)
            
            # --- Apply masks ---
            scores = apply_masks(scores, tokens, ctx['word_num'], self.vocab.token_type)
            
            # --- Select ---
            next_tok = select_token(scores, temperature)
            
            if next_tok == EOS:
                tokens.append(EOS)
                break
            
            tokens.append(next_tok)
            
            if explain:
                self._record_explanation(next_tok, ctx, scores, 
                                          rule_explanations, concept_act)
        
        result = {'tokens': tokens}
        if explain:
            result['explanations'] = self.explanations
        return result
    
    def _record_explanation(self, token_id, ctx, scores, 
                             rule_explanations, concept_act):
        """Записать объяснение для шага генерации."""
        try:
            text = self.vocab.decode([token_id]).strip()
        except:
            text = str(token_id)
        
        # Top-3 reasons this token was chosen
        reasons = []
        
        # 1. Head contributions
        try:
            hs = self.heads.individual_scores(ctx)
            head_names = ['morph', 'syntax', 'transition', 'concept', 'contra', 'semantic']
            head_contribs = []
            for i in range(6):
                val = hs[i, token_id] if token_id < hs.shape[1] else -1e9
                if val > -10:
                    head_contribs.append((head_names[i], val))
            head_contribs.sort(key=lambda x: -x[1])
            for name, val in head_contribs[:3]:
                reasons.append(f"head:{name}={val:.2f}")
        except:
            pass
        
        # 2. Rule contributions
        if token_id in rule_explanations:
            for rule_name in rule_explanations[token_id]:
                reasons.append(f"rule:{rule_name}")
        
        # 3. Concept contributions
        if concept_act and self.ag:
            for node_id, energy in concept_act.items():
                if self.ag.L1_OFFSET <= node_id < self.ag.L2_OFFSET:
                    members = self.ag.concept_members(node_id)
                    if token_id in members:
                        cname = self.ag.concept_name(node_id)
                        reasons.append(f"concept:{cname}={energy:.2f}")
        
        self.explanations.append({
            'step': len(self.explanations),
            'token_id': int(token_id),
            'text': text,
            'reasons': reasons[:5],
            'pos_in_word': int(ctx.get('pos_in_word', -1)),
            'word_num': int(ctx.get('word_num', -1)),
        })
    
    def print_explanations(self, result):
        """Вывести генерацию с объяснениями."""
        tokens = result['tokens']
        text = self.decode_tokens(tokens)
        
        print(f"\nГенерация: {text}\n")
        print("Потокосемантическая трассировка:\n")
        
        for exp in result.get('explanations', []):
            reasons_str = ', '.join(exp['reasons'])
            print(f"  [{exp['step']:3d}] w{exp['word_num']:3d} "
                  f"piw={exp['pos_in_word']:2d} "
                  f"'{exp['text']:10s}' | {reasons_str}")
        
        # Статистика
        reasons_flat = [r for exp in result['explanations'] 
                        for r in exp['reasons']]
        
        from collections import Counter
        head_counts = Counter(r for r in reasons_flat if r.startswith('head:'))
        rule_counts = Counter(r for r in reasons_flat if r.startswith('rule:'))
        concept_counts = Counter(r for r in reasons_flat if r.startswith('concept:'))
        
        print(f"\nСтатистика причин:")
        if head_counts:
            print(f"  Heads: {dict(head_counts.most_common(3))}")
        if rule_counts:
            print(f"  Rules: {dict(rule_counts.most_common(3))}")
        if concept_counts:
            print(f"  Concepts: {dict(concept_counts.most_common(3))}")
