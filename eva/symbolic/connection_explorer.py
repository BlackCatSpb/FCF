"""
ConnectionExplorer — показывает что и как связано.
Принцип: "всегда есть связь" — любой выход прослеживается до входа.
"""
import sys, math
import numpy as np
from collections import defaultdict

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')


class ConnectionExplorer:
    """
    Исследователь связей. Берёт слово, показывает все цепочки:
    - Какие концепты активируются
    - Какие правила срабатывают
    - Какие ассоциации ведут к чему
    - Всё всегда прослеживается до seed
    """
    
    def __init__(self, heads_obj, assoc_graph, rule_engine, tokenizer):
        self.heads = heads_obj
        self.ag = assoc_graph
        self.rules = rule_engine
        self.hv = tokenizer
    
    def explore(self, seed_text, depth=3):
        """Полное исследование: от слова через все связи."""
        # 1. Find the token
        seed_tid = None
        for tid in range(156, 4096):
            if self.hv.decode([tid]).strip() == seed_text:
                seed_tid = tid
                break
        
        if seed_tid is None:
            # Partial match
            for tid in range(156, 4096):
                t = self.hv.decode([tid]).strip()
                if seed_text in t.lower():
                    seed_tid = tid
                    break
        
        if seed_tid is None:
            print(f"'{seed_text}' not found in vocabulary")
            return
        
        seed_text_actual = self.hv.decode([seed_tid]).strip()
        print(f"\n{'='*60}")
        print(f"ИССЛЕДОВАНИЕ: '{seed_text_actual}' (tid={seed_tid})")
        print(f"{'='*60}")
        
        cid = self.ag.get_concept(seed_tid)
        mid = self.ag.get_meta(cid) if cid else None
        
        print(f"\n1. ИЕРАРХИЯ:")
        print(f"   Токен: {seed_text_actual}")
        print(f"   Концепт: {self.ag.concept_name(cid) if cid else 'None'}")
        print(f"   Мета: {self.ag.meta_name(mid) if mid else 'None'}")
        
        # 2. PMI associations
        print(f"\n2. АССОЦИАЦИИ (PMI) от '{seed_text_actual}':")
        act = self.ag.activate(seed_tid, max_depth=depth, decay=0.5)
        
        # Show concepts sorted by activation
        concepts = [(n, e) for n, e in act.items() 
                    if self.ag.L1_OFFSET <= n < self.ag.L2_OFFSET]
        concepts.sort(key=lambda x: -x[1])
        
        for cn, ce in concepts[:8]:
            cn_name = self.ag.concept_name(cn)
            # Get tokens from this concept
            tokens_texts = []
            for t in self.ag.concept_members(cn)[:4]:
                tokens_texts.append(self.hv.decode([t]).strip())
            print(f"   → {cn_name:12s} (act={ce:.3f}) [{', '.join(tokens_texts)}]")
        
        # 3. Transition paths
        print(f"\n3. ПЕРЕХОДЫ (что следует за '{seed_text_actual}'):")
        if seed_tid < self.heads.log_prob_csr.shape[0]:
            row = self.heads.log_prob_csr[seed_tid].tocoo()
            trans = [(int(c), float(d)) for c, d in zip(row.col, row.data) if c < 4096]
            trans.sort(key=lambda x: -x[1])
            for col, prob in trans[:6]:
                text = self.hv.decode([col]).strip()
                full = self.hv.decode([seed_tid, col]).strip()
                pct = np.exp(prob) * 100
                print(f"   → {text:10s} ({pct:.1f}%) = \"{full}\"")
        
        # 4. Potential new words from associated concepts
        print(f"\n4. НОВЫЕ СЛОВА из ассоциированных концептов:")
        from eva.symbolic.constrained_decoder import ConstrainedDecoder
        decoder = ConstrainedDecoder(self.heads, self.ag)
        
        for cn, ce in concepts[:5]:
            if ce < 0.05:
                continue
            cn_name = self.ag.concept_name(cn)
            for _ in range(2):
                tokens = decoder.generate_with_target(
                    cn, context=[2], temperature=0.3, novelty=True)
                text = self.hv.decode(tokens).strip()
                if text and text != seed_text_actual:
                    tokens_list = ', '.join([
                        self.hv.decode([t]).strip() 
                        for t in self.ag.concept_members(cn)[:3]
                    ])
                    print(f"   [{cn_name:12s}] {text}")
                    break
        
        # 5. Applicable rules  
        print(f"\n5. ПРАВИЛА (извлечены из heads):")
        # Create a context where seed_tid is the prev_token
        meta = self.hv.metadata_from_ids([2, seed_tid])
        ctx = dict(meta[-1])
        
        applicable = 0
        for rule in self.rules.rules:
            if rule.applicable(ctx):
                applicable += 1
                print(f"   ✓ {rule.name}: {rule.description[:60]}")
        if applicable == 0:
            print(f"   (в этом контексте правила не применимы)")
        
        # 6. Connection graph: seed → association → concept → new word
        print(f"\n6. ГРАФ СВЯЗЕЙ (seed → ассоциация → слово):")
        # Pick top 3 associated concepts
        seen_words = set()
        for cn, ce in concepts[:3]:
            cn_name = self.ag.concept_name(cn)
            # Try to generate a word from this concept
            for _ in range(3):
                tokens = decoder.generate_with_target(
                    cn, context=[2], temperature=0.3, novelty=True)
                text = self.hv.decode(tokens).strip()
                if text and text not in seen_words and len(text) > 1:
                    seen_words.add(text)
                    print(f"   {seed_text_actual} → [{cn_name}] → \"{text}\"")
                    break
        
        print(f"\n{'='*60}")
        print(f"ИТОГО: '{seed_text_actual}' → {len(concepts)} ассоциаций "
              f"→ {len(seen_words)} новых слов")
        print(f"{'='*60}")
