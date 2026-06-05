"""
LinguisticRuleExtractor v2 — правильная фильтрация:
  WORD_STARTER → WORD_CONT = аффиксация (внутри слова)
  WORD_STARTER → WORD_STARTER = синтаксис (между словами)
"""
import math, re
import numpy as np
from collections import defaultdict


class AffixRule:
    """Правило аффиксации: причина → следствие."""
    def __init__(self, name, description, condition_fn, score_fn,
                 cause_text="", effect_text="", weight=1.0):
        self.name = name
        self.description = description
        self.condition_fn = condition_fn
        self.score_fn = score_fn
        self.cause_text = cause_text
        self.effect_text = effect_text
        self.weight = weight
    def applicable(self, ctx):
        return self.condition_fn(ctx)
    def score(self, ctx, token_id):
        if not self.condition_fn(ctx):
            return 0.0
        return self.weight * self.score_fn(ctx, token_id)
    def explain(self, ctx, token_id=None):
        s = self.score(ctx, token_id) if token_id is not None else 0
        return (f"  [{self.name}] cause: {self.cause_text} | "
                f"+{s:.2f} because: {self.effect_text}")


class LinguisticRuleExtractor:
    """
    Извлекает ПРИЧИННО-СЛЕДСТВЕННЫЕ правила:
    
    Аффиксация (piw=0 → piw=1): P → R (префикс → корень)
      - Фильтр: prev is STARTER, curr is CONT
      - Пример: по(2) → яв(3) = "появ" (начало "появился")
    
    Синтаксис (word_n → word_{n+1}): W → W'
      - Фильтр: both are STARTER
      - Пример: по(2) → лесу(2) = "по лесу" (предлог + существительное)
    """
    
    def __init__(self, heads_obj, tokenizer):
        self.heads = heads_obj
        self.hv = tokenizer
        self.csr = heads_obj.log_prob_csr
        self.V = min(4096, self.csr.shape[0])
        self.tt = tokenizer.token_type  # type shortcut
        
        self.affixation_rules = []   # piw=0 → piw=1 (word-internal)
        self.syntax_rules = []       # wn → wn+1 (cross-word)
        self.affix_profiles = {}     # tid -> {next_tid: prob} same-word
        self.syntax_profiles = {}    # tid -> {next_tid: prob} new-word
    
    def analyze(self):
        print("=" * 60)
        print("LINGUISTIC RULE EXTRACTOR (cause→effect analysis)")
        print("=" * 60)
        
        V, csr, tt = self.V, self.csr, self.tt
        
        # Step 1: Separate affixation from syntax transitions
        print("\n1. Separating affixation vs syntax transitions...")
        
        for tid in range(156, V):
            if tid >= len(tt) or tt[tid] != 2:  # only WORD_STARTER at piw=0
                continue
            
            text = self.hv.decode([tid]).strip()
            if not text:
                continue
            
            row = csr[tid].tocoo()
            
            for col, prob in zip(row.col, row.data):
                col = int(col)
                if col >= V or col >= len(tt):
                    continue
                
                p = float(prob)
                
                if tt[col] == 3:
                    # WORD_CONT = same word continuation = AFFIXATION
                    if tid not in self.affix_profiles:
                        self.affix_profiles[tid] = {}
                    self.affix_profiles[tid][col] = p
                elif tt[col] == 2:
                    # WORD_STARTER = new word = SYNTAX
                    if tid not in self.syntax_profiles:
                        self.syntax_profiles[tid] = {}
                    self.syntax_profiles[tid][col] = p
        
        # Count
        n_affix = sum(len(p) for p in self.affix_profiles.values())
        n_syntax = sum(len(p) for p in self.syntax_profiles.values())
        print(f"  Affixation transitions (same-word): {n_affix}")
        print(f"  Syntax transitions (cross-word): {n_syntax}")
        
        # Step 2: Find true prefixes (productive word-initial tokens)
        print("\n2. Finding productive prefixes...")
        
        prefix_scores = []
        for tid, profile in self.affix_profiles.items():
            n_roots = len(profile)
            if n_roots < 3:
                continue
            text = self.hv.decode([tid]).strip()
            # Normalize: sum of transition probabilities
            total = sum(math.exp(p) for p in profile.values())
            prefix_scores.append((tid, text, n_roots, total))
        
        prefix_scores.sort(key=lambda x: -x[2])
        
        print(f"\n  TOP 20 PREFIXES (most productive):")
        for tid, text, n_roots, total in prefix_scores[:20]:
            top_roots = sorted(self.affix_profiles[tid].items(), 
                              key=lambda x: -x[1])[:5]
            roots_text = []
            for rt, rp in top_roots:
                try:
                    rt_text = self.hv.decode([rt]).strip()
                    full = self.hv.decode([tid, rt]).strip()
                    roots_text.append(f"{rt_text}({full})")
                except:
                    roots_text.append(f"#{rt}")
            print(f"  '{text:10s}' (tid={tid:4d}) → {n_roots} roots")
            print(f"    e.g.: {', '.join(roots_text)}")
        
        # Step 3: Extract affixation rules from prefix patterns
        print("\n3. Extracting affixation rules...")
        
        for tid, text, n_roots, total in prefix_scores[:15]:
            profile = self.affix_profiles[tid]
            profile_normalized = {k: math.exp(v)/max(total, 1e-10) 
                                 for k, v in profile.items()}
            
            # Create a rule with captured tid and profile
            rule = self._make_affix_rule(tid, text, profile)
            if rule:
                self.affixation_rules.append(rule)
        
        print(f"  Generated {len(self.affixation_rules)} affixation rules")
        
        # Step 4: Extract syntax rules (cross-word transitions)
        print("\n4. Extracting syntax rules...")
        
        for tid in list(self.syntax_profiles.keys())[:20]:
            profile = self.syntax_profiles[tid]
            if len(profile) < 3:
                continue
            text = self.hv.decode([tid]).strip()
            
            rule = self._make_syntax_rule(tid, text, profile)
            if rule:
                self.syntax_rules.append(rule)
        
        print(f"  Generated {len(self.syntax_rules)} syntax rules")
        
        return self.affixation_rules + self.syntax_rules
    
    def _make_affix_rule(self, tid, text, profile):
        """Create a rule: prefix P → root R should follow affixation pattern."""
        # Sort roots by probability
        sorted_roots = sorted(profile.items(), key=lambda x: -x[1])[:10]
        prob_dict = dict(sorted_roots)
        total_p = sum(math.exp(v) for _, v in sorted_roots)
        
        def condition_fn(ctx, _tid=tid):
            # Applies when prev token is this prefix
            return (ctx.get('pos_in_word', -1) == 0 and 
                    ctx.get('prev_token_id') == _tid)
        
        def score_fn(ctx, token_id, _pd=prob_dict, _tp=total_p):
            # Boost if token_id is a known root for this prefix
            p = _pd.get(token_id)
            if p is not None:
                return (math.exp(p) / max(_tp, 1e-10)) * 5.0
            return -1.0
        
        root_texts = [self.hv.decode([c]).strip() 
                     for c, _ in sorted_roots[:5]]
        
        return AffixRule(
            name=f"affix_{text}",
            description=f"'{text}' at piw=0 → word continues with root",
            condition_fn=condition_fn,
            score_fn=score_fn,
            cause_text=f"prefix '{text}' at start of word",
            effect_text=f"root continues with {root_texts}",
            weight=0.7
        )
    
    def _make_syntax_rule(self, tid, text, profile):
        """Create a rule: word W → next word W' should follow syntax pattern."""
        sorted_succ = sorted(profile.items(), key=lambda x: -x[1])[:10]
        prob_dict = dict(sorted_succ)
        total_p = sum(math.exp(v) for _, v in sorted_succ)
        
        def condition_fn(ctx, _tid=tid):
            # Applies when prev token is this word
            return (ctx.get('pos_in_word', -1) == 0 and 
                    ctx.get('prev_token_id') == _tid and
                    (ctx.get('flags', 0) >> 1) & 1)  # word end
        
        def score_fn(ctx, token_id, _pd=prob_dict, _tp=total_p):
            p = _pd.get(token_id)
            if p is not None:
                return (math.exp(p) / max(_tp, 1e-10)) * 5.0
            return -1.0
        
        succ_texts = [self.hv.decode([c]).strip() 
                     for c, _ in sorted_succ[:5]]
        
        return AffixRule(
            name=f"syntax_{text}",
            description=f"'{text}' → next word from {succ_texts}",
            condition_fn=condition_fn,
            score_fn=score_fn,
            cause_text=f"word '{text}' completed",
            effect_text=f"next word from distribution: {succ_texts}",
            weight=0.5
        )
    
    def full_formation_chain(self, prefix_tid, max_depth=3):
        """Show full word formation chain: prefix → root → suffix..."""
        if prefix_tid not in self.affix_profiles:
            return f"No affixation data for token {prefix_tid}"
        
        text = self.hv.decode([prefix_tid]).strip()
        print(f"\n--- Word formation from '{text}' ---")
        
        # Level 1: prefix → roots
        roots = self.affix_profiles[prefix_tid]
        for rt, rp in sorted(roots.items(), key=lambda x: -x[1])[:8]:
            rt_text = self.hv.decode([rt]).strip()
            full = self.hv.decode([prefix_tid, rt]).strip()
            pct = math.exp(rp) * 100
            print(f"  {text} + {rt_text:10s} = \"{full:15s}\" ({pct:.1f}%)")
            
            if max_depth >= 2:
                # Level 2: root → continuations
                if rt < self.csr.shape[0]:
                    row = self.csr[rt].tocoo()
                    for col2, p2 in zip(row.col, row.data):
                        col2 = int(col2)
                        if col2 >= 4096:
                            continue
                        if col2 < len(self.tt) and self.tt[col2] == 3:
                            t2 = self.hv.decode([col2]).strip()
                            full2 = self.hv.decode([prefix_tid, rt, col2]).strip()
                            pct2 = math.exp(p2) * 100
                            print(f"    + {t2:10s} = \"{full2:15s}\" ({pct2:.1f}%)")
                            if max_depth >= 3:
                                if col2 < self.csr.shape[0]:
                                    row3 = self.csr[col2].tocoo()
                                    for col3, p3 in zip(row3.col, row3.data):
                                        col3 = int(col3)
                                        if col3 >= 4096:
                                            continue
                                        if col3 < len(self.tt) and self.tt[col3] in (2, 3):
                                            t3 = self.hv.decode([col3]).strip()
                                            full3 = self.hv.decode([prefix_tid, rt, col2, col3]).strip()
                                            pct3 = math.exp(p3) * 100
                                            print(f"      + {t3:10s} = \"{full3:15s}\" ({pct3:.1f}%)")
                                        break
                            break
                    break
