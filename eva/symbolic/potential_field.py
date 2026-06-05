"""
PotentialField — генерация в потенциальном поле.
Принцип:
  1. Правила = жёсткие стены (отсекают невозможное)
  2. Потенциалы = мягкие веса внутри разрешённого
  3. Выбор = всегда в пределах правил
"""
import sys, math
import numpy as np
from collections import defaultdict

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.generation_loop import apply_masks, select_token, V, BOS, EOS

# Блокируемые токены (рефлекторно, вне правил)
REPLACEMENT_TOKENS = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
    130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
    140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 228, 229, 230, 231, 232, 233, 234, 235, 236,
    237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
    247, 248, 249, 250, 251, 252, 253, 254, 255, 256,
    257, 258, 259, 260, 261, 262, 272, 274, 277, 298,
    299, 334, 1498}
IGNORED = {0, 1, 2, 4, 5}  # PAD, UNK, BOS, SEP, MASK
BANNED = {3}  # EOS — только когда разрешено


class PotentialField:
    """
    Потенциальное поле: правила → стены, heads + концепты → потенциалы.
    
    generate_step(ctx, rules, heads, assoc_graph):
      1. rules.filter(ctx) → valid_tokens (булева маска)
      2. heads.scores(ctx) → raw_scores
      3. assoc_graph.activate(ctx.prev_token) → concept_boost
      4. raw_scores + concept_boost = potential
      5. potential[~valid] = -inf  (стены!)
      6. select from potential
    """
    
    def __init__(self, heads_obj, rules, assoc_graph=None, tokenizer=None):
        self.heads = heads_obj
        self.rules = rules  # list of AffixRule
        self.ag = assoc_graph
        self.hv = tokenizer or HierarchicalVocab()
        self.tt = self.hv.token_type
        self.V = min(4101, max(len(self.tt) + 1, 4101))
        self.repeat_window = 4
        # Use DISCRIMINATIVE profile instead of all concept members
        self.concept_profile_topk = 20
        self.concept_profile_minweight = 0.3
    
    def _concept_tokens(self, cid):
        """
        Возвращает токены концепта с дискриминативным весом.
        Если аггрегировано всё — возвращает [(tid, weight)].
        """
        if self.ag is None:
            return []
        profile = self.ag.get_profile(cid, top_k=self.concept_profile_topk,
                                       min_weight=self.concept_profile_minweight)
        if profile:
            return profile
        # Fallback: all members
        return self.ag.cid_to_tids.get(cid, [])
    
    def valid_mask(self, ctx):
        """
        Правила → стена. Возвращает булеву маску (V,) где True = разрешено.
        """
        mask = np.ones(self.V, dtype=bool)
        
        # 1. Всегда блокируем: спецтокены, replacement
        for t in IGNORED | BANNED | REPLACEMENT_TOKENS:
            if t < self.V:
                mask[t] = False
        for t in range(4096, self.V):
            mask[t] = False
        
        pos_in_word = ctx.get('pos_in_word', -1)
        word_num = ctx.get('word_num', -1)
        flags = ctx.get('flags', 0)
        is_word_end = (flags >> 1) & 1
        is_special = (flags >> 5) & 1
        
        # 2. ПРАВИЛО: На piw=0 (начало слова) — только WORD_STARTER
        if pos_in_word == 0:
            for tid in range(self.V):
                if tid < len(self.tt) and self.tt[tid] != 2:
                    mask[tid] = False
            # EOS разрешён только после min слов
            if word_num < 3:
                mask[3] = False
        
        # 3. ПРАВИЛО: На piw>=1 (внутри слова) — только WORD_CONT или BYTE
        if pos_in_word >= 1:
            for tid in range(self.V):
                if tid < len(self.tt) and self.tt[tid] == 2:
                    mask[tid] = False  # нельзя начать новое слово
                if tid < len(self.tt) and self.tt[tid] == 0:
                    mask[tid] = False  # нельзя спецтокены
            # EOS нельзя внутри слова
            mask[3] = False
        
        # 4. ПРАВИЛО: На конце слова — разрешён EOS
        if is_word_end and word_num >= 2:
            mask[3] = True
        
        # 5. ПРАВИЛО: Нельзя повторять один токен >3 раз подряд
        prev_tokens = ctx.get('context_tokens', [])
        if len(prev_tokens) >= 3:
            last = prev_tokens[-1]
            # Count consecutive repeats
            count = 1
            for t in reversed(prev_tokens[:-1]):
                if t == last:
                    count += 1
                else:
                    break
            if count >= 3 and last < self.V:
                mask[last] = False  # блокируем повтор
        
        # 6. Правила аффиксации (от AffixRule)
        for rule in self.rules:
            if not rule.applicable(ctx):
                continue
            # Если правило жёсткое (weight близок к 1), блокируем несоответствия
            if rule.weight > 0.8:
                for tid in range(self.V):
                    if mask[tid] and rule.score(ctx, tid) <= -1:
                        mask[tid] = False
        
        # 7. Всегда разрешаем EOS если он был разрешён ранее
        if word_num < 3:
            mask[3] = False
        
        return mask
    
    def compute_potential(self, ctx, valid_mask, concept_activation=None,
                          concept_history=None, step=0, target_concept=None):
        """
        Потенциалы внутри разрешённого пространства.
        step: номер шага генерации.
        target_concept: int — целевой концепт (0..n_clusters-1).
          Если задан, токены из этого концепта получают сильный boost.
        """
        scores = np.full(self.V, -np.float32(-np.inf), dtype=np.float32)
        
        # 1. Head scores (базовый потенциал)
        try:
            head_scores = self.heads.individual_scores(ctx)  # (6, V)
            w = np.array([1.0, 1.0, 3.0, 0.5, 0.2, 0.5], dtype=np.float32)
            base = np.dot(w, head_scores)
            scores = base.copy()
        except:
            pass
        
        # 2. Target concept boost (дискриминативный: только high-profile токены)
        if target_concept is not None and self.ag:
            target_cid = 4096 + target_concept
            # Use discriminative profile (centroid-close tokens = essence)
            essence = self._concept_tokens(target_cid)
            if essence:
                for tid in essence:
                    if tid < self.V:
                        scores[tid] += 15.0  # сильный приоритет сущности
            else:
                # Fallback: boost all members
                for tid in self.ag.cid_to_tids.get(target_cid, []):
                    if tid < self.V:
                        scores[tid] += 8.0
        
        # 3. Concept activation (PMI association) — дискриминативные токены
        if concept_activation is not None and self.ag:
            for node_id, energy in concept_activation.items():
                if self.ag.L1_OFFSET <= node_id < self.ag.L2_OFFSET:
                    essence = self._concept_tokens(node_id)
                    if essence:
                        for tid in essence:
                            if tid < self.V:
                                scores[tid] += energy * 2.0
                    else:
                        for tid in self.ag.cid_to_tids.get(node_id, []):
                            if tid < self.V:
                                scores[tid] += energy * 1.0
                elif node_id < 4096:
                    scores[node_id] += energy * 0.5
        
        # 5. Concept repetition penalty
        if concept_history is not None and len(concept_history) >= 2:
            recent = set(concept_history[-self.repeat_window:])
            for ci in range(self.ag.n_clusters):
                if ci in recent:
                    for tid in self.ag.cid_to_tids.get(4096 + ci, []):
                        if tid < self.V:
                            scores[tid] -= 5.0
        
        # 6. Rule scores (мягкие потенциалы внутри правил)
        for rule in self.rules:
            if rule.applicable(ctx):
                for tid in np.where(valid_mask)[0]:
                    r = rule.score(ctx, int(tid))
                    if r > 0:
                        scores[tid] += r
        
        # 7. Стена: ~valid = -inf
        scores[~valid_mask] = -np.inf
        
        return scores
    
    def generate_step(self, ctx, concept_activation=None, concept_history=None,
                      step=0, target_concept=None):
        """Один шаг генерации: правила → стены, потенциалы → выбор."""
        valid = self.valid_mask(ctx)
        
        # Если всё заблокировано — форсируем EOS
        if not valid.any():
            return 3, {"reason": "all blocked, force EOS"}
        
        potential = self.compute_potential(ctx, valid, concept_activation, 
                                           concept_history, step=step,
                                           target_concept=target_concept)
        
        # Выбор из потенциального поля
        next_tok = select_token(potential, temperature=0.3)
        
        # Объяснение
        explanation = {"valid_tokens": int(valid.sum()), "chosen": int(next_tok)}
        try:
            explanation['chosen_text'] = self.hv.decode([next_tok]).strip()
        except:
            explanation['chosen_text'] = str(next_tok)
        
        # Почему выбран именно этот?
        reasons = []
        if concept_activation and self.ag:
            for nid, e in concept_activation.items():
                if self.ag.L1_OFFSET <= nid < self.ag.L2_OFFSET:
                    if next_tok in self.ag.concept_members(nid):
                        reasons.append(f"concept:{self.ag.concept_name(nid)}={e:.2f}")
        for rule in self.rules:
            if rule.applicable(ctx) and rule.score(ctx, next_tok) > 0:
                reasons.append(f"rule:{rule.name}")
        explanation['reasons'] = reasons
        
        return next_tok, explanation


class PotentialGenerator:
    """
    Полная генерация в потенциальном поле.
    Каждый шаг: правила → стена, heads + концепты → потенциал, выбор.
    """
    
    def __init__(self, heads_obj, rule_extractor, assoc_graph=None):
        rules = getattr(rule_extractor, 'affixation_rules', []) + \
                getattr(rule_extractor, 'syntax_rules', [])
        self.field = PotentialField(heads_obj, rules, assoc_graph)
        self.ag = assoc_graph
        self.hv = HierarchicalVocab()
        self.history = []
    
    def _token_to_concept(self, tid):
        """Convert token_id to concept_id (0..n_clusters-1, or -1)."""
        if self.ag is None or tid >= 4096 or tid < 0:
            return -1
        cid = self.ag.get_concept(tid)
        if cid is not None and self.ag.L1_OFFSET <= cid < self.ag.L2_OFFSET:
            return cid - self.ag.L1_OFFSET
        return -1
    
    def _find_type2_token(self, word):
        """
        Найти WORD_STARTER (type=2) токен для заданного слова.
        Возвращает token_id или None.
        """
        bpe_tokens = self.hv.encode(word)
        if not bpe_tokens:
            return None
        # Ищем type 2 токен для первого символа seed-слова
        first_char = self.hv.decode([bpe_tokens[0]]).strip()
        for t in range(4096):
            if t < len(self.hv.token_type) and self.hv.token_type[t] == 2:
                if self.hv.decode([t]).strip() == first_char:
                    return t
        return None
    
    def generate(self, max_tokens=100, seed=None, target_word=None):
        if seed is not None and isinstance(seed, (int, float)):
            np.random.seed(seed)
        
        self.history = []
        tokens = [2]  # BOS
        
        # Seed word (expand to type-2 tokens if needed)
        if seed is not None and isinstance(seed, str):
            tok = self._find_type2_token(seed)
            if tok is not None and tok < 4096:
                tokens.append(tok)
        
        concept_hist = []
        explanations = []
        step = 0
        
        # Target concept
        target_concept = None
        if target_word is not None and self.ag is not None:
            tok = self._find_type2_token(target_word)
            if tok is not None:
                tc = self.ag.get_concept(tok)
                if tc is not None and self.ag.L1_OFFSET <= tc < self.ag.L2_OFFSET:
                    target_concept = tc - self.ag.L1_OFFSET
                    print("Target concept: %d (%s)" % (target_concept, 
                          self.ag.concept_name(tc) if hasattr(self.ag, 'concept_name') else ''))
        
        while len(tokens) < max_tokens:
            meta = self.hv.metadata_from_ids(tokens)
            ctx = dict(meta[-1])
            ctx['context_tokens'] = tokens[-5:-1] if len(tokens) > 1 else tokens
            
            # Concept activation
            concept_act = None
            if self.ag and len(tokens) > 1:
                prev = tokens[-1]
                if prev < 4096:
                    concept_act = self.ag.activate(prev, max_depth=2, decay=0.5)
            
            # Generate step
            next_tok, explanation = self.field.generate_step(
                ctx, concept_activation=concept_act, 
                concept_history=concept_hist if concept_hist else None,
                step=step, target_concept=target_concept
            )
            step += 1
            
            tokens.append(next_tok)
            explanations.append(explanation)
            
            # Update concept history
            if self.ag and next_tok < 4096:
                nc = self._token_to_concept(next_tok)
                if nc >= 0:
                    concept_hist.append(nc)
            
            if next_tok == 3:
                break
        
        result = {
            'tokens': tokens,
            'text': self.hv.decode(tokens),
            'explanations': explanations,
        }
        return result
    
    def print_trace(self, result):
        """Показать трассу генерации."""
        print("\n=== ТРАССА ГЕНЕРАЦИИ (потенциальное поле) ===")
        print(f"Итог: {result['text']}\n")
        
        for i, exp in enumerate(result.get('explanations', [])):
            reasons = ', '.join(exp.get('reasons', [])) or 'heads'
            print(f"  [{i:3d}] '{exp['chosen_text']:10s}' "
                  f"| V={exp['valid_tokens']:4d} valid"
                  f" | {reasons}")
        print(f"\nТокенов: {len(result['tokens'])}")
