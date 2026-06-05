"""
MorphologicalRuleAnalyzer — извлекает ПРАВИЛА морфологии и синтаксиса из heads.
Не просто статистика, а выведенные закономерности:
- Какие аффиксы с какими корнями сочетаются
- Какие синтаксические конструкции допустимы
- Какие переходы между частями речи обязательны

Результат: RuleEngine — компонует веса из правил + данных.
"""
import math
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional


class MorphologicalRule:
    """Одно морфологическое правило с понятным объяснением."""
    
    def __init__(self, name, description, condition_fn, score_fn, weight=1.0):
        self.name = name
        self.description = description
        self.condition_fn = condition_fn  # ctx -> bool (правило применимо?)
        self.score_fn = score_fn          # (ctx, token_id) -> score_contribution
        self.weight = weight              # насколько правило жёсткое (0..1)
    
    def applicable(self, ctx):
        return self.condition_fn(ctx)
    
    def score(self, ctx, token_id):
        if not self.condition_fn(ctx):
            return 0.0
        return self.weight * self.score_fn(ctx, token_id)
    
    def explain(self, ctx, token_id=None):
        parts = [f"Rule: {self.name}"]
        parts.append(f"  Why: {self.description}")
        if token_id is not None:
            s = self.score(ctx, token_id)
            parts.append(f"  Score: {s:.3f}")
        return '\n'.join(parts)


class MorphologicalRuleAnalyzer:
    """
    Анализирует heads-данные и извлекает правила:
    
    Уровень 1: Позиции в слове (piw)
      - piw=0: PREFIXES и STEM_STARTS
      - piw=1: STEM продолжение (корни)
      - piw=2+: SUFFIXES и окончания
    
    Уровень 2: Аффиксные цепочки
      - prefix(X) + root(Y) + suffix(Z) = слово
      - Какие комбинации валидны
    
    Уровень 3: Части речи
      - Какие piw-паттерны соответствуют каким POS
      - noun: stem + case_suffix
      - verb: prefix + stem + tense_suffix + person_suffix
    
    Уровень 4: Синтаксис
      - transition POS_i → POS_j
      - Какие структуры обязательны (subject → verb)
    """
    
    def __init__(self, morph_logprob, syntax_logprob, log_prob_csr, tokenizer):
        """
        morph_logprob: dict piw->ndarray(V,) — token log probs per position-in-word
        syntax_logprob: dict wn->ndarray(V,) — token log probs per word-number
        log_prob_csr: sparse matrix (V, V) — transition log probs
        tokenizer: HierarchicalVocab or similar with .token_type and .decode
        """
        self.csr = log_prob_csr
        self.tok = tokenizer
        
        # Convert dicts to arrays if needed
        if isinstance(morph_logprob, dict):
            piws = sorted(morph_logprob.keys())
            self.max_piw = max(piws)
            arr = np.zeros((self.max_piw + 1, morph_logprob[0].shape[0]), dtype=np.float32)
            for piw, v in morph_logprob.items():
                arr[piw] = v
            self.morph_logprob = arr
        else:
            self.morph_logprob = morph_logprob
            self.max_piw = self.morph_logprob.shape[0] - 1
        
        if isinstance(syntax_logprob, dict):
            wns = sorted(syntax_logprob.keys())
            self.max_wn = max(wns)
            first_val = syntax_logprob[wns[0]]
            arr2 = np.zeros((self.max_wn + 1, first_val.shape[0]), dtype=np.float32)
            for wn, v in syntax_logprob.items():
                arr2[wn] = v
            self.syntax_logprob = arr2
        else:
            self.syntax_logprob = syntax_logprob
            self.max_wn = self.syntax_logprob.shape[0] - 1
        
        self.V = self.morph_logprob.shape[1]
        
        # Результаты анализа
        self.affix_profiles = {}   # piw -> {token_id: {next_piw_token_dist}}
        self.prefix_candidates = []   # tokens that appear at piw=0
        self.suffix_candidates = {}   # piw -> tokens common at that position
        self.root_candidates = set()  # tokens common at piw=1
        
        self.pos_patterns = {}    # piw_sequence -> pos_label
        self.syntax_patterns = [] # (prev_pos, current_pos) -> frequency
        
        self.rules = []  # extracted MorphologicalRules
        
    def analyze(self):
        """Главный метод: извлекает все правила из данных."""
        print("=" * 60)
        print("MorphologicalRuleAnalyzer: извлечение правил из heads")
        print("=" * 60)
        
        self._analyze_positional_distributions()
        self._analyze_affix_combinations()
        self._analyze_syntactic_patterns()
        self._build_rules()
        
        return self
    
    def _analyze_positional_distributions(self):
        """Анализ распределений по позициям в слове."""
        V = self.V
        
        for piw in range(min(6, self.max_piw + 1)):
            logprobs = self.morph_logprob[piw]
            # Токены с вероятностью > threshold
            threshold = np.max(logprobs) - 5.0  # top exp(-5)
            top_mask = logprobs > threshold
            top_tokens = np.where(top_mask)[0]
            
            # Фильтруем только WORD_STARTER для piw=0, но для piw>0 берём все
            if piw == 0:
                valid = [t for t in top_tokens if t < len(self.tok.token_type) 
                         and self.tok.token_type[t] == 2]
            else:
                valid = [t for t in top_tokens if t < len(self.tok.token_type)]
            
            texts = []
            for t in valid[:10]:
                try:
                    texts.append(self.tok.decode([t]).strip())
                except:
                    texts.append(str(t))
            
            entropy = self._compute_entropy(logprobs)
            print(f"\n  piw={piw}: {len(valid)} high-prob tokens, "
                  f"H={entropy:.2f}")
            print(f"    e.g. {texts}")
            
            if piw == 0:
                self.prefix_candidates = valid
            else:
                self.suffix_candidates[piw] = valid
        
        # piw=1 = root candidates
        piw1_logprobs = self.morph_logprob[1]
        threshold = np.max(piw1_logprobs) - 4.0
        self.root_candidates = set(np.where(piw1_logprobs > threshold)[0])
        
        print(f"\n  Root candidates (piw=1): {len(self.root_candidates)}")
    
    def _compute_entropy(self, logprobs):
        """Entropy of a categorical distribution."""
        probs = np.exp(logprobs)
        probs = probs / (probs.sum() + 1e-10)
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def _analyze_affix_combinations(self):
        """Анализ: какие аффиксы с какими корнями сочетаются."""
        print("\n--- Affix combinations ---")
        
        # Для piw=0: какие tokens чаще всего идут на piw=1?
        for tid in self.prefix_candidates[:20]:
            tid = int(tid)
            if tid >= self.csr.shape[0]:
                continue
            
            # Проверяем куда переходит этот токен
            row = self.csr[tid].tocoo()
            successors = [(col, data) for col, data in zip(row.col, row.data)
                         if col < self.V]
            successors.sort(key=lambda x: -x[1])
            
            if successors:
                top_succ = successors[:5]
                top_texts = []
                for s, p in top_succ:
                    try:
                        txt = self.tok.decode([s]).strip()
                        top_texts.append(f"'{txt}'({p:.2f})")
                    except:
                        top_texts.append(f"#{s}({p:.2f})")
                
                try:
                    txt = self.tok.decode([tid]).strip()
                except:
                    txt = str(tid)
                print(f"  '{txt}' (piw=0) → {top_texts}")
    
    def _analyze_syntactic_patterns(self):
        """Анализ синтаксических паттернов."""
        print("\n--- Syntactic patterns ---")
        
        # For each word position, what tokens are likely?
        for wn in range(min(5, self.max_wn + 1)):
            logprobs = self.syntax_logprob[wn]
            threshold = np.max(logprobs) - 5.0
            top_mask = logprobs > threshold
            top_tokens = np.where(top_mask)[0]
            
            valid = [t for t in top_tokens if t < len(self.tok.token_type)
                     and self.tok.token_type[t] == 2]
            
            texts = []
            for t in valid[:8]:
                try:
                    texts.append(self.tok.decode([t]).strip())
                except:
                    texts.append(str(t))
            
            print(f"  wn={wn}: {texts}")
    
    def _build_rules(self):
        """Строим конкретные правила из найденных паттернов."""
        print("\n--- Building rules ---")
        
        V = self.V
        
        # === Rule 1: Prefix-verb compatibility ===
        # Если токен на piw=0 — это часто префикс (по-, при-, за-, вы- и т.д.),
        # то слово вероятно будет глаголом
        prefix_ids = []
        for tid in self.prefix_candidates[:30]:
            tid = int(tid)
            try:
                t = self.tok.decode([tid]).strip()
                # Russian prefixes are short (2-4 chars) and end with vowel or hard sign
                if len(t) <= 4 and any(t.endswith(c) for c in ['о', 'а', 'е', 'и', 'у', 'ъ', 'ь', 'й']):
                    prefix_ids.append(tid)
            except:
                pass
        
        print(f"  Prefix candidates: {len(prefix_ids)}")
        
        # Создаём правило: "После префикса ожидается глагольный корень"
        def prefix_rule_condition(ctx):
            return ctx.get('pos_in_word', -1) == 1  # после префикса (piw=1)
        
        V_actual = V
        def prefix_rule_score(ctx, token_id):
            # Boost tokens that are common roots after prefixes
            if token_id < V_actual and token_id in self.root_candidates:
                return 1.0
            return 0.0
        
        self.rules.append(MorphologicalRule(
            name="prefix_expects_verb_root",
            description="After a prefix (piw=0), the next token (piw=1) "
                       "should be a verb root",
            condition_fn=prefix_rule_condition,
            score_fn=prefix_rule_score,
            weight=0.7
        ))
        
        # === Rule 2: First word of sentence capitalisation ===
        # wn=0 должен быть WORD_STARTER с заглавной
        def capital_rule_condition(ctx):
            return ctx.get('word_num', -1) == 0
        
        wn0_logprobs = self.syntax_logprob[0]
        top_at_wn0 = set(np.where(wn0_logprobs > np.max(wn0_logprobs) - 3.0)[0])
        
        def capital_rule_score(ctx, token_id):
            if token_id < V_actual and token_id in top_at_wn0:
                return 1.0
            return 0.0
        
        self.rules.append(MorphologicalRule(
            name="sentence_start",
            description="First word of sentence uses specific tokens",
            condition_fn=capital_rule_condition,
            score_fn=capital_rule_score,
            weight=0.8
        ))
        
        # === Rule 3: Mid-word morph consistency ===
        # На piw=1 (после начала слова), токен должен быть WORD_CONT или BYTE
        def midword_rule_condition(ctx):
            return ctx.get('pos_in_word', -1) >= 1
        
        def midword_rule_score(ctx, token_id):
            if token_id < len(self.tok.token_type):
                tt = self.tok.token_type[token_id]
                # WORD_CONT (3) or BYTE (1) — продолжение слова
                if tt in (1, 3):
                    return 1.0
                elif tt == 2:  # WORD_STARTER — начинает новое слово
                    return -2.0  # штраф
            return 0.0
        
        self.rules.append(MorphologicalRule(
            name="midword_continuation",
            description="In the middle of a word, only WORD_CONT or BYTE tokens",
            condition_fn=midword_rule_condition,
            score_fn=midword_rule_score,
            weight=0.9
        ))
        
        # === Rule 4: Transition coherence ===
        # Если предыдущий токен скорее всего PUNCTUATION → следующий должен быть
        # WORD_STARTER (слово) или другой punct
        def punct_rule_condition(ctx):
            prev = ctx.get('prev_token_id', -1)
            if prev < 0 or prev >= V_actual:
                return False
            if prev < len(self.tok.token_type):
                return self.tok.token_type[prev] == 2  # WORD_STARTER = мог быть punct
            return False
        
        def punct_rule_score(ctx, token_id):
            if token_id < len(self.tok.token_type):
                if self.tok.token_type[token_id] == 2:
                    return 1.0
            return 0.0
        
        self.rules.append(MorphologicalRule(
            name="word_after_punct",
            description="After punctuation, start a new word",
            condition_fn=punct_rule_condition,
            score_fn=punct_rule_score,
            weight=0.6
        ))
        
        print(f"  Total rules: {len(self.rules)}")


class RuleEngine:
    """
    Compositional weight engine.
    Веса = Σ правил + head scores + concept activation.
    """
    
    def __init__(self, heads_obj, morph_analyzer, assoc_graph=None):
        self.heads = heads_obj
        self.rules = morph_analyzer.rules if morph_analyzer else []
        self.ag = assoc_graph
        self.V = 4101
    
    def compute_scores(self, ctx, concept_activation=None):
        """
        Композиционный счёт: rules + heads + concepts.
        
        Returns:
            scores: (V,) вектор логов
            explanations: Dict[int, str] — почему каждый токен получил свой счёт
        """
        V = self.V
        
        # 1. Head scores (базовая статистика)
        head_scores = np.zeros((6, V), dtype=np.float32)
        try:
            head_scores = self.heads.individual_scores(ctx)
        except:
            pass
        
        # Default weights
        weights = np.array([1.0, 1.0, 3.0, 0.5, 0.2, 0.5], dtype=np.float32)
        
        base_scores = np.dot(weights, head_scores)  # (V,)
        
        # 2. Rule scores
        rule_scores = np.zeros(V, dtype=np.float32)
        rule_explanations = {}
        
        for rule in self.rules:
            if rule.applicable(ctx):
                for tid in range(min(V, 4096)):
                    s = rule.score(ctx, tid)
                    if s != 0.0:
                        rule_scores[tid] += s
                        if tid not in rule_explanations:
                            rule_explanations[tid] = []
                        rule_explanations[tid].append(rule.name)
        
        # 3. Concept activation (if available)
        concept_scores = np.zeros(V, dtype=np.float32)
        if concept_activation and self.ag:
            for node_id, energy in concept_activation.items():
                if self.ag.L1_OFFSET <= node_id < self.ag.L2_OFFSET:
                    # This is a concept — boost its member tokens
                    members = self.ag.concept_members(node_id)
                    for tid in members:
                        if tid < V:
                            concept_scores[tid] += energy * 2.0
                elif node_id < 4096:
                    # Direct token activation
                    concept_scores[node_id] += energy
        
        # 4. Compose
        final_scores = base_scores + rule_scores * 3.0 + concept_scores * 2.0
        
        return final_scores, rule_explanations
    
    def explain_token(self, token_id, ctx, concept_activation=None):
        """Объяснить, почему конкретный токен получил свой счёт."""
        scores, explanations = self.compute_scores(ctx, concept_activation)
        
        if token_id >= len(scores):
            return f"Token {token_id}: out of range"
        
        total = scores[token_id]
        reasons = []
        
        # Head scores
        try:
            head_scores = self.heads.individual_scores(ctx)
            head_total = sum(head_scores[:, token_id])
            reasons.append(f"  heads: {head_total:.3f}")
        except:
            pass
        
        # Rule explanations
        if token_id in explanations:
            for rule_name in explanations[token_id]:
                reasons.append(f"  rule '{rule_name}': +3.0")
        
        # Concept boost
        if concept_activation:
            for node_id, energy in concept_activation.items():
                if self.ag and self.ag.L1_OFFSET <= node_id < self.ag.L2_OFFSET:
                    members = self.ag.concept_members(node_id)
                    if token_id in members:
                        reasons.append(f"  concept '{self.ag.concept_name(node_id)}': "
                                      f"+{energy*2.0:.3f}")
        
        try:
            text = ctx.get('tokenizer', lambda x: str(x))(token_id)
        except:
            text = str(token_id)
        
        result = [f"Token {token_id} ('{text}'): total={total:.3f}"]
        result.extend(reasons)
        return '\n'.join(result)
