"""
constrained_decoder — wraps GenerationLoop with concept constraints.
Позволяет генерацию, ограниченную концептами:
1. Контекст → ConceptTransition → следующий концепт
2. Выбираем неочевидный токен из этого концепта
3. Проверяем грамматическую корректность через heads
"""
import sys, math
from typing import Optional
import numpy as np

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.generation_loop import GenerationLoop, apply_masks, select_token, V, BOS, EOS

CONCEPT_BOOST = 3.0   # насколько бустим токены из целевого концепта
NOVELTY_TEMP = 0.7    # температура для выбора "неочевидного" токена внутри концепта


class ConstrainedDecoder(GenerationLoop):
    """GenerationLoop + ConceptGraph: генерация, ограниченная концептами."""

    def __init__(self, heads_obj, concept_graph, transformer=None,
                 max_tokens=200, device=None):
        super().__init__(heads_obj, transformer, None, max_tokens, device)
        self.cg = concept_graph
        self.target_concept = None   # текущий целевой концепт
        self.concept_history = []    # (cid, log_prob) для отслеживания
        self.novelty_mode = False    # если True — выбираем неочевидные токены

    def generate(self, temperature=0.0, seed=None, return_compact=False,
                 novelty=False, target_concept=None) -> list:
        """
        Генерация с концептуальными ограничениями.
        
        Args:
            novelty: если True, выбирает НЕ самый вероятный токен из концепта
            target_concept: если задан, принудительно генерировать в этом концепте
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.novelty_mode = novelty
        self.target_concept = target_concept
        self.concept_history = []
        
        tokens = [BOS]
        if return_compact:
            compact_frames = []
        
        while len(tokens) < self.max_tokens:
            meta = self.vocab.metadata_from_ids(tokens)
            ctx = dict(meta[-1])
            ctx['context_tokens'] = tokens[-4:-1] if len(tokens) > 1 else tokens
            word_num = ctx['word_num']
            
            # --- Concept prediction ---
            if self.target_concept is None and len(tokens) > 1:
                prev_tid = tokens[-1]
                transitions = self.cg.predict_next(prev_tid)
                if transitions:
                    # Выбираем топ-3 концепта 
                    top = transitions[:3]
                    if self.novelty_mode and len(top) > 1:
                        # Выбираем НЕ самый вероятный
                        chosen = top[np.random.randint(1, min(3, len(top)))]
                    else:
                        chosen = top[0]
                    self.target_concept = chosen[0]
                    self.concept_history.append(chosen)
                else:
                    self.target_concept = None
            
            # --- Веса ---
            weights = self._compute_weights(ctx)
            
            # --- Head scores ---
            scores = np.zeros(V, dtype=np.float32)
            try:
                head_scores = self.heads.individual_scores(ctx)
                scores = np.dot(weights, head_scores)
            except Exception:
                scores = np.ones(V, dtype=np.float32) * -1e9
                scores[EOS] = 0
            
            # --- Концептуальный буст ---
            if self.target_concept is not None:
                boost = self.cg.concept_scores(self.target_concept, V)
                scores = scores + boost * CONCEPT_BOOST
            
            # --- Применяем маски ---
            scores = apply_masks(scores, tokens, word_num, self.vocab.token_type)
            
            # --- Выбор токена ---
            if self.novelty_mode and self.target_concept is not None:
                # В novelty-режиме: выбираем из концепта с повышенной температурой
                members = self.cg.get_members(self.target_concept)
                if members:
                    # Сначала фильтруем маскированные токены
                    member_scores = np.array([scores[t] for t in members])
                    valid = np.isfinite(member_scores)
                    if valid.any():
                        member_ids = [m for m, v in zip(members, valid) if v]
                        member_vals = member_scores[valid]
                        # Выбираем через softmax с температурой
                        member_vals = member_vals - member_vals.max()
                        probs = np.exp(np.clip(member_vals / NOVELTY_TEMP, -50, 50))
                        total = probs.sum()
                        if total > 0:
                            probs /= total
                            idx = np.random.choice(len(member_ids), p=probs)
                            next_tok = member_ids[idx]
                        else:
                            next_tok = select_token(scores, 0.0)
                    else:
                        next_tok = select_token(scores, 0.0)
                else:
                    next_tok = select_token(scores, 0.0)
            else:
                next_tok = select_token(scores, temperature)
            
            if next_tok == EOS:
                tokens.append(EOS)
                break
            
            tokens.append(next_tok)
            
            # Сброс концепта после завершения слова
            meta2 = self.vocab.metadata_from_ids(tokens)
            flags = meta2[-1]['flags']
            is_word_end = (flags >> 1) & 1
            if is_word_end:
                self.target_concept = None  # после слова — новый концепт
            
            if return_compact:
                compact_frames.append(next_tok)
        
        result = {'tokens': tokens}
        if return_compact:
            result['compact'] = np.array(compact_frames, dtype=np.uint16)
        result['concept_history'] = self.concept_history
        return result if return_compact else tokens

    def generate_with_target(self, target_concept, context=[BOS], 
                              temperature=0.0, novelty=False, max_tokens=100):
        """Генерация с принудительным целевым концептом."""
        original_target = self.target_concept
        self.target_concept = target_concept
        self.novelty_mode = novelty
        
        # Создаём начальный контекст
        tokens = list(context)
        
        while len(tokens) < max_tokens:
            meta = self.vocab.metadata_from_ids(tokens)
            ctx = dict(meta[-1])
            ctx['context_tokens'] = tokens[-4:-1] if len(tokens) > 1 else tokens
            word_num = ctx['word_num']
            
            weights = self._compute_weights(ctx)
            
            scores = np.zeros(V, dtype=np.float32)
            try:
                head_scores = self.heads.individual_scores(ctx)
                scores = np.dot(weights, head_scores)
            except Exception:
                scores = np.ones(V, dtype=np.float32) * -1e9
                scores[EOS] = 0
            
            # Концептуальный буст
            if self.target_concept is not None:
                boost = self.cg.concept_scores(self.target_concept, V)
                scores = scores + boost * CONCEPT_BOOST
            
            scores = apply_masks(scores, tokens, ctx['word_num'], self.vocab.token_type)
            
            if novelty and self.target_concept is not None:
                members = self.cg.get_members(self.target_concept)
                if members:
                    member_scores = np.array([scores[t] for t in members])
                    valid = np.isfinite(member_scores)
                    if valid.any():
                        member_ids = [m for m, v in zip(members, valid) if v]
                        member_vals = member_scores[valid]
                        member_vals = member_vals - member_vals.max()
                        probs = np.exp(np.clip(member_vals / NOVELTY_TEMP, -50, 50))
                        total = probs.sum()
                        if total > 0:
                            probs /= total
                            idx = np.random.choice(len(member_ids), p=probs)
                            next_tok = member_ids[idx]
                        else:
                            next_tok = select_token(scores, 0.0)
                    else:
                        next_tok = select_token(scores, 0.0)
                else:
                    next_tok = select_token(scores, 0.0)
            else:
                next_tok = select_token(scores, temperature)
            
            if next_tok == EOS:
                tokens.append(EOS)
                break
            
            tokens.append(next_tok)
            
            # Проверяем конец слова
            meta2 = self.vocab.metadata_from_ids(tokens)
            flags = meta2[-1]['flags']
            is_word_end = (flags >> 1) & 1
            if is_word_end:
                break  # сгенерировали одно слово — хватит
            
        self.target_concept = original_target
        return tokens
    
    def generate_alternative(self, context, exclude_tokens=set(), temperature=0.0):
        """
        Из известного контекста сгенерировать АЛЬТЕРНАТИВНОЕ продолжение.
        Берёт последний токен контекста, находит его концепт,
        выбирает другой токен из того же концепта.
        """
        context = list(context)
        if len(context) < 2:
            return self.generate(temperature=temperature)
        
        prev_tid = context[-1]
        cid = self.cg.get_concept(prev_tid)
        if cid is None:
            # Если концепта нет — обычная генерация
            return self.generate(temperature=temperature)
        
        # Выбираем альтернативный токен из того же концепта
        alt_tid = self.cg.sample_alternative(cid, exclude_tokens | {prev_tid})
        if alt_tid is None:
            # Нет альтернатив — используем оригинал
            alt_tid = prev_tid
        
        # Генерируем с альтернативным токеном
        alt_context = context[:-1] + [alt_tid]
        gen_tokens = list(alt_context)
        
        # Завершаем слово и предложение с альтернативным контекстом
        while len(gen_tokens) < self.max_tokens:
            meta = self.vocab.metadata_from_ids(gen_tokens)
            ctx = dict(meta[-1])
            ctx['context_tokens'] = gen_tokens[-4:-1] if len(gen_tokens) > 1 else gen_tokens
            
            weights = self._compute_weights(ctx)
            
            scores = np.zeros(V, dtype=np.float32)
            try:
                head_scores = self.heads.individual_scores(ctx)
                scores = np.dot(weights, head_scores)
            except Exception:
                scores = np.ones(V, dtype=np.float32) * -1e9
                scores[EOS] = 0
            
            scores = apply_masks(scores, gen_tokens, ctx['word_num'], self.vocab.token_type)
            next_tok = select_token(scores, temperature)
            
            if next_tok == EOS:
                gen_tokens.append(EOS)
                break
            gen_tokens.append(next_tok)
        
        return gen_tokens

    def compare_generations(self, context_tokens, temperature=0.0, n_alternatives=3):
        """Сравнить обычную генерацию и альтернативные."""
        print("=== Обычная генерация ===")
        normal = self.generate(temperature=temperature)
        print(self.decode_tokens(normal))
        
        print("\n=== Альтернативные генерации ===")
        for i in range(n_alternatives):
            alt = self.generate_alternative(context_tokens, temperature=temperature)
            print(f"  [{i+1}] {self.decode_tokens(alt)}")
        
        return normal
