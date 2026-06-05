"""
PatternLearner — самоорганизация шаблонов.
Находит повторяющиеся концепт-последовательности в корпусе
и использует их как лекала для генерации.
"""
import numpy as np
from collections import defaultdict


class PatternLearner:
    def __init__(self, hv, ag, gates=None):
        self.hv = hv
        self.ag = ag
        self.gates = gates
        self.patterns = {}
        self.patterns_by_s_type = defaultdict(list)
        self.patterns_by_first = defaultdict(list)
        self.total_sentences = 0
        self._tt = hv.token_type
        
        # Статистика s_type по первому слову предложения
        # Уровни: концепт → s_type, токен → s_type
        self.concept_s_type_dist = defaultdict(lambda: defaultdict(int))
        self.token_s_type_dist = defaultdict(lambda: defaultdict(int))  # tid → s_type
    
    def learn(self, text_hierarchy):
        """
        Извлекает концепт-последовательности из всех предложений корпуса.
        Группирует идентичные последовательности, считает частоту.
        
        text_hierarchy: TextHierarchy с загруженным корпусом
        """
        if not text_hierarchy or not hasattr(text_hierarchy, 'sentences'):
            print("PatternLearner: no sentences to learn from")
            return
        
        seq_counter = defaultdict(lambda: {'freq': 0, 'samples': [], 's_type': 'statement'})
        total = 0
        
        for sent in text_hierarchy.sentences:
            sent_text = sent.text if hasattr(sent, 'text') else str(sent)
            tokens = self.hv.encode(' ' + sent_text)
            concepts = []
            seen_tids = []
            prev_c = None
            for t in tokens:
                if t >= 4096:
                    continue
                if self._tt[t] != 2:
                    continue
                text = self.hv.decode([t]).strip()
                if not text or len(text) <= 1:
                    continue
                if text[0].isascii() and text[0].isalpha():
                    continue
                c = self.ag.get_concept(t)
                if c is None:
                    continue
                cluster = c - self.ag.L1_OFFSET
                if not (0 <= cluster < self.ag.n_clusters):
                    continue
                if prev_c is None or cluster != prev_c:
                    concepts.append(cluster)
                    seen_tids.append(t)
                    prev_c = cluster
            
            if len(concepts) < 2:
                continue
            
            seq_key = tuple(concepts + [-1])
            s_type = sent.s_type if hasattr(sent, 's_type') and sent.s_type else 'statement'
            seq_counter[seq_key]['freq'] += 1
            seq_counter[seq_key]['s_type'] = s_type
            if len(seq_counter[seq_key]['samples']) < 5:
                sample_words = [self.hv.decode([t]).strip() for t in seen_tids]
                seq_counter[seq_key]['samples'].append(' '.join(sample_words[:6]))
            
            # Статистика: первый концепт → s_type
            first_c = concepts[0]
            self.concept_s_type_dist[first_c][s_type] += 1
            
            # Статистика: первый токен → s_type
            if seen_tids:
                self.token_s_type_dist[seen_tids[0]][s_type] += 1
            
            total += 1
        
        self.total_sentences = total
        
        # Фильтр: только частые (>=3 повторов) и длиной 2-5 концептов
        min_freq = 3
        
        self.patterns = {}
        self.patterns_by_s_type = defaultdict(list)
        self.patterns_by_first = defaultdict(list)
        
        for seq_key, info in seq_counter.items():
            if info['freq'] < min_freq:
                continue
            n_concepts = sum(1 for c in seq_key if c >= 0)
            if n_concepts < 2 or n_concepts > 5:
                continue
            
            self.patterns[seq_key] = {
                'freq': info['freq'],
                's_type': info['s_type'],
                'samples': info['samples'],
                'n_words': n_concepts,
                'weight': min(1.0, info['freq'] / 20.0),
                'effective_freq': info['freq'] * min(1.0, info['freq'] / 20.0),
                'success_count': 0,
                'fail_count': 0,
                'consecutive': 0,
            }
            self.patterns_by_s_type[info['s_type']].append(seq_key)
            first_c = seq_key[0]
            if first_c >= 0:
                self.patterns_by_first[first_c].append(seq_key)
        
        print("PatternLearner: %d sentences → %d patterns (min_freq=%d)" % (
            self.total_sentences, len(self.patterns), min_freq))
        if self.patterns:
            top = sorted(self.patterns.items(), key=lambda x: -x[1]['freq'])[:5]
            for seq, info in top:
                words = [self.hv.decode([t]).strip() for t in seq if t >= 0]
                print("  [%d] %s → %s" % (info['freq'], list(seq), info['samples'][0] if info['samples'] else ''))
    
    def match(self, seed_word=None, seed_concept=None, s_type=None):
        """
        Находит лучший шаблон для seed-слова.
        Возвращает tuple(concept_sequence) или None.
        
        Приоритет: effective_freq (freq × weight), затем s_type.
        """
        if not self.patterns:
            return None
        
        cid = None
        if seed_word:
            tokens = self.hv.encode(' ' + seed_word)
            for t in tokens:
                if t < 4096 and self._tt[t] == 2:
                    cid = self.ag.get_concept(t)
                    if cid is not None:
                        cid = cid - self.ag.L1_OFFSET
                        break
        elif seed_concept is not None:
            cid = seed_concept
        
        if cid is None or cid < 0:
            return None
        
        candidates = self.patterns_by_first.get(cid, [])
        if not candidates:
            return None
        
        def sort_key(seq_key):
            info = self.patterns[seq_key]
            eff = info.get('effective_freq', info['freq'])
            bonus = 0.0
            if s_type and info['s_type'] == s_type:
                bonus = 0.5
            return eff + bonus * eff
        
        candidates.sort(key=sort_key, reverse=True)
        return candidates[0]
    
    def report_outcome(self, seq_key, success, word_count=0, match_ratio=0.0):
        """
        Самоусиление/затухание шаблона по результату генерации.
        
        seq_key: кортеж концептов
        success: True если EOS достигнут до max_tokens
        word_count: сколько слов сгенерировано
        match_ratio: доля слов, совпавших с концептами шаблона (0.0-1.0)
        """
        if seq_key not in self.patterns:
            return
        info = self.patterns[seq_key]
        
        # Успех: усиливаем
        if success and match_ratio >= 0.5:
            delta = 0.1 * (0.5 + 0.5 * match_ratio)
            info['weight'] = min(1.0, info['weight'] + delta)
            info['success_count'] = info.get('success_count', 0) + 1
            info['consecutive'] = info.get('consecutive', 0) + 1
            # Цепная реакция: после 3+ успехов подряд усиливаем ещё больше
            if info['consecutive'] >= 3:
                info['weight'] = min(1.0, info['weight'] + 0.05)
        else:
            # Неудача: затухание
            decay = 0.05
            if not success:
                decay = 0.1  # сильнее если не дошли до EOS
            info['weight'] = max(0.1, info['weight'] - decay)
            info['fail_count'] = info.get('fail_count', 0) + 1
            info['consecutive'] = 0
        
        # Обновляем effective_freq для match()
        info['effective_freq'] = info['freq'] * info['weight']
    
    # ─── Самоанализ ─────────────────────────────────────────────┬─
    #                                                              │
    # При reinforcement каждое успешное прохождение пути           │
    # увеличивает effective_freq на 10%.                           │
    # При неудаче — уменьшает на 5-10%.                            │
    # После 3+ успехов подряд — цепная реакция (+5% доп).         │
    #──────────────────────────────────────────────────────────────┘
    
    def to_forced_path(self, seq_key):
        if seq_key not in self.patterns:
            return None
        info = self.patterns[seq_key]
        seq = list(seq_key)
        # Определяем s_type через статистику первого концепта
        first_c = seq[0] if seq else -1
        s_type = self.detect_s_type(first_c) if first_c >= 0 else 'statement'
        return {
            's_type': s_type,
            'concept_sequence': seq,
            'token_sequence': None,
            'word_count': info['n_words'],
            'text': info['samples'][0] if info['samples'] else 'pattern',
        }
    
    def get_top_patterns(self, n=10):
        """Возвращает top-n шаблонов по частоте."""
        return sorted(self.patterns.items(), key=lambda x: -x[1]['freq'])[:n]
    
    def describe(self, seq_key):
        """Читаемое описание шаблона."""
        if seq_key not in self.patterns:
            return None
        info = self.patterns[seq_key]
        seq = [c for c in seq_key if c >= 0]
        meta_names = []
        for c in seq:
            mid = self.ag.cid_to_mid.get(self.ag.L1_OFFSET + c)
            mname = self.ag.meta_name(mid) if mid is not None else '?'
            meta_names.append(mname)
        return {
            'concepts': seq,
            'meta': meta_names,
            'freq': info['freq'],
            's_type': info['s_type'],
            'sample': info['samples'][0] if info['samples'] else '',
            'weight': info['weight'],
        }
    
    def detect_s_type(self, first_concept=None, first_tid=None):
        """
        Определяет тип предложения по первому токену (если есть статистика)
        или по первому концепту. Без хардкода.
        """
        if first_tid is not None:
            dist = self.token_s_type_dist.get(first_tid, {})
            if dist:
                return max(dist, key=dist.get)
        if first_concept is not None:
            dist = self.concept_s_type_dist.get(first_concept, {})
            if dist:
                return max(dist, key=dist.get)
        return 'statement'
    
    def get_s_type_distribution(self, level='concept'):
        """
        Возвращает статистику для отладки.
        level='concept': концепт → s_type
        level='token': токен → s_type (только top)
        """
        if level == 'token':
            result = {}
            for tid, dist in self.token_s_type_dist.items():
                total = sum(dist.values())
                text = self.hv.decode([tid]).strip()
                result[text] = {k: v/total for k, v in dist.items()}
            return result
        result = {}
        for c, dist in self.concept_s_type_dist.items():
            total = sum(dist.values())
            result[c] = {k: v/total for k, v in dist.items()}
        return result
