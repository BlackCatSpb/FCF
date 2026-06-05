"""
StructuralRules — бинарные (0/1) правила переходов для каждого уровня иерархии.
Без частот. Если переход хоть раз встретился в корпусе — он структурно возможен.

Уровни:
- sentence_type: тип предложения → какие типы могут следовать
- cross_concept: последний концепт предложения → какие концепты могут быть первыми в следующем
- chapter_intro: первые концепты глав
"""
import json, os, sys
from collections import defaultdict

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')


class StructuralRules:
    
    def __init__(self):
        # sentence_type transition: str -> set[str]
        self.s_type_trans = {}
        # cross-sentence concept: int (cluster idx) -> set[int]
        self.cross_concept = {}
        # L1 offset for ID conversion
        self.l1_offset = 48
        
    def build(self, sentences, ag=None):
        """
        Build rules from parsed TextHierarchy sentences.
        sentences: list of SentenceInfo
        ag: AssociationGraph (for concept_id -> cluster idx mapping)
        """
        # Sentence type transitions
        s_types = [s.s_type for s in sentences if s.s_type]
        type_map = defaultdict(set)
        for i in range(len(s_types) - 1):
            type_map[s_types[i]].add(s_types[i+1])
        self.s_type_trans = {k: set(v) for k, v in type_map.items()}
        
        # Cross-sentence concept transitions
        conc_map = defaultdict(set)
        for i in range(len(sentences) - 1):
            c1 = sentences[i].last_content_cid
            c2 = sentences[i+1].first_content_cid
            if c1 >= 0 and c2 >= 0:
                conc_map[c1].add(c2)
        self.cross_concept = {int(k): set(int(x) for x in v) for k, v in conc_map.items()}
        
        print(f"StructuralRules built:")
        print(f"  sentence_type: {len(self.s_type_trans)} types, "
              f"total {sum(len(v) for v in self.s_type_trans.values())} transitions")
        print(f"  cross_concept: {len(self.cross_concept)} source concepts, "
              f"total {sum(len(v) for v in self.cross_concept.values())} transitions")
        
        return self
    
    def save(self, path):
        """Save to JSON. Sets -> lists for serialization."""
        data = {
            'l1_offset': self.l1_offset,
            's_type_trans': {k: list(v) for k, v in self.s_type_trans.items()},
            'cross_concept': {str(k): [int(x) for x in v] for k, v in self.cross_concept.items()},
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {path}")
    
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.l1_offset = data.get('l1_offset', 48)
        self.s_type_trans = {k: set(v) for k, v in data['s_type_trans'].items()}
        self.cross_concept = {int(k): set(int(x) for x in v) 
                              for k, v in data['cross_concept'].items()}
        print(f"Loaded {path}: {len(self.s_type_trans)} type rules, "
              f"{len(self.cross_concept)} concept rules")
        return self
    
    def get_valid_next_types(self, prev_type):
        """Какие типы предложений могут следовать за prev_type."""
        return self.s_type_trans.get(prev_type, set())
    
    def get_valid_next_concepts(self, prev_concept_cluster):
        """Какие концепты (cluster idx) могут быть первыми в след. предложении."""
        return self.cross_concept.get(prev_concept_cluster, set())
    
    def print_summary(self):
        print("\n=== STRUCTURAL RULES ===")
        print(f"Sentence type transitions:")
        for t1, t2s in sorted(self.s_type_trans.items()):
            print(f"  {t1:12s} -> {sorted(t2s)}")
        print(f"Cross-sentence concept transitions (top 10 by out-degree):")
        sorted_rules = sorted(self.cross_concept.items(), key=lambda x: -len(x[1]))
        for c, next_cs in sorted_rules[:10]:
            print(f"  C{c:2d} -> [{len(next_cs)} clusters]: "
                  f"{sorted(next_cs)[:8]}{'...' if len(next_cs) > 8 else ''}")
        print(f"  ... ({len(self.cross_concept)} source concepts total)")

if __name__ == '__main__':
    from eva.symbolic.text_hierarchy import TextHierarchy
    from eva.symbolic.bpe_tokenizer import HierarchicalVocab
    from eva.symbolic.association_graph import AssociationGraph
    from eva.symbolic.heads import HeadsEnsemble
    
    hv = HierarchicalVocab()
    heads = HeadsEnsemble('real_data/v8/heads_meta.pkl', 'real_data/v8')
    ag = AssociationGraph(n_clusters=48, n_metas=12)
    ag.build(heads.log_prob_csr, hv.token_type, decode_fn=hv.decode)
    
    th = TextHierarchy(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru.txt', hv)
    sents = th.parse()
    th.analyze_concepts(ag)
    
    rules = StructuralRules()
    rules.build(sents, ag)
    rules.print_summary()
    rules.save(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\structural_rules.json')
