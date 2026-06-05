"""
TextHierarchy — многоуровневый парсер текста.
Строит иерархию: Том → Часть → Глава → Параграф → Предложение → Слово → Токен.

Каждый уровень хранит структурные переходы (что может следовать за чем)
и семантические метки (тема, тип предложения, грамматические роли).
"""
import re, sys, os, json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
from eva.symbolic.bpe_tokenizer import HierarchicalVocab


ROMAN = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,'X':10,
         'XI':11,'XII':12,'XIII':13,'XIV':14,'XV':15,'XVI':16,'XVII':17,'XVIII':18,
         'XIX':19,'XX':20,'XXI':21,'XXII':22,'XXIII':23,'XXIV':24,'XXV':25,
         'XXVI':26,'XXVII':27,'XXVIII':28,'XXIX':29,'XXX':30,
         'XXXI':31,'XXXII':32,'XXXIII':33,'XXXIV':34,'XXXV':35,
         'XXXVI':36,'XXXVII':37,'XXXVIII':38,'XXXIX':39,'XL':40,
         'XLI':41,'XLII':42,'XLIII':43,'XLIV':44,'XLV':45,'XLVI':46,'XLVII':47,'XLVIII':48,
         'L':50,'LI':51,'LII':52,'LIII':53,'LIV':54,'LV':55,'LVI':56,'LVII':57,'LVIII':58,
         'LX':60,'LXI':61,'LXII':62,'LXIII':63,'LXIV':64,'LXV':65,'LXVI':66,'LXVII':67,
         'LXX':70,'LXXI':71}
ROMAN_RE = re.compile(r'^[IVXLCDM]+\s*$')
VOLUME_RE = re.compile(r'^Том\s+(первый|второй|третий|четвертый)', re.IGNORECASE)
PART_RE = re.compile(r'^Часть\s+(первая|вторая|третья|четвертая|пятая)', re.IGNORECASE)
EPILOGUE_RE = re.compile(r'^Эпилог', re.IGNORECASE)


@dataclass
class SentenceInfo:
    id: int
    text: str
    volume: int          # 0-3
    part: int            # 0-4
    chapter: int         # 0-354
    paragraph: int       # within chapter
    position: int        # within paragraph
    
    # Sentence-level properties
    s_type: str = 'statement'  # statement / question / exclamation / dialogue / french
    has_direct_speech: bool = False
    has_french: bool = False
    word_count: int = 0
    
    # Token-level data
    tokens: List[int] = field(default_factory=list)
    type2_tokens: List[int] = field(default_factory=list)
    type2_texts: List[str] = field(default_factory=list)
    
    # Structural: first content word's concept
    first_content_cid: int = -1
    last_content_cid: int = -1


@dataclass
class ParagraphInfo:
    id: int
    chapter: int
    position: int
    sentences: List[SentenceInfo] = field(default_factory=list)
    
    # Paragraph-level
    first_speaker: str = ''
    has_dialogue: bool = False
    topic: str = ''


class TextHierarchy:
    """
    Строит иерархию текста:
      Том → Часть → Глава → Параграф → Предложение
    
    Для каждого предложения сохраняет:
      - Токены (BPE)
      - Type-2 токены (word starters)
      - Тип предложения
      - Позицию в иерархии
    """
    
    def __init__(self, corpus_path, hv=None):
        self.corpus_path = corpus_path
        self.hv = hv or HierarchicalVocab()
        
        self.volumes = []        # list of volume names
        self.parts = []          # list of (vol_idx, part_name)
        self.chapters = []       # list of (vol_idx, part_idx, chapter_num)
        self.paragraphs = []     # list[ParagraphInfo]
        self.sentences = []      # list[SentenceInfo]
        
        self.volume_of_sent = {}   # sent_id -> volume_idx
        self.chapter_of_sent = {}  # sent_id -> chapter_idx
        self.paragraph_of_sent = {} # sent_id -> paragraph_idx
        
    def parse(self):
        """Parse full corpus into hierarchy."""
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            lines = [l.rstrip('\n') for l in f]
        
        current_vol = -1
        current_part = -1
        current_chapter = -1
        current_para = 0
        sent_id = 0
        
        # Find Том/Часть positions
        vol_starts = []
        part_starts = []
        chapter_lines = []
        
        for i, l in enumerate(lines):
            s = l.strip()
            if VOLUME_RE.match(s):
                vol_starts.append(i)
                self.volumes.append(s)
            elif PART_RE.match(s) or EPILOGUE_RE.match(s):
                part_starts.append((len(self.volumes)-1 if self.volumes else 0, s))
            elif ROMAN_RE.match(s) and len(s) <= 5:
                chapter_lines.append(i)
        
        # Collect parts per volume
        current_vol_parts = []
        for v_idx in range(len(self.volumes)):
            v_start = vol_starts[v_idx]
            v_end = vol_starts[v_idx+1] if v_idx+1 < len(vol_starts) else len(lines)
            parts_in_v = [(i, lines[i].strip()) for i in range(v_start, v_end) 
                         if PART_RE.match(lines[i].strip())]
            for p_line, p_name in parts_in_v:
                self.parts.append((v_idx, p_name, p_line))
        
        # Assign chapters to volumes/parts
        for ch_idx, ch_line in enumerate(chapter_lines):
            assigned_vol = -1
            for v_idx in range(len(self.volumes)):
                v_start = vol_starts[v_idx]
                v_end = vol_starts[v_idx+1] if v_idx+1 < len(vol_starts) else len(lines)
                if v_start <= ch_line < v_end:
                    assigned_vol = v_idx
                    break
            assigned_part = -1
            for p_idx, (pv, pn, pl) in enumerate(self.parts):
                if pv == assigned_vol and pl <= ch_line:
                    next_part_line = self.parts[p_idx+1][2] if p_idx+1 < len(self.parts) else len(lines)
                    if ch_line < next_part_line:
                        assigned_part = p_idx
                        break
            chapter_num = ch_idx + 1
            self.chapters.append((ch_idx, assigned_vol, assigned_part, chapter_num, ch_line))
        
        # Build chapter line index
        chapter_at_line = {}
        for ch_idx, v_idx, p_idx, ch_num, ch_line in self.chapters:
            chapter_at_line[ch_line] = ch_idx
        
        # Now parse sentences within chapters
        current_vol = -1
        current_part = -1
        current_chapter = -1
        current_para = 0
        
        for i, l in enumerate(lines):
            s = l.strip()
            if not s:
                continue
            if VOLUME_RE.match(s) or PART_RE.match(s) or EPILOGUE_RE.match(s) or (ROMAN_RE.match(s) and len(s) <= 5):
                continue
            
            # Determine chapter for this sentence
            ch_idx = -1
            for ch_i, v_i, p_i, ch_n, ch_l in self.chapters:
                if ch_l <= i:
                    ch_idx = ch_i
            
            # Group into paragraphs (blank-line separated)
            # We use line i-1 being empty as paragraph boundary
            # But since we skip empty lines, we need prev_line check
            # For simplicity: each non-empty line is a paragraph with 1+ sentences
            # Actually in this text format, each line IS a sentence (or French phrase)
            
            # Detect sentence type
            s_type = 'statement'
            has_direct = bool(re.search(r'[—–\-]\s*[«"`\']', s) or re.search(r'[!?]\s*[—–]', s))
            has_french = bool(re.search(r'[àéèêëîïôùûüçœæ]', s, re.IGNORECASE))
            
            if s.endswith('?') or '?' in s:
                s_type = 'question'
            elif s.endswith('!') or '!' in s:
                s_type = 'exclamation'
            if has_direct:
                s_type = 'dialogue'
            if has_french and len(s) > 20:
                # If predominantly French
                french_chars = sum(1 for c in s if c in 'àéèêëîïôùûüçœæ')
                if french_chars > len(s) * 0.05:
                    s_type = 'french'
            
            # Tokenize
            tokens = self.hv.encode(s)
            type2_tids = [t for t in tokens if t < 4096 and self.hv.token_type[t] == 2]
            type2_texts = [self.hv.decode([t]).strip() for t in type2_tids]
            
            # First/last content word concept
            fc_cid = -1
            lc_cid = -1
            if type2_tids:
                from eva.symbolic.association_graph import AssociationGraph
                # We'll compute concepts later; store tids for now
            
            sent = SentenceInfo(
                id=sent_id,
                text=s,
                volume=ch_idx // 50 if ch_idx >= 0 else -1,  # approximate
                part=ch_idx // 20 if ch_idx >= 0 else -1,
                chapter=ch_idx,
                paragraph=current_para,
                position=len(self.sentences) - (sum(1 for p in self.paragraphs for _ in p.sentences)),
                s_type=s_type,
                has_direct_speech=has_direct,
                has_french=has_french,
                word_count=len(type2_tids),
                tokens=tokens,
                type2_tokens=type2_tids,
                type2_texts=type2_texts,
            )
            
            self.sentences.append(sent)
            self.volume_of_sent[sent_id] = ch_idx // 50 if ch_idx >= 0 else -1
            self.chapter_of_sent[sent_id] = ch_idx
            sent_id += 1
        
        print(f"Parsed {len(self.sentences)} sentences in {len(self.chapters)} chapters, {len(self.volumes)} volumes")
        return self.sentences
    
    def analyze_concepts(self, assoc_graph):
        """Tag each sentence with concept IDs for first/last content word."""
        for sent in self.sentences:
            for tid in sent.type2_tokens:
                cid = assoc_graph.get_concept(tid)
                if cid is not None:
                    if sent.first_content_cid < 0:
                        sent.first_content_cid = cid - assoc_graph.L1_OFFSET
                    sent.last_content_cid = cid - assoc_graph.L1_OFFSET
    
    def build_structural_rules(self):
        """
        Build BINARY structural rules for each level.
        Returns dict: level -> {key -> set of valid next keys}
        Levels: 'chapter_topic', 'sentence_type', 'concept_seq'
        """
        rules = {}
        
        # Sentence type transitions
        s_type_seq = [s.s_type for s in self.sentences]
        type_trans = defaultdict(set)
        for i in range(len(s_type_seq) - 1):
            type_trans[s_type_seq[i]].add(s_type_seq[i+1])
        rules['sentence_type'] = dict(type_trans)
        
        # Concept bigram transitions across sentences
        concept_pairs = defaultdict(set)
        for sent in self.sentences:
            if sent.first_content_cid >= 0:
                concept_pairs['ANY'].add(sent.first_content_cid)
        for i in range(len(self.sentences) - 1):
            s1 = self.sentences[i]
            s2 = self.sentences[i+1]
            if s1.last_content_cid >= 0 and s2.first_content_cid >= 0:
                concept_pairs[(s1.last_content_cid,)].add(s2.first_content_cid)
        rules['concept_cross_sentence'] = dict(concept_pairs)
        
        # Chapter topic structure
        # Detect topic transitions from first/last content words
        for ch_idx in set(self.chapter_of_sent.values()):
            if ch_idx < 0:
                continue
            ch_sents = [s for sid, s in enumerate(self.sentences) if self.chapter_of_sent.get(sid) == ch_idx]
            topics = set()
            for s in ch_sents:
                for t in s.type2_tokens:
                    topics.add(t)
        # Chapter-type transitions
        rules['chapter_topic'] = {'ANY': set(range(50))}  # simplified
        
        return rules
    
    def get_sentences_by_chapter(self, chapter_idx):
        return [s for sid, s in enumerate(self.sentences) if self.chapter_of_sent.get(sid) == chapter_idx]
    
    def get_chapter_text(self, chapter_idx):
        sents = self.get_sentences_by_chapter(chapter_idx)
        return ' '.join(s.text for s in sents if s.text)
    
    def print_hierarchy(self, max_sents=10):
        print(f"\n=== TEXT HIERARCHY ===")
        print(f"Volumes: {len(self.volumes)}")
        print(f"Chapters: {len(self.chapters)}")
        print(f"Sentences: {len(self.sentences)}")
        
        print(f"\nSample sentences:")
        for i, s in enumerate(self.sentences[:max_sents]):
            print(f"  [{s.id}] V{s.volume} Ch{s.chapter} P{s.paragraph} "
                  f"[{s.s_type:10s}] {s.text[:80]}")


if __name__ == '__main__':
    hv = HierarchicalVocab()
    th = TextHierarchy(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru.txt', hv)
    sents = th.parse()
    th.print_hierarchy()
    
    rules = th.build_structural_rules()
    print(f"\nStructural rules:")
    for level, trans in rules.items():
        print(f"  {level}: {sum(len(v) for v in trans.values())} transitions")
        if level == 'sentence_type':
            for k, v in trans.items():
                print(f"    {k:12s} -> {sorted(v)[:5]}")
