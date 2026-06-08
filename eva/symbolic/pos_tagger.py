"""POS tagger for Russian words using pymorphy3."""
import pymorphy3
from functools import lru_cache

# Initialize morphological analyzer once
_morph = pymorphy3.MorphAnalyzer()

# Map pymorphy3 POS tags to simplified categories
POS_MAP = {
    'NOUN': 'NOUN',
    'ADJF': 'ADJ', 'ADJS': 'ADJ', 'COMP': 'ADJ',
    'VERB': 'VERB', 'INFN': 'VERB', 'PRTF': 'VERB', 'PRTS': 'VERB', 'GRND': 'VERB',
    'NUMR': 'NUM', 'NUMB': 'NUM',
    'ADVB': 'ADV',
    'PREP': 'PREP',
    'CONJ': 'CONJ',
    'PRCL': 'PART',
    'INTJ': 'INTJ',
    'NPRO': 'PRON',
    'PRED': 'PRED',
}


@lru_cache(maxsize=10000)
def get_pos(word: str) -> str:
    """Get simplified POS tag for a word.

    Returns one of: NOUN, ADJ, VERB, NUM, ADV, PREP, CONJ, PART, INTJ, PRON, PRED, UNK
    """
    if not word or len(word) < 2:
        return 'UNK'

    parsed = _morph.parse(word.lower())
    if not parsed:
        return 'UNK'

    # Take the most likely parse
    best = parsed[0]
    pos_tag = best.tag.POS

    if pos_tag is None:
        return 'UNK'

    return POS_MAP.get(pos_tag, 'UNK')


@lru_cache(maxsize=10000)
def get_morph_features(word: str) -> dict:
    """Get morphological features for a word.

    Returns dict with keys: pos, gender, number, case, person, tense
    """
    if not word or len(word) < 2:
        return {'pos': 'UNK'}

    parsed = _morph.parse(word.lower())
    if not parsed:
        return {'pos': 'UNK'}

    best = parsed[0]
    tag = best.tag

    features = {
        'pos': POS_MAP.get(tag.POS, 'UNK'),
        'gender': tag.gender,  # masc, femn, neut
        'number': tag.number,  # sing, plur
        'case': tag.case,      # nomn, gent, datv, accs, ablt, loct
        'person': tag.person,  # 1per, 2per, 3per
        'tense': tag.tense,    # pres, past, futr
    }

    return features


def check_agreement(word1: str, word2: str) -> bool:
    """Check if two adjacent words have compatible morphological features.

    Returns True if words can be adjacent (no obvious agreement violation).
    """
    f1 = get_morph_features(word1)
    f2 = get_morph_features(word2)

    pos1, pos2 = f1['pos'], f2['pos']

    # ADJ + NOUN: must agree in gender, number, case
    if pos1 == 'ADJ' and pos2 == 'NOUN':
        if f1['gender'] and f2['gender'] and f1['gender'] != f2['gender']:
            return False
        if f1['number'] and f2['number'] and f1['number'] != f2['number']:
            return False
        if f1['case'] and f2['case'] and f1['case'] != f2['case']:
            return False
        return True

    # NOUN + VERB: must agree in number (and gender for past tense)
    if pos1 == 'NOUN' and pos2 == 'VERB':
        if f1['number'] and f2['number'] and f1['number'] != f2['number']:
            return False
        if f2['tense'] == 'past' and f1['gender'] and f2['gender'] and f1['gender'] != f2['gender']:
            return False
        return True

    # PREP + NOUN: no specific agreement check (case government is complex)
    if pos1 == 'PREP' and pos2 == 'NOUN':
        return True

    # Default: assume compatible
    return True


# POS transition probabilities (learned from corpus)
# These are typical Russian POS bigram frequencies
POS_BIGRAMS = {
    ('NOUN', 'VERB'): 0.15,
    ('VERB', 'NOUN'): 0.12,
    ('ADJ', 'NOUN'): 0.10,
    ('PREP', 'NOUN'): 0.09,
    ('NOUN', 'ADJ'): 0.05,
    ('CONJ', 'NOUN'): 0.04,
    ('VERB', 'ADV'): 0.04,
    ('ADV', 'VERB'): 0.03,
    ('NOUN', 'PREP'): 0.03,
    ('PRON', 'VERB'): 0.03,
    ('VERB', 'PREP'): 0.03,
    ('NUM', 'NOUN'): 0.02,
    ('NOUN', 'CONJ'): 0.02,
    ('PART', 'VERB'): 0.02,
    ('VERB', 'PRON'): 0.02,
}


def pos_transition_score(prev_pos: str, next_pos: str) -> float:
    """Get transition score for POS bigram.

    Returns a score in [0, 1] where higher is more likely.
    """
    return POS_BIGRAMS.get((prev_pos, next_pos), 0.01)
