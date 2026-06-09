"""ConceptTokenizer — hierarchical semantic extractor.

Architecture:
- BPE tokens preserved for backward compatibility
- Added pymorphy3 morphological layer: root + affix decomposition
- Each word: root → core concept vector + affix shifts → word vector
- Metadata: concept_id, is_anchor, root, prefix, suffix, ending, pos, features
- Hierarchical: word → phrase → sentence metadata extraction

Token ID layout:
   0: PAD
   1: UNK
   2: BOS
   3: EOS
   4: WORD_OPEN    — marks word start
   5: WORD_CLOSE   — marks word end
   6: SENT_OPEN    — marks sentence start
   7: SENT_CLOSE   — marks sentence end
   8+: BPE subword tokens (from pretrained BPE model)
"""

import os, re, json, math
import numpy as np
from collections import defaultdict
from tokenizers import Tokenizer
from typing import List, Optional

from eva.symbolic.concept_net import ConceptSkeleton
from eva.symbolic.pos_tagger import get_morph_features, get_pos

BPE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'bpe_tokenizer.json')
SKELETON_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'concept_skeleton.json')

# Token types (contextual)
TYPE_SPECIAL = 0   # PAD, UNK, BOS, EOS, WORD_OPEN, WORD_CLOSE, SENT_OPEN, SENT_CLOSE
TYPE_WORD_START = 2  # first BPE token after WORD_OPEN
TYPE_WORD_CONT = 3   # subsequent BPE tokens within a word


class ConceptTokenizer:
    """Tokenizer with explicit boundaries and concept metadata.

    Usage:
        tok = ConceptTokenizer()
        tok.initialize()
        ids = tok.encode("Война и мир.")
        meta = tok.metadata_from_ids(ids)
        text = tok.decode(ids)
    """

    def __init__(self, bpe_path=None, skeleton_path=None):
        self.bpe_path = bpe_path or BPE_PATH
        self.skeleton_path = skeleton_path or SKELETON_PATH

        # Special token IDs
        self.PAD = 0
        self.UNK = 1
        self.BOS = 2
        self.EOS = 3
        self.WORD_OPEN = 4
        self.WORD_CLOSE = 5
        self.SENT_OPEN = 6
        self.SENT_CLOSE = 7
        self.N_SPECIAL = 8

        # BPE tokenizer (loaded from file)
        self.bpe = None
        self.bpe_vocab_size = 0
        self.V = 0  # total vocab size (special + BPE)

        # Concept skeleton
        self.skeleton = None

        # Precomputed: word → (concept_id, is_anchor)
        self._word_cache = {}

    def initialize(self):
        """Load BPE model and ConceptNet skeleton."""
        # Load BPE
        if not os.path.exists(self.bpe_path):
            raise FileNotFoundError(f"BPE tokenizer not found at {self.bpe_path}")
        self.bpe = Tokenizer.from_file(self.bpe_path)
        self.bpe_vocab_size = self.bpe.get_vocab_size()
        self.V = self.N_SPECIAL + self.bpe_vocab_size

        # Load or build ConceptNet skeleton
        self.skeleton = ConceptSkeleton()
        if os.path.exists(self.skeleton_path):
            self.skeleton.load(self.skeleton_path)
        else:
            print("Concept skeleton not found, building from ConceptNet...")
            self.skeleton.build()
            self.skeleton.save(self.skeleton_path)

        # Build BPE token properties
        self._build_bpe_properties()

        return self

    def _build_bpe_properties(self):
        """Compute properties of BPE tokens: can_start_word, can_continue_word."""
        self.can_start_word = np.zeros(self.V, dtype=bool)
        self.can_continue_word = np.zeros(self.V, dtype=bool)

        for i in range(self.bpe_vocab_size):
            tid = i + self.N_SPECIAL
            decoded = self.bpe.decode([i])
            # A BPE token can start a word if its decoded form doesn't start
            # with a space (in character-level BPE, no spaces in tokens)
            # For character-level BPE, ALL tokens are word-internal
            self.can_start_word[tid] = True  # all BPE tokens can potentially start
            self.can_continue_word[tid] = True  # all can continue

        # Special tokens: only boundary markers
        self.token_type_arr = np.zeros(self.V, dtype=np.uint8)
        for i in range(self.N_SPECIAL):
            self.token_type_arr[i] = TYPE_SPECIAL
        for i in range(self.N_SPECIAL, self.V):
            self.token_type_arr[i] = TYPE_WORD_START  # default, context-dependent

    def encode(self, text: str) -> List[int]:
        """Encode text into token sequence with explicit boundaries.

        Returns:
            List of token IDs including BOS, EOS, WORD_OPEN/CLOSE, SENT_OPEN/CLOSE
        """
        result = [self.BOS]

        # Split into sentences (simple heuristic)
        sentences = re.split(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z])', text.strip())

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            result.append(self.SENT_OPEN)
            words = sent.split()
            for word in words:
                # Split trailing punctuation from clean word
                clean = word.rstrip('.,!?;:()[]{}«»—–-…\"\'')
                punct = word[len(clean):]

                if clean:
                    result.append(self.WORD_OPEN)
                    bpe_ids = self.bpe.encode(clean).ids
                    result.extend(tid + self.N_SPECIAL for tid in bpe_ids)
                    result.append(self.WORD_CLOSE)

                # Punctuation as separate single-character words
                for p in punct:
                    result.append(self.WORD_OPEN)
                    pt_ids = self.bpe.encode(p).ids
                    if pt_ids:
                        result.append(pt_ids[0] + self.N_SPECIAL)
                    result.append(self.WORD_CLOSE)

            result.append(self.SENT_CLOSE)

        result.append(self.EOS)
        return result

    def encode_words(self, words: List[str]) -> List[int]:
        """Encode a pre-split list of words with SENT_OPEN/CLOSE."""
        result = [self.BOS, self.SENT_OPEN]
        for word in words:
            clean = word.strip('.,!?;:()[]{}«»—–-…\"\'')
            if clean:
                result.append(self.WORD_OPEN)
                bpe_ids = self.bpe.encode(clean.lower()).ids
                result.extend(tid + self.N_SPECIAL for tid in bpe_ids)
                result.append(self.WORD_CLOSE)
        result.append(self.SENT_CLOSE)
        result.append(self.EOS)
        return result

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text.

        Handles ByteLevel BPE per-word. Inserts spaces between words
        (but not between word and trailing punctuation). Inserts sentence
        boundaries as ". " or "? " etc.
        """
        result = []
        in_word = False
        word_buf = []
        last_was_sent_close = False

        for tid in ids:
            if tid == self.WORD_OPEN:
                in_word = True
                word_buf = []
            elif tid == self.WORD_CLOSE:
                in_word = False
                if word_buf:
                    w = self.bpe.decode(word_buf).strip()
                    if w:
                        result.append(w)
            elif tid == self.SENT_CLOSE:
                last_was_sent_close = True
            elif tid == self.SENT_OPEN:
                if last_was_sent_close and result:
                    # Add period if sentence didn't end with punctuation
                    if result[-1] not in '.!?…':
                        result.append('.')
                last_was_sent_close = False
            elif tid >= self.N_SPECIAL:
                if in_word:
                    word_buf.append(tid - self.N_SPECIAL)
                else:
                    w = self.bpe.decode([tid - self.N_SPECIAL]).strip()
                    if w:
                        result.append(w)
            # Skip PAD, UNK, BOS, EOS, WORD_CLOSE already handled

        # Reconstruct with spaces
        if not result:
            return ''
        text = result[0]
        for w in result[1:]:
            if not w:
                continue
            if w in '.,!?;:()[]{}«»—–-…\u2014':
                text += w  # no space before punctuation
            elif text and text[-1] in '«(—–\u2014':
                text += w  # no space after opening
            else:
                text += ' ' + w
        return text

    def token_type(self, tid: int, pos_in_word: int) -> int:
        """Get token type based on position in word.
        After WORD_OPEN (piw=0): TYPE_WORD_START or TYPE_SPECIAL
        After word start (piw>0): TYPE_WORD_CONT"""
        if tid < self.N_SPECIAL:
            return TYPE_SPECIAL
        if pos_in_word <= 0:
            return TYPE_WORD_START
        return TYPE_WORD_CONT

    def word_info(self, word: str):
        """Get concept info for a word. Returns (concept_id, is_anchor, anchor_word)
        or (None, False, None) if unknown."""
        wl = word.strip('.,!?;:()[]{}«»—–-…\"\'').lower()
        if wl in self._word_cache:
            return self._word_cache[wl]

        cid = self.skeleton.concept_of(wl) if self.skeleton else None
        is_anchor = False
        anchor = wl
        if cid is not None:
            c = self.skeleton.concepts.get(cid, {})
            anchor = c.get('anchor', wl)
            is_anchor = (anchor == wl)

        result = (cid, is_anchor, anchor)
        self._word_cache[wl] = result
        return result

    def metadata_from_ids(self, ids: List[int]) -> List[dict]:
        """Compute per-position metadata from a token sequence.

        Returns:
            List of dicts with: pos_in_word, word_len, word_num, concept_id,
            is_anchor, anchor_word, token_type, flags, prev_token_id
        """
        L = len(ids)
        result = [{} for _ in range(L)]

        word_num = -1
        pos_in_word = -1
        current_word_tokens = []
        current_concept_id = None
        current_is_anchor = False
        current_anchor = None

        for t in range(L):
            tid = ids[t]
            prev_tid = ids[t - 1] if t > 0 else tid

            if tid == self.WORD_OPEN:
                word_num += 1
                pos_in_word = 0
                current_word_tokens = [tid]
                current_concept_id = None
                current_is_anchor = False
                current_anchor = None

            elif tid == self.WORD_CLOSE:
                pos_in_word = -1  # not in a word

            elif tid == self.SENT_OPEN or tid == self.SENT_CLOSE:
                pos_in_word = -1
                word_num = -1  # reset for next sentence
                current_word_tokens = []

            elif tid < self.N_SPECIAL:  # other special tokens
                pos_in_word = -1

            else:  # BPE token
                if pos_in_word >= 0:
                    pos_in_word += 1
                else:
                    pos_in_word = 1  # shouldn't happen, but be safe
                current_word_tokens.append(tid)

            # Compute token type
            tt = self.token_type(tid, max(0, pos_in_word))

            # Build flags
            flags = 0
            if tid == self.WORD_OPEN: flags |= 1 << 0  # word_start
            if tid == self.WORD_CLOSE: flags |= 1 << 1  # word_end
            if tid == self.SENT_OPEN or (t > 0 and ids[t-1] == self.SENT_OPEN):
                flags |= 1 << 2  # sent_start
            if tid == self.SENT_CLOSE or tid == self.EOS:
                flags |= 1 << 3  # sent_end
            if tid < self.N_SPECIAL:
                flags |= 1 << 5  # is_special

            result[t] = {
                'pos_in_word': max(0, pos_in_word),
                'word_len': len(current_word_tokens),
                'word_num': max(0, word_num),
                'pos_in_sent': t,
                'sent_len': L,
                'flags': flags,
                'token_id': tid,
                'prev_token_id': prev_tid,
                'token_type': tt,
            }

        # Pass 2: resolve concept info for each word
        # We need to extract words from the sequence to look up concept_ids
        word_idx = -1
        current_word = []
        current_start = -1

        for t in range(L):
            tid = ids[t]
            if tid == self.WORD_OPEN:
                current_word = []
                current_start = t
            elif tid == self.WORD_CLOSE:
                if current_word:
                    word_text = self.bpe.decode(
                        [tid - self.N_SPECIAL for tid in current_word])
                    cid, is_anchor, anchor = self.word_info(word_text)
                    # Apply to all positions of this word
                    for p in range(current_start, t + 1):
                        result[p]['concept_id'] = cid
                        result[p]['is_anchor'] = is_anchor
                        result[p]['anchor_word'] = anchor
                current_word = []
            elif tid >= self.N_SPECIAL:
                current_word.append(tid)
            elif tid in (self.SENT_OPEN, self.SENT_CLOSE):
                pass

        return result

    def __len__(self):
        return self.V

    # ── Morphological parsing layer ─────────────────────────────

    @staticmethod
    def morph_parse(word):
        """Parse word into morphological components.

        Returns:
            dict with: root, prefix, suffix, ending, pos, features, normalized
            or None if word can't be parsed.
        """
        w = word.strip('.,!?;:()[]{}«»—–-…\u201d\u201c"\'').lower()
        if not w or len(w) < 2:
            return None

        features = get_morph_features(w)
        pos = features.get('pos', 'UNK')

        # Use pymorphy3 to get normal form (lemma)
        from eva.symbolic.pos_tagger import _morph
        parsed = _morph.parse(w)
        if not parsed:
            return None

        best = parsed[0]
        normal_form = best.normal_form
        tag = best.tag

        # Decompose word into morphemes (heuristic for Russian)
        # Root approximation: take normal_form as root candidate
        # For now, root = normal_form, affixes are derived from differences
        root = normal_form

        # Detect common prefixes
        prefix = ''
        for p in ['пре', 'при', 'пере', 'про', 'раз', 'рас', 'вз', 'воз',
                  'вос', 'из', 'ис', 'вы', 'от', 'о', 'об', 'обо', 'под',
                  'над', 'за', 'на', 'по', 'до', 'с', 'со', 'в', 'во',
                  'у', 'без', 'бес', 'через', 'чрез']:
            if w.startswith(p) and not normal_form.startswith(p):
                prefix = p
                break

        # Detect common suffixes
        suffix = ''
        remaining = w[len(prefix):] if prefix else w
        for s in ['к', 'ок', 'ек', 'ик', 'ник', 'тель', 'чик', 'щик',
                  'ств', 'еств', 'ость', 'ность', 'ени', 'ани',
                  'изм', 'ист', 'атор', 'тор',
                  'лив', 'чив', 'ист', 'оват', 'еват',
                  'ну', 'а', 'я']:
            if remaining.endswith(s) and len(remaining) > len(s) + 2:
                suffix = s
                break

        # Detect ending (last 1-2 chars after removing root-suffix, heuristic)
        ending = ''
        base = remaining[:-len(suffix)] if suffix else remaining
        for e in ['ого', 'его', 'ому', 'ему', 'ым', 'им', 'ой', 'ей',
                  'ую', 'юю', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие',
                  'а', 'я', 'о', 'е', 'ы', 'и', 'у', 'ю',
                  'ой', 'ей', 'ых', 'их', 'ам', 'ям']:
            if base.endswith(e) and len(base) > len(e) + 1:
                ending = e
                break

        return {
            'word': w,
            'root': root,
            'prefix': prefix,
            'suffix': suffix,
            'ending': ending,
            'pos': pos,
            'normal_form': normal_form,
            'features': features,
        }

    @staticmethod
    def morph_root_vector(word, cs):
        """Get root concept vector for a word's morphological root.

        Returns:
            (cid, vector) or (None, None) if root not found.
        """
        morph = ConceptTokenizer.morph_parse(word)
        if morph is None:
            return None, None

        # Try root (normal form) first
        root = morph['normal_form']
        cid = cs.word_to_cid.get(root)
        if cid is not None:
            v = cs.concept_vector(cid)
            if v is not None:
                return cid, v

        # Try original word
        cid = cs.word_to_cid.get(morph['word'])
        if cid is not None:
            v = cs.concept_vector(cid)
            if v is not None:
                return cid, v

        return None, None

    @staticmethod
    def word_morph_vector(word, cs, affix_shifts=None):
        """Assemble word vector from root concept + affix shifts.

        word_vector = root_concept_vector + prefix_shift + suffix_shift + ending_shift

        Args:
            word: input word
            cs: ConceptSpace instance
            affix_shifts: dict {affix: np.array(128D)} or None

        Returns:
            np.array(128D) or None if root not found
        """
        morph = ConceptTokenizer.morph_parse(word)
        if morph is None:
            return None

        cid, root_v = ConceptTokenizer.morph_root_vector(word, cs)
        if root_v is None:
            return None

        vec = root_v.copy()
        if affix_shifts:
            for affix_type in ('prefix', 'suffix', 'ending'):
                affix = morph.get(affix_type, '')
                if affix and affix in affix_shifts:
                    vec += affix_shifts[affix] * 0.3  # affix shift is weaker than root

        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec

    def morph_metadata(self, text):
        """Extract hierarchical morphological metadata from text.

        Returns:
            list of dicts, one per word: {word, root, prefix, suffix,
            ending, pos, features, cid, root_vector, word_vector}
        """
        result = []
        words = text.strip().split()
        for w in words:
            entry = ConceptTokenizer.morph_parse(w)
            if entry is None:
                result.append({'word': w, 'pos': 'UNK'})
                continue
            result.append(entry)
        return result


def train_character_bpe(corpus_path, vocab_size=8192, save_path=None):
    """Train a character-level BPE tokenizer (no ByteLevel)."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    save_path = save_path or BPE_PATH

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    # Whitespace pre-tokenizer: splits on whitespace, each "word" is independent
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train([corpus_path], trainer)
    tokenizer.save(save_path)
    print(f"Character BPE saved to {save_path} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


if __name__ == '__main__':
    tok = ConceptTokenizer()
    tok.initialize()

    print(f"Total vocab size: {len(tok)}")
    print(f"BPE vocab size: {tok.bpe_vocab_size}")
    print(f"Concepts: {tok.skeleton.n_concepts}")

    # Test encoding/decoding
    test_text = "Война и мир. Шедевр мировой литературы."
    ids = tok.encode(test_text)
    decoded = tok.decode(ids)

    print(f"\nOriginal: {test_text}")
    print(f"Encoded length: {len(ids)}")
    print(f"Decoded: {decoded}")
    print(f"Roundtrip OK: {test_text == decoded}")

    # Show metadata
    meta = tok.metadata_from_ids(ids)
    word_starts = [m for m in meta if m['flags'] & 1]
    print(f"\nWord-level metadata:")
    for m in word_starts:
        print(f"  word[{m['word_num']}] concept={m.get('concept_id')} "
              f"anchor={m.get('anchor_word')!r} is_anchor={m.get('is_anchor')}")
