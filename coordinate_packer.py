"""
COORDINATE PACKER — EVA Symbolic
384-d перфокарта: детерминированная упаковка текста в координаты.

Каждый h[t] — 384-мерный вектор с жёстко закреплёнными значениями:
  +1.0 = бит установлен
  -1.0 = бит сброшен
   0.0 = зарезервировано (для будущего трансформера)

384 dimension allocation:
┌─────────────┬──────────┬──────────────────────────────────────┐
│ Поле        │  Бит     │  Назначение                         │
├─────────────┼──────────┼──────────────────────────────────────┤
│ TOKEN       │  0-12    │  token_id (13 бит, покрывает 0-8191)│
│ POS_WORD    │  13-20   │  позиция в слове (до 256)           │
│ LEN_WORD    │  21-28   │  длина слова (до 256)               │
│ NUM_WORD    │  29-36   │  номер слова в предложении (до 256) │
│ POS_SENT    │  37-45   │  позиция в предложении (до 512)     │
│ LEN_SENT    │  46-54   │  длина предложения (до 512)         │
├─────────────┼──────────┼──────────────────────────────────────┤
│ FLAGS       │  55-72   │  бинарные флаги (18 шт):            │
│             │  55      │  word_start                          │
│             │  56      │  word_end                            │
│             │  57      │  sent_start                          │
│             │  58      │  sent_end                            │
│             │  59      │  is_bpe_subword (vs character)       │
│             │  60      │  is_special_token                    │
│             │  61      │  is_punctuation                      │
│             │  62      │  is_digit                            │
│             │  63      │  is_letter                           │
│             │  64      │  is_capitalized                      │
│             │  65      │  has_word_left_context               │
│             │  66      │  has_word_right_context              │
│             │  67      │  has_sent_left_context               │
│             │  68      │  has_sent_right_context              │
│             │  69-72   │  (запас)                             │
├─────────────┼──────────┼──────────────────────────────────────┤
│ META        │  73-80   │  тип BPE-токена (8 бит):           │
│             │          │    0 = char inside word              │
│             │          │    1 = char single-letter word       │
│             │          │    2 = BPE subword (start)           │
│             │          │    3 = BPE subword (continuation)    │
│             │          │    4 = BPE single-token word         │
│             │          │    5 = punctuation                   │
│             │          │    6 = special marker (boundary)     │
│             │          │    7 = reserved                      │
│ CONTEXT     │  81-88   │  n-gram fingerprint (8 бит):       │
│             │          │    hash(prev_tokens, next_tokens)    │
│ ID_MISC     │  89-96   │  идентификатор текста (8 бит):     │
│             │          │    book_id, para_id, ...             │
├─────────────┼──────────┼──────────────────────────────────────┤
│ RESERVED    │  97-383  │  зарезервировано для трансформера   │
│             │          │  (287 измерений, инициализируются 0) │
└─────────────┴──────────┴──────────────────────────────────────┘

Итого: 97 бит метаданных, 287 зарезервировано. Всего 384.
"""
import numpy as np

class CoordinatePacker:
    V = 4101       # vocab size
    DIM = 384
    
    # ─── Bit allocation ───
    TOKEN_BITS = 13      # covers 0-8191 (vocab 0-4100)
    POS_WORD_BITS = 8    # 0-255
    LEN_WORD_BITS = 8    # 0-255
    NUM_WORD_BITS = 8    # 0-255
    POS_SENT_BITS = 9    # 0-511
    LEN_SENT_BITS = 9    # 0-511
    
    # Offsets
    OFF_TOKEN = 0                       # dims 0-12
    OFF_POS_WORD = OFF_TOKEN + 13       # dims 13-20
    OFF_LEN_WORD = OFF_POS_WORD + 8     # dims 21-28
    OFF_NUM_WORD = OFF_LEN_WORD + 8     # dims 29-36
    OFF_POS_SENT = OFF_NUM_WORD + 8     # dims 37-45
    OFF_LEN_SENT = OFF_POS_SENT + 9     # dims 46-54
    OFF_FLAGS = OFF_LEN_SENT + 9        # dims 55-72 (18 flags)
    OFF_META = OFF_FLAGS + 18           # dims 73-80
    OFF_CONTEXT = OFF_META + 8          # dims 81-88
    OFF_ID = OFF_CONTEXT + 8            # dims 89-96
    OFF_RESERVED = OFF_ID + 8           # dims 97-383 (287 dims)
    
    # ─── Flag indices ───
    F_WORD_START = 0
    F_WORD_END = 1
    F_SENT_START = 2
    F_SENT_END = 3
    F_BPE_SUBWORD = 4
    F_SPECIAL = 5
    F_PUNCTUATION = 6
    F_DIGIT = 7
    F_LETTER = 8
    F_CAPITALIZED = 9
    F_HAS_WORD_LEFT = 10
    F_HAS_WORD_RIGHT = 11
    F_HAS_SENT_LEFT = 12
    F_HAS_SENT_RIGHT = 13
    F_RESERVED_0 = 14
    F_RESERVED_1 = 15
    F_RESERVED_2 = 16
    F_RESERVED_3 = 17
    
    N_FLAGS = 18
    
    def __init__(self):
        # Precompute bit masks for speed
        self._token_masks = [(dim, 1 << bit) for dim, bit in 
                            [(self.OFF_TOKEN + b, b) for b in range(self.TOKEN_BITS)]]
        self._pos_word_masks = [(self.OFF_POS_WORD + b, 1 << b) for b in range(self.POS_WORD_BITS)]
        self._len_word_masks = [(self.OFF_LEN_WORD + b, 1 << b) for b in range(self.LEN_WORD_BITS)]
        self._num_word_masks = [(self.OFF_NUM_WORD + b, 1 << b) for b in range(self.NUM_WORD_BITS)]
        self._pos_sent_masks = [(self.OFF_POS_SENT + b, 1 << b) for b in range(self.POS_SENT_BITS)]
        self._len_sent_masks = [(self.OFF_LEN_SENT + b, 1 << b) for b in range(self.LEN_SENT_BITS)]
    
    def _int_to_dims(self, value: int, n_bits: int, offset: int) -> dict:
        """Pack a small integer into ±1.0 dimensions (binary encoding).
        
        value: 0..(2^n_bits - 1)
        Returns: {dim_index: ±1.0}
        """
        result = {}
        for b in range(n_bits):
            bit_val = (value >> b) & 1
            result[offset + b] = 1.0 if bit_val else -1.0
        return result
    
    def _dims_to_int(self, coords: np.ndarray, n_bits: int, offset: int) -> int:
        """Unpack ±1.0 dimensions back to integer.
        Threshold: > 0 -> 1, ≤ 0 -> 0
        """
        value = 0
        for b in range(n_bits):
            if coords[offset + b] > 0.0:
                value |= (1 << b)
        return value
    
    def pack_token(self, token_id: int, pos_in_word: int = 0,
                   word_len: int = 0, word_num: int = 0,
                   pos_in_sent: int = 0, sent_len: int = 0,
                   flags: int = 0, meta_type: int = 0,
                   context_hash: int = 0, text_id: int = 0) -> np.ndarray:
        """
        Pack one token + metadata into 384-dim coordinate vector.
        All values: ±1.0 for metadata, 0.0 for reserved.
        """
        h = np.zeros(self.DIM, dtype=np.float32)
        
        # Token ID (13 bits)
        for dim, mask in self._token_masks:
            h[dim] = 1.0 if (token_id & mask) else -1.0
        
        # Position in word (8 bits)
        for dim, mask in self._pos_word_masks:
            h[dim] = 1.0 if (pos_in_word & mask) else -1.0
        
        # Word length (8 bits)
        for dim, mask in self._len_word_masks:
            h[dim] = 1.0 if (word_len & mask) else -1.0
        
        # Word number (8 bits)
        for dim, mask in self._num_word_masks:
            h[dim] = 1.0 if (word_num & mask) else -1.0
        
        # Position in sentence (9 bits)
        for dim, mask in self._pos_sent_masks:
            h[dim] = 1.0 if (pos_in_sent & mask) else -1.0
        
        # Sentence length (9 bits)
        for dim, mask in self._len_sent_masks:
            h[dim] = 1.0 if (sent_len & mask) else -1.0
        
        # Flags (18 bits)
        for f in range(self.N_FLAGS):
            h[self.OFF_FLAGS + f] = 1.0 if (flags >> f) & 1 else -1.0
        
        # Meta type (8 bits)
        for b in range(8):
            h[self.OFF_META + b] = 1.0 if (meta_type >> b) & 1 else -1.0
        
        # Context hash (8 bits)
        for b in range(8):
            h[self.OFF_CONTEXT + b] = 1.0 if (context_hash >> b) & 1 else -1.0
        
        # Text ID (8 bits)
        for b in range(8):
            h[self.OFF_ID + b] = 1.0 if (text_id >> b) & 1 else -1.0
        
        # Reserved dims (97-383) stay 0.0
        
        return h
    
    def unpack_token(self, h: np.ndarray) -> dict:
        """
        Unpack 384-dim coordinate back to all metadata.
        100% deterministic, threshold at 0.0.
        """
        return {
            'token_id': self._dims_to_int(h, self.TOKEN_BITS, self.OFF_TOKEN),
            'pos_in_word': self._dims_to_int(h, self.POS_WORD_BITS, self.OFF_POS_WORD),
            'word_len': self._dims_to_int(h, self.LEN_WORD_BITS, self.OFF_LEN_WORD),
            'word_num': self._dims_to_int(h, self.NUM_WORD_BITS, self.OFF_NUM_WORD),
            'pos_in_sent': self._dims_to_int(h, self.POS_SENT_BITS, self.OFF_POS_SENT),
            'sent_len': self._dims_to_int(h, self.LEN_SENT_BITS, self.OFF_LEN_SENT),
            'flags': self._dims_to_int(h, self.N_FLAGS, self.OFF_FLAGS),
            'meta_type': self._dims_to_int(h, 8, self.OFF_META),
            'context_hash': self._dims_to_int(h, 8, self.OFF_CONTEXT),
            'text_id': self._dims_to_int(h, 8, self.OFF_ID),
        }
    
    def encode_sentence(self, ids, word_boundaries=None,
                        sent_start=0, sent_end=None,
                        text_id=0) -> np.ndarray:
        """
        Encode a complete sentence into a trajectory [L, 384].
        
        Args:
            ids: list of token IDs (including special tokens)
            word_boundaries: list of (start, end) tuples
            sent_start: index of sentence start token
            sent_end: index of sentence end token
            text_id: source text identifier
        
        Returns:
            trajectory: [L, 384] numpy array
        """
        L = len(ids)
        if sent_end is None:
            sent_end = L - 1
        
        if word_boundaries is None:
            word_boundaries = self._infer_word_boundaries(ids)
        
        # Number of words
        n_words = len(word_boundaries)
        
        trajectory = np.zeros((L, self.DIM), dtype=np.float32)
        
        for t in range(L):
            tid = ids[t]
            
            # Determine which word this token belongs to
            word_idx = -1
            pos_in_word = -1
            word_len = 0
            for wi, (ws, we) in enumerate(word_boundaries):
                if ws <= t <= we:
                    word_idx = wi
                    pos_in_word = t - ws
                    word_len = we - ws + 1
                    break
            
            # Compute flags
            flags = 0
            if word_idx >= 0:
                # This token is inside a word
                if pos_in_word == 0:
                    flags |= (1 << self.F_WORD_START)
                if pos_in_word == word_len - 1:
                    flags |= (1 << self.F_WORD_END)
                if word_idx > 0:
                    flags |= (1 << self.F_HAS_WORD_LEFT)
                if word_idx < n_words - 1:
                    flags |= (1 << self.F_HAS_WORD_RIGHT)
            else:
                # Boundary/special token (WORD_OPEN, WORD_CLOSE, etc.)
                flags |= (1 << self.F_SPECIAL)
            
            # Determine if this is start/end of sentence
            if word_idx >= 0:
                if t == 0 or word_idx == 0:
                    flags |= (1 << self.F_SENT_START)
                if t == L - 1 or word_idx == n_words - 1:
                    flags |= (1 << self.F_SENT_END)
            
            # Outside sentence span (before SENT_OPEN, after SENT_CLOSE)
            if t < sent_start:
                pass  # no sent flags
            if t >= sent_end:
                pass  # no sent flags
            
            # Meta type
            meta_type = self._get_meta_type(tid, word_idx, pos_in_word)
            
            # Token type flags
            if self._is_punctuation(tid):
                flags |= (1 << self.F_PUNCTUATION)
            if self._is_digit(tid):
                flags |= (1 << self.F_DIGIT)
            if self._is_letter(tid):
                flags |= (1 << self.F_LETTER)
            
            # Simple context hash (optional)
            ctx = self._context_hash(ids, t)
            
            trajectory[t] = self.pack_token(
                token_id=tid,
                pos_in_word=max(0, pos_in_word),
                word_len=max(1, word_len),
                word_num=max(0, word_idx),
                pos_in_sent=t,
                sent_len=L,
                flags=flags,
                meta_type=meta_type,
                context_hash=ctx,
                text_id=text_id,
            )
        
        return trajectory
    
    def decode_sentence(self, trajectory: np.ndarray) -> list:
        """Decode [L, 384] trajectory back to token IDs. 100% reversible."""
        L = trajectory.shape[0]
        ids = []
        for t in range(L):
            info = self.unpack_token(trajectory[t])
            ids.append(info['token_id'])
        return ids
    
    # ─── Helper methods ───
    
    def _infer_word_boundaries(self, ids):
        """Simple heuristic: treat consecutive letters as words."""
        WORD_OPEN = 157
        WORD_CLOSE = 158
        boundaries = []
        in_word = False
        start = 0
        for i, tid in enumerate(ids):
            if tid == WORD_OPEN:
                in_word = True
                start = i + 1
            elif tid == WORD_CLOSE and in_word:
                boundaries.append((start, i))
                in_word = False
        return boundaries
    
    def _get_meta_type(self, tid, word_idx, pos_in_word):
        WORD_OPEN = 157
        WORD_CLOSE = 158
        SENT_OPEN = 159
        SENT_CLOSE = 160
        
        if tid in (WORD_OPEN, WORD_CLOSE, SENT_OPEN, SENT_CLOSE):
            return 6  # special marker
        if self._is_punctuation(tid):
            return 5
        if pos_in_word == 0 and word_idx >= 0:
            return 2 if self._has_multiple_chars(tid) else 1
        if pos_in_word > 0:
            return 0 if not self._is_complete_word else 3
        return 7  # reserved
    
    def _is_punctuation(self, tid):
        return tid == 156  # PUNCT token or similar
    
    def _is_digit(self, tid):
        return 48 <= tid <= 57  # ASCII digits
    
    def _is_letter(self, tid):
        """Is this a letter token (character or BPE subword)?"""
        # BPE tokens start at some offset; characters are 4-155
        return 4 <= tid <= 155 or tid >= 161  # excluding special 156-160
    
    def _has_multiple_chars(self, tid):
        """Does this BPE token represent multiple characters?"""
        return tid >= 161  # BPE subwords
    
    def _is_complete_word(self, tid):
        return False  # placeholder
    
    def _context_hash(self, ids, t, window=3):
        """Simple hash of surrounding tokens for context fingerprint."""
        h = 0
        for i in range(max(0, t-window), min(len(ids), t+window+1)):
            if i != t:
                h ^= (ids[i] & 0xFF) << ((i - t + window) & 7)
        return h & 0xFF
    
    def verify(self, num_tests=10000):
        """Verify 100% reversibility on random valid metadata."""
        import random
        errors = 0
        for _ in range(num_tests):
            token_id = random.randint(0, self.V - 1)
            pos_w = random.randint(0, 255)
            len_w = random.randint(1, 255)
            num_w = random.randint(0, 255)
            pos_s = random.randint(0, 511)
            len_s = random.randint(1, 511)
            flags = random.randint(0, (1 << self.N_FLAGS) - 1)
            meta = random.randint(0, 255)
            ctx = random.randint(0, 255)
            tid = random.randint(0, 255)
            
            h = self.pack_token(token_id, pos_w, len_w, num_w, pos_s, len_s,
                                flags, meta, ctx, tid)
            info = self.unpack_token(h)
            
            if (info['token_id'] != token_id or
                info['pos_in_word'] != pos_w or
                info['word_len'] != len_w or
                info['word_num'] != num_w or
                info['pos_in_sent'] != pos_s or
                info['sent_len'] != len_s or
                info['flags'] != flags or
                info['meta_type'] != meta or
                info['context_hash'] != ctx or
                info['text_id'] != tid):
                errors += 1
        
        return errors == 0, f"0/{num_tests} errors" if errors == 0 else f"{errors}/{num_tests} errors"


# ─── Test ───
if __name__ == '__main__':
    cp = CoordinatePacker()
    
    ok, msg = cp.verify(10000)
    print(f"Verify 100% reversibility: {ok} ({msg})")
    
    # Test with a real sentence
    print("\n" + "="*60)
    print("TEST: encode -> decode roundtrip")
    print("="*60)
    
    # Simulate tokenized sentence
    test_ids = [2, 334, 2616, 676, 3379, 405, 1497, 1989, 3]
    # 2=BOS, 334='на', 2616=' ули', 676='це', 3379=' хорош', 405='ая', 
    # 1497=' пог', 1989='ода', 3=EOS
    
    boundaries = [(1, 3), (4, 5), (6, 7)]  # слов: (на), (улице), (хорошая), (погода)
    # Adjusted for [BOS, na, uli, tse, kho, rosh, aya, po, goda, EOS]
    # Actually these are BPE tokens, not characters. Let me use proper boundaries.
    boundaries = [(1, 1), (2, 3), (4, 5), (6, 7)]
    # Word 1: token 1 (334='на')
    # Word 2: tokens 2-3 (2616=' ули', 676='це')
    # Word 3: tokens 4-5 (3379=' хорош', 405='ая')
    # Word 4: tokens 6-7 (1497=' пог', 1989='ода')
    
    traj = cp.encode_sentence(test_ids, boundaries, text_id=0)
    
    print(f"  Input IDs:     {test_ids}")
    print(f"  Trajectory:    [{traj.shape[0]} x {traj.shape[1]}]")
    print(f"  Norms per pos: {[f'{traj[t].sum():.1f}' for t in range(len(test_ids))]}")
    
    decoded = cp.decode_sentence(traj)
    print(f"  Decoded IDs:   {decoded}")
    print(f"  Exact match:   {decoded == test_ids}")
    
    # Show individual dims for position 0
    print(f"\n  Dims 0-12 (token_id, pos=1):")
    bits = ''.join(['1' if traj[1, d] > 0 else '0' for d in range(13)])
    print(f"    binary: {bits} = {int(bits[::-1], 2)}")
    print(f"    expected: {test_ids[1]} (334)")
    
    # Show flags for a boundary position
    print(f"\n  Flags for pos 0 (BOS):")
    flag_names = ['word_start','word_end','sent_start','sent_end',
                  'bpe_subword','special','punct','digit','letter',
                  'capitalized','has_word_left','has_word_right',
                  'has_sent_left','has_sent_right']
    for f in range(min(14, cp.N_FLAGS)):
        val = traj[0, cp.OFF_FLAGS + f]
        if val > 0:
            print(f"    {flag_names[f]}: ON")
    
    print(f"\n  All dims nonzero: {(traj != 0).sum()}")
    print(f"  Dims 0-96 nonzero:   {(traj[:, :97] != 0).sum()}/{97*len(test_ids)}")
    print(f"  Dims 97-383 nonzero: {(traj[:, 97:] != 0).sum()}/{(384-97)*len(test_ids)}")
    print(f"    (all reserved dims should be 0 -> {((traj[:, 97:] != 0).sum() == 0)})")
