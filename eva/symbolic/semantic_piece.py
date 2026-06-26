"""SemanticPiece — VSA-native многоуровневая токенизация.

Слои (снизу вверх):
  BPE (bootstrapping, seed vectors)
  Char VSA (CharEnvelope) — Unicode codepoint → HD vector
  Morph VSA (STDP char→char bind) — обучение морфем без хардкода
  Word VSA (Harmonizer) — compose(bind(morphⱼ, roleⱼ))
  Sent VSA (EntityField) — cross-level char↔word↔sent↔para

BPE остаётся для инициализации seed vectors.
Все уровни работают в едином latent_dim пространстве.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R


class MorphSTDP:
    """STDP-driven morpheme discovery from char n-gram sequences.

    charID → HD vector (from CharEnvelope)
    char bigram bind: STDP strengthens bind(c₁, ρ(c₂)) when co-occurring
    High co-occurrence → "soft morpheme" → fixed as morph vector
    """

    def __init__(self, dim: int, morph_lr: float = 0.03, cohesion_threshold: float = 0.6):
        self.dim = dim
        self.morph_lr = morph_lr
        self.cohesion_threshold = cohesion_threshold

        # char → HD vector (populated by CharEnvelope)
        self.char_vecs: Dict[int, np.ndarray] = {}

        # char bigram bind → cohesion score
        self.char_bigram_cohesion: Dict[Tuple[int, int], float] = defaultdict(float)
        self.bigram_counts: Counter = Counter()

        # Discovered morphemes: morph_id → HD vector
        self.morphemes: Dict[int, np.ndarray] = {}
        self.morph_to_chars: Dict[int, List[int]] = {}  # morph → constituent char IDs

        # Role vectors for bigram binding (quasi-orthogonal per position)
        rng = _R.rng('morph_roles')
        self.role_left = rng.randn(dim).astype(np.float32)
        self.role_left /= np.linalg.norm(self.role_left)
        self.role_right = rng.randn(dim).astype(np.float32)
        self.role_right /= np.linalg.norm(self.role_right)

        # Trie for O(L) decompose (rebuilt on new morphs)
        self._decompose_trie: Dict[int, Dict[int, int]] = {}

    def bind_char(self, c1: int, c2: int) -> np.ndarray:
        """VSA bind of char pair: bind(c1, ρ(c2)) + bind(ρ(c1), c2)."""
        v1 = self.char_vecs.get(c1)
        v2 = self.char_vecs.get(c2)
        if v1 is None or v2 is None:
            return np.zeros(self.dim, dtype=np.float32)
        from eva.symbolic.concept_space import _hybrid_bind
        bound = _hybrid_bind(v1, self.role_right)
        bound = bound + _hybrid_bind(self.role_left, v2)
        bn = float(np.linalg.norm(bound))
        return bound / bn if bn > 1e-10 else bound

    def _ensure_char_vec(self, cp: int) -> np.ndarray:
        """Lazy-init a char vector if missing (seed-based, like CharEnvelope.ensure)."""
        if cp not in self.char_vecs:
            from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
            v = _R.rng(f'char_init_{cp}').randn(self.dim).astype(np.float32)
            v /= max(np.linalg.norm(v), 1e-10)
            self.char_vecs[cp] = v
        return self.char_vecs[cp]

    def observe(self, char_ids: List[int], lr: float = 0.01):
        """STM + STDP update from a char sequence (e.g., a word's spelling)."""
        # Auto-populate char_vecs for any unseen characters
        for cp in char_ids:
            self._ensure_char_vec(cp)
        for i in range(len(char_ids) - 1):
            c1, c2 = char_ids[i], char_ids[i + 1]
            key = (c1, c2)
            self.bigram_counts[key] += 1

            # STDP: bind(c1, ρ(c2)) → pull toward current char vectors
            bound = self.bind_char(c1, c2)
            if np.linalg.norm(bound) < 1e-10:
                continue

            # Strengthen char→char association
            self.char_bigram_cohesion[key] = (
                1.0 - lr) * self.char_bigram_cohesion[key] + lr * 1.0

        # Decay older bigrams (O(N+M) via set lookup, was O(N*M))
        decay = 0.999
        current_bigrams = {(char_ids[i], char_ids[i + 1]) for i in range(len(char_ids) - 1)}
        for k in list(self.char_bigram_cohesion.keys()):
            if k not in current_bigrams:
                self.char_bigram_cohesion[k] *= decay
                if self.char_bigram_cohesion[k] < 0.01:
                    del self.char_bigram_cohesion[k]

    def discover_morphemes(self, min_cohesion: float | None = None) -> int:
        """Pop-out: merge char bigrams with cohesion > threshold into morphs."""
        threshold = min_cohesion or self.cohesion_threshold
        n_new = 0

        # Find high-cohesion bigrams
        candidates = [(k, v) for k, v in self.char_bigram_cohesion.items()
                      if v > threshold and self.bigram_counts.get(k, 0) > 1]
        candidates.sort(key=lambda x: -x[1])

        used_chars = set()
        for (c1, c2), cohesion in candidates:
            if c1 in used_chars or c2 in used_chars:
                continue
            if c1 in self.char_vecs and c2 in self.char_vecs:
                morph_id = abs(hash((c1, c2))) % (2**31 - 1)
                if morph_id not in self.morphemes:
                    bound = self.bind_char(c1, c2)
                    bn = float(np.linalg.norm(bound))
                    if bn > 1e-10:
                        self.morphemes[morph_id] = (bound / bn).astype(np.float16)
                        self.morph_to_chars[morph_id] = [c1, c2]
                        used_chars.add(c1)
                        used_chars.add(c2)
                        n_new += 1
        if n_new > 0:
            self._rebuild_decompose_trie()
        return n_new

    def _rebuild_decompose_trie(self):
        """Build prefix-trie from morph_to_chars for O(L) decompose.

        Leaf nodes store {'_id': morph_id}; internal nodes are plain dicts.
        """
        trie: dict = {}
        for morph_id, chars in self.morph_to_chars.items():
            node = trie
            for ch in chars:
                node = node.setdefault(ch, {})
            node['_id'] = morph_id
        self._decompose_trie = trie

    def decompose(self, char_ids: List[int]) -> List[Tuple[int, str]]:
        """Decompose char sequence into known morphemes + residue chars (O(L) via trie)."""
        result = []
        i = 0
        while i < len(char_ids):
            node = self._decompose_trie
            best_morph_id = None
            best_len = 0
            j = i
            while j < len(char_ids) and char_ids[j] in node:
                node = node[char_ids[j]]
                if '_id' in node:
                    best_morph_id = node['_id']
                    best_len = j - i + 1
                j += 1
            if best_morph_id is not None:
                result.append((best_morph_id, 'MORPH'))
                i += best_len
            else:
                result.append((char_ids[i], 'CHAR'))
                i += 1
        return result


class CharEnvelope:
    """Character HD vectors — unified CharEnvelope.

    Each Unicode codepoint gets a unit-norm HD vector.
    Supports STDP learning, LFU eviction, word envelope composition,
    and VSA modulation of word vectors with char-level context.
    """

    def __init__(self, dim: int, max_chars: Optional[int] = None):
        self.dim = dim
        self.max_chars = max_chars
        self.vecs: Dict[int, np.ndarray] = {}
        self._access_count: Dict[int, int] = {}
        self.context_traces: Dict[int, np.ndarray] = {}

    def ensure(self, cp: int) -> np.ndarray:
        if cp not in self.vecs:
            if self.max_chars is not None and len(self.vecs) >= self.max_chars:
                evict = min(self._access_count, key=self._access_count.get)
                self.vecs.pop(evict, None)
                self._access_count.pop(evict, None)
            v = _R.rng(f'char_init_{cp}').randn(self.dim).astype(np.float32)
            v /= max(np.linalg.norm(v), 1e-10)
            v_f16 = v.astype(np.float16)
            vn = float(np.linalg.norm(v_f16))
            self.vecs[cp] = (v_f16 / vn).astype(np.float16) if vn > 1e-10 else v_f16
        self._access_count[cp] = self._access_count.get(cp, 0) + 1
        v = self.vecs[cp]
        return v.astype(np.float32) if hasattr(v, 'astype') else v

    def word_envelope(self, word_text: str):
        if not word_text:
            return None
        from eva.symbolic.concept_space import _hybrid_bind
        result = None
        for i, ch in enumerate(word_text):
            cv = self.ensure(ord(ch))
            shifted = np.roll(cv, i)
            result = shifted if result is None else _hybrid_bind(result, shifted)
        nrm = float(np.linalg.norm(result))
        return result / nrm if nrm > 1e-10 else result

    def modulate(self, word_vec, char_env, strength=0.05):
        from eva.symbolic.concept_space import _hybrid_bind
        bound = _hybrid_bind(char_env, word_vec)
        result = word_vec + bound * strength
        n = float(np.linalg.norm(result))
        return result / n if n > 1e-10 else result

    def stdp_update(self, cp_trace: List[int], lr: float = 0.01):
        """STDP: chars that co-occur in the same trace attract each other."""
        if len(cp_trace) < 2:
            return
        center = len(cp_trace) // 2
        c_center = self.ensure(cp_trace[center])
        for i, cp in enumerate(cp_trace):
            if i == center:
                continue
            v = self.ensure(cp)
            dist = abs(i - center)
            w = max(0.0, 1.0 - dist * 0.2)
            delta = (c_center - v) * lr * w
            v_new = v + delta
            vn = float(np.linalg.norm(v_new))
            if vn > 1e-10:
                self.vecs[cp] = (v_new / vn).astype(np.float16)
