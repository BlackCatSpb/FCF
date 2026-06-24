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

    def observe(self, char_ids: List[int], lr: float = 0.01):
        """STM + STDP update from a char sequence (e.g., a word's spelling)."""
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

        # Decay older bigrams
        decay = 0.999
        for k in list(self.char_bigram_cohesion.keys()):
            if k not in [(char_ids[i], char_ids[i + 1]) for i in range(len(char_ids) - 1)]:
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
        return n_new

    def decompose(self, char_ids: List[int]) -> List[Tuple[int, str]]:
        """Decompose char sequence into known morphemes + residue chars."""
        result = []
        i = 0
        while i < len(char_ids):
            found = False
            for morph_id, chars in self.morph_to_chars.items():
                if i + len(chars) <= len(char_ids) and char_ids[i:i + len(chars)] == chars:
                    result.append((morph_id, 'MORPH'))
                    i += len(chars)
                    found = True
                    break
            if not found:
                result.append((char_ids[i], 'CHAR'))
                i += 1
        return result


class CharEnvelope:
    """Character HD vectors — updated from STDP char→char bind learning.

    Each Unicode codepoint gets a unit-norm HD vector.
    Vectors are refined by STDP: chars that co-occur in similar contexts
    are pulled together.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vecs: Dict[int, np.ndarray] = {}
        self.context_traces: Dict[int, np.ndarray] = {}  # for STDP eligibility

    def ensure(self, cp: int) -> np.ndarray:
        """Get (or create) char vector."""
        if cp not in self.vecs:
            v = _R.rng('char_init').randn(self.dim).astype(np.float32)
            v /= max(np.linalg.norm(v), 1e-10)
            self.vecs[cp] = v.astype(np.float16)
        v = self.vecs[cp]
        return v.astype(np.float32) if hasattr(v, 'astype') else v

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
            w = max(0.0, 1.0 - dist * 0.2)  # Gaussian-ish decay
            delta = (c_center - v) * lr * w
            v_new = v + delta
            vn = float(np.linalg.norm(v_new))
            if vn > 1e-10:
                self.vecs[cp] = (v_new / vn).astype(np.float16)
