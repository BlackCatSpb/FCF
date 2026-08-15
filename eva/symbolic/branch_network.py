"""CollocationMatrix — λ_d + PMI матрица языка.

Единый механизм осмысленной генерации без softmax.

colloc[context][target] = (1-β) · λ_d^(-α·dist·F^(d)_n) + β · PMI(context, target)

Где:
  λ_d^(-dist) — VSA-prior: семантически близкие векторы → высокий вес
  PMI(context, target) — насколько сочетание информативнее случайного
  β = (λ_d-1)/λ_d — скорость, с которой данные перевешивают λ_d-структуру

Никакого softmax. Никаких эмпирических констант.
"""

from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import numpy as np
from eva.symbolic.fibonacci_utils import FibonacciUtils as _FU
if TYPE_CHECKING:
    from eva.symbolic.concept_space import ConceptVectorStore


# ── λ_d-производные конфигурации уровней ──

def _level_lam(d: int) -> float:
    return 2.0 if d == 1 else _FU.get_lambda(d)


def _level_capacity(d: int) -> int:
    return _FU.get_generalized(d + 8, d)


def _mixing_ratio(lam: float) -> float:
    return (lam - 1.0) / lam


def _pmi(count_st: int, count_s: int, count_t: int, total: int) -> float:
    """PMI = log(P(s,t) / (P(s)·P(t))), clamped to [0, 1].  При total=1 — максимум."""
    if count_st < 1 or count_s < 1 or count_t < 1 or total < 1:
        return 0.0
    if total == 1:
        return 1.0  # единственное наблюдение — максимальная информативность
    p_st = count_st / total
    p_s = count_s / total
    p_t = count_t / total
    if p_s * p_t <= 0.0:
        return 0.0
    pmi = math.log2(p_st / (p_s * p_t))
    max_pmi = math.log2(total / min(count_s, count_t))
    if max_pmi <= 0.0:
        return 0.0
    return min(1.0, pmi / max_pmi)


@dataclass
class LevelConfig:
    d: int
    lam: float
    capacity: int
    alpha: float       # λ_d-распад по косинусному расстоянию
    mix_rate: float    # (λ_d-1)/λ_d — доля PMI

    @classmethod
    def build(cls, d: int) -> LevelConfig:
        lam = _level_lam(d)
        cap = _level_capacity(d)
        ref = _FU.get_generalized(d + 4, d)
        alpha = (lam - 1.0) / max(ref, 1)
        return cls(d=d, lam=lam, capacity=cap,
                   alpha=alpha, mix_rate=_mixing_ratio(lam))


class CollocationMatrix:
    """λ_d + PMI матрица сочетаемости. Единственный источник для генерации.

    Каждый уровень (1–4) хранит разреженную матрицу colloc[s][t].
    Вес = λ_d-prior + PMI-коррекция. Никакого softmax.

    Уровень 2 использует concept ID (cid) в качестве ключей, остальные —
    хэш-ключи от векторов.  CID-ключи STDP-safe: векторы могут меняться,
    а ключ (cid) остаётся тем же.
    """

    def __init__(self, dim: int = 768, lam: float | None = None,
                 concept_store: ConceptVectorStore | None = None):
        self.dim = dim
        self.lam = lam or _FU.get_lambda(2)
        self._concept_store = concept_store  # для λ_d-prior по cid

        # Конфигурация уровней
        self.levels: dict[int, LevelConfig] = {}
        for d in range(1, 5):
            cfg = LevelConfig.build(d)
            if d == 2 and lam is not None:
                cfg.lam = lam
                cfg.alpha = (lam - 1.0) / max(_FU.get_generalized(d + 4, d), 1)
                cfg.mix_rate = _mixing_ratio(lam)
            self.levels[d] = cfg

        # ── Хэш-основанное хранение (уровни 1, 3, 4) ──
        # level → {hash(s) → {hash(t) → count}}
        self._counts: dict[int, dict[int, Counter]] = {1: {}, 2: {}, 3: {}, 4: {}}
        # level → {hash(s) → {hash(t) → penalty}}
        self._decays: dict[int, dict[int, Counter]] = {1: {}, 2: {}, 3: {}, 4: {}}
        # level → total transitions (vec-based)
        self._total: Counter = Counter()
        # level → {hash(t) → total_incoming}
        self._incoming: dict[int, Counter] = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}

        # ── CID-основанное хранение (уровень 2 — STDP-safe) ──
        # level → {src_cid → {tgt_cid → count}}
        self._counts_cid: dict[int, dict[int, Counter]] = {1: {}, 2: {}, 3: {}, 4: {}}
        # level → {tgt_cid → total_incoming}
        self._incoming_cid: dict[int, Counter] = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
        # level → total cid-based transitions
        self._total_cid: Counter = Counter()
        # level → {src_cid → {tgt_cid → penalty}}
        self._decays_cid: dict[int, dict[int, Counter]] = {1: {}, 2: {}, 3: {}, 4: {}}

    # ── Единый вес коллокации ──

    def __call__(self, level: int, s: np.ndarray | int, t: np.ndarray | int) -> float:
        """colloc[s][t] = (1-β)·λ_d-prior·damping + β·PMI.

        Если s и t — int, используются cid-ключи (STDP-safe, уровень 2).
        Если np.ndarray — классические хэш-ключи.
        """
        if isinstance(s, (int, np.integer)):
            return self._score_cid(level, int(s), int(t))
        return self._score_vec(level, s, t)

    def _score_vec(self, level: int, s: np.ndarray, t: np.ndarray) -> float:
        """Вес коллокации по хэш-ключам от векторов (уровни 1, 3, 4)."""
        cfg = self.levels[level]

        # λ_d-prior: семантическая близость из VSA
        cos = float(np.dot(s, t) / (np.linalg.norm(s) * np.linalg.norm(t) + 1e-10))
        dist = 1.0 - cos
        prior = float(cfg.lam ** (-cfg.alpha * dist * cfg.capacity))

        # Штраф: damping = λ_d^(-decay)
        sk = self._key(s)
        tk = self._key(t)
        decay_st = 0
        if sk in self._decays[level] and tk in self._decays[level][sk]:
            decay_st = self._decays[level][sk][tk]
        damping = float(cfg.lam ** (-decay_st))
        prior_damped = prior * damping

        # PMI
        src = self._counts[level].get(sk)
        count_st = max(0, (src.get(tk, 0) if src is not None else 0) - decay_st)
        count_s = sum(src.values()) if src is not None else 0
        count_t = self._incoming[level].get(tk, 0)
        pmi = _pmi(count_st, count_s, count_t, self._total[level])

        if pmi <= 0.0:
            return prior_damped
        return (1.0 - cfg.mix_rate) * prior_damped + cfg.mix_rate * pmi

    def _score_cid(self, level: int, src_cid: int, tgt_cid: int) -> float:
        """Вес коллокации по cid-ключам (уровень 2, STDP-safe)."""
        cfg = self.levels[level]
        # Проверка границ: cid может быть > vocab_size (BOS/OOV из BPE)
        V = self._concept_store.size if self._concept_store is not None else 0
        if src_cid < 0 or src_cid >= V or tgt_cid < 0 or tgt_cid >= V:
            return 0.0
        if not self._concept_store.valid[src_cid] or not self._concept_store.valid[tgt_cid]:
            return 0.0

        # λ_d-prior через concept vectors (STDP может их менять, но cos
        # пересчитывается на лету — это нормально)
        s = self._concept_store.data[src_cid]
        t = self._concept_store.data[tgt_cid]
        cos = float(np.dot(s, t) / (np.linalg.norm(s) * np.linalg.norm(t) + 1e-10))
        dist = 1.0 - cos
        prior = float(cfg.lam ** (-cfg.alpha * dist * cfg.capacity))

        # Штраф
        decay_st = 0
        src_d = self._decays_cid[level].get(src_cid)
        if src_d is not None:
            decay_st = src_d.get(tgt_cid, 0)
        damping = float(cfg.lam ** (-decay_st))
        prior_damped = prior * damping

        # PMI по cid-счётчикам (STDP не меняет cid)
        src = self._counts_cid[level].get(src_cid)
        count_st = max(0, (src.get(tgt_cid, 0) if src is not None else 0) - decay_st)
        count_s = sum(src.values()) if src is not None else 0
        count_t = self._incoming_cid[level].get(tgt_cid, 0)
        total = self._total_cid[level]
        pmi = _pmi(count_st, count_s, count_t, total)

        if pmi <= 0.0:
            return prior_damped
        return (1.0 - cfg.mix_rate) * prior_damped + cfg.mix_rate * pmi

    # ── Обучение: STDP-наблюдение переходов ──

    def observe(self, level: int, s: np.ndarray | int, t: np.ndarray | int,
                weight: float = 1.0) -> None:
        """Записать переход s → t.  s/t могут быть int (cid) или ndarray."""
        if isinstance(s, (int, np.integer)):
            return self._observe_cid(level, int(s), int(t), weight)
        self._observe_vec(level, s, t, weight)

    def _observe_vec(self, level: int, s: np.ndarray, t: np.ndarray,
                     weight: float = 1.0) -> None:
        sk = self._key(s)
        tk = self._key(t)
        if sk not in self._counts[level]:
            self._counts[level][sk] = Counter()
        self._counts[level][sk][tk] += weight
        self._incoming[level][tk] += weight
        self._total[level] += weight

    def _observe_cid(self, level: int, src_cid: int, tgt_cid: int,
                     weight: float = 1.0) -> None:
        """Записать переход по cid (STDP-safe — ключ не зависит от вектора)."""
        V = self._concept_store.size if self._concept_store is not None else 0
        if src_cid < 0 or src_cid >= V or tgt_cid < 0 or tgt_cid >= V:
            return
        if not self._concept_store.valid[src_cid] or not self._concept_store.valid[tgt_cid]:
            return
        if src_cid not in self._counts_cid[level]:
            self._counts_cid[level][src_cid] = Counter()
        self._counts_cid[level][src_cid][tgt_cid] += weight
        self._incoming_cid[level][tgt_cid] += weight
        self._total_cid[level] += weight

    def decay(self, level: int, s: np.ndarray | int, t: np.ndarray | int,
              strength: float = 1.0) -> None:
        """Наказать переход s → t.  s/t могут быть int (cid) или ndarray."""
        if isinstance(s, (int, np.integer)):
            return self._decay_cid(level, int(s), int(t), strength)
        self._decay_vec(level, s, t, strength)

    def _decay_vec(self, level: int, s: np.ndarray, t: np.ndarray,
                   strength: float = 1.0) -> None:
        sk = self._key(s)
        tk = self._key(t)
        if sk not in self._decays[level]:
            self._decays[level][sk] = Counter()
        self._decays[level][sk][tk] += strength

    def _decay_cid(self, level: int, src_cid: int, tgt_cid: int,
                   strength: float = 1.0) -> None:
        V = self._concept_store.size if self._concept_store is not None else 0
        if src_cid < 0 or src_cid >= V or tgt_cid < 0 or tgt_cid >= V:
            return
        if not self._concept_store.valid[src_cid] or not self._concept_store.valid[tgt_cid]:
            return
        if src_cid not in self._decays_cid[level]:
            self._decays_cid[level][src_cid] = Counter()
        self._decays_cid[level][src_cid][tgt_cid] += strength

    def hormonal_learn(self, level: int,
                       prev: np.ndarray | int, generated: np.ndarray | int,
                       is_match: bool, surprise: float,
                       expected: np.ndarray | None = None,
                       novelty: float = 0.0) -> None:
        """Гормональное обучение: поощрение/наказание перехода.

        DA (is_match=True) → observe — укрепить правильную ветвь.
        NA (surprise > 0.3) → decay — ослабить неожиданную ветвь.
        ACh (novelty) → усилить observe при новизне (повторные observe).
        """
        if is_match:
            self.observe(level, prev, generated)
            if novelty > 0.3:
                extra = min(int(novelty * 3), 5)
                for _ in range(extra):
                    self.observe(level, prev, generated)
            if expected is not None:
                self.observe(level, prev, expected)
        elif expected is not None:
            self.decay(level, prev, generated, strength=min(surprise * 2, 5.0))
            self.observe(level, prev, expected)
        elif surprise > 0.3:
            self.decay(level, prev, generated, strength=surprise)

    # ── Генерация: выбор следующего элемента без softmax ──

    def generate(self, level: int, context: np.ndarray,
                 candidates: list[np.ndarray],
                 temperature: float = 0.5,
                 top_k: int = 5,
                 exclude: np.ndarray | None = None,
                 rng: np.random.Generator | None = None) -> int:
        """Выбрать следующий элемент из candidates пропорционально colloc[context].

        context может быть ndarray (вектор) или int (cid).  Для level=2
        используйте cid, для остальных — вектор.

        Никакого softmax.  Только colloc[context][target] как вес.
        """
        weights = np.array([
            self(level, context, t) for t in candidates
        ], dtype=np.float64)

        # Top-K: учитывать только K лучших
        if 0 < top_k < len(weights):
            cutoff = np.sort(weights)[-top_k]
            weights[weights < cutoff] = 0.0

        # Исключить self (cos ≈ 1)
        if exclude is not None:
            for i, t in enumerate(candidates):
                if float(np.dot(t, exclude)) > 0.99:
                    weights[i] = 0.0

        # Пропорциональный выбор без softmax
        total = float(weights.sum())
        if total < 1e-12:
            weights[:] = 1.0
            total = float(len(weights))

        # Temperature: возвести в степень (не softmax!)
        if temperature > 0.0 and abs(temperature - 1.0) > 1e-10:
            weights = weights ** (1.0 / temperature)
            total = float(weights.sum())

        if rng is not None:
            r = float(rng.random()) * total
        else:
            r = float(np.random.random()) * total
        cumsum = 0.0
        for i in range(len(weights)):
            cumsum += weights[i]
            if r < cumsum:
                return i
        return len(weights) - 1

    # ── N-грам контекст ──

    @staticmethod
    def ngram_context(seq: list[np.ndarray], lam: float) -> np.ndarray:
        """λ_d-взвешенный bundle последовательности векторов."""
        ctx = np.zeros_like(seq[0]) if seq else np.array([], dtype=np.float32)
        for k, v in enumerate(reversed(seq)):
            ctx += (lam ** (-k)) * v
        nrm = float(np.linalg.norm(ctx))
        if nrm > 1e-10:
            ctx /= nrm
        return ctx

    # ── Внутреннее ──

    def _key(self, v: np.ndarray) -> int:
        """Детерминированный хеш первых 256 dims (int32@1e3)."""
        coarse = np.round(v[:256] * 1e3).astype(np.int32)
        # Используем FNV-1a для детерминированности (вместо hash())
        h = 2166136261
        for b in coarse.tobytes():
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def reset_level(self, level: int):
        self._counts[level].clear()
        self._incoming[level].clear()
        self._counts_cid[level].clear()
        self._incoming_cid[level].clear()

    def __len__(self) -> int:
        return sum(len(v) for v in self._counts.values())

    def __repr__(self) -> str:
        total = sum(self._total.values())
        return (f'<CollocationMatrix lam={self.lam:.4f} '
                f'transitions={total} entries={len(self)}>')
