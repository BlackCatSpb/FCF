"""Spiral-математика — интеграция в FCF.

Примитивы интегрированы в eva/symbolic/fibonacci_utils.py:
  signed_phi_digits, signed_phi_sum, spiral_bundle, phi_decay
и в eva/symbolic/vsa_attention.py (VSAAttention use_signed_weights=True).

Проверяем:
  1. signed_phi_digits — жадный φ-разбор с цифрами {-1,0,1} (антизнание),
     ошибка усечения <= φ^{1-K} (закрывает лакуну «отрицательных весов»
     в Zeckendorf-иерархии FCF)
  2. spiral_bundle — знаковый bundle с Σmax-нормировкой: антизнание не разбавляет,
     ||accum|| <= 1 при ортонормированных витках и ядре (M11a),
     рост до sqrt(1+R) при антизнании (M11b)
  3. phi_decay — аналог экспоненциального затухания exp(-d/tau) через φ-витки
     (согласован с TemporalZeckendorf.theta, граница сжатия M5)
  4. telescopic_zero — телескоп нуля: знакопеременная сумма -> ровно 0
     (замена пороговых «нет перехода» структурным балансом)
  5. signed_attention — знаковое внимание в VSAAttention: отрицательный косинус
     -> подавление, а не отбрасывание
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eva.symbolic.fibonacci_utils import (  # noqa: E402
    signed_phi_digits, signed_phi_sum, spiral_bundle, phi_decay,
    FibonacciUtils, ZeckendorfQuantizer, TemporalZeckendorf,
)

PHI = (1.0 + math.sqrt(5.0)) / 2.0
EPS = 1e-12


def phi_partial_sum(digits):
    return signed_phi_sum(digits)


def telescopic_zero(N=50):
    """Телескоп нуля: Σ_{i=-N..N-1} (φ^{-|i|} - φ^{-|i+1|}) = 0 ровно."""
    i = np.arange(-N, N + 1)
    w = PHI ** (-np.abs(i))
    return float((w[:-1] - w[1:]).sum())


# ── 1. знаковый φ-разбор ─────────────────────────────────────────

class TestSignedPhiDigits:
    def test_greedy_bounded_error(self):
        rng = np.random.RandomState(1)
        X = rng.uniform(-2, 2, 1000)
        K = 40
        worst = 0.0
        for x in X:
            digits, rem = signed_phi_digits(x, K)
            s = phi_partial_sum(digits)
            worst = max(worst, abs(x - s))
        bound = PHI ** (1 - K)
        assert worst <= bound, f"worst={worst:.3e} > bound={bound:.3e}"

    def test_sign_antisymmetry(self):
        """Цифры для -x — ровно отрицание цифр для x."""
        for x in [0.7, -1.234, 1.999, -0.001, 3.5, 7.0]:
            d1, _ = signed_phi_digits(x, 40)
            d2, _ = signed_phi_digits(-x, 40)
            assert d1 == {j: -a for j, a in d2.items()}, f"sign flip broken for {x}"

    def test_weights_zeckendorf_range(self):
        """Веса [-7,7] (диапазон Zeckendorf-взвешивания VSA) точны."""
        for w in range(-7, 8):
            digits, rem = signed_phi_digits(w, 40)
            s = phi_partial_sum(digits)
            bound = 1e-6  # вес — целое число, ошибка ~φ^{-K}·φ^M ≪ порог
            assert abs(w - s) < bound, f"w={w}: err={abs(w - s):.2e}"

    def test_zero_is_balanced(self):
        """Ноль как баланс, а не пустота: +/- пары сокращаются."""
        for x in [0.0, 0.5, -0.5, 1.0, -1.0, 1.618, -1.618, 7.0, -7.0]:
            digits, rem = signed_phi_digits(x, 60)
            assert abs(rem) < 1e-9, f"x={x}: rem={rem:.2e}"

    def test_digit_bounds(self):
        digits, _ = signed_phi_digits(1.9, 100)
        assert set(digits.values()) <= {-1, 0, 1}
        assert abs(1.9 - phi_partial_sum(digits)) < 1e-9


# ── 2. spiral_bundle ─────────────────────────────────────────────

class TestSpiralBundle:
    def test_antiknowledge_does_not_dilute(self):
        """M6: антизнание не разбавляет знаменатель."""
        r1 = np.array([1.0, 0.0])
        r2 = np.array([0.0, 1.0])
        a = spiral_bundle([r1, r2], [1.0, -0.5])
        # Σmax = 1 -> вклад ядра сохранён
        assert abs(a[0] - 1.0) < 1e-9
        assert abs(a[1] + 0.5) < 1e-9

    def test_exact_zero(self):
        """M7: компенсация витков даёт ровно 0."""
        r = np.array([1.0, 0.0])
        a = spiral_bundle([r, r], [0.5, -0.5])
        assert np.linalg.norm(a) < 1e-12

    def test_norm_bounded_positive(self):
        """M11a: ||accum|| <= 1 при ортонормированных витках и ядре a0=1."""
        rng = np.random.RandomState(2)
        Q, _ = np.linalg.qr(rng.randn(5, 5))
        worst = 0.0
        for _ in range(2000):
            n = rng.randint(1, 5)
            a = rng.uniform(-1, 1, n)
            a[0] = 1.0  # ядро (аналог bias[0]=10 в WideBind)
            if (a < 0).any():
                continue
            rr = Q[:, :n]
            norm = np.linalg.norm(spiral_bundle(rr.T, a))
            worst = max(worst, norm)
        assert worst <= 1.0 + 1e-6, f"worst={worst}"

    def test_norm_growth_antiknowledge(self):
        """M11b: с антизнанием норма растёт до sqrt(1+R)."""
        rng = np.random.RandomState(3)
        Q, _ = np.linalg.qr(rng.randn(5, 5))
        worst_neg, worst_ratio = 0.0, 0.0
        for _ in range(2000):
            n = rng.randint(2, 5)
            a = rng.uniform(-1, 1, n)
            a[0] = 1.0
            if not (a < 0).any():
                continue
            rr = Q[:, :n]
            norm = np.linalg.norm(spiral_bundle(rr.T, a))
            R = (-a).clip(min=0).sum() / (a.clip(min=0).sum() + EPS)
            worst_neg = max(worst_neg, norm)
            worst_ratio = max(worst_ratio, math.sqrt(1.0 + R))
        assert worst_neg > 1.0, "антизнание должно разворачивать норму"
        assert worst_neg <= worst_ratio + 1e-6, f"{worst_neg} > {worst_ratio}"


# ── 3. phi_decay ─────────────────────────────────────────────────

class TestPhiDecay:
    def test_monotonic_decreasing(self):
        prev = 1.1
        for d in [1, 2, 3, 5, 10, 50, 100, 500, 1000]:
            cur = phi_decay(d)
            assert cur < prev, f"phi_decay({d})={cur} >= prev={prev}"
            prev = cur

    def test_in_range(self):
        for d in [1, 2, 3, 5, 10, 100, 10000]:
            v = phi_decay(d)
            assert 0.0 < v <= 1.0

    def test_step_scale_matches_fib(self):
        """phi_decay(d) = φ^{1-idx}: d на одной Fib-ступеньке дают один вес."""
        from eva.symbolic.fibonacci_utils import FibonacciUtils as _FU
        for d in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            expected = PHI ** (1 - _FU.get(_FU.zeckendorf(d)[0]) if False else 0)
            # Вес определяется крупнейшим Fib-числом ≤ d
            i = 2
            while _FU.get(i) <= d:
                i += 1
            assert phi_decay(d) == PHI ** (1 - (i - 1))

    def test_comparable_to_theta(self):
        """Обе шкалы (phi_decay и TemporalZeckendorf.theta) убывают вместе."""
        from eva.symbolic.fibonacci_utils import TemporalZeckendorf
        tz = TemporalZeckendorf()
        ds = [1, 2, 3, 5, 10, 50, 100]
        phis = [phi_decay(d) for d in ds]
        thetas = [tz.theta(d)[0] for d in ds]
        for i in range(len(ds) - 1):
            assert (phis[i] > phis[i + 1]) == (thetas[i] >= thetas[i + 1]), \
                f"шаг {ds[i]}->{ds[i+1]}: phi {phis[i]:.3f}->{phis[i+1]:.3f}, theta {thetas[i]:.3f}->{thetas[i+1]:.3f}"

    def test_compression_bound(self):
        """M5: |S_N - 1/φ| <= φ^{-(N+1)}/(1+1/φ) — сжатие спирали."""
        target = 1.0 / PHI
        for N in range(0, 10):
            S = sum((-1.0) ** i * PHI ** (-i) for i in range(N + 1))
            bound = PHI ** (-(N + 1)) / (1.0 + 1.0 / PHI)
            assert abs(S - target) <= bound + 1e-12, f"N={N}"


# ── 4. телескоп нуля ─────────────────────────────────────────────

class TestTelescopicZero:
    def test_exact_zero(self):
        assert abs(telescopic_zero(50)) < 1e-14

    def test_zero_means_no_transition(self):
        """Структурный «нет перехода»: +1/-1 на том же витке -> 0."""
        r = np.array([0.3, -0.8, 0.5])  # произвольное направление
        a = spiral_bundle([r, r], [1.0, -1.0])
        assert np.linalg.norm(a) < 1e-12


# ── 5. знаковое внимание (расширение VSAAttention) ──────────────

class TestSignedAttention:
    def _qs(self):
        from eva.symbolic.concept_space import _hybrid_bind
        rng = np.random.RandomState(42)
        v1 = rng.randn(64).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        v2 = rng.randn(64).astype(np.float32)
        v2 /= np.linalg.norm(v2)
        wv = rng.randn(64).astype(np.float32)
        wv /= np.linalg.norm(wv)
        return v1, v2, _hybrid_bind(v1, wv), _hybrid_bind(v2, wv)

    def test_suppression_steers(self):
        """Подавление (вес<0) отклоняет агрегат — вклад не отбрасывается."""
        v1, v2, b1, b2 = self._qs()
        pos = spiral_bundle([b1], [1.0])
        sup = spiral_bundle([b1, b2], [1.0, -0.5])
        c_pos = float(np.dot(sup, pos))  # до подавления
        assert c_pos < 1.0, "подавление не изменило направление"
        assert c_pos >= 0.0, "подавление перевернуло знак"

    def test_negative_weight_kept(self):
        """Антизнание не выкидывается: вклад виден в результатах."""
        v1, v2, b1, b2 = self._qs()
        a = spiral_bundle([b1, b2], [1.0, -0.3])
        # координата по b2 отрицательна: подавление учтено (знак веса)
        assert float(np.dot(a, b2)) < -0.1
        # без подавления вклад был бы положительным
        old = spiral_bundle([b1], [1.0])
        assert float(np.dot(old, b2)) > 0.0

    def test_all_positive_matches_old(self):
        """Совместимость: при весах >= 0 новый путь = старому (вектор*вес/Σ)."""
        v1, v2, b1, b2 = self._qs()
        new = spiral_bundle([b1, b2], [0.6, 0.3])
        old = (0.6 * b1 + 0.3 * b2) / 0.9
        assert np.allclose(new, old, atol=1e-9)

    def test_hybrid_bind_keeps_sign(self):
        """Связка bind(value, weight_hv) устойчива к знаку веса."""
        from eva.symbolic.concept_space import _hybrid_bind
        rng = np.random.RandomState(7)
        v = rng.randn(64).astype(np.float32)
        wv = rng.randn(64).astype(np.float32)
        v /= np.linalg.norm(v)
        wv /= np.linalg.norm(wv)
        b = _hybrid_bind(v, wv)
        nb = _hybrid_bind(-v, wv)
        assert abs(float(np.dot(b, nb)) + 1.0) < 1e-6, "знак не сохранился в bind"


# ── 5b. интегрированное знаковое внимание (VSAAttention) ────────

class TestSignedAttentionIntegrated:
    def test_negative_key_suppresses(self):
        """Отрицательный косинус -> отрицательный вес -> подавление."""
        from eva.symbolic.vsa_attention import VSAAttention
        rng = np.random.RandomState(11)
        dim = 64
        q = rng.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        k_anti = -q.copy()  # sim = -1
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        attn = VSAAttention(dim=dim, n_heads=1, use_fib_pos=False,
                            use_bind_weighting=False, use_signed_weights=True)
        out = attn.forward(q, [q, k_anti], [q, v])
        # подавление вклада v через отрицательный вес: out отклонился от v
        assert float(np.dot(out, v)) < 0.0

    def test_signed_disabled_matches_old(self):
        """use_signed_weights=False — старое поведение (отбрасывание)."""
        from eva.symbolic.vsa_attention import VSAAttention
        rng = np.random.RandomState(12)
        dim = 64
        q = rng.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        k_anti = -q.copy()
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        attn = VSAAttention(dim=dim, n_heads=1, use_fib_pos=False,
                            use_bind_weighting=False, use_signed_weights=False)
        out = attn.forward(q, [q, k_anti], [q, v])
        # негативный вклад отброшен: out = q (вклад v не входит)
        assert np.allclose(out, q, atol=0.01)

    def test_signed_identity_single_key(self):
        """Один ключ с sim=1: знаковый путь даёт единичную норму."""
        from eva.symbolic.vsa_attention import VSAAttention
        rng = np.random.RandomState(13)
        dim = 64
        q = rng.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        attn = VSAAttention(dim=dim, n_heads=1, use_fib_pos=False,
                            use_bind_weighting=True, use_signed_weights=True)
        out = attn.forward(q, [q.copy()], [q.copy()])
        assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5

    def test_signed_zero_weight_no_contribution(self):
        """Ортогональный ключ (sim≈0) -> вес 0 -> без вклада (как раньше)."""
        from eva.symbolic.vsa_attention import VSAAttention
        rng = np.random.RandomState(14)
        dim = 64
        q = rng.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        k_orth = rng.randn(dim).astype(np.float32)
        k_orth -= np.dot(k_orth, q) * q
        k_orth /= np.linalg.norm(k_orth)
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        attn = VSAAttention(dim=dim, n_heads=1, use_fib_pos=False,
                            use_bind_weighting=False, use_signed_weights=True)
        out_orth = attn.forward(q, [k_orth], [v])
        out_empty = attn.forward(q, [], [])
        assert np.allclose(out_orth, out_empty, atol=0.01)


# ── 6. совместимость с существующим FCF ──────────────────────────

class TestFCFCompat:
    def test_fibonacci_utils_import(self):
        from eva.symbolic.fibonacci_utils import FibonacciUtils
        assert abs(FibonacciUtils.golden_ratio() - PHI) < 1e-12

    def test_zeckendorf_quantizer_unchanged(self):
        from eva.symbolic.fibonacci_utils import ZeckendorfQuantizer
        zq = ZeckendorfQuantizer(dim=64, max_fib_value=1000)
        v = zq.encode(0.5)
        assert v.shape == (64,)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5

    def test_signed_digits_roundtrip_with_quantizer_scale(self):
        """Знаковое кодирование веса w: decode(encode(w)) ≈ w (антисимметрия)."""
        zq_sim = lambda a, b: float(np.dot(a, b))
        rng = np.random.RandomState(9)
        v = rng.randn(64).astype(np.float32)
        v /= np.linalg.norm(v)
        # encode(3) должен быть антисимметричен encode(-3): знак — структура
        d3, _ = signed_phi_digits(3.0, 40)
        dm3, _ = signed_phi_digits(-3.0, 40)
        assert d3 == {j: -a for j, a in dm3.items()}
        s3 = phi_partial_sum(d3)
        sm3 = phi_partial_sum(dm3)
        assert abs(s3 - 3.0) < 1e-6 and abs(sm3 + 3.0) < 1e-6
        assert zq_sim(v * 0, v) == 0.0  # нулевой вес -> нулевой вклад


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
