"""AlphabetBasis — алфавит как λ_d-формализованный базис в ℤ₈^d.

Алфавит — не внешние данные (BPE-модель), а часть кода.
Каждая буква — детерминированный VSA-вектор, порождённый
из λ_d через permute(basis_seed, position).

Из букв → морфемы → слова — всё выводится из той же λ_d-иерархии.
Это формализованный потенциал: модель не «узнаёт» алфавит,
она уже содержит его как математическую структуру.
"""

from __future__ import annotations
import numpy as np
from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _REG

# ── Русский алфавит: 33 буквы + ё + пробел + знаки ──

RUSSIAN_LETTERS: str = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
ALPHABET_SYMBOLS: str = RUSSIAN_LETTERS + " .,!?:;—()-«»\"'\n\t"

# λ_d-веса для декодирования слова из букв
# (читается из FormulaCoefficients при инициализации)
_LAM: float = 0.0  # устанавливается при первом вызове init()


class AlphabetBasis:
    """Детерминированные VSA-векторы для каждого символа алфавита.

    Каждый символ s на позиции p ∈ [0, N) получает вектор:

        vec(s) = permute(base_vector, p)

    где base_vector — фиксированный VSA-шум из SeedRegistry,
    permute — циклический сдвиг на элемент группы ℤ₈^d.

    Символы квазиортогональны: cos(vec(a), vec(б)) ≈ 0 для a ≠ б.
    """

    dim: int
    _vectors: dict[str, np.ndarray]
    _base: np.ndarray

    def __init__(self, dim: int = 768):
        self.dim = dim
        self._vectors = {}
        rng = _REG.rng('alphabet_basis')
        self._base = rng.randn(dim).astype(np.float32)
        self._base /= np.linalg.norm(self._base)
        self._build()

    def _build(self):
        """Построить векторы для всех символов через permute."""
        for pos, ch in enumerate(ALPHABET_SYMBOLS):
            # permute = циклический сдвиг на λ_d-масштабированную позицию
            shift = int((pos + 1) * _LAM * 137) % self.dim if _LAM > 0 else pos
            vec = np.roll(self._base, shift)
            self._vectors[ch] = vec

    def __contains__(self, ch: str) -> bool:
        return ch in self._vectors

    def __getitem__(self, ch: str) -> np.ndarray:
        """Вектор для символа (O(1))."""
        return self._vectors[ch]

    def __len__(self) -> int:
        return len(self._vectors)

    def keys(self) -> set[str]:
        return set(self._vectors.keys())

    # ── Композиция ──

    def word_vector(self, word: str, lam: float | None = None) -> np.ndarray:
        """Вектор слова как λ_d-взвешенный bundle букв.

        word = "дом"
        vec(word) = w₁·vec(д) + w₂·vec(о) + w₃·vec(м)

        где w_k = λ_d^{-k} / Σ λ_d^{-j} — нормированные позиционные веса.
        """
        word = word.lower()
        letters = [ch for ch in word if ch in self._vectors]
        if not letters:
            return self._base.copy()

        l = lam or (_LAM if _LAM > 0 else 1.618)
        depth = len(letters)
        weights = np.array([l ** (-k) for k in range(depth)], dtype=np.float32)
        weights /= weights.sum()

        vec = np.zeros(self.dim, dtype=np.float32)
        for w, ch in zip(weights, letters):
            vec += w * self._vectors[ch]

        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec

    def morpheme_vector(self, word: str, lam: float | None = None) -> np.ndarray:
        """Вектор морфемы как VSA bind букв с позиционным permute.

        vec(морфема) = bind(permute(vec_letter₁, 1),
                             permute(vec_letter₂, 2), …)

        bind = поэлементное умножение в частотной области (FFT).
        """
        word = word.lower()
        letters = [ch for ch in word if ch in self._vectors]
        if not letters:
            return self._base.copy()

        # VSA bind через FFT: bind(a,b) = ifft(fft(a) * fft(b))
        fft_acc = np.fft.fft(self._vectors[letters[0]])
        for k, ch in enumerate(letters[1:], 1):
            permuted = np.roll(self._vectors[ch], k * 7)
            fft_acc *= np.fft.fft(permuted)

        vec = np.fft.ifft(fft_acc).real.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec


# ── Глобальный экземпляр (ленивая инициализация) ──
_INSTANCE: AlphabetBasis | None = None


def get_basis(dim: int = 768) -> AlphabetBasis:
    global _INSTANCE
    if _INSTANCE is None or _INSTANCE.dim != dim:
        _INSTANCE = AlphabetBasis(dim)
    return _INSTANCE


def init(lam: float):
    """Установить λ_d для позиционных весов (вызывается при rebuild)."""
    global _LAM
    _LAM = lam
    # перестроить с новым сдвигом
    global _INSTANCE
    _INSTANCE = None
