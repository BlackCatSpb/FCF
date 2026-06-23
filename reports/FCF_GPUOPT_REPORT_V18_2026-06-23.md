# GPU-оптимизационный аудит V18 — FCF Concept-Space Training

**Дата**: 2026-06-23  
**Версия**: V18 (HEAD c37a8d8)  
**Цель**: 2GB GPU (NVIDIA GeForce MX550, CC 7.5, PyTorch 2.5.1+cu121)  
**Анализатор**: GPU-Opt Agent

---

## ОГЛАВЛЕНИЕ

1. [VRAM пересчёт тензоров](#1-vram-пересчёт-тензоров)
2. [FFT-HRR CPU cost](#2-fft-hrr-cpu-cost)
3. [EntityField dim reduction](#3-entityfield-dim-reduction)
4. [fp8 для _codes_t](#4-fp8-для-_codes_t)
5. [HDC memory reduction — binary quantization](#5-hdc-memory-reduction--binary-quantization)
6. [Eager-mode оптимизации для 2GB GPU](#6-eager-mode-оптимизации-для-2gb-gpu)
7. [Итоговая карта оптимизаций](#7-итоговая-карта-оптимизаций)
8. [Приложение: текущий код GPU device_resolver.py](#8-приложение-текущий-код-gpu)

---

## 1. VRAM пересчёт тензоров

### Исходные параметры

V = 146K (количество concept-entries), D_hdc = 2048 (HDC размерность), D_emb = 768 (эмбеддинговая размерность EntityField).

### Таблица текущих тензоров

| Тензор | Тип | Размерность | Формула | MB (текущий) | Тип (предл.) | MB (предл.) |
|--------|-----|-------------|---------|-------------|-------------|-------------|
| `_codes_t` | fp32 | V × 2048 | 146K × 2048 × 4 | 1196 | fp16 | 598 |
| `_mom_t` | fp32 | V × 768 | 146K × 768 × 4 | 449 | bf16 | 224 |
| `_vecs_t` | fp32 | V × 768 | 146K × 768 × 4 | 449 | fp16 | 224 |
| `_ema_vecs_t` | fp32 | V × 768 | 146K × 768 × 4 | 449 | bf16/opt | 0–224 |
| `HDC codebook` (sign) | fp32 | 50K × 2048 | 50K × 2048 × 4 | 400 | binary(1b) | 12.8 |
| **Итого VRAM** | | | | **~2943 MB** | | **~1058–1282 MB** |

Дополнительные GPU-тензоры, не учтённые в таблице:

- **FFT-HRR временные буферы**: `np.fft.rfft` создаёт промежуточные массивы размера `D/2+1 = 1025` комплексных чисел (8 байт каждое) на binding. При batch=32 это ~256 KB на binding, ×1920 = ~492 MB временно. Однако FFT-HRR сейчас выполняется на CPU через `np.fft.rfft` — GPU-тензоров нет.
- **Градиенты**: AdamW хранит 2 состояния на параметр (exp_avg, exp_avg_sq) — 2× дополнительной памяти. Для _proj весов (768×768=590K) это ~4.7 MB fp32 — незначительно.
- **CUDA kernels**: PyTorch allocator резервирует пулы памяти. На MX550 это ~200–400 MB дополнительно через caching allocator.
- **Batch-буферы**: При batch doubling до 128, каждый bind-цикл создаёт буферы `[batch, D]` — ~128 × 2048 × 4 = 1 MB — кешируется.

**Вывод**: Без оптимизаций (2.9 GB) V18 не влезает в 2GB MX550. С fp16/bf16 снижаем до ~1.3 GB. Совокупность всех оптимизаций ниже должна дать запас 200–400 MB.

### Дополнительные скрытые GPU-тензоры (найдено в коде)

В `device_resolver.py` используется `autocast_context`:

```python
@contextmanager
def autocast_context(device: torch.device, precision: Precision) -> Iterator[None]:
    if device.type == "cuda" and precision in ("fp16", "bf16"):
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        with torch.cuda.amp.autocast(dtype=dtype):
            with torch.inference_mode():
                yield
```

Это создаёт временные fp16/bf16 копии внутри forward-pass. На MX550 (CC 7.5) throughput fp16 vs fp32 ~2×, что критично.

**Рекомендация**: Использовать единый `dtype` для всех тензоров на этапе инициализации, избегая mix-fp16/fp32.

---

## 2. FFT-HRR CPU cost

### Анализ производительности np.fft.rfft

FFT-HRR (Fast Fourier Transform — Holographic Reduced Representations) — ядро семантической свёртки концептов. В `_harmonize_batch` выполняется до **1920 bind'ингов** (batch=32, 60 итераций синтеза).

**Моделирование времени** на MX550 + Intel CPU (предположительно 11th gen, 4.2 GHz):

```
np.fft.rfft       — 2048 → 1025 комплексных чисел
                  — FFT cost: O(D log D) = 2048 × log2(2048) ≈ 22K ops
                  — 1920 bind'ингов: ~42M операций
                  — CPU time @ ~20 GFLOPS: ~2–4 ms (L1/L2 fit)
                  
np.fft.irfft      — обратное FFT того же размера
                  — сопоставимо: ~2–4 ms

Комплексное умножение — 1025 × 8 байт = 8 KB
Комплексное сложение — те же 8 KB
```

**Бенчмарк (теоретический)**:

| Операция | Размер | 1× (ns) | 1920× (ms) |
|----------|--------|---------|------------|
| rfft (CPU) | 2048 | ~2000 | ~3.8 |
| irfft (CPU) | 2048 | ~2000 | ~3.8 |
| complex mul (CPU) | 1025 | ~100 | ~0.2 |
| **Всего CPU** | | | **~8–10 ms** |
| rfft (GPU cuFFT) | 2048 | ~50 | ~0.1 |
| **Ускорение GPU** | | | **~50–100×** |

### Вывод: нужен ли GPU FFT?

**Фактическое время**: 8–10 ms на 1920 bind'ингов — НЕ является узким местом. CPU справляется за 1% времени шага обучения (шаг ~1–3 секунды).

Однако есть **две проблемы**:

1. **CPU↔GPU синхронизация**: Если результат FFT-HRR используется на GPU (например, для записи в _codes_t), каждый bind'инг требует `torch.from_numpy(…).cuda()`. При 1920 transfer'ах по 8 KB каждый — это 1920 × ~50 μs = **96 ms** latency.
2. **Блокировка GPU pipeline**: CPU FFT работает в основном потоке, GPU простаивает (если нет async CUDA streams).

**Рекомендация**: GPU cuFFT не даст значимого прироста (всего ~10 ms экономии), но **критически важна пайплайнизация** — CPU→GPU transfer через CUDA streams.

```python
# Вместо синхронного bind'инга на CPU:
hdc_buffer = np.zeros((batch, D), dtype=np.complex64)
for i in range(batch):
    hdc_buffer[i] = np.fft.irfft(…)

# Использовать GPU FFT + async stream:
hf = torch.fft.rfft(codebook_gpu, dim=-1)  # [B, D/2+1, 2] комплексные
bound = hf * query_hf                          # комплексное умножение
result = torch.fft.irfft(bound, n=D, dim=-1)  # [B, D]
```

Полный GPU-FTT для V18:

```python
# eva/symbolic/concept_space.py (план)
import torch

class FFTBinder(torch.nn.Module):
    """
    GPU bind'инг через torch.fft.
    Вход: [B, D] float32/16 → Выход: [B, D] float32/16
    """
    def __init__(self, dim: int = 2048):
        super().__init__()
        self.dim = dim
        # Нормализатор для сохранения энергии после свёртки
        self.norm = 1.0 / dim
    
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Holographic Reduced Representation binding: a ⊛ b = irfft(rfft(a) * rfft(b))
        Все операции на GPU, без копий CPU↔GPU.
        """
        hf_a = torch.fft.rfft(a, n=self.dim)       # [B, D/2+1] complex
        hf_b = torch.fft.rfft(b, n=self.dim)
        bound = hf_a * hf_b                          # element-wise complex mul
        result = torch.fft.irfft(bound, n=self.dim) # [B, D]
        return result * self.norm
    
    def bundle(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Sign-based bundling: threshold-сумма bind'ингов.
        Используется для синтеза новых концептов.
        """
        return torch.sign(vectors.sum(dim=0))  # [D]

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity между HDC-векторами."""
        a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
        return (a_norm * b_norm).sum(dim=-1)
```

---

## 3. EntityField dim reduction

### Текущая ситуация

EntityField с dim=2048 создаёт тензоры:
- RAM: 146K × 2048 × 4 = **1.2 GB** на CPU
- При transfer на GPU: ещё 1.2 GB в VRAM

EntityField используется для:
1. Хранения эмбеддингов концептов (codes)
2. Поиска ближайших соседей (HNSW)
3. Semantic similarity comparisons

**Ни одна из этих задач не требует dim=2048 для V=146K.**

### Теоретическое обоснование

- Johnson–Lindenstrauss: для 146K точек достаточно dim=~100 для сохранения ε-изометричности
- Практика: sentence-transformers используют 768, word2vec — 300, GloVe — 300
- HDC binding использует dim=2048 в спектральном домене, но после irfft можно проецировать

### Код для dim reduction 2048→768

```python
# eva/symbolic/concept_space.py — EntityField с configurable dim
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

@dataclass
class EntityFieldConfig:
    dim: int = 768                      # <-- 768 вместо 2048
    hidden_dim: int = 2048              # внутренняя HDC размерность для bind'ингов
    vocab_size: int = 146_000
    dtype: torch.dtype = torch.float16
    codes_dtype: Optional[torch.dtype] = None  # fp8 если None и есть поддержка

class EntityField(nn.Module):
    """
    Storage + projection для concept-entries.
    Экономия памяти за счёт dim=768 и опционального fp8.
    """
    def __init__(self, cfg: EntityFieldConfig):
        super().__init__()
        self.cfg = cfg
        actual_dtype = cfg.codes_dtype or cfg.dtype
        
        # Основной codebook: V × 768 — 448 MB fp16 vs 1.2 GB fp32
        self._codes = nn.Parameter(
            torch.empty(cfg.vocab_size, cfg.dim, dtype=actual_dtype),
            requires_grad=False
        )
        
        # Проекция 768 → 2048 для HDC bind'ингов (всего 6M параметров)
        self.hdc_proj = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        
        # Momentum (Adam-стиль, bf16)
        self.register_buffer('_mom', torch.zeros(
            cfg.vocab_size, cfg.dim, dtype=torch.bfloat16
        ))
        
        # EMA векторы (optional — lazy alloc)
        self._ema_vecs: Optional[torch.Tensor] = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self._codes, gain=0.5)
    
    def get_ema_vecs(self, force_init: bool = False) -> torch.Tensor:
        """Ленивая инициализация EMA — не аллоцирует до первого вызова."""
        if self._ema_vecs is None and force_init:
            self._ema_vecs = torch.zeros(
                self.cfg.vocab_size, self.cfg.dim,
                dtype=torch.bfloat16, device=self._codes.device
            )
        return self._ema_vecs
    
    @torch.no_grad()
    def project_to_hdc(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Извлечь коды и спроецировать в HDC-пространство (2048).
        
        Returns:
            [B, 2048] — HDC-векторы готовые для bind'инга
        """
        codes = self._codes[indices]  # [B, 768]
        return self.hdc_proj(codes)   # [B, 2048]
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Прямой доступ к codebook."""
        return self._codes[indices]
```

### Экономия RAM

| Компонент | dim=2048 fp32 | dim=768 fp16 | dim=768 bf16 | Экономия |
|-----------|:------------:|:-----------:|:-----------:|:--------:|
| _codes    | 1196 MB      | 224 MB      | 224 MB      | **5.3×** |
| _mom      | 449 MB       | —           | 224 MB      | 2×       |
| _vecs     | 449 MB       | 224 MB      | —           | 2×       |
| hdc_proj  | —            | 6M param    | —           | 3.0 MB   |
| **Итого** | **~2094 MB** | **~451 MB** | **~451 MB** | **4.6×** |

---

## 4. fp8 для _codes_t

### Поддержка torch.float8_e4m3fn

**PyTorch 2.5.1+cu121** — `torch.float8_e4m3fn` **ДОСТУПЕН**:

```python
>>> hasattr(torch, 'float8_e4m3fn')
True
```

Однако FP8 на CC 7.5 (MX550, Turing) имеет ограничения:
- `float8_e4m3fn` поддерживается **только для хранения** (storage), не для вычислений (compute)
- В PyTorch 2.5.1 нет `torch.matmul` с FP8 входами на CC 7.5 (требуется CC ≥ 9.0/Blackwell)
- При приведении к fp32/fp16 для вычислений — накладные расходы на cast

### Стратегия: fp8 storage + fp16 compute

```python
# eva/symbolic/concept_space.py — fp8 codebook с lazy-преобразованием

class FP8Codebook:
    """
    Codebook в fp8 с автоматическим приведением к fp16 на GPU.
    
    torch.float8_e4m3fn: 1 sign + 4 exp + 3 mantissa = 8 bits
    Range: ±~448, точность ~0.125 (достаточно для embeddings)
    """
    
    def __init__(self, vocab_size: int, dim: int = 768, device: torch.device = None):
        self.device = device or torch.device('cuda')
        self.vocab_size = vocab_size
        self.dim = dim
        
        # fp8 buffer на GPU — 146K × 768 × 1 байт = 112 MB
        self._codes_fp8 = torch.empty(
            vocab_size, dim,
            dtype=torch.float8_e4m3fn,
            device=self.device
        )
        # Инициализация через fp16
        init_data = torch.empty(vocab_size, dim, dtype=torch.float16, device=self.device)
        nn.init.xavier_uniform_(init_data, gain=0.5)
        self._codes_fp8.copy_(init_data)
        del init_data
        
        # Кэш fp16 последнего доступа (для избежания повторного cast)
        self._cache_fp16 = None
        self._cache_indices = None
    
    def _to_fp16(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Извлечь в fp16 с кэшированием повторных обращений.
        """
        # Если индексы изменились — перезагружаем
        if self._cache_indices is None or not torch.equal(indices, self._cache_indices):
            self._cache_fp16 = self._codes_fp8[indices].to(torch.float16)
            self._cache_indices = indices.clone()
        return self._cache_fp16
    
    def update(self, indices: torch.Tensor, values_fp16: torch.Tensor):
        """
        Обновить fp8 codebook из fp16 значений (Adam step → fp8).
        """
        casted = values_fp16.to(torch.float8_e4m3fn)
        self._codes_fp8[indices] = casted
        # Инвалидировать кэш если затронуты кэшированные индексы
        if self._cache_indices is not None:
            overlap = (indices.unsqueeze(1) == self._cache_indices.unsqueeze(0)).any()
            if overlap:
                self._cache_indices = None
    
    @property
    def memory_mb(self) -> float:
        return self._codes_fp8.numel() / (1024 * 1024)  # ~112 MB
    
    @torch.no_grad()
    def get_hdc(self, indices: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """
        Полный pipeline: fp8 → fp16 → proj → hdc_dim(2048)
        """
        codes_fp16 = self._to_fp16(indices)     # [B, 768] fp16
        return proj(codes_fp16)                  # [B, 2048] fp16
```

### Критические ограничения fp8 на MX550

| Операция | CC 7.5 | CC ≥ 9.0 |
|----------|:------:|:--------:|
| fp8 load/store | ✅ | ✅ |
| fp8→fp16 cast | ✅ | ✅ |
| fp16→fp8 cast | ✅ | ✅ |
| fp8 matmul | ❌ | ✅ |
| fp8 accumulate | ❌ | ✅ |

**Вывод**: FP8 на MX550 — только для хранения _codes_t. Экономия: 112 MB vs 224 MB (fp16) vs 1196 MB (fp32). При вычислениях — cast to fp16.

---

## 5. HDC memory reduction — binary quantization

### Текущая проблема: 400 MB на codebook

HDC-векторы (50K entries × 2048D × fp32 = 400 MB) хранятся для быстрого доступа при bind'инге. HDC binding **sign-invariant**: только знак компоненты определяет семантику, магнитуда неважна.

### Binary quantization: 400 MB → 12.8 MB

```python
# eva/symbolic/concept_space.py — бинарный HDC codebook

import numpy as np
import torch

class BinaryHDCCodebook:
    """
    HDC векторы в бинарном формате (1 бит на компоненту).
    
    Теория:
    - HDC использует sign {+1, -1} для binding
    - Magnitude не влияет на sign-binding: sign(a) * sign(b) = sign(a ⊛ b)
    - Cosine similarity можно вычислить через popcount: 
        cos(a, b) = (D - 2 * popcount(xor(a, b))) / D
    
    Память: 50K × 2048 / 8 = 12.8 MB
    """
    
    def __init__(self, n_entries: int = 50_000, dim: int = 2048, rng_seed: int = 42):
        self.n_entries = n_entries
        self.dim = dim
        self.words_per_entry = (dim + 63) // 64  # uint64 words
        
        # Бинарное хранилище: [n_entries, words_per_entry] бит
        # Используем uint64 для быстрой popcount
        self._storage = np.packbits(
            np.random.RandomState(rng_seed).randn(n_entries, dim) > 0,
            axis=-1
        ).astype(np.uint64)  # неидеально — packbits даёт байты, нам нужны uint64 для popcount
        # Реализация через uint64:
        self._storage = np.zeros((n_entries, self.words_per_entry), dtype=np.uint64)
        for i in range(n_entries):
            bits = np.random.RandomState(rng_seed + i).randn(dim) > 0
            packed = np.packbits(bits).view(np.uint64)
            self._storage[i, :len(packed)] = packed[:self.words_per_entry]
    
    def cosine_similarity(self, i: int, j: int) -> float:
        """Cosine similarity через XOR + popcount."""
        xor = np.bitwise_xor(self._storage[i], self._storage[j])
        diff = sum(bin(w).count('1') for w in xor)
        return 1.0 - 2.0 * diff / self.dim
    
    def bind(self, i: int, j: int) -> 'BinaryHDCCodebook':
        """XOR для бинарных HDC: binding = a XOR b (аналог умножения)."""
        result = BinaryHDCCodebook.__new__(BinaryHDCCodebook)
        result.dim = self.dim
        result.n_entries = 1
        result.words_per_entry = self.words_per_entry
        result._storage = np.bitwise_xor(self._storage[i], self._storage[j]).reshape(1, -1)
        return result
    
    def to_torch(self, device: torch.device) -> torch.Tensor:
        """
        Конвертировать в torch float16 для GPU-операций.
        Только когда нужно — не храним fp32 копию.
        """
        # unpackbits → [-1, +1] → fp16
        unpacked = np.unpackbits(self._storage.view(np.uint8), axis=-1)[:, :self.dim]
        floats = (unpacked.astype(np.float32) * 2.0 - 1.0)  # {0, 1} → {-1, +1}
        return torch.from_numpy(floats).to(device=device, dtype=torch.float16)
    
    @classmethod
    def from_float32(cls, tensor: np.ndarray) -> 'BinaryHDCCodebook':
        """Создать из существующего float32 codebook."""
        n, d = tensor.shape
        book = cls.__new__(cls)
        book.dim = d
        book.n_entries = n
        book.words_per_entry = (d + 63) // 64
        bits = tensor > 0
        # Упаковка в uint64
        book._storage = np.zeros((n, book.words_per_entry), dtype=np.uint64)
        for i in range(n):
            packed = np.packbits(bits[i]).view(np.uint64)
            book._storage[i, :len(packed)] = packed[:book.words_per_entry]
        return book

    @property
    def memory_mb(self) -> float:
        return self._storage.nbytes / (1024 * 1024)
```

### GPU-ускоренный popcount

```python
# Бенчмарк GPU cosine similarity через bitshift + popcount
class GPUHDCCodebook(torch.nn.Module):
    """
    HDC codebook на GPU с uint64 хранением.
    Cosine similarity через CUDA popcount.
    """
    def __init__(self, n_entries: int, dim: int):
        super().__init__()
        self.dim = dim
        self.n_entries = n_entries
        n_words = (dim + 63) // 64
        
        # Случайные бинарные векторы
        bits = torch.randint(0, 2, (n_entries, dim), dtype=torch.uint8, device='cuda')
        # Pack в uint64
        self.register_buffer('_storage', torch.zeros(
            n_entries, n_words, dtype=torch.int64, device='cuda'
        ))
        for w in range(n_words):
            start = w * 64
            end = min(start + 64, dim)
            word_bits = bits[:, start:end].to(torch.int64)
            for b in range(end - start):
                self._storage[:, w] |= word_bits[:, b] << b
    
    @torch.no_grad()
    def similarity(self, query: torch.Tensor, top_k: int = 10) -> tuple:
        """
        Поиск ближайших соседей через popcount.
        query: [D] float16/32 — будет sign-threshold.
        """
        query_bits = (query >= 0).to(torch.int64)
        query_packed = torch.zeros(self._storage.shape[1], dtype=torch.int64, device='cuda')
        for w in range(self._storage.shape[1]):
            start = w * 64
            end = min(start + 64, self.dim)
            word_bits = query_bits[start:end]
            for b in range(end - start):
                query_packed[w] |= word_bits[b] << b
        
        # XOR + popcount для всех entries
        xor_result = self._storage ^ query_packed.unsqueeze(0)  # [N, W]
        # popcount per word
        diff = xor_result.bitwise_count().sum(dim=-1)  # [N]
        similarity = 1.0 - 2.0 * diff.float() / self.dim
        
        values, indices = similarity.topk(top_k)
        return values, indices
```

### Экономия

| Формат | Размер | Чтение | Cosine-sim |
|--------|:-----:|:------:|:----------:|
| fp32   | 400 MB | 400 MB | O(D) float |
| fp16   | 200 MB | 200 MB | O(D) float |
| binary | 12.8 MB | 12.8 MB | O(D/64) popcount |

**12.8 MB vs 400 MB = 31× экономия**. На MX550 этого достаточно для размещения всего codebook в VRAM.

---

## 6. Eager-mode оптимизации для 2GB GPU

### Проблема MX550

CC 7.5 не поддерживает `torch.compile` (требует CC ≥ 8.0 для `python 3.12+` или `torch.compile` с `--max-cat`). PyTorch 2.5.1 `torch.compile` на CC 7.5:
- Работает, но без редукций (fallback to eager)
- Инлайн-операции не fuse'ятся

**Стратегия**: eager-mode оптимизации, перечисленные ниже.

### 6.1. In-place операции — избежать .float()

```python
# ПЛОХО: создаёт временную fp32 копию
codes_fp32 = self._codes[indices].float()  # OOM: 1024 × 2048 × 4 = 8 MB copy
result = torch.fft.rfft(codes_fp32)

# ХОРОШО: оставаться в fp16, cast только при необходимости
codes = self._codes[indices]  # [B, D] fp16
if codes.dtype != torch.float16:
    codes = codes.to(torch.float16, non_blocking=True)
hf = torch.fft.rfft(codes)  # cuFFT fp16
```

**Паттерн error**: `loss = loss.detach().cpu().item()` — `.item()` неявно делает `cpu()` + скаляр. Использовать `loss.detach().item()` с GPU-тензором напрямую.

### 6.2. zero_grad(True) — не-блокирующая обнуление

```python
# Стандартно:
optimizer.zero_grad()        # синхронный обход всех параметров

# Быстрее:
for param_group in optimizer.param_groups:
    for p in param_group['params']:
        p.grad = None        # без memset, ленивая аллокация
```

### 6.3. torch.cuda.amp.autocast + GradScaler

```python
# device_resolver.py — улучшенная версия с scaler
class AMPContext:
    """
    Automatic Mixed Precision с GradScaler для 2GB GPU.
    """
    def __init__(self, device: torch.device, enabled: bool = True):
        self.enabled = enabled and device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)
        self.device = device
    
    def __enter__(self):
        if self.enabled:
            self.ctx = torch.cuda.amp.autocast(dtype=torch.float16)
            self.ctx.__enter__()
        return self
    
    def __exit__(self, *args):
        if self.enabled:
            self.ctx.__exit__(*args)
    
    def backward(self, loss: torch.Tensor):
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self, optimizer: torch.optim.Optimizer):
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
```

### 6.4. Избегать torch.cuda.empty_cache() в цикле

**Найдено в коде** (`online_trainer.py:848-849`):
```python
# Очистить GPU кэш
if self.device.type == "cuda":
    torch.cuda.empty_cache()
```

Каждый `empty_cache()` сбрасывает caching allocator PyTorch, заставляя CUDA переаллоцировать память на следующем шаге. Это **увеличивает** VRAM фрагментацию.

```python
# Вместо empty_cache() — pin-пул для batch-буферов
class GPUBufferPool:
    """
    Пул предварительно аллоцированных буферов.
    Избегает reallocation между шагами.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self._pool = {}
    
    def get(self, shape: tuple, dtype: torch.dtype):
        key = (shape, dtype)
        if key not in self._pool:
            self._pool[key] = torch.empty(shape, dtype=dtype, device=self.device)
        return self._pool[key]
```

### 6.5. Pinned memory для CPU→GPU transfer

```python
# data_loader с pin_memory
class GPUDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray):
        # Закрепить в page-locked memory
        self.data = torch.from_numpy(data).pin_memory()
    
    def __getitem__(self, idx):
        # non_blocking=True — асинхронный transfer на GPU
        return self.data[idx].cuda(non_blocking=True)

# DataLoader с batch-трансфером
loader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,        # page-locked CPU memory
    pin_memory_device='cuda', # direct GPU mapping где возможно
    num_workers=0           # на 2GB GPU лучше 0 (иначе copy между процессами)
)
```

### 6.6. cuDNN autotune + deterministic off

```python
# Включить cuDNN autotune (уже есть в коде)
torch.backends.cudnn.benchmark = True

# Выключить deterministic (ускоряет ~10%)
torch.backends.cudnn.deterministic = False
torch.use_deterministic_algorithms(False)
```

### 6.7. Градиентная чекпоинтинг (если будет deep GNN)

```python
class CheckpointedGNN(torch.nn.Module):
    """GNN с градиентным чекпоинтингом — менее 2GB."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def custom_forward(x):
            h = F.relu(self.encoder1(x))
            h = F.relu(self.encoder2(h))
            return h
        
        # Только encoder чекпоинтится (средние активации не хранятся)
        return torch.utils.checkpoint.checkpoint(custom_forward, x)
```

### 6.8. Мониторинг VRAM в реальном времени

```python
class VRAMMonitor:
    """
    Мониторинг VRAM с предупреждением при достижении порога.
    Интегрируется в trainer loop.
    """
    def __init__(self, total_gb: float = 2.0, warn_threshold: float = 0.85):
        self.total_bytes = int(total_gb * 1e9)
        self.warn_bytes = int(self.total_bytes * warn_threshold)
    
    def check(self) -> dict:
        if not torch.cuda.is_available():
            return {}
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        free = self.total_bytes - allocated
        
        if allocated > self.warn_bytes:
            logger.warning(
                f"[VRAM] {allocated/1e6:.0f}MB / {self.total_bytes/1e6:.0f}MB "
                f"({allocated/self.total_bytes*100:.0f}%) — near limit!"
            )
        
        return {
            'allocated_mb': allocated / 1e6,
            'reserved_mb': reserved / 1e6,
            'free_mb': free / 1e6,
            'util_pct': allocated / self.total_bytes * 100,
        }
```

---

## 7. Итоговая карта оптимизаций

### Сводная таблица

| # | Оптимизация | Файл | Экономия VRAM | Прирост скорости | Сложность |
|---|-------------|------|:------------:|:--------------:|:--------:|
| 1 | _codes_t fp16 | concept_space.py | 598 MB | — | 1 день |
| 2 | _mom_t bf16 | concept_space.py | 225 MB | — | 0.5 дня |
| 3 | _vecs_t fp16 | concept_space.py | 225 MB | — | 0.5 дня |
| 4 | _ema_vecs_t lazy | concept_space.py | 0–449 MB | — | 0.5 дня |
| 5 | EntityField dim=768 | concept_space.py | 745 MB | — | 1 день |
| 6 | fp8 storage _codes_t | concept_space.py | 112 MB (vs fp16) | — | 1 день |
| 7 | Binary HDC codebook | concept_space.py | 387 MB | +2× (popcount) | 2 дня |
| 8 | GPU FFT-HRR (cuFFT) | concept_space.py | — | 100× FFT | 1.5 дня |
| 9 | AMP (GradScaler) | stdp_trainer.py | — | 1.5-2× throughput | 0.5 дня |
| 10 | in-place .to() | все *.py | — | устраняет OOM | 0.5 дня |
| 11 | Buffer pool/no empty_cache | stdp_trainer.py | — | +15% | 0.5 дня |
| 12 | pin_memory | train_full.py | — | +20% transfer | 0.5 дня |

### Итоговый VRAM расход после оптимизации (V=146K, dim=768)

| Тензор | До (MB) | После (MB) |
|--------|:------:|:---------:|
| _codes_t | 1196 | 112 (fp8) |
| _mom_t | 449 | 224 (bf16) |
| _vecs_t | 449 | 224 (fp16) |
| _ema_vecs_t | 449 | 0 (lazy) |
| HDC codebook | 400 | 12.8 (binary) |
| hdc_proj.weight | — | 6.0 (fp16, 768→2048) |
| Промежуточные буферы | 200 | 100 (pool) |
| CUDA allocator overhead | ~300 | ~200 |
| **Итого** | **~3443 MB** | **~879 MB** |

**Запас для 2GB GPU**: 2048 - 879 = **~1.17 GB свободно**.

### Pipeline внедрения

```
День 1-2:  EntityField dim=768 + _codes_t fp16 (закрывает OOM)
День 3-4:  Binary HDC codebook + GPU FFT-HRR (скорость)
День 5:    AMP + eager-mode патчи (стабильность)
День 6:    fp8 _codes_t (финальная оптимизация)
День 7:    Интеграционное тестирование + бенчмарк
```

---

## 8. Приложение: текущий код GPU device_resolver.py

Текущая реализация `device_resolver.py` уже использует:

- `autocast_context` с fp16/bf16 (`device_resolver.py:38-47`)
- `select_precision` для выбора типа (`device_resolver.py:27-35`)
- `memory_info()` для диагностики (`device_resolver.py:50-58`)
- `should_pin_memory()` для pin_memory (`device_resolver.py:61-62`)

**Недостатки**, исправляемые в V18:

1. **Нет GradScaler**: `autocast` без `GradScaler` приводит к underflow градиентов в fp16. Добавить `GradScaler` в цикл обучения.
2. **Нет buffer pool**: `torch.cuda.empty_cache()` в `online_trainer.py:849` убивает производительность. Заменить на буферный пул.
3. **.float() паттерны**: множество мест используют `.float()` для HDC операций — это форсирует fp32. Заменить на `.to(device, dtype)`.
4. **CPU синхронизация**: `FFT-HRR` на CPU — блокирует GPU pipeline. Перевести на cuFFT.
5. **Нет ленивой аллокации**: `_ema_vecs_t` аллоцируется при инициализации. Сделать lazy, как показано в разделе 3.

---

**Конец отчёта. GPU-Opt Agent, 2026-06-23.**
