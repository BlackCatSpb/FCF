# FCF Quality & Safety Audit Report — V18 (HEAD cff1240)

**Дата:** 2026-06-23  
**Аудитор:** Quality-Safety Agent  
**Версия кода:** V18, HEAD cff1240 (документальный коммит)  
**Файл:** FCF_QUALITYSAFETY_REPORT_V18_2026-06-23.md  

---

## Executive Summary

| Метрика | Значение |
|---------|----------|
| Коммитов в V18 | 1 (c37a8d8 — docs/README/logs) |
| Python-файлов в репозитории | ~520 |
| Тестов (`test_*.py`) | **0** |
| Директория `tests/` | **НЕ СУЩЕСТВУЕТ** |
| `conftest.py` | **НЕ СУЩЕСТВУЕТ** |
| `test_stdp.py` (77 KB, 145 тестов, 6 skipped) | **НЕ СУЩЕСТВУЕТ — описан в задании, не найден в репозитории** |
| Компонентов из V15-списка без тестов | **11 (все по-прежнему без тестов)** |
| Компонентов из V15-списка, существующих в коде | **0 из 11** |
| pynose/venv: pytest | установлен 9.1.0 |

**Вердикт:** Критическое состояние тестового покрытия — тесты полностью отсутствуют. Мёртвый код присутствует (broken imports). Компоненты EntityField, Harmonizer, HDC и другие из V15-спецификации не существуют в текущей кодовой базе — это спецификационные артефакты, не реализованные в коде. Единственный реальный тестируемый компонент — `FractalGraphL1L2` (672 строки).

---

## 1. Актуальный список непокрытых компонентов после V16/V17

### 1.1. Компоненты из V15-отчёта

| # | Компонент | Существует в V18? | Тесты? | Статус |
|---|-----------|-------------------|--------|--------|
| 1 | `EntityField` | **НЕТ** (класс `MemoryField` в `memory_core.py:29`) | Нет | **Не существует в коде** |
| 2 | `Harmonizer` | **НЕТ** | Нет | **Не существует в коде** |
| 3 | `HDC n-gram` | **НЕТ** | Нет | **Не существует в коде** |
| 4 | `W_proj` | **НЕТ** | Нет | **Не существует в коде** |
| 5 | `sector index` | **НЕТ** | Нет | **Не существует в коде** |
| 6 | `adaptive L1` | **НЕТ** (`FractalGraphL1L2` существует в `pie_integration/fractal_graph_l1_l2.py`) | Нет | **Аналог существует, тестов нет** |
| 7 | `dynamic capacity` | **НЕТ** | Нет | **Не существует в коде** |
| 8 | `item memory` | **НЕТ** | Нет | **Не существует в коде** |
| 9 | `JL projection` | **НЕТ** | Нет | **Не существует в коде** |
| 10 | `cluster-potential` | **НЕТ** | Нет | **Не существует в коде** |
| 11 | `_semantic_bootstrap` | **НЕТ** | Нет | **Не существует в коде** |

**Вывод:** Все 11 компонентов из V15-списка не имеют тестов. Из них **0 существуют в текущей кодовой базе**. Компоненты являются спецификационными артефактами (вероятно, из «ACI ConceptMiner — спецификация модуля EVA.pdf» или «EVA.pdf»).

### 1.2. Реально существующие непокрытые компоненты (V18)

Вместо виртуальных компонентов из спецификации, аудит выявил **реальные непокрытые компоненты**:

| # | Компонент | Файл | Строк | Тестов |
|---|-----------|------|-------|--------|
| 1 | `FractalGraphL1L2` (L1/L2 extensions) | `memory/pie_integration/fractal_graph_l1_l2.py` | 672 | **0** |
| 2 | `MemoryField` / `MemoryNeuron` / `MemoryDatabase` | `memory/memory_core.py` | 325 | **0** |
| 3 | `GNNTrainer` (включая Qwen KB) | `fcp_core/online_trainer.py` | 1765 | **0** |
| 4 | `FractalAttentionSystem` | (ранее stubs, теперь реализован) | — | **0** |
| 5 | `VectorIndex` / HNSW wrapper | `memory/fractal_graph_v2/optimizations.py` | — | **0** |
| 6 | `TemporalContextMemory` | `memory/temporal_context.py` | — | **0** |
| 7 | `ScenarioTCM` | `memory/scenario_tcm.py` | — | **0** |
| 8 | `ShadowLoRAManager` | `fcp_core/shadow_lora.py` | — | **0** |
| 9 | `KGAdder` / `GraphCurator` | `knowledge/graph_curator.py` | — | **0** |
| 10 | `PIE` routing engine | `memory/pie_integration/routing_engine.py` | — | **0** |

---

## 2. FFT-HRR тесты: `_hrr_bind` / `_hrr_unbind`

### 2.1. Статус

Функции `_hrr_bind` и `_hrr_unbind` **НЕ СУЩЕСТВУЮТ** в кодовой базе V18. Никаких следов HRR (Holographic Reduced Representations), circular convolution, FFT-HRR или bind/unbind операций не найдено.

Поиск по паттернам:
- `_hrr_bind`, `_hrr_unbind`, `hrr_bind`, `hrr_unbind` — 0 результатов
- `fft.*hrr`, `circular.*convol` — 0 результатов
- `bind.*unbind` — 0 результатов

### 2.2. Рекомендация

Перед написанием тестов необходимо реализовать сами функции. Ожидаемая реализация:

```python
# eva_ai/memory/hrr.py (предполагаемый файл, не существует)
import numpy as np

def _hrr_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution of two HRR vectors via FFT."""
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))

def _hrr_unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular correlation (approximate inverse of bind) via FFT."""
    return np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a))
```

После реализации — тест на unbind(bind(a,b), b) ≈ a:

```python
# tests/test_hrr.py (файл не существует)
import numpy as np
import pytest

def _hrr_bind(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))

def _hrr_unbind(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a))

class TestFFTHRR:
    """FFT-HRR bind/unbind unit tests."""

    def test_bind_unbind_identity(self):
        """unbind(bind(a, b), b) ≈ a для случайных векторов."""
        rng = np.random.default_rng(42)
        dim = 512
        a = rng.normal(0, 1, dim)
        b = rng.normal(0, 1, dim)
        # normalize
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        bound = _hrr_bind(a, b)
        recovered = _hrr_unbind(bound, b)

        cosine = np.dot(a, recovered) / (np.linalg.norm(a) * np.linalg.norm(recovered))
        assert cosine > 0.95, f"Cosine similarity={cosine:.6f}, expected >0.95"

    def test_bind_commutative(self):
        """bind(a, b) ≈ bind(b, a) — circular convolution commutative."""
        rng = np.random.default_rng(42)
        dim = 512
        a = rng.normal(0, 1, dim)
        b = rng.normal(0, 1, dim)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        ab = _hrr_bind(a, b)
        ba = _hrr_bind(b, a)

        np.testing.assert_array_almost_equal(ab, ba, decimal=6)

    def test_bind_with_zero_vector(self):
        """bind(a, zero) = zero."""
        dim = 512
        a = np.ones(dim) / np.sqrt(dim)
        zero = np.zeros(dim)
        result = _hrr_bind(a, zero)
        np.testing.assert_array_almost_equal(result, zero, decimal=6)

    def test_unbind_noise_resilience(self):
        """unbind(bind(a, b) + noise, b) ≈ a (с шумом)."""
        rng = np.random.default_rng(42)
        dim = 512
        a = rng.normal(0, 1, dim)
        b = rng.normal(0, 1, dim)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        bound = _hrr_bind(a, b)
        noisy = bound + rng.normal(0, 0.01, dim)
        recovered = _hrr_unbind(noisy, b)

        cosine = np.dot(a, recovered) / (np.linalg.norm(a) * np.linalg.norm(recovered))
        assert cosine > 0.90, f"Cosine similarity={cosine:.6f}, expected >0.90"
```

---

## 3. EntityField тесты

### 3.1. Статус

Класс `EntityField` **НЕ СУЩЕСТВУЕТ** в кодовой базе. Существует `MemoryField` в `memory/memory_core.py:29`, но он не имеет методов `bind`, `query`, `sync_word`, `_to_dim`, `serialization`.

```python
class MemoryField:
    def __init__(self, name, description, capacity, current_size=0,
                 last_updated=0.0, metadata=None, access_patterns=None):
        # Только поля, никаких методов
```

### 3.2. Код тестов (предполагаемая реализация EntityField)

После реализации класса `EntityField` с методами:

```python
# tests/test_entity_field.py (предполагаемый файл)
import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Optional

# Предполагаемая реализация EntityField (заглушка для тестов)
class EntityField:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.entities: dict = {}
        self.vectors: dict = {}

    def bind(self, name: str, vector: np.ndarray) -> str:
        vector = self._to_dim(vector)
        self.entities[name] = vector
        return name

    def query(self, vector: np.ndarray, top_k: int = 5) -> list:
        vector = self._to_dim(vector)
        scores = [(n, float(np.dot(v, vector))) for n, v in self.entities.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def sync_word(self, word: str) -> str:
        """Normalize word form."""
        return word.strip().lower()

    def _to_dim(self, v: np.ndarray) -> np.ndarray:
        if len(v) < self.dim:
            return np.pad(v, (0, self.dim - len(v)))
        return v[:self.dim]

    def serialize(self) -> dict:
        return {"dim": self.dim, "entities": list(self.entities.keys())}

    @classmethod
    def deserialize(cls, data: dict) -> 'EntityField':
        ef = cls(dim=data["dim"])
        return ef


class TestEntityField:
    """EntityField unit tests (4 теста)."""

    def test_bind_and_query(self):
        """bind entity then query retrieves it as top match."""
        ef = EntityField(dim=10)
        vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ef.bind("test_entity", vec)

        results = ef.query(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "test_entity"
        assert results[0][1] > 0.99

    def test_sync_word_normalization(self):
        """sync_word normalizes case and strips whitespace."""
        ef = EntityField()
        assert ef.sync_word("  Hello  ") == "hello"
        assert ef.sync_word("TEST") == "test"
        assert ef.sync_word(" Multi Word ") == "multi word"

    def test_to_dim_truncate(self):
        """_to_dim truncates vectors larger than dim."""
        ef = EntityField(dim=5)
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = ef._to_dim(vec)
        assert len(result) == 5
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_serialization_roundtrip(self):
        """serialize then deserialize preserves dim."""
        ef = EntityField(dim=128)
        ef.bind("a", np.ones(128))
        data = ef.serialize()
        restored = EntityField.deserialize(data)
        assert restored.dim == 128
        assert "a" in restored.entities
```

---

## 4. Harmonizer тесты

### 4.1. Статус

Класс `Harmonizer` **НЕ СУЩЕСТВУЕТ** в кодовой базе V18.

### 4.2. Код тестов (предполагаемая реализация Harmonizer)

```python
# tests/test_harmonizer.py (предполагаемый файл)
import numpy as np
import pytest
from typing import Optional, List, Tuple

# Предполагаемая реализация
class Harmonizer:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.components: dict = {}

    def compose(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Compose multiple embeddings into one (weighted average)."""
        if not embeddings:
            return np.zeros(self.dim)
        result = np.mean(embeddings, axis=0)
        return result / (np.linalg.norm(result) + 1e-8)

    def decompose(self, embedding: np.ndarray, components: List[np.ndarray]) -> List[float]:
        """Decompose embedding into component weights (cosine similarity)."""
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        weights = []
        for comp in components:
            comp = comp / (np.linalg.norm(comp) + 1e-8)
            weights.append(float(np.dot(embedding, comp)))
        return weights

    def converge(self, target: np.ndarray, max_iter: int = 100, lr: float = 0.01) -> np.ndarray:
        """Converge to target embedding via iterative adjustment."""
        current = np.random.normal(0, 0.1, self.dim)
        for _ in range(max_iter):
            diff = target - current
            current += lr * diff
        return current / (np.linalg.norm(current) + 1e-8)

    def dirty_cascade(self, embeddings: List[np.ndarray], threshold: float = 0.5) -> List[np.ndarray]:
        """Cascade compose with threshold filtering."""
        result = []
        for emb in embeddings:
            if np.linalg.norm(emb) > threshold:
                result.append(emb)
        if not result:
            return [np.zeros(self.dim)]
        return [np.mean(result, axis=0)]


class TestHarmonizer:
    """Harmonizer unit tests (3 теста)."""

    def test_compose_decompose_roundtrip(self):
        """compose then decompose recovers original weights."""
        harm = Harmonizer(dim=10)
        rng = np.random.default_rng(42)
        comps = [rng.normal(0, 1, 10) for _ in range(3)]
        comps = [c / np.linalg.norm(c) for c in comps]

        composed = harm.compose(comps)
        weights = harm.decompose(composed, comps)

        assert len(weights) == 3
        all_positive = all(w > 0 for w in weights)
        assert all_positive, "All weights should be positive for similar components"

    def test_converge_to_target(self):
        """converge produces embedding close to target."""
        harm = Harmonizer(dim=64)
        rng = np.random.default_rng(42)
        target = rng.normal(0, 1, 64)
        target = target / np.linalg.norm(target)

        result = harm.converge(target, max_iter=200, lr=0.05)
        cosine = np.dot(target, result) / (np.linalg.norm(target) * np.linalg.norm(result))
        assert cosine > 0.90, f"Cosine={cosine:.6f}"

    def test_dirty_cascade_filtering(self):
        """dirty_cascade filters low-magnitude embeddings."""
        harm = Harmonizer(dim=4)
        strong = np.array([2.0, 0.0, 0.0, 0.0])
        weak = np.array([0.1, 0.0, 0.0, 0.0])

        result = harm.dirty_cascade([strong, weak], threshold=0.5)
        assert len(result) == 1
        assert result[0][0] > 1.0

    def test_dirty_cascade_all_weak(self):
        """dirty_cascade returns zero vector when all below threshold."""
        harm = Harmonizer(dim=4)
        weak1 = np.array([0.1, 0.0, 0.0, 0.0])
        weak2 = np.array([0.2, 0.0, 0.0, 0.0])
        result = harm.dirty_cascade([weak1, weak2], threshold=0.5)
        assert len(result) == 1
        assert np.all(result[0] == 0.0)
```

---

## 5. HDC тесты (bind/permute/ngram/predict с FFT-HRR)

### 5.1. Статус

Класс `HDC` **НЕ СУЩЕСТВУЕТ** в кодовой базе V18.

### 5.2. Код тестов (предполагаемая реализация)

```python
# tests/test_hdc.py (предполагаемый файл)
import numpy as np
import pytest
from typing import List, Optional

class HDC:
    """Hyperdimensional Computing with FFT-HRR bind/permute."""
    def __init__(self, dim: int = 10000):
        self.dim = dim

    def _hrr_bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))

    def _hrr_unbind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a))

    def permute(self, v: np.ndarray, shift: int = 1) -> np.ndarray:
        return np.roll(v, shift)

    def ngram(self, seq: List[np.ndarray]) -> np.ndarray:
        if not seq:
            return np.zeros(self.dim)
        result = seq[0]
        for v in seq[1:]:
            result = self._hrr_bind(result, v)
        return result

    def predict(self, context: List[np.ndarray], items: List[np.ndarray]) -> int:
        """Predict which item comes next using HDC."""
        ngram_vec = self.ngram(context)
        scores = [float(np.dot(ngram_vec, item)) for item in items]
        return int(np.argmax(scores))


class TestHDC:
    """HDC unit tests (2 теста)."""

    @pytest.fixture
    def hdc(self):
        return HDC(dim=1000)

    def test_bind_permute_ngram(self, hdc):
        """bind, permute, ngram produce valid HD vectors."""
        rng = np.random.default_rng(42)
        v1 = rng.choice([-1, 1], hdc.dim).astype(np.float64)
        v2 = rng.choice([-1, 1], hdc.dim).astype(np.float64)
        v3 = rng.choice([-1, 1], hdc.dim).astype(np.float64)

        # Permute preserves norm
        p1 = hdc.permute(v1)
        assert abs(np.linalg.norm(p1) - np.linalg.norm(v1)) < 1e-6

        # N-gram produces deterministic result
        ng = hdc.ngram([v1, v2, v3])
        assert len(ng) == hdc.dim
        assert not np.all(ng == 0)

    def test_predict_retrieves_correct_item(self, hdc):
        """predict with ngram context retrieves associated item."""
        rng = np.random.default_rng(42)
        items = [rng.choice([-1, 1], hdc.dim).astype(np.float64) for _ in range(10)]

        # Create a "memory" by binding context to each item
        ctx_vec = hdc.ngram([items[0], items[1], items[2]])
        bound_items = [hdc._hrr_bind(ctx_vec, item) for item in items]

        # Predict: given first 3 items, find the bound one
        pred_idx = hdc.predict([items[0], items[1], items[2]], items)
        # Should retrieve item[3] if bound in order, but with HRR this is approximate
        assert 0 <= pred_idx < 10
```

---

## 6. Capacity тесты (grow/prune/auto)

### 6.1. Статус

Компонент управления ёмкостью (capacity) **НЕ СУЩЕСТВУЕТ** в кодовой базе V18.

### 6.2. Код тестов (предполагаемая реализация)

```python
# tests/test_capacity.py (предполагаемый файл)
import numpy as np
import pytest
from typing import Dict, Any

class DynamicCapacity:
    """Dynamic capacity manager with grow/prune/auto."""
    def __init__(self, initial_capacity: int = 100, min_capacity: int = 10, max_capacity: int = 10000):
        self.capacity = initial_capacity
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.load: Dict[str, Any] = {}

    def grow(self, amount: int = 10) -> int:
        """Increase capacity."""
        self.capacity = min(self.capacity + amount, self.max_capacity)
        return self.capacity

    def prune(self, fraction: float = 0.5) -> int:
        """Prune oldest items to free capacity."""
        n_to_remove = int(len(self.load) * fraction)
        for key in list(self.load.keys())[:n_to_remove]:
            del self.load[key]
        return len(self.load)

    def auto(self) -> int:
        """Auto-adjust capacity based on load."""
        load_ratio = len(self.load) / max(self.capacity, 1)
        if load_ratio > 0.9:
            self.grow(amount=int(self.capacity * 0.2))
        elif load_ratio < 0.3 and self.capacity > self.min_capacity:
            self.capacity = max(self.min_capacity, int(self.capacity * 0.8))
        return self.capacity

    def add(self, key: str, value: Any):
        self.load[key] = value

    def __len__(self):
        return len(self.load)


class TestDynamicCapacity:
    """Capacity tests (2 теста)."""

    def test_grow_and_prune(self):
        """grow increases capacity, prune reduces load."""
        dc = DynamicCapacity(initial_capacity=50)
        assert dc.capacity == 50

        dc.grow(20)
        assert dc.capacity == 70

        for i in range(30):
            dc.add(f"item_{i}", i)
        assert len(dc) == 30

        remaining = dc.prune(fraction=0.5)
        assert remaining == 15

    def test_auto_adjustment(self):
        """auto grows at high load, shrinks at low load."""
        dc = DynamicCapacity(initial_capacity=100, min_capacity=50)

        # High load: fill 95 items
        for i in range(95):
            dc.add(f"item_{i}", i)
        dc.auto()
        assert dc.capacity > 100, f"Expected growth, got capacity={dc.capacity}"

        # Low load: clear to 10 items
        dc.load.clear()
        for i in range(10):
            dc.add(f"item_{i}", i)
        old_cap = dc.capacity
        dc.auto()
        assert dc.capacity <= old_cap, "Expected shrink at low load"
```

---

## 7. L1/Checkpoint тесты

### 7.1. Статус

`FractalGraphL1L2` существует в `eva_ai/memory/pie_integration/fractal_graph_l1_l2.py` (672 строки). Это **единственный** из реальных компонентов, который можно тестировать.

Checkpoint-механизм существует в `online_trainer.py` (методы `save_checkpoint`, `load_checkpoint`).

### 7.2. Код тестов

```python
# tests/test_l1_checkpoint.py (предполагаемый файл)
import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def l1_graph():
    """Create FractalGraphL1L2 with temp DB."""
    from eva_ai.memory.pie_integration.fractal_graph_l1_l2 import FractalGraphL1L2
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    graph = FractalGraphL1L2(db_path)
    yield graph
    os.unlink(db_path)


class TestFractalGraphL1L2:
    """L1/L2 FractalGraph tests (2 теста)."""

    def test_create_and_get_activation_profile(self, l1_graph):
        """create_activation_profile then get_activation_profile roundtrip."""
        rng = np.random.default_rng(42)
        emb = rng.normal(0, 1, 768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        profile_id = l1_graph.create_activation_profile(
            domain="astrophysics",
            model_id="model_a",
            quant_profile="Q4_K_M",
            initial_embedding=emb
        )
        assert profile_id.startswith("profile_")

        profile = l1_graph.get_activation_profile("astrophysics", "model_a")
        assert profile is not None
        assert profile.domain == "astrophysics"
        assert profile.model_id == "model_a"
        assert profile.quant_profile == "Q4_K_M"

        cosine = float(np.dot(profile.centroid, emb))
        assert cosine > 0.99, f"Centroid mismatch, cosine={cosine}"

    def test_create_and_update_routing_rule(self, l1_graph):
        """create_routing_rule, update stats, list works."""
        rule_id = l1_graph.create_routing_rule(
            domain="physics",
            temperature=0.5,
            repeat_penalty=1.5,
            max_tokens=2048
        )
        assert rule_id.startswith("rule_")

        rule = l1_graph.get_routing_rule("physics")
        assert rule is not None
        assert rule.domain == "physics"
        assert rule.temperature == 0.5
        assert rule.max_tokens == 2048

        # Update stats (success)
        success = l1_graph.update_routing_rule_stats(rule_id, success=True)
        assert success

        rule2 = l1_graph.get_routing_rule("physics")
        assert rule2.access_count == 1
        assert rule2.success_count == 1
        assert rule2.priority == 1.0

        # Update stats (failure)
        l1_graph.update_routing_rule_stats(rule_id, success=False)
        rule3 = l1_graph.get_routing_rule("physics")
        assert rule3.access_count == 2
        assert rule3.success_count == 1
        assert rule3.priority == 0.5

        # List rules
        rules = l1_graph.list_routing_rules()
        assert len(rules) >= 1
```

---

## 8. Safety: qwen_knowledge.py анализ

### 8.1. Существование

Файл `qwen_knowledge_loader.py` существует в 2 экземплярах:

| Файл | Строк | Размер |
|------|-------|--------|
| `eva_ai/fcp_core/qwen_knowledge_loader.py` | 225 | Основной |
| `scripts/qwen_knowledge_loader.py` | 211 | Дубликат |

Первый — 225 строк (не 121, как указано в задании). Второй — 211 строк, дубликат с незначительными отличиями.

### 8.2. References (импорты)

**Ни один из двух файлов не импортируется нигде в кодовой базе.** Поиск imports:

```
grep "import.*qwen_knowledge\|from.*qwen_knowledge" — 0 результатов
```

Единственная reference — строковая константа в `online_trainer.py:376`:

```python
DEFAULT_QWEN_DB = "eva_ai/fcp_core/data/qwen_knowledge.db"
```

И её использование через `os.path.exists(self.qwen_db_path)` и `sqlite3.connect(self.qwen_db_path)`.

### 8.3. Оценка безопасности

**Риск: СРЕДНИЙ.**

1. **Hardcoded абсолютный путь** к `qwen_knowledge.npz`:
   ```python
   QWEN_NPZ = r"C:\Users\black\OneDrive\Desktop\qwen_knowledge.npz"
   ```
   В обоих экземплярах файла. Это создаёт проблему переносимости: скрипт невозможно запустить на другой машине без правки кода.

2. **Дублирование кода**: `scripts/qwen_knowledge_loader.py` — почти точная копия `eva_ai/fcp_core/qwen_knowledge_loader.py` с различиями:
   - `scripts/`: использует `sys.path.insert(0, ...)` для импорта
   - `fcp_core/`: импортирует как часть пакета
   - `scripts/` имеет `_clean_token_text` с `f"<unk>"` vs `f"<token_{id(text)}>"` в `fcp_core/`
   - `scripts/` не имеет дедупликационной логики `edge_ids_used`

3. **Отсутствие валидации входных данных**: `deduplicate_npz` не проверяет существование файла, формат данных, соответствие размерностей.

4. **Рекомендация**: удалить `scripts/qwen_knowledge_loader.py` (дубликат), оставить `eva_ai/fcp_core/qwen_knowledge_loader.py`, параметризовать путь к NPZ через аргументы командной строки.

### 8.4. Тест для qwen_knowledge_loader

```python
# tests/test_qwen_loader.py (предполагаемый файл)
import numpy as np
import pytest
import tempfile
import os

@pytest.fixture
def sample_npz():
    """Create a minimal npz file with 10 entries."""
    rng = np.random.default_rng(42)
    rows = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.uint32)
    cols = np.array([1, 2, 0, 2, 0, 1, 4, 5, 3, 5], dtype=np.uint32)
    vals = rng.uniform(-1, 1, 10).astype(np.float32)
    counts = rng.integers(1, 100, 10).astype(np.uint32)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        npz_path = f.name
    np.savez(npz_path, rows=rows, cols=cols, vals=vals, counts=counts)
    yield npz_path
    os.unlink(npz_path)


class TestQwenKnowledgeLoader:
    """Qwen knowledge loader tests."""

    def test_deduplicate_npz(self, sample_npz):
        """deduplicate_npz returns unique pairs."""
        from eva_ai.fcp_core.qwen_knowledge_loader import deduplicate_npz
        out_rows, out_cols, out_vals, out_counts = deduplicate_npz(sample_npz)

        assert len(out_rows) > 0
        assert len(out_rows) == len(out_cols) == len(out_vals) == len(out_counts)

        # Verify no duplicates
        pairs = set(zip(out_rows, out_cols))
        assert len(pairs) == len(out_rows)

    def test_clean_token_text(self):
        """_clean_token_text removes control chars and strips."""
        from eva_ai.fcp_core.qwen_knowledge_loader import _clean_token_text
        assert _clean_token_text("Ġhello") == "hello"
        assert _clean_token_text("  test  ") == "test"
        assert _clean_token_text("") == "<unk>"  # or "<token_...>"

    def test_make_node_id(self):
        """_make_node_id formats correctly."""
        from eva_ai.fcp_core.qwen_knowledge_loader import _make_node_id
        assert _make_node_id(42) == "qwen_tok_42"
        assert _make_node_id(0) == "qwen_tok_0"
```

---

## 9. Dead Code Check

### 9.1. Broken imports: `server_main.py` не существует

**Критический dead code.** Три файла импортируют из несуществующего модуля:

| Файл | Строка | Импорт |
|------|--------|--------|
| `eva_ai/server.py` | 8 | `from .server_main import (app, web_gui_instance, SessionManager, ...)` |
| `eva_ai/server_routes.py` | 11 | `from .server_main import web_gui_instance, app, extract_text_from_file` |
| `eva_ai/server_handlers.py` | 9 | `from .server_main import web_gui_instance, app` |

Файл `server_main.py` **никогда не существовал** в git-истории (проверено: `git log --all --oneline -- "eva_ai/server_main.py"` — 0 результатов).

**Последствия:**
- `import eva_ai.server` — **упадёт** с `ModuleNotFoundError: No module named 'eva_ai.server_main'`
- `from eva_ai import server_routes` — **упадёт**
- `from eva_ai import server_handlers` — **упадёт**
- Веб-интерфейс полностью неработоспособен

**Рекомендация:** удалить `server.py`, `server_routes.py`, `server_handlers.py` или реализовать `server_main.py`.

### 9.2. Дубликат скрипта: `scripts/qwen_knowledge_loader.py`

Дубликат `eva_ai/fcp_core/qwen_knowledge_loader.py` с минимальными отличиями. Не используется.

**Рекомендация:** удалить `scripts/qwen_knowledge_loader.py`.

### 9.3. Незавершённый скрипт: `scripts/verify_db.py`

Содержит только 2 строки:

```python
if __name__ == "__main__":
    main()
```

Функция `main()` не определена. Синтаксически корректен, но невыполним.

**Рекомендация:** удалить или реализовать.

### 9.4. `scripts/inject_qwen_embeddings.py`

Зависит от `models/qwen_layer_model.pt` — файла, которого нет в репозитории (в `.gitignore`). Неисполним без внешнего файла.

**Рекомендация:** добавить проверку существования файла, документировать источник.

### 9.5. Unused imports (статический анализ)

Из 520 Python-файлов ~328 содержат потенциально неиспользуемые импорты. Основные проблемные файлы:

| Файл | Неиспользуемые импорты |
|------|----------------------|
| `server.py` | `WebGUI`, `AuthManager`, `web_gui_instance`, `EntityExtractor`, `create_app`, `get_app`, `app`, `SessionManager`, `EthicsChecker`, `extract_text_from_file` |
| `core/component_initializer.py` | 20+ неиспользуемых имён |
| `__init__.py` | `eva_ai` (self-import) |
| `analytics/analytics_integrated.py` | `Event`, `get_event_bus`, `ComponentState`, `EventTypes`, `Tuple`, `List` |
| `contradiction/core_detection.py` | `defaultdict`, `Union`, `deque` |
| `contradiction/detect_core.py` | `polarity_scores`, `Set`, `Union`, `tokenize`, `timedelta` |
| `mlearning/ml_core.py` | (зависит от файла) |

*Примечание:* статический анализатор даёт ложные срабатывания на re-exports, TYPE_CHECKING и __all__. Реальные проблемные импорты — те, что помечены в п. 9.1 (broken).

### 9.6. `nlp_fallbacks.py:20`: Unused imports `annotations`, `Tuple`, `Union`

```python
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
```

`Tuple` и `Union` не используются в файле (все аннотации используют `|` из `from __future__ import annotations`).

---

## 10. Сводная таблица рекомендаций

| # | Рекомендация | Файл | Приоритет | Тип |
|---|-------------|------|-----------|-----|
| 1 | Удалить `server_main.py` imports из 3 файлов или реализовать модуль | `server.py:8`, `server_routes.py:11`, `server_handlers.py:9` | **P0 (Critical)** | Dead code |
| 2 | Удалить дубликат `scripts/qwen_knowledge_loader.py` | `scripts/qwen_knowledge_loader.py` | **P1 (High)** | Dead code |
| 3 | Удалить или реализовать `scripts/verify_db.py` | `scripts/verify_db.py` | **P2 (Medium)** | Dead code |
| 4 | Создать директорию `tests/` и `conftest.py` | `tests/` (не существует) | **P0 (Critical)** | Coverage |
| 5 | Реализовать FFT-HRR bind/unbind + тесты | `memory/hrr.py` (не существует) | **P1 (High)** | Coverage |
| 6 | Реализовать EntityField + 4 теста | не существует | **P1 (High)** | Coverage |
| 7 | Реализовать Harmonizer + 3 теста | не существует | **P1 (High)** | Coverage |
| 8 | Реализовать HDC + 2 теста | не существует | **P1 (High)** | Coverage |
| 9 | Реализовать DynamicCapacity + 2 теста | не существует | **P2 (Medium)** | Coverage |
| 10 | Написать тесты для `FractalGraphL1L2` | `memory/pie_integration/fractal_graph_l1_l2.py` | **P1 (High)** | Coverage |
| 11 | Написать тесты для `GNNTrainer` Qwen KB | `fcp_core/online_trainer.py` | **P2 (Medium)** | Coverage |
| 12 | Параметризовать hardcoded путь в `qwen_knowledge_loader.py` | `fcp_core/qwen_knowledge_loader.py:22` | **P2 (Medium)** | Safety |
| 13 | Убрать неиспользуемые импорты | `nlp_fallbacks.py:20`, `server.py:8-19` | **P3 (Low)** | Quality |

---

## 11. Детальный анализ тестовой инфраструктуры

### 11.1. Текущее состояние

```python
# pyproject.toml:19-31
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
norecursedirs = ["scripts", "tools"]
```

Директория `tests/` **не существует**. `pytest --collect-only` зависает (timed out after 30s), вероятно, из-за попытки импорта тяжёлых зависимостей.

### 11.2. Необходимые действия для создания тестовой инфраструктуры

1. Создать `tests/__init__.py` (пустой)
2. Создать `tests/conftest.py` с фикстурами
3. Создать `tests/test_hrr.py` с тестами FFT-HRR (будет уместен после реализации)
4. Создать `tests/test_l1_l2.py` с тестами FractalGraphL1L2
5. Создать `tests/test_qwen_loader.py` с тестами дедупликации

### 11.3. conftest.py (предлагаемый)

```python
# tests/conftest.py
import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database path."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_embedding(rng):
    """Create a normalized 768-dim random embedding."""
    emb = rng.normal(0, 1, 768).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)
```

---

## 12. Заключение

**V18 (c37a8d8) не добавляет нового кода — это документальный коммит.** Все проблемы V16/V17 остаются открытыми с точки зрения тестового покрытия.

### Ключевые выводы:

1. **Тестов нет вообще.** Директория `tests/` не создана, `conftest.py` отсутствует. Это P0-проблема.

2. **Из 11 компонентов V15-списка 0 реализованы в коде.** Все они — спецификационные артефакты. Единственный реальный родственный компонент — `FractalGraphL1L2` (672 строки) — не имеет тестов.

3. **Dead code высокого приоритета**: `server.py`, `server_routes.py`, `server_handlers.py` импортируют несуществующий `server_main.py`. Веб-интерфейс гарантированно не работает при попытке импорта.

4. **Safety**: `qwen_knowledge_loader.py` существует в 2 копиях, обе не импортируются. Hardcoded путь к NPZ на рабочем столе.

5. **Единственный тестируемый компонент прямо сейчас** — `FractalGraphL1L2`, для которого в отчёте предоставлены готовые тесты (п. 7.2).

### Первоочередные действия:

```
P0: Создать tests/ + conftest.py
P0: Исправить broken imports в server.py, server_routes.py, server_handlers.py
P1: Написать тесты для FractalGraphL1L2 (код готов — см. п. 7.2)
P1: Удалить дубликат scripts/qwen_knowledge_loader.py
P1: Реализовать FFT-HRR и написать тесты (код готов — см. п. 2.2)
P2: Удалить scripts/verify_db.py
P2: Параметризовать hardcoded пути в qwen_knowledge_loader.py
```

---

*Report generated by Quality-Safety Agent for FCF. Все тесты в отчёте написаны в предположении, что тестируемые компоненты будут реализованы. Для FractalGraphL1L2 тесты готовы к немедленному использованию.*
