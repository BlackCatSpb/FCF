# EVA — Свёрточная оптимизация для FCF

## 6 инноваций, адаптированных под координатное пространство

### 1. Coordinate-Aware Depthwise Convolution

Стандартный Conv1D: `y[i] = Σ w[k] * x[i+k]` — веса фиксированы для всех позиций.

**Coordinate-aware**: `y[i] = Σ f(coord_i, coord_{i+k}) * x[i+k]`

Где `f(a,b) = MLP([a, b])` — вес зависит не от позиции, а от **расстояния в ℝ¹²⁸**.
Это та же идея что и manifold bias, но в свёрточном kernel.

```python
class CoordinateConv1D(nn.Module):
    def __init__(self, dim, kernel_size=3):
        self.weight_net = nn.Sequential(
            nn.Linear(dim * 2, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):  # [B, L, D]
        out = torch.zeros_like(x)
        for k in range(-1, 2):  # kernel_size=3
            shifted = torch.roll(x, -k, dims=1)
            # Вес = функция от коорд. расстояния
            combined = torch.cat([x, shifted], dim=-1)
            weight = self.weight_net(combined).sigmoid()
            out += weight * shifted
        return out
```

**Выигрыш**: kernel адаптируется к СЕМАНТИЧЕСКОЙ близости, не позиционной.

---

### 2. Boundary-Guided Convolution

Свёртка видит `<W>` и `</W>` — НИКОГДА не пересекает границы слов.

```python
class BoundaryGuidedConv(nn.Module):
    def forward(self, x, boundary_mask):
        # boundary_mask[i,j] = 1 если i и j в ОДНОМ слове
        out = F.conv1d(x, self.weight)  # стандартный conv
        out = out * boundary_mask       # обнуляем пересечения границ
        return out
```

**Выигрыш**: чистая intra-word информация без паразитного смешивания слов.

---

### 3. Word-Weight Modulated Convolution

Важные слова (сущ, глаголы) влияют сильнее. Weight encoder даёт коэффициент.

```python
class WeightModulatedConv(nn.Module):
    def forward(self, x, word_weights):
        out = self.conv(x)
        # weight[b,i] умножает вклад позиции i
        return out * word_weights.unsqueeze(-1)
```

---

### 4. Gated Fractal Convolution (GLU)

Половина каналов — фильтр, половина — gate. Gate решает ЧТО пропустить.

```python
class GatedConv(nn.Module):
    def __init__(self, dim, kernel_size):
        self.conv_filter = nn.Conv1d(dim, dim//2, kernel_size, padding='same')
        self.conv_gate = nn.Conv1d(dim, dim//2, kernel_size, padding='same')
    
    def forward(self, x):
        f = self.conv_filter(x)
        g = self.conv_gate(x).sigmoid()
        return f * g  # element-wise gate
```

---

### 5. Dilated Fractal Convolution

Тот же фрактальный принцип что и внимание, но свёртками:

| Уровень | Операция | Dilation | Зона видимости |
|---------|----------|----------|----------------|
| Символ | Conv1d k=3 | 1 | ±1 символ |
| Биграмма | DilatedConv | 2 | ±3 символа |
| Слово | DilatedConv | 4 | ±7 символов |
| Фраза | DilatedConv | 8 | ±15 символов |

```python
class FractalConv(nn.Module):
    def __init__(self, dim):
        self.levels = nn.ModuleList([
            nn.Conv1d(dim, dim//4, 3, padding=2**l, dilation=2**l)
            for l in range(4)  # 4 уровня: dilation 1,2,4,8
        ])
    def forward(self, x):
        return torch.cat([conv(x) for conv in self.levels], dim=1)
```

**VRAM**: O(L) на всех уровнях вместо O(L²) у attention. Для L=256 экономия 64×.

---

### 6. Hybrid Block: FractalConv + Attention

Финальная архитектура одного Transformer Block:

```
Вход x [B, L, D]
  │
  ├─→ FractalConv (dilations 1,2,4,8) ─→ локальные паттерны [B, L, D]
  │
  ├─→ Attention (только на граничных токенах) ─→ глобальные связи
  │       │
  │       └─ sparse: только <W>, </W>, <S>, </S> + top-10% по весу
  │
  └─→ Gate: conv_out ⊙ σ(attention_out) ─→ смешивание
        │
        └─→ FFN SwiGLU ─→ Выход
```

**Математика блока:**
```
x_local  = FractalConv(Norm(x))      # O(L) — локальные паттерны
x_global = SparseAttention(Norm(x))  # O(L·K) — глобальные связи, K<<L
x = x + Gate(x_local, x_global)      # x_local ⊙ σ(x_global)
x = x + SwiGLU(Norm(x))
```

**Итоговая экономия:**

| Метрика | Сейчас (Attention-only) | Conv + Sparse Attention |
|---------|------------------------|------------------------|
| Сложность | O(L²) | O(L) + O(L·K), K≈L/4 |
| VRAM для L=256 | ~40 MB на слой | ~5 MB на слой |
| Макс длина блока | 128 (VRAM лимит) | **512+** |
| Скорость шага (батч 8) | ~360 мс | **~120 мс (×3)** |
