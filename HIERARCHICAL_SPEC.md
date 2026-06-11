# Hierarchical FractalField v2 — Finite Anchor Matrix + BMSSP Field Mask

## 1. Основа

**BPE словарь конечен** (V = 32K). Это не баг, а фича — конечность даёт:
- Kонечную матрицу якорей
- Предсказуемую сложность BMSSP
- Вариативность через комбинаторику полей, а не через параметры

## 2. Архитектура

### 2.1 FractalField code split

```
z ∈ R^L (L = 512)
z = [z_c | z_a | z_m]

z_c ∈ R^256 — identity концепта (медленная пластичность, lr_c = 0.01×)
z_a ∈ R^128 — внимание/маска (быстрая, lr_a = 1.0×)
    z_a = [z_a_doc | z_a_block | z_a_sent]
          [48d     | 48d       | 32d     ]
z_m ∈ R^128 — мета-пластичность (lr_m = 0.1×)

Базис B ∈ R^(L×D), D=384, QTQ = I
v = normalize(z @ B)  — вектор полной размерности
```

## 2. BPE-структура как морфологическая маска

QWEN с 146K словарём: «князь» → [кня, зь], «князя» → [кня, зя].
BPE хранит морфологию в самой токенизации: корень [кня] — identity,
суффикс [зь/зя/зем] — грамматическое поле.

Наша SentencePiece (32K): «князь» → [▁князь], «князя» → [▁князя].
Каждая форма — отдельный ID, морфологическая связь потеряна.

**Решение: НЕ менять токенизатор. Восстановить морфологию через H-матрицу.**

```
H[▁князь, ▁князя] > θ  — потому что они в одинаковых контекстах
H[▁князь, ▁князем] > θ
H[▁князь, ▁война]  = 0  — разные поля

⇒ field[▁князь] ∩ field[▁князя] ≠ ∅
   Они в одном морфологическом поле = "корень князь"
⇒ field[▁князь] ∩ field[▁война] = ∅
   Разные семантические поля

Маска H → BMSSP автоматически выделяет:
  - морфологические группы (падежи одного слова)
  - семантические поля (синонимы, тематические кластеры)
  - синтаксические роли (субъект, объект)
```

**BPE-разбивка QWEN — это заранее вычисленная H-матрица,**
**зашитая в словарь. Наша H будет вычислена из данных.**
**Итог тот же, но наш — обучаемый, а не жёстко заданный.**

### 2.2 Finite Anchor Matrix

```
N_a = 1024  — максимальное количество якорей (2^10, степень 2 для BMSSP)

Якоря = top-N_a существительных по frequency × centrality (отбираются при init)

Матрица H ∈ R^(N_a × N_a):
  H[i,j] = PMI(anchor_i, anchor_j)  — семантическая сила связи
  H[i,j] = 0 если PMI < θ (спарсенность ~90%)

Свойства:
  H[i,j] > H[j,i] → anchor_i властвует над anchor_j
  H — не симметрична (направленная связь)
  Хранится как CSC/CSR sparse (90%+ нулей)

field[cid] ∈ {0,1}^N_a — бинарный вектор для каждого токена:
  field[cid][i] = BMSSP(cid, anchor_i) < boundary_i
  boundary_i = adaptive(PMI, distance, frequency)

Хранение: N_a / 8 bytes на токен = 32K × 128 байт = 4MB (битовая упаковка)
```

### 2.3 BMSSP как ядро маски

```
Algorithm: BMSSP(l, B, S) → (B', U)
  l = уровень (0=предложение, 1=блок, 2=документ)
  B = граница (макс. семантическое расстояние от якоря)
  S = источники (якорь + его дополнения)

  level 2 (документ):
    якорь = тема документа
    FIND(Pivots) → подтемы ← H[i, :] > θ_2
    ↓ BMSSP(1, B_2, Pivots)
    
  level 1 (блок):
    якорь = главная сущность абзаца
    FIND(Pivots) → ключевые слова ← H[i, :] > θ_1  
    ↓ BMSSP(0, B_1, Pivots)
    
  level 0 (предложение):
    якорь = главное существительное
    U = концепты внутри boundary
    Релаксация: для каждого u ∈ U, если d[u] + w_uv < d[v] → d[v] обновляется

Где w_uv = 1 / (1 + PMI(u,v))  — вес ребра в графе концептов (PMI = сила связи)

BOUNDARY: для каждого level l boundary_l = f(PMI_anchor_x, position_rank_in_field)

Функция f:
  - Уровень документа: boundary_2 = 0.8 (широкое поле)
  - Уровень блока: boundary_1 = 0.5  
  - Уровень предложения: boundary_0 = 0.3 (узкое поле)

Маска: для токена cid в контексте якоря a:
  mask[cid] = 1 если BMSSP(l, boundary_l, {a}) содержит cid
  mask[cid] = 0 иначе
```

### 2.4 Иерархия якорей: дополнение vs доминирование

```
Каждый якорь может быть:
  - Главным (dominator): имеет H[i,j] > H[j,i] для большинства j
  - Дополнением (subordinate): H[j,i] > H[i,j] для главного j
  - Равным (peer): H[i,j] ≈ H[j,i]

Из requirement: "каждое существительное-якорь может быть дополнением другого якоря"

Это означает что H образует DAG (частичный порядок):
  dominator → subordinate → sub-subordinate ...
  
При BMSSP: на каждом уровне рекурсии, S (источники) = 
  {anchor_l | anchor_l ∈ Pivots_of_anchor_{l+1}}
  
Т.е. якорь уровня документа порождает якоря блоков, которые порождают
якоря предложений. Каждый subordinate — дополнение своего dominator.
```

### 2.5 Сдвиг внутри поля

```
Для активной маски (где mask=1), z_a концепта сдвигается к z_a якоря:

z_a_shift[cid] = Σ_{anchor ∈ active_fields} w_a * (z_a_anchor - z_a_cid)

w_a = softmax(H[anchor, :] ∩ field[cid]) — вес якоря относительно поля

mask_continuous = softmax(z_a_shift[cid] · z_a_candidate, τ)
При генерации: mask_continuous > 0.5 → 1, иначе 0
```

### 2.6 Мета-пластичность

```
z_m ∈ R^128 управляет:
  lr_mod = 1/(1+exp(-z_m·w_lr))     — множитель learning rate
  th_mod = tanh(z_m·w_th)            — сдвиг inhibition threshold
  gate_k = 1/(1+exp(-z_m·w_k))       — бинарный gate для каждого поля k
  pause = 1/(1+exp(-z_m·w_pause))    — полное замораживание (пластичность выкл)

Если pause > 0.9 → z не обновляется вообще (консолидация)

w_lr, w_th, w_k, w_pause ∈ R^(128×1) — обучаемые, инициализируются нулями
```

## 3. Изменения в коде

### 3.1 FractalField (concept_space.py)

```python
class FractalField:
    def __init__(self, dim=384, latent_dim=512, n_anchors=1024):
        self.dim = dim
        self.latent_dim = latent_dim
        self.l_c = 256    # identity
        self.l_a = 128    # attention (48+48+32)
        self.l_m = 128    # meta
        self.n_anchors = n_anchors
        
        # Basis (unchanged)
        rng = np.random.RandomState(42)
        mat = rng.randn(latent_dim, dim).astype(np.float32)
        Q, _ = np.linalg.qr(mat, mode='reduced')
        self.basis = Q.astype(np.float32)
        
        # Anchor matrix H (sparse, built from lattice PMI)
        self.H = None           # N_a × N_a sparse CSR
        self.anchor_ids = []    # list of anchor concept IDs
        self.anchor_rank = {}   # anchor_id → index in H
        
        # Binary field vectors: V × N_a bits → packed as uint8
        self.fields = None      # np.ndarray (V, N_a // 8), dtype=uint8
        
        # Meta weights (trainable)
        self.meta_w_lr = np.zeros(self.l_m, dtype=np.float32)
        self.meta_w_th = np.zeros(self.l_m, dtype=np.float32)
        self.meta_w_pause = np.zeros(self.l_m, dtype=np.float32)
        self.meta_b_lr = np.float32(0.0)
        self.meta_b_th = np.float32(0.0)
        self.meta_b_pause = np.float32(0.0)
        
        # Per-field gate weights (n_anchors)
        self.meta_w_gate = np.zeros((self.l_m, n_anchors), dtype=np.float32)
        
        self.codes = {}
        self._matrix_dirty = True
    
    def split_code(self, z):
        """Разделить код на подпространства."""
        return z[:self.l_c], z[self.l_c:self.l_c+self.l_a], z[self.l_c+self.l_a:]
    
    def get_field(self, cid):
        """Бинарный вектор поля для концепта."""
        if self.fields is None or cid >= len(self.fields):
            return None
        # Распаковать из uint8 в bool
        byte_idx = np.arange(self.n_anchors // 8)
        bit_idx = np.arange(self.n_anchors)
        return (self.fields[cid, byte_idx] >> (bit_idx % 8)) & 1
    
    def set_field(self, cid, bits):
        """Установить поле для концепта."""
        if self.fields is None:
            self.fields = np.zeros((len(self.codes), self.n_anchors // 8), dtype=np.uint8)
        packed = np.packbits(bits.astype(np.uint8))
        self.fields[cid] = packed
```

### 3.2 BMSSP как метод FractalField

```python
def bmssp(self, level, boundary, sources, visited=None):
    """Bounded Multi-Source Shortest Path.
    
    Args:
        level: 0=предложение, 1=блок, 2=документ
        boundary: макс. семантическое расстояние
        sources: set of concept IDs (якоря + дополнения)
        visited: set of already-visited concepts (рекурсия)
    
    Returns:
        B': фактическая граница
        U: set концептов внутри поля
    """
    if visited is None:
        visited = set()
    
    if level == 0:
        # BASE: level 0 — предложение, прямой поиск
        return self._bmssp_base(boundary, sources, visited)
    
    # FIND(Pivots): ключевые узлы-дополнения через H
    pivots = self._find_pivots(sources, level)
    weights = {}  # weight = PMI-based distance
    
    # D.INITIALIZE с M = 2^((l-1)*t)
    dist = {s: 0.0 for s in sources}
    for p in pivots:
        dist[p] = self._distance_to_sources(p, sources)
    
    # D = priority queue (min distance)
    D = SortedList(sources, key=lambda x: dist[x])
    B_l = min(dist.values())
    U = set()
    
    while len(U) < self.n_anchors * 2**level and D:
        B_i, S_i = D.pop(0)
        B_i, U_i = self.bmssp(level - 1, B_i, {S_i}, visited | U)
        U |= U_i
        
        K = []
        for u in U_i:
            for v in self._neighbors(u, boundary):
                if dist.get(u, float('inf')) + self._edge_weight(u, v) < dist.get(v, float('inf')):
                    dist[v] = dist[u] + self._edge_weight(u, v)
                    if dist[v] < boundary:
                        D.add(v)
                        if dist[v] < B_i:
                            K.append((v, dist[v]))
        D.batch_prepend(K)
    
    visited |= U
    return min(B_i, boundary), U
```

### 3.3 Обновление field[] при train_from_text

```python
# В train_from_text, после STDP по всем парам:
for idx, cid in enumerate(ids):
    # Для каждого токена в линии:
    # 1. Вычислить BMSSP от каждого якоря, чья активность > θ
    active_anchors = self._active_anchors(cid, context=ids)
    
    # 2. Обновить field[cid] для найденных якорей
    for anchor_id in active_anchors:
        level = self._anchor_level(anchor_id)
        boundary = self._anchor_boundary(anchor_id, level)
        B, U = self.cs.fractal.bmssp(level, boundary, {anchor_id})
        for cid_u in U:
            bit = self.cs.fractal.anchor_rank[anchor_id]
            self.cs.fractal.fields[cid_u, bit // 8] |= (1 << (bit % 8))
```

### 3.4 Маска в генерации (CrystalGenerator.generate)

```python
def generate(self, seed_word, max_words=25, temperature=0.3, top_k=20):
    ids = self._encode_input(seed_word)
    current = list(ids)
    
    for step in range(max_words):
        # 1. Определить активные якоря для последних концептов
        context = current[-3:]
        active = set()
        for cid in context:
            field_bits = self.cs.fractal.get_field(cid)
            if field_bits is not None:
                active |= set(np.where(field_bits)[0])
        
        # 2. Для каждого кандидата = маска + сдвиг
        candidates = self._get_candidates(current)
        scores = []
        for cid in candidates:
            field_c = self.cs.fractal.get_field(cid)
            if field_c is None:
                continue
            
            # Маска = совпадение полей
            if active and field_c is not None:
                field_overlap = np.bitwise_and(active, np.where(field_c)[0])
                mask_score = len(field_overlap) / max(len(active), 1)
            else:
                mask_score = 0.0
            
            # Сдвиг внутри поля
            _, z_a_c, _ = self.cs.fractal.split_code(self.cs.fractal.codes.get(cid))
            z_a_shifted = self._compute_shift(cid, active)
            shift_score = np.dot(z_a_shifted, z_a_c) if z_a_shifted is not None else 0.0
            
            # Векторная семантика
            v_cid = self.cs.concept_vectors.get(cid)
            v_last = self.cs.concept_vectors.get(current[-1])
            vec_score = np.dot(v_cid, v_last) if v_cid is not None and v_last is not None else 0.0
            
            score = vec_score * 0.5 + mask_score * 0.3 + shift_score * 0.2
            scores.append((cid, score))
        
        scores.sort(key=lambda x: -x[1])
        # top_k выборка + temperature
        ...
```

## 4. Параметры

| Параметр | Значение | Почему |
|----------|----------|--------|
| N_a | 1024 | 2^10, degree of 2 для BMSSP, ~1K core якорей |
| l_c | 256 | ~половина кода на identity |
| l_a | 128 | 48+48+32 для 3 уровней |
| l_m | 128 | достаточно для модуляции 1024 gates |
| V | 32000 | BPE словарь, конечный |
| H спарсенность | ~90% | только сильные PMI-связи |

## 5. Вариативность

Количество возможных масок = 2^N_a = 2^1024 ≈ 10^308.
Физически храним только V × N_a бит = 4MB.
Вариативность генерации = комбинаторика полей, а не параметров.

## 6. Сложность BMSSP

- Уровень 0 (предложение): O(|E_0| + |V_0| log |V_0|) — граф PMI-связей токенов
- Уровень 1 (блок): 2 × Уровень 0 + O(N_a)
- Уровень 2 (документ): 4 × Уровень 1 + O(N_a²)
- Итого: O(|E| log |V| + N_a²) — без N² трансформера
- |E| ≈ 10^6 (PMI-рёбра), N_a² ≈ 10^6
- На порядок быстрее attention матрицы 32K×32K
