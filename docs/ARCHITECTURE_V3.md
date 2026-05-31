# EVA Symbolic — Расширенная архитектура (v3)

## 0. Общая картина

```
Текст ──→ Координатная траектория [L×128] ──→ Знание (core)
                                                     ↓
                              Шум/вариативность траектории
                              ├── Концепции (что МОЖЕТ БЫТЬ важным)
                              └── Противоречия (что НЕ МОЖЕТ БЫТЬ точным)
                                                     ↓
                        Мыслительный процесс = итеративная конвергенция
                        (поиск концепций + разрешение противоречий → схождение)
```

**Ключевая идея (дополнение):** 
Траектория = сигнал (знание) + шум (концепции ⊕ противоречия). 
- T1 (Navigator) учится воспроизводить сигнал — «знание»
- ConceptHead учится извлекать концепты из residuals (траектория − предсказание T1)
- ContradictionHead учится детектировать противоречия из residuals
- Thought process = итеративное уменьшение residuals до конвергенции

---

## 1. Архитектура компонентов

### 1.1. Shared Encoder (существующий, доработка)

**UnifiedMultidimensionalTransformer** — 6 слоёв, 128dim. Без изменений архитектуры.
Доработка: добавить возможность forward_with_latent(z) для KCA-цикла.

```python
# Новый метод (необходим для KCA):
def forward_with_latent(self, z, token_ids=None):
    """
    z: [D] — латентный код (координата)
    token_ids: [L] — опциональный контекст
    Возвращает: (logits [V], c_out [D])
    Применяется, когда KCA оптимизирует z по градиенту SRG.
    """
```

### 1.2. Heads на Shared Encoder (все НОВЫЕ)

#### Head 1: TrajectoryBoundaryPredictor
```python
class TrajectoryBoundaryPredictor(nn.Module):
    """h → (delta_end, delta_next, conn_vector) — все ℝ¹²⁸"""
    def __init__(self, d_model=128):
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256), nn.SiLU(),
            nn.Linear(256, 128*3)
        )
    def forward(self, h):
        out = self.mlp(h)  # [B, L, 384]
        return out[..., :128], out[..., 128:256], out[..., 256:]
```

#### Head 2: BoundaryValidator
```python
class BoundaryValidator(nn.Module):
    """h + z_current → softmax(word_boundary, sentence_boundary)"""
    def __init__(self, d_model=128):
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, 64), nn.SiLU(),
            nn.Linear(64, 2), nn.Softmax(dim=-1)
        )
    def forward(self, h, z_current):
        inp = torch.cat([h, z_current], dim=-1)
        return self.mlp(inp)
```

#### Head 3: ConceptHead
```python
class ConceptHead(nn.Module):
    """h → concept_probability [0,1] — training target от ConceptScorer"""
    def __init__(self, d_model=128):
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, h):
        return self.mlp(h).squeeze(-1)
```

#### Head 4: ContradictionHead
```python
class ContradictionHead(nn.Module):
    """h → contradiction_probability [0,1] — training target от ContradictionScorer"""
    def __init__(self, d_model=128):
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, h):
        return self.mlp(h).squeeze(-1)
```

#### Head 5: MetaWeighter
```python
class MetaWeighter(nn.Module):
    """context_hidden → [w_gen, w_know, w_conc, w_contr] — softmax с bias на знания"""
    def __init__(self, d_model=128, warmup_steps=1000):
        self.proj = nn.Linear(d_model, 64)
        self.weight_net = nn.Linear(64, 4)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        self.warmup_steps = warmup_steps
    
    def forward(self, context_hidden):
        # bias: знания (индекс 1) в приоритете
        bias = torch.tensor([0.0, 1.0, 0.0, 0.0], device=h.device) * self.temperature
        h = torch.silu(self.proj(context_hidden))
        return torch.softmax(self.weight_net(h) + bias, dim=-1)
```

### 1.3. Standalone модули (существующие, без изменений)

| Модуль | Статус | Файл |
|--------|--------|------|
| TensorPotentialField | ✅ готов | potential_fields.py |
| RecursiveTensorPotentialField | ✅ готов | potential_fields.py |
| WordValenceField | ✅ готов | potential_fields.py |
| SemanticRelevanceGate | ✅ готов | potential_fields.py |
| KCACycle | ✅ готов | potential_fields.py |
| GradientFlowSolver | ✅ готов | potential_fields.py |
| TrajectoryStore | ✅ готов | trajectory_store.py |
| SelfReflection | ✅ готов | self_reflection.py |
| ConceptScorer | ✅ готов (новый) | concept_miner.py |
| ContradictionScorer | ✅ готов (новый) | contradiction_filter.py |

### 1.4. H2K Pipeline (НОВЫЙ, целиком)

```python
class HypothesisBuffer:
    """Буфер на 100 гипотез (траекторий до подтверждения)."""
    buffer: deque(maxlen=100)
    def add(trajectory, srg_conf, message_id, attn_weights, word_coords, symbol_ids)
    def find(message_id) -> item
    def remove(message_id)

class HypothesisValidator:
    """Валидация гипотезы → превращение в знание через EWC."""
    def __init__(self, srg, trajectory_store, tensor_potential, word_valence)
    def update_fisher(self) — накопление Fisher Information Matrix
    def validate(hypothesis, srg_conf, user_feedback=None) -> bool
      - combined = SRG + user_feedback (взвешенно)
      - если combined > threshold:
        → сохранить в TrajectoryStore
        → обновить TPF.P с EWC-регуляризацией
        → обновить WordValenceField
        → вернуть True
```

---

## 2. Пайплайн генерации (Section 6 спецификации, дополненный)

```python
def generate_text(model, prompt, max_len=128, temperature=0.8):
    model.eval()
    with torch.no_grad():
        # 1. Кодируем промпт
        input_ids, coord_stream = model.encode_prompt(prompt)
        generated = list(input_ids[0].numpy())
        context_hidden = coord_stream.mean(dim=1)
        
        # 2. MetaWeighter: адаптивные веса
        w = model.meta_weighter(context_hidden)
        w_gen, w_know, w_conc, w_contr = w[0].tolist()

        # 3. Thought Loop (ДОПОЛНЕНИЕ: итеративная конвергенция)
        for thought_step in range(max_thought_steps):
            for _ in range(max_len):
                h, coord_stream = model.transformer(input_ids, coord_stream)
                
                # 3a. BoundaryPredictor
                end, nxt, conn = model.boundary_predictor(h[:, -1, :])
                
                # 3b. GradientFlowSolver (внутри слова)
                z_current = coord_stream[:, -1, :]
                for t in range(10):
                    z_current, converged = model.flow_solver.step(z_current, t)
                    if converged: break
                
                # 3c. Предсказание следующей координаты
                delta = model.traj_predictor(h[:, -1, :])
                z_pred = z_current + delta
                
                # 3d. Потенциалы → logits
                last_sym = generated[-1]
                bias_tpf = model.tensor_potential.get_bias(last_sym, input_ids[0])
                bias_wvf = model.word_valence.get_valence_bias(
                    coord_stream.mean(dim=1), input_ids[0])
                logits_pot = potential_guided_logits(
                    z_pred, model.sym_coords, bias_tpf, bias_wvf)
                
                # 3e. Concept + Contradiction
                prob_conc = model.concept_head(h[:, -1, :]).item()
                level_contra = model.contradiction_head(h[:, -1, :]).item()
                bias_conc = bias_tpf * prob_conc
                penalty_contra = bias_tpf * level_contra
                
                # 3f. Финальные logits (4 источника)
                logits_gen = model.coord_decoder(h[:, -1, :])
                final_logits = (
                    w_gen * logits_gen 
                    + w_know * logits_pot 
                    + w_conc * bias_conc 
                    - w_contr * penalty_contra
                )
                
                # 3g. Семплинг
                probs = F.softmax(final_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                if next_token == model.eos_id: break
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            
            # 4. SRG + KCA (самооценка после каждой мысли)
            # ДОПОЛНЕНИЕ: concept/contradiction heads влияют на KCA
            srg_conf = model.srg.evaluate(c_query, c_response, logits)
            if srg_conf < threshold or prob_conc > 0.7 or level_contra > 0.7:
                # Запускаем KCA для уточнения
                z_opt = model.kca.optimize(z_current, c_query)
                # Повторяем генерацию с уточнённого z
                continue  # следующий thought_step
            
            # 5. H2K (сохраняем гипотезу)
            model.hypothesis_buffer.add(...)
            break  # конвергенция достигнута
    
    # 6. Финализация: H2K валидация
    for hyp in model.hypothesis_buffer:
        model.hypothesis_validator.validate(hyp, srg_conf)
    
    return model.decode(generated)
```

---

## 3. Training Data Pipeline (ДОПОЛНЕНИЕ — чего нет в спецификации)

### 3.1. Генерация labels для ConceptHead + ContradictionHead

```python
def generate_training_labels(ids, trajectory, word_boundaries, word_centroids, 
                               sentence_centroid, cv, store):
    # Concept labels (от ConceptScorer)
    concept_labels = concept_scorer.score_trajectory(
        word_centroids, connection_coords, sentence_centroid)
    
    # Contradiction labels (от ContradictionScorer)
    contra_labels = contra_scorer.score_trajectory(
        trajectory, word_boundaries, sentence_centroid, word_centroids, ids)
    
    # Boundary labels (из разметки)
    boundary_labels = make_boundary_labels(ids, word_boundaries, cv)
    
    # Delta labels (из trajectory difference)
    delta_labels = trajectory[1:] - trajectory[:-1]
    
    return {
        'concept': concept_labels,       # [num_words] — sparse float
        'contradiction': contra_labels,  # [L] — sparse float
        'boundary': boundary_labels,     # [L, 2] — word/sentence boundary
        'delta': delta_labels,           # [L-1, 128] — coordinate deltas
    }
```

### 3.2. Multi-task loss

```python
loss = (
    λ_traj * MSE(delta_pred, delta_true)          # TrajectoryBoundaryPredictor
    + λ_bound * BCE(boundary_pred, boundary_true)   # BoundaryValidator
    + λ_conc * BCE(concept_pred, concept_true)      # ConceptHead (sparse!)
    + λ_contra * BCE(contra_pred, contra_true)      # ContradictionHead (sparse!)
    + λ_ce * CE(decoder_logits, next_token)         # CoordinateDecoder (aux)
    + λ_comp * composition_loss(hidden)             # RecursiveTPF (existing)
)
```

**Ключевое дополнение:** ConceptHead и ContradictionHead учим ТОЛЬКО на позициях, где есть событие (concept > 0.6 или contradiction > 0.5). Остальные позиции маскируем — иначе heads схлопнутся в 0.

---

## 4. Итеративный мыслительный процесс (НОВОЕ — расширение идеи)

То, что пользователь назвал "мыслительный процесс — это непрерывность поиска концепций, противоречий, схождения данных", реализуется как:

```
Уровень 0: Авторегрессия (1 forward pass / token)
Уровень 1: Thought Loop (N итераций генерации с SRG-оценкой)
Уровень 2: KCA Cycle (M шагов градиентного спуска по z)
Уровень 3: H2K Consolidation (сохранение подтверждённых траекторий)
```

Каждый уровень — грубее/медленнее предыдущего. Система останавливается, когда:
1. SRG > threshold (уверенность)
2. ConceptHead < threshold (нет новых концептов)
3. ContradictionHead < threshold (нет противоречий)
4. ‖z_new − z_old‖ < ε (конвергенция)

---

## 5. План реализации (приоритет)

| Этап | Что делать | Новые файлы | Зависит от |
|------|-----------|-------------|-----------|
| **0** | Генерация training data (раздел 3) | `train_data_pipeline.py` | ConceptScorer ✅, ContradictionScorer ✅ |
| **1** | SRG + KCA (интеграция в генерацию) | — (дописать unified_transformer.py) | существующий SRG + KCA |
| **2** | 5 heads: BoundaryPredictor, BoundaryValidator, ConceptHead, ContradictionHead, MetaWeighter | `heads.py` | — |
| **3** | potential_guided_logits + новый генерационный цикл (раздел 2) | — (дописать enhanced_generate) | Этап 2 |
| **4** | H2K Pipeline (HypothesisBuffer + HypothesisValidator + EWC) | `h2k_pipeline.py` | Этап 1 |
| **5** | Multi-task training loop | `train_v3.py` | Всё выше |
| **6** | Thought Loop (итеративная конвергенция, раздел 4) | — (дописать генерацию) | Этап 5 |

---

## 6. Что упущено в спецификации (и добавлено)

1. **Residual learning:** ConceptHead и ContradictionHead учатся на residuals траектории, не на raw hidden states
2. **Sparse training:** heads должны учиться только на значимых позициях, иначе class imbalance убивает
3. **Thought Loop:** генерация не 1 pass, а N итераций с SRG-остановкой
4. **Training data pipeline:** полный конвейер ConceptScorer + ContradictionScorer → train labels
5. **Multi-task loss:** единая функция потерь для всех heads + существующих компонентов
6. **Единый shared encoder** с 6 heads вместо трёх отдельных трансформеров
7. **CoordinateDecoder остаётся**, но его вес регулируется MetaWeighter (bias на знания)
