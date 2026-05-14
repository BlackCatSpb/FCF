"""
StateGrammar — расширенная детализация всех механизмов.

Добавляет к каждому из 11 механизмов углублённую математику и частности:
1.  Valence: полярность, асимметрия, шкалирование Тверски
2.  Temporal: skip-connections, forget gate, distant dependency attention
3.  Negation: скопальное отрицание, excluded middle, контрарность vs контрадикторность
4.  Superposition: декогеренция, интерференция, базис измерения
5.  Validator: типо-зависимые правила, explainability, контрпримеры
6.  Inheritance: множественное наследование, default reasoning, прототипы
7.  Emergence: авто-именование, проверка обобщения, консолидация
8.  Distance: взвешенные рёбра, альтернативные пути, статистика расстояний
9.  Conservation: фазовые переходы, Noether-подобная симметрия, breaking detection
10. SelfReference: парадокс Рассела, well-foundedness, стратификация
11. Entropy: KL-дивергенция, mutual information, entropy rate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math, random, time
from loguru import logger


class StateType(Enum):
    SYMBOL = "symbol"; CONCEPT = "concept"; RELATION = "relation"
    CONTEXT = "context"; COMPOSITE = "composite"; EMERGENT = "emergent"
    NEGATED = "negated"; SUPERPOSED = "superposed"


class TransformRule(Enum):
    COMPOSE = "compose"; SPECIFY = "specify"; GENERALIZE = "generalize"
    ANALOGIZE = "analogize"; CONTRAST = "contrast"; SEQUENCE = "sequence"
    NEGATE = "negate"; COLLAPSE = "collapse"; INHERIT = "inherit"


# ============================================================
# 1. VALENCE — полярность, асимметрия, шкала Тверски
# ============================================================

class Polarity(Enum):
    POSITIVE = 1; NEUTRAL = 0; NEGATIVE = -1


@dataclass
class ValenceProfile:
    alpha: float; polarity: Polarity = Polarity.NEUTRAL
    confidence: float = 1.0; asymmetry: float = 0.0

class StateValenceV2(nn.Module):
    """Валентность v2: полярность ±, асимметрия, шкала Тверски Tversky(A,B) = |A∩B|/(|A∩B|+α|A\B|+β|B\A|)."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(dim + 2, 128), nn.SiLU(), nn.Linear(128, 3))
        self.polarity_net = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Tanh())

    def profile(self, z: torch.Tensor, intensity: float = 0.5) -> ValenceProfile:
        inp = torch.cat([z, torch.full((z.shape[0], 2), intensity, device=z.device)], dim=-1)
        raw = self.scorer(inp)
        alpha = torch.sigmoid(raw[:, 0]).mean().item()
        asymmetry = torch.sigmoid(raw[:, 1]).mean().item()
        confidence = torch.sigmoid(raw[:, 2]).mean().item()
        pol_raw = self.polarity_net(z).mean().item()
        polarity = Polarity.POSITIVE if pol_raw > 0.33 else Polarity.NEGATIVE if pol_raw < -0.33 else Polarity.NEUTRAL
        return ValenceProfile(alpha=alpha, polarity=polarity, confidence=confidence, asymmetry=asymmetry)

    def tversky_similarity(self, z_a: torch.Tensor, z_b: torch.Tensor, alpha: float = 0.5, beta: float = 0.5) -> float:
        a, b = z_a.flatten(), z_b.flatten()
        intersection = torch.sum(torch.min(a.abs(), b.abs()))
        a_not_b = torch.sum(F.relu(a.abs() - b.abs()))
        b_not_a = torch.sum(F.relu(b.abs() - a.abs()))
        return float(intersection / (intersection + alpha * a_not_b + beta * b_not_a + 1e-8))

    def modulate(self, z_np: np.ndarray, alpha: float, polarity: Polarity = Polarity.NEUTRAL) -> np.ndarray:
        z = z_np * np.clip(alpha, 0.0, 1.0)
        if polarity == Polarity.NEGATIVE:
            z = z * 0.5 + np.random.randn(*z.shape).astype(np.float32) * 0.01
        return z


# ============================================================
# 2. TEMPORAL — skip-connections, forget gate, attention over history
# ============================================================

class TemporalChainV2(nn.Module):
    """Марковская цепь v2: skip-connections, forget gate, sparse attention over history."""

    def __init__(self, dim: int = 2560, num_heads: int = 8, max_len: int = 64):
        super().__init__(); self.dim = dim; self.max_len = max_len
        self.forget_gate = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.skip_proj = nn.Linear(dim, dim, bias=False)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*2, batch_first=True, dropout=0.0), num_layers=2)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.distant_attn = nn.MultiheadAttention(dim, 4, batch_first=True)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        B, T, D = states.shape
        pos = self.pos_embed[:, :T, :].expand(B, -1, -1)
        x = states + pos
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=states.device)
        out = self.transformer(x, mask=mask)

        gate = self.forget_gate(states)
        skip = self.skip_proj(states) * gate + states * (1 - gate)
        out = out + skip * 0.1

        if T > 2:
            query = out[:, -1:, :]; keys = out[:, :-1, :]
            distant, _ = self.distant_attn(query, keys, keys)
            out[:, -1, :] = out[:, -1, :] + distant[:, 0, :] * 0.1

        return out

    def transition_entropy(self, states: torch.Tensor) -> float:
        out = self(states)
        sim = F.cosine_similarity(out[:, :-1], out[:, 1:], dim=-1)
        probs = torch.softmax(sim / 0.1, dim=-1)
        return float(-torch.sum(probs * torch.log(probs + 1e-10)) / np.log(len(probs)))

    def predict_next(self, history: List[np.ndarray]) -> np.ndarray:
        if len(history) > self.max_len: history = history[-self.max_len:]
        x = torch.from_numpy(np.stack(history)).float().unsqueeze(0)
        with torch.no_grad(): out = self(x)
        return out[0, -1, :].numpy()


# ============================================================
# 3. NEGATION — скопальное, excluded middle, контрарность
# ============================================================

class NegationAlgebraV2(nn.Module):
    """Отрицание v2: скопальное (под-аспекты), excluded middle, контрарность vs контрадикторность."""

    def __init__(self, dim: int = 2560, num_scopes: int = 8):
        super().__init__()
        self.dim = dim; self.scope_dim = dim // num_scopes
        self.z_universal = nn.Parameter(torch.randn(dim) * 0.02)
        self.scope_net = nn.Sequential(nn.Linear(dim * 2, 128), nn.SiLU(), nn.Linear(128, num_scopes), nn.Sigmoid())
        self.alpha_net = nn.Sequential(nn.Linear(dim * 2, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())

    def negate_full(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]; univ = self.z_universal.unsqueeze(0).expand(B, -1)
        alpha = self.alpha_net(torch.cat([z, univ], dim=-1))
        n = univ - alpha * z
        return n / (torch.norm(n, dim=-1, keepdim=True) + 1e-8)

    def negate_scopal(self, z: torch.Tensor, scope_indices: Optional[List[int]] = None) -> torch.Tensor:
        B = z.shape[0]
        if scope_indices is None:
            scopes = self.scope_net(torch.cat([z, self.z_universal.unsqueeze(0).expand(B, -1)], dim=-1))
        else:
            scopes = torch.zeros(B, 8, device=z.device)
            for idx in scope_indices: scopes[:, idx] = 1.0

        result = z.clone()
        for s in range(8):
            start, end = s * self.scope_dim, (s + 1) * self.scope_dim
            result[:, start:end] = self.z_universal[start:end] - scopes[:, s:s+1] * z[:, start:end]
        return result / (torch.norm(result, dim=-1, keepdim=True) + 1e-8)

    def excluded_middle_satisfaction(self, z: torch.Tensor) -> float:
        """Насколько близко z ∨ ¬z к универсальному состоянию."""
        with torch.no_grad():
            nz = self.negate_full(z)
            middle = (z + nz) / 2.0
            return float(F.cosine_similarity(middle, self.z_universal.unsqueeze(0)).item())

    def is_contrary(self, z_a: torch.Tensor, z_b: torch.Tensor, threshold: float = -0.3) -> bool:
        """Контрарность: не могут быть истинны вместе."""
        return float(F.cosine_similarity(z_a, z_b).item()) < threshold

    def is_contradictory(self, z_a: torch.Tensor, z_b: torch.Tensor) -> bool:
        """Контрадикторность: одно = ¬другое."""
        n_a = self.negate_full(z_a)
        return float(F.cosine_similarity(n_a, z_b).item()) > 0.8


# ============================================================
# 4. SUPERPOSITION — декогеренция, интерференция, базис измерения
# ============================================================

class SuperpositionCollapseV2(nn.Module):
    """Суперпозиция v2: декогеренция, интерференция, ортогональный базис измерения."""

    def __init__(self, dim: int = 2560, max_components: int = 8):
        super().__init__(); self.dim = dim; self.max_components = max_components
        self.measure_basis = nn.Parameter(torch.randn(dim // 64, dim) * 0.02)
        self.measure_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.decoherence_rate = nn.Parameter(torch.tensor(0.01))

    def superpose(self, components: List[np.ndarray], amplitudes: Optional[List[float]] = None) -> np.ndarray:
        if amplitudes is None: amplitudes = [1.0 / len(components)] * len(components)
        total = sum(a * c for a, c in zip(amplitudes, components))
        return total / (np.linalg.norm(total) + 1e-8)

    def collapse(self, z_superposed: np.ndarray, z_context: np.ndarray) -> np.ndarray:
        zs = torch.from_numpy(z_superposed).float().unsqueeze(0)
        zc = torch.from_numpy(z_context).float().unsqueeze(0)
        with torch.no_grad(): return self.measure_net(torch.cat([zs, zc], dim=-1)).squeeze(0).numpy()

    def decohere(self, z_superposed: np.ndarray, time_steps: int = 10) -> np.ndarray:
        """Естественная декогеренция без измерения."""
        z = torch.from_numpy(z_superposed).float()
        rate = torch.sigmoid(self.decoherence_rate)
        for _ in range(time_steps):
            noise = torch.randn_like(z) * 0.01
            z = (1 - rate) * z + rate * noise
            z = z / (torch.norm(z) + 1e-8)
        return z.numpy()

    def interference_pattern(self, components: List[np.ndarray]) -> np.ndarray:
        n = len(components); pattern = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    pattern[i, j] = float(np.dot(components[i].flatten(), components[j].flatten()) /
                                          (np.linalg.norm(components[i]) * np.linalg.norm(components[j]) + 1e-8))
        return pattern

    def collapse_uncertainty(self, z_superposed: torch.Tensor) -> float:
        z = z_superposed.flatten()
        probs = torch.softmax(z[:100] / 0.1, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return float(entropy / np.log(100))


# ============================================================
# 5. VALIDATOR — типо-зависимые правила, explainability
# ============================================================

class CompositionalValidatorV2(nn.Module):
    """Валидатор v2: типо-зависимые правила, explainability, контрпримерное обучение."""

    def __init__(self, dim: int = 2560, num_types: int = 5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 3, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 2))
        self.explainer = nn.Sequential(nn.Linear(dim * 3, 128), nn.SiLU(), nn.Linear(128, 64))
        self.type_weights = nn.Parameter(torch.ones(num_types, 2))

    def validate(self, z_a: np.ndarray, z_b: np.ndarray, z_context: np.ndarray,
                 type_a: int = 0, type_b: int = 0) -> Tuple[float, float]:
        za = torch.from_numpy(z_a).float().unsqueeze(0); zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zc = torch.from_numpy(z_context).float().unsqueeze(0)
        with torch.no_grad():
            out = self.net(torch.cat([za, zb, zc], dim=-1))
            type_factor = self.type_weights[type_a] * self.type_weights[type_b]
            logits = out + type_factor.mean()
            probs = torch.softmax(logits, dim=-1)
        return float(probs[0, 0].item()), float(probs[0, 1].item())

    def explain(self, z_a: np.ndarray, z_b: np.ndarray, z_context: np.ndarray) -> np.ndarray:
        """Вектор объяснения: какие компоненты повлияли на решение."""
        za = torch.from_numpy(z_a).float().unsqueeze(0); zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zc = torch.from_numpy(z_context).float().unsqueeze(0)
        with torch.no_grad():
            return self.explainer(torch.cat([za, zb, zc], dim=-1)).squeeze(0).numpy()

    def generate_counterexample(self, z_a: np.ndarray, z_b: np.ndarray,
                                 z_context: np.ndarray, valid: bool) -> np.ndarray:
        """Генерирует контрпример: минимально изменённую версию с противоположной валидностью."""
        target = torch.tensor([0.0, 1.0]) if valid else torch.tensor([1.0, 0.0])
        za_t = torch.from_numpy(z_a).float().clone().requires_grad_(True)
        zb_t = torch.from_numpy(z_b).float().clone().detach()
        zc_t = torch.from_numpy(z_context).float().clone().detach()
        opt = torch.optim.Adam([za_t], lr=0.01)
        for _ in range(50):
            opt.zero_grad()
            out = self.net(torch.cat([za_t.unsqueeze(0), zb_t.unsqueeze(0), zc_t.unsqueeze(0)], dim=-1))
            loss = F.cross_entropy(out, target.unsqueeze(0))
            loss.backward(); opt.step()
        return za_t.detach().numpy()


# ============================================================
# 6. INHERITANCE — множественное, default reasoning, прототипы
# ============================================================

@dataclass
class InheritanceNodeV2:
    concept_id: str; state_vector: np.ndarray; parent_ids: List[str] = field(default_factory=list)
    inherited_rules: Set[TransformRule] = field(default_factory=set)
    overridden_rules: Dict[TransformRule, str] = field(default_factory=dict)
    default_values: Dict[str, np.ndarray] = field(default_factory=dict)
    prototype_vector: Optional[np.ndarray] = None
    instance_count: int = 0

class StateInheritanceGraphV2:
    """Множественное наследование, default reasoning, прототипы."""

    def __init__(self):
        self.nodes: Dict[str, InheritanceNodeV2] = {}
        self.children: Dict[str, List[str]] = defaultdict(list)

    def add(self, concept_id: str, vector: np.ndarray, parents: List[str] = None,
            defaults: Dict[str, np.ndarray] = None):
        node = InheritanceNodeV2(concept_id=concept_id, state_vector=vector.copy(),
                                  parent_ids=parents or [], default_values=defaults or {})
        self.nodes[concept_id] = node
        for p in (parents or []): self.children[p].append(concept_id)

    def resolve_multiple_inheritance(self, concept_id: str, attr: str) -> Optional[np.ndarray]:
        """C3-линеаризация для разрешения конфликтов множественного наследования."""
        if concept_id not in self.nodes: return None
        node = self.nodes[concept_id]
        if attr in node.default_values: return node.default_values[attr]
        linearization = self._c3_linearize(concept_id)
        for ancestor in linearization:
            if ancestor in self.nodes and attr in self.nodes[ancestor].default_values:
                return self.nodes[ancestor].default_values[attr]
        return None

    def _c3_linearize(self, concept_id: str) -> List[str]:
        if concept_id not in self.nodes: return [concept_id]
        node = self.nodes[concept_id]
        if not node.parent_ids: return [concept_id]
        parent_linearizations = [self._c3_linearize(p) for p in node.parent_ids]
        result = [concept_id]
        while any(parent_linearizations):
            found = False
            for pl in parent_linearizations:
                if not pl: continue
                candidate = pl[0]
                if all(candidate not in other[1:] for other in parent_linearizations):
                    result.append(candidate)
                    for pl2 in parent_linearizations:
                        if pl2 and pl2[0] == candidate: pl2.pop(0)
                    found = True; break
            if not found: break
        return result

    def compute_prototype(self, concept_id: str,
                          instance_vectors: List[np.ndarray]) -> np.ndarray:
        """Прототип категории — усреднение экземпляров."""
        if concept_id not in self.nodes: return np.zeros(2560, dtype=np.float32)
        if instance_vectors:
            proto = np.mean(instance_vectors, axis=0)
            self.nodes[concept_id].prototype_vector = proto / (np.linalg.norm(proto) + 1e-8)
            self.nodes[concept_id].instance_count = len(instance_vectors)
        return self.nodes[concept_id].prototype_vector or self.nodes[concept_id].state_vector

    def typicality(self, concept_id: str, instance: np.ndarray) -> float:
        """Насколько типичен экземпляр для категории."""
        proto = self.nodes.get(concept_id)
        if proto is None or proto.prototype_vector is None: return 0.5
        return float(np.dot(instance.flatten(), proto.prototype_vector.flatten()) /
                     (np.linalg.norm(instance) * np.linalg.norm(proto.prototype_vector) + 1e-8))


# ============================================================
# 7. EMERGENCE — авто-именование, обобщение, консолидация
# ============================================================

class EmergentGenesisV2:
    """Рождение концептов v2: авто-именование, проверка обобщения, консолидация."""

    def __init__(self, emergence_threshold: int = 10):
        self.emergence_threshold = emergence_threshold
        self.composition_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.composition_vectors: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
        self.emergent: Dict[str, EmergentConcept] = {}
        self._next_id = 0

    def record_composition(self, concept_a: str, concept_b: str, z_result: np.ndarray):
        key = (min(concept_a, concept_b), max(concept_a, concept_b))
        self.composition_counts[key] += 1
        self.composition_vectors[key].append(z_result.copy())
        if len(self.composition_vectors[key]) > 50: self.composition_vectors[key].pop(0)
        if self.composition_counts[key] >= self.emergence_threshold: self._try_emerge(key)

    def _try_emerge(self, key: Tuple[str, str]):
        vectors = self.composition_vectors[key]
        if len(vectors) < 5: return
        centroid = np.mean(vectors, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        variance = float(np.mean([np.linalg.norm(v - centroid) for v in vectors]))
        stability = 1.0 / (1.0 + variance)
        if stability > 0.5 and not self._already_exists(centroid):
            concept_id = f"emergent_{self._next_id}"; self._next_id += 1
            name = self._auto_name(key[0], key[1])
            self.emergent[concept_id] = EmergentConcept(
                concept_id=concept_id, state_vector=centroid,
                parent_a=key[0], parent_b=key[1],
                composition_count=self.composition_counts[key],
                stability=stability, first_seen=time.time(), last_seen=time.time())
            logger.info(f"[Emergence] {name} ({concept_id}): {key[0]}⊕{key[1]}, stability={stability:.3f}")

    def _auto_name(self, a: str, b: str) -> str:
        """Авто-именование: a_b или b_a в зависимости от порядка."""
        return f"{a}_{b}"

    def _already_exists(self, vector: np.ndarray) -> bool:
        for c in self.emergent.values():
            if np.dot(vector, c.state_vector) / (np.linalg.norm(vector)*np.linalg.norm(c.state_vector)+1e-8) > 0.85:
                return True
        return False

    def consolidate(self, similarity_threshold: float = 0.95):
        """Объединить похожие эмерджентные концепты."""
        ids = list(self.emergent.keys()); merged = 0
        for i in range(len(ids)):
            if ids[i] not in self.emergent: continue
            for j in range(i + 1, len(ids)):
                if ids[j] not in self.emergent: continue
                ci, cj = self.emergent[ids[i]], self.emergent[ids[j]]
                sim = np.dot(ci.state_vector, cj.state_vector) / (np.linalg.norm(ci.state_vector)*np.linalg.norm(cj.state_vector)+1e-8)
                if sim > similarity_threshold:
                    total = ci.composition_count + cj.composition_count
                    ci.state_vector = (ci.composition_count*ci.state_vector + cj.composition_count*cj.state_vector) / total
                    ci.state_vector /= np.linalg.norm(ci.state_vector) + 1e-8
                    ci.composition_count = total
                    del self.emergent[ids[j]]; merged += 1
        if merged: logger.info(f"[Emergence] Consolidated {merged} similar concepts")

    def check_generalization(self, emergent_id: str,
                             existing_concepts: Dict[str, np.ndarray]) -> Optional[str]:
        """Проверяет: не является ли эмерджентный концепт обобщением существующего."""
        if emergent_id not in self.emergent: return None
        ev = self.emergent[emergent_id].state_vector
        for cid, cv in existing_concepts.items():
            sim = np.dot(ev, cv) / (np.linalg.norm(ev)*np.linalg.norm(cv)+1e-8)
            if sim > 0.9: return cid
        return None


# ============================================================
# 8. DISTANCE — взвешенные рёбра, альтернативные пути, статистика
# ============================================================

class TransformDistanceV2:
    """Метрика v2: взвешенные рёбра по надёжности правила, альтернативные пути, распределение."""

    def __init__(self):
        self.graph: Dict[int, Dict[int, List[Tuple[float, TransformRule]]]] = defaultdict(lambda: defaultdict(list))
        self.rule_reliability: Dict[TransformRule, float] = defaultdict(lambda: 1.0)
        self.usage_count: Dict[TransformRule, int] = defaultdict(int)

    def add_transform(self, from_id: int, to_id: int, rule: TransformRule, success: bool = True):
        base_cost = {TransformRule.COMPOSE:1.0, TransformRule.SPECIFY:0.5, TransformRule.GENERALIZE:0.5,
                     TransformRule.ANALOGIZE:2.0, TransformRule.CONTRAST:2.0, TransformRule.SEQUENCE:1.5,
                     TransformRule.NEGATE:3.0}.get(rule, 1.0)
        self.usage_count[rule] += 1
        if not success: self.rule_reliability[rule] = self.rule_reliability[rule]*0.9 + 0.1*0.0
        else: self.rule_reliability[rule] = self.rule_reliability[rule]*0.99 + 0.01*1.0
        cost = base_cost / (self.rule_reliability[rule] + 0.1)
        self.graph[from_id][to_id].append((cost, rule))

    def best_distance(self, from_id: int, to_id: int) -> Tuple[float, List[TransformRule]]:
        """Дейкстра с учётом надёжности правил."""
        import heapq
        if from_id == to_id: return 0.0, []
        dist = {from_id: 0.0}; path_rules = {from_id: []}
        pq = [(0.0, from_id)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')): continue
            if u == to_id: return d, path_rules[u]
            for v, edges in self.graph.get(u, {}).items():
                for cost, rule in edges:
                    nd = d + cost
                    if nd < dist.get(v, float('inf')):
                        dist[v] = nd; path_rules[v] = path_rules[u] + [rule]
                        heapq.heappush(pq, (nd, v))
        return float('inf'), []

    def distance_distribution(self, from_id: int) -> Dict[str, float]:
        """Распределение расстояний до всех достижимых узлов."""
        dists = {}
        for to_id in self.graph:
            if to_id == from_id: continue
            d, rules = self.best_distance(from_id, to_id)
            if d != float('inf'): dists[str(to_id)] = d
        if not dists: return {"mean": 0, "std": 0, "max": 0}
        vals = list(dists.values())
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "max": float(np.max(vals)), "count": len(vals)}


# ============================================================
# 9. CONSERVATION — фазовые переходы, Noether-подобная симметрия
# ============================================================

class ConservationLawsV2:
    """Законы сохранения v2: фазовые переходы, симметрия → сохранение, breaking detection."""

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon
        self.invariants: Dict[TransformRule, Dict] = {}
        self.phase_history: Dict[TransformRule, List[float]] = defaultdict(list)

    def discover(self, rule: TransformRule, before: List[np.ndarray], after: List[np.ndarray]) -> np.ndarray:
        diffs = np.array([a - b for a, b in zip(after, before)])
        mean_diff = np.mean(np.abs(diffs), axis=0)
        mask = mean_diff < self.epsilon
        self.invariants[rule] = {"mask": mask, "count": int(np.sum(mask)), "fraction": float(np.mean(mask))}
        self.phase_history[rule].append(float(np.mean(mask)))
        return mask

    def detect_phase_transition(self, rule: TransformRule, window: int = 20) -> Optional[int]:
        """Обнаруживает резкое изменение числа инвариантов."""
        hist = self.phase_history.get(rule, [])
        if len(hist) < window * 2: return None
        before = np.mean(hist[-window*2:-window])
        after = np.mean(hist[-window:])
        if abs(after - before) > 0.1: return len(hist) - window
        return None

    def symmetry_to_conservation(self, rule: TransformRule, z: np.ndarray,
                                  transform: Callable) -> float:
        """Noether-подобная теорема: если transform(z) ≈ z при композиции, то что-то сохраняется."""
        z_t = transform(z.copy())
        sim = np.dot(z.flatten(), z_t.flatten()) / (np.linalg.norm(z)*np.linalg.norm(z_t)+1e-8)
        conserved = self.invariants.get(rule, {}).get("fraction", 0.0)
        return float(sim * conserved)


# ============================================================
# 10. SELF-REFERENCE — Рассел, well-foundedness, стратификация
# ============================================================

class SelfReferenceV2:
    """Самореферентность v2: парадокс Рассела, well-foundedness, стратификация."""

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6):
        self.max_iterations = max_iterations; self.tolerance = tolerance

    def find_fixed_point(self, f, z_init: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        z = z_init.copy()
        for i in range(self.max_iterations):
            z_new = f(z)
            if np.linalg.norm(z_new - z) < self.tolerance: return z_new, i, True
            z = z_new
        return z, self.max_iterations, False

    def self_reference_degree(self, z: np.ndarray, f) -> float:
        z_f = f(z)
        return float((np.dot(z.flatten(), z_f.flatten()) / (np.linalg.norm(z)*np.linalg.norm(z_f)+1e-8) + 1.0) / 2.0)

    def detect_russell_paradox(self, z_set: np.ndarray, membership_fn: Callable) -> bool:
        """Проверяет парадокс Рассела: содержит ли множество само себя."""
        z_member = membership_fn(z_set, z_set)
        z_not_member = 1.0 - z_member
        return abs(z_member - z_not_member) < 0.1

    def is_well_founded(self, z: np.ndarray, f, max_depth: int = 20) -> bool:
        """Проверяет well-foundedness: нет бесконечно убывающей цепочки z → f(z) → f(f(z))..."""
        visited = set(); current = z.tobytes() if hasattr(z, 'tobytes') else hash(z.tostring())
        for _ in range(max_depth):
            z = f(z)
            key = z.tobytes() if hasattr(z, 'tobytes') else hash(z.tostring())
            if key in visited: return False
            visited.add(key)
        return True

    def stratification_level(self, z: np.ndarray, f,
                             max_levels: int = 10) -> int:
        """Определяет стратификационный уровень состояния."""
        level = 0; current = z.copy()
        for _ in range(max_levels):
            next_z = f(current)
            diff = np.linalg.norm(next_z - current)
            if diff < self.tolerance: return level
            current = next_z; level += 1
        return max_levels


# ============================================================
# 11. ENTROPY — KL, mutual information, entropy rate
# ============================================================

class InformationEntropyV2:
    """Информационная энтропия v2: KL-дивергенция, mutual information, entropy rate."""

    def __init__(self, dim: int = 2560, num_bins: int = 50):
        self.dim = dim; self.num_bins = num_bins
        self.global_dist: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._samples: List[np.ndarray] = []

    def update_global(self, vectors: List[np.ndarray]):
        self._samples.extend(vectors)
        if len(self._samples) > 10000: self._samples = self._samples[-10000:]
        if len(self._samples) >= 100:
            all_vecs = np.stack(self._samples)
            self.global_dist = (np.mean(all_vecs, axis=0), np.std(all_vecs, axis=0) + 1e-8)

    def estimate_entropy(self, z: np.ndarray) -> float:
        z = z.flatten(); probs = np.abs(z) / (np.sum(np.abs(z)) + 1e-10)
        probs = np.clip(probs, 1e-10, 1.0); probs = probs / np.sum(probs)
        return float(-np.sum(probs * np.log2(probs)))

    def composition_surprise(self, z_a: np.ndarray, z_b: np.ndarray, z_result: np.ndarray) -> float:
        return self.estimate_entropy(z_a) + self.estimate_entropy(z_b) - self.estimate_entropy(z_result)

    def kl_divergence(self, z_p: np.ndarray, z_q: np.ndarray) -> float:
        """D_KL(P || Q) между двумя распределениями состояний."""
        p = np.abs(z_p.flatten()); p = np.clip(p, 1e-10, 1.0); p = p / np.sum(p)
        q = np.abs(z_q.flatten()); q = np.clip(q, 1e-10, 1.0); q = q / np.sum(q)
        return float(np.sum(p * np.log2(p / q)))

    def mutual_information(self, z_a: np.ndarray, z_b: np.ndarray) -> float:
        """I(A; B) = H(A) + H(B) - H(A, B)."""
        h_a = self.estimate_entropy(z_a); h_b = self.estimate_entropy(z_b)
        joint = np.concatenate([z_a.flatten(), z_b.flatten()])
        h_joint = self.estimate_entropy(joint)
        return h_a + h_b - h_joint

    def entropy_rate(self, sequence: List[np.ndarray]) -> float:
        """Энтропия на шаг для последовательности состояний."""
        if len(sequence) < 2: return 0.0
        rates = []
        for i in range(len(sequence) - 1):
            joint = np.concatenate([sequence[i].flatten(), sequence[i+1].flatten()])
            mi = self.mutual_information(sequence[i], sequence[i+1])
            h_cond = self.estimate_entropy(sequence[i+1]) - mi
            rates.append(h_cond)
        return float(np.mean(rates))


# ============================================================
# ОБЪЕДИНЯЮЩИЙ КЛАСС V2
# ============================================================

@dataclass
class EmergentConcept:
    concept_id: str; state_vector: np.ndarray
    parent_a: str; parent_b: str; composition_count: int = 1
    stability: float = 0.0; first_seen: float = 0.0; last_seen: float = 0.0


class StateGrammar:
    """Полная грамматика состояний — 11 механизмов v2 с углублённой математикой."""

    def __init__(self, dim: int = 2560):
        self.dim = dim
        self.valence = StateValenceV2(dim)
        self.temporal = TemporalChainV2(dim)
        self.negation = NegationAlgebraV2(dim)
        self.superposition = SuperpositionCollapseV2(dim)
        self.validator = CompositionalValidatorV2(dim)
        self.inheritance = StateInheritanceGraphV2()
        self.emergence = EmergentGenesisV2()
        self.distance = TransformDistanceV2()
        self.conservation = ConservationLawsV2()
        self.self_ref = SelfReferenceV2()
        self.info_entropy = InformationEntropyV2(dim)

    def compose(self, z_a: np.ndarray, z_b: np.ndarray, z_context: np.ndarray,
                alpha_a: float = 1.0, alpha_b: float = 1.0) -> Dict:
        za = torch.from_numpy(z_a).float().unsqueeze(0); zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zc = torch.from_numpy(z_context).float().unsqueeze(0)

        profile_a = self.valence.profile(za, alpha_a)
        profile_b = self.valence.profile(zb, alpha_b)

        za_mod = torch.from_numpy(self.valence.modulate(z_a, profile_a.alpha, profile_a.polarity)).float().unsqueeze(0)
        zb_mod = torch.from_numpy(self.valence.modulate(z_b, profile_b.alpha, profile_b.polarity)).float().unsqueeze(0)

        valid_score, invalid_score = self.validator.validate(z_a, z_b, z_context)
        validity = valid_score / (valid_score + invalid_score + 1e-8)

        z_ctx = ContextualComposerV2(self.dim)(za_mod, zb_mod, zc)
        result = z_ctx.squeeze(0).detach().numpy()

        delta_I = self.info_entropy.composition_surprise(z_a, z_b, result)
        collapsed = self.superposition.collapse(result, z_context)
        uncertainty = self.superposition.collapse_uncertainty(z_ctx)

        tversky = self.valence.tversky_similarity(za, zb)
        mutual_info = self.info_entropy.mutual_information(z_a, z_b)

        neg_a = self.negation.negate_full(za)
        is_contradictory = self.negation.is_contradictory(za, zb)
        em_sat = self.negation.excluded_middle_satisfaction(za)

        self_ref_deg = self.self_ref.self_reference_degree(z_a, lambda x: x * 0.99)
        is_paradox = self.self_ref.detect_russell_paradox(z_a, lambda a, b: float(F.cosine_similarity(
            torch.from_numpy(a).float(), torch.from_numpy(b).float(), dim=-1)))

        return {
            "result": result, "validity": round(validity, 3),
            "delta_I": round(delta_I, 3),
            "tversky_similarity": round(tversky, 3),
            "mutual_information": round(mutual_info, 3),
            "contradictory": is_contradictory,
            "excluded_middle": round(em_sat, 3),
            "self_ref_degree": round(self_ref_deg, 3),
            "russell_paradox": is_paradox,
            "uncertainty": uncertainty,
            "profile_a": (round(profile_a.alpha, 3), profile_a.polarity.name),
            "profile_b": (round(profile_b.alpha, 3), profile_b.polarity.name),
        }

    def analyze(self, state_sequence: List[np.ndarray], context: np.ndarray) -> Dict:
        if len(state_sequence) < 2: return {}
        stacked = torch.from_numpy(np.stack(state_sequence)).float().unsqueeze(0)
        transition = self.temporal(stacked)
        coherence = float(1.0 / (1.0 + torch.norm(transition[:, 1:] - stacked[:, 1:], dim=-1).mean()))
        entropy_rate = self.info_entropy.entropy_rate(state_sequence)
        next_pred = self.temporal.predict_next(state_sequence)
        transition_entropy = self.temporal.transition_entropy(stacked)
        well_founded = all(self.self_ref.is_well_founded(s, lambda x: x*0.99) for s in state_sequence[:5])
        return {"coherence": float(coherence), "entropy_rate": round(entropy_rate, 3),
                "transition_entropy": round(transition_entropy, 3),
                "next_prediction": next_pred, "well_founded": well_founded}

    def formalize(self) -> Dict[str, str]:
        desc = {}
        for rt in TransformRule:
            d, rules = self.distance.best_distance(0, 1) if rt == TransformRule.COMPOSE else (0, [])
            desc[rt.value] = f"reliability={self.distance.rule_reliability[rt]:.2f}, usage={self.distance.usage_count[rt]}"
        desc["emergent"] = f"{len(self.emergence.emergent)} concepts"
        desc["conservation"] = str({rt.value: round(self.conservation.invariants.get(rt, {}).get('fraction', 0), 3)
                                   for rt in TransformRule if rt in self.conservation.invariants})
        return desc


class ContextualComposerV2(nn.Module):
    def __init__(self, dim: int = 2560):
        super().__init__(); self.dim = dim
        self.encoder = nn.Sequential(nn.Linear(dim*3, dim), nn.SiLU(), nn.Linear(dim, dim//2), nn.SiLU(), nn.Linear(dim//2, dim))
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True); self.norm = nn.LayerNorm(dim)
    def forward(self, z_A, z_B, z_C):
        ctx = self.encoder(torch.cat([z_A, z_B, z_C], dim=-1))
        s = torch.stack([z_A, z_B, ctx], dim=1); a, _ = self.attn(s, s, s)
        return self.norm(a.mean(dim=1))
