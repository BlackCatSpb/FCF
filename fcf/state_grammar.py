"""
StateGrammar — полная грамматика состояний (11 механизмов).

Механизмы:
1.  StateValence — градуальная интенсивность состояний
2.  TemporalChain — марковская цепь переходов
3.  NegationAlgebra — осмысленное отрицание
4.  SuperpositionCollapse — квантово-подобная семантика
5.  CompositionalValidator — блокировка бессмысленных комбинаций
6.  StateInheritanceGraph — иерархия наследования правил
7.  EmergentGenesis — рождение новых концептов
8.  TransformDistance — метрика на основе правил трансформации
9.  ConservationLaws — инварианты композиции
10. SelfReference — неподвижные точки и самореферентность
11. InformationEntropy — мера интересности композиции

Математический фундамент: алгебраическая система S = (Z, ⊕, ⊗, ¬, M, V),
где Z — пространство состояний, ⊕ — композиция, ⊗ — контекстное произведение,
¬ — отрицание, M — метрика, V — валидатор.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math
from loguru import logger


# ======================== БАЗОВЫЕ ТИПЫ ========================

class StateType(Enum):
    SYMBOL = "symbol"
    CONCEPT = "concept"
    RELATION = "relation"
    CONTEXT = "context"
    COMPOSITE = "composite"
    EMERGENT = "emergent"
    NEGATED = "negated"
    SUPERPOSED = "superposed"


class TransformRule(Enum):
    COMPOSE = "compose"; SPECIFY = "specify"; GENERALIZE = "generalize"
    ANALOGIZE = "analogize"; CONTRAST = "contrast"; SEQUENCE = "sequence"
    NEGATE = "negate"; COLLAPSE = "collapse"; INHERIT = "inherit"


# ======================== 1. VALENCE ========================

class StateValence(nn.Module):
    """Градуальная интенсивность: z_effective = α · z, α ∈ [0,1]."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim + 1, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, z: torch.Tensor, intensity_hint: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        hint = torch.full((z.shape[0], 1), intensity_hint, device=z.device)
        alpha = self.scorer(torch.cat([z, hint], dim=-1))
        return z * alpha, alpha

    def modulate(self, z_np: np.ndarray, alpha: float) -> np.ndarray:
        return z_np * np.clip(alpha, 0.0, 1.0)


# ======================== 2. TEMPORAL DYNAMICS ========================

class TemporalChain(nn.Module):
    """Марковская цепь: P(z_t | z_{t-1}, ..., z_0) через causal transformer."""

    def __init__(self, dim: int = 2560, num_heads: int = 8, max_len: int = 64):
        super().__init__()
        self.dim = dim; self.max_len = max_len
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads,
                dim_feedforward=dim * 2, batch_first=True, dropout=0.0), num_layers=2)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        B, T, D = states.shape
        pos = self.pos_embed[:, :T, :].expand(B, -1, -1)
        x = states + pos
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=states.device)
        return self.transformer(x, mask=mask)

    def predict_next(self, history: List[np.ndarray]) -> np.ndarray:
        if len(history) > self.max_len:
            history = history[-self.max_len:]
        x = torch.from_numpy(np.stack(history)).float().unsqueeze(0)
        with torch.no_grad():
            out = self(x)
        return out[0, -1, :].numpy()

    def transition_matrix(self, states: torch.Tensor) -> torch.Tensor:
        """Матрица переходов P(z_i → z_j) для анализа причинности."""
        out = self(states)
        sim = F.cosine_similarity(out[:, :-1], out[:, 1:], dim=-1)
        n = sim.shape[0]
        P = torch.zeros(n, n, device=states.device)
        for i in range(min(n, len(sim))):
            if i < n:
                P[i, min(i + 1, n - 1)] = float(sim[i])
        return P / (P.sum(dim=-1, keepdim=True) + 1e-8)


# ======================== 3. NEGATION ALGEBRA ========================

class NegationAlgebra(nn.Module):
    """z_not(A) = Proj(z_universal - α · z_A), z_not(A) ≠ -z_A."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.z_universal = nn.Parameter(torch.randn(dim) * 0.02)
        self.alpha_net = nn.Sequential(
            nn.Linear(dim * 2, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())

    def negate(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        univ = self.z_universal.unsqueeze(0).expand(B, -1)
        alpha = self.alpha_net(torch.cat([z, univ], dim=-1))
        negated = univ - alpha * z
        return negated / (torch.norm(negated, dim=-1, keepdim=True) + 1e-8)

    def double_negation_test(self, z_np: np.ndarray) -> float:
        """Проверяет: z ≈ ¬(¬z) с некоторой потерей информации."""
        z = torch.from_numpy(z_np).float().unsqueeze(0)
        with torch.no_grad():
            not_z = self.negate(z)
            not_not_z = self.negate(not_z)
            return float(F.cosine_similarity(z, not_not_z).item())


# ======================== 4. SUPERPOSITION & COLLAPSE ========================

class SuperpositionCollapse(nn.Module):
    """Суперпозиция смыслов; контекст коллапсирует в конкретное значение."""

    def __init__(self, dim: int = 2560, max_components: int = 8):
        super().__init__()
        self.dim = dim; self.max_components = max_components
        self.measure_net = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))

    def superpose(self, components: List[np.ndarray],
                  amplitudes: Optional[List[float]] = None) -> np.ndarray:
        if amplitudes is None:
            amplitudes = [1.0 / len(components)] * len(components)
        total = sum(a * c for a, c in zip(amplitudes, components))
        return total / (np.linalg.norm(total) + 1e-8)

    def collapse(self, z_superposed: np.ndarray,
                 z_context: np.ndarray) -> np.ndarray:
        zs = torch.from_numpy(z_superposed).float().unsqueeze(0)
        zc = torch.from_numpy(z_context).float().unsqueeze(0)
        with torch.no_grad():
            collapsed = self.measure_net(torch.cat([zs, zc], dim=-1))
        return collapsed.squeeze(0).numpy()

    def collapse_uncertainty(self, z_superposed: torch.Tensor) -> float:
        """Энтропия суперпозиции до коллапса — мера неопределённости."""
        z = z_superposed.flatten()
        probs = torch.softmax(z[:100] / 0.1, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return float(entropy / np.log(100))


# ======================== 5. COMPOSITIONAL VALIDATOR ========================

class CompositionalValidator(nn.Module):
    """Классификатор: валидна ли композиция A ⊕ B в контексте C?"""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, 256), nn.SiLU(), nn.Linear(256, 128),
            nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())

    def validate(self, z_a: np.ndarray, z_b: np.ndarray,
                 z_context: np.ndarray) -> float:
        za = torch.from_numpy(z_a).float().unsqueeze(0)
        zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zc = torch.from_numpy(z_context).float().unsqueeze(0)
        with torch.no_grad():
            return float(self.net(torch.cat([za, zb, zc], dim=-1)).item())

    def train_on_examples(self, valid_pairs: List, invalid_pairs: List,
                           epochs: int = 100, lr: float = 1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        K = min(len(valid_pairs), len(invalid_pairs), 32)
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(K):
                optimizer.zero_grad()
                za, zb, zc = [torch.from_numpy(x).float().unsqueeze(0)
                              for x in valid_pairs[i]]
                pred_v = self.net(torch.cat([za, zb, zc], dim=-1))
                loss_v = F.binary_cross_entropy(pred_v, torch.ones_like(pred_v))

                za2, zb2, zc2 = [torch.from_numpy(x).float().unsqueeze(0)
                                 for x in invalid_pairs[i]]
                pred_i = self.net(torch.cat([za2, zb2, zc2], dim=-1))
                loss_i = F.binary_cross_entropy(pred_i, torch.zeros_like(pred_i))

                (loss_v + loss_i).backward(); optimizer.step()
                total_loss += loss_v.item() + loss_i.item()
            if epoch % 20 == 0:
                logger.debug(f"[Validator] epoch={epoch}, loss={total_loss:.4f}")


# ======================== 6. STATE INHERITANCE GRAPH ========================

@dataclass
class InheritanceNode:
    concept_id: str; state_vector: np.ndarray; parent_ids: List[str] = field(default_factory=list)
    inherited_rules: Set[TransformRule] = field(default_factory=set)
    overridden_rules: Dict[TransformRule, str] = field(default_factory=dict)


class StateInheritanceGraph:
    """DAG is_a: "рожь" → "злак" → "растение" → "живое"."""

    def __init__(self):
        self.nodes: Dict[str, InheritanceNode] = {}
        self.children: Dict[str, List[str]] = defaultdict(list)

    def add(self, concept_id: str, vector: np.ndarray, parents: List[str] = None):
        node = InheritanceNode(concept_id=concept_id, state_vector=vector.copy(),
                               parent_ids=parents or [])
        self.nodes[concept_id] = node
        for p in (parents or []):
            self.children[p].append(concept_id)

    def find_nearest_common_ancestor(self, id_a: str, id_b: str) -> Optional[str]:
        if id_a not in self.nodes or id_b not in self.nodes:
            return None
        ancestors_a = set(); queue_a = [id_a]
        while queue_a:
            n = queue_a.pop(0)
            if n in ancestors_a: continue
            ancestors_a.add(n)
            queue_a.extend(self.nodes[n].parent_ids)
        queue_b = [id_b]
        while queue_b:
            n = queue_b.pop(0)
            if n in ancestors_a:
                return n
            queue_b.extend(self.nodes[n].parent_ids)
        return None

    def inherit_rules(self, concept_id: str) -> Set[TransformRule]:
        """Подняться по дереву и собрать все унаследованные правила."""
        if concept_id not in self.nodes:
            return set()
        rules = set()
        visited = set()
        queue = [concept_id]
        while queue:
            n = queue.pop(0)
            if n in visited: continue
            visited.add(n)
            if n in self.nodes:
                node = self.nodes[n]
                rules.update(node.inherited_rules)
                for r in node.overridden_rules:
                    rules.discard(r)
                queue.extend(node.parent_ids)
        return rules


# ======================== 7. EMERGENT GENESIS ========================

@dataclass
class EmergentConcept:
    concept_id: str; state_vector: np.ndarray
    parent_a: str; parent_b: str; composition_count: int = 1
    stability: float = 0.0; first_seen: float = 0.0; last_seen: float = 0.0


class EmergentGenesis:
    """Рождение новых концептов из частых композиций."""

    def __init__(self, emergence_threshold: int = 10, similarity_threshold: float = 0.85):
        self.emergence_threshold = emergence_threshold
        self.similarity_threshold = similarity_threshold
        self.composition_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.composition_vectors: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
        self.emergent: Dict[str, EmergentConcept] = {}
        self._next_id = 0

    def record_composition(self, concept_a: str, concept_b: str,
                           z_result: np.ndarray):
        key = (min(concept_a, concept_b), max(concept_a, concept_b))
        self.composition_counts[key] += 1
        self.composition_vectors[key].append(z_result.copy())
        if len(self.composition_vectors[key]) > 50:
            self.composition_vectors[key].pop(0)

        if self.composition_counts[key] >= self.emergence_threshold:
            self._try_emerge(key)

    def _try_emerge(self, key: Tuple[str, str]):
        vectors = self.composition_vectors[key]
        if len(vectors) < 5:
            return
        centroid = np.mean(vectors, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        variance = float(np.mean([np.linalg.norm(v - centroid) for v in vectors]))
        stability = 1.0 / (1.0 + variance)

        if stability > 0.5 and not self._already_exists(centroid):
            concept_id = f"emergent_{self._next_id}"
            self._next_id += 1
            import time
            self.emergent[concept_id] = EmergentConcept(
                concept_id=concept_id, state_vector=centroid,
                parent_a=key[0], parent_b=key[1],
                composition_count=self.composition_counts[key],
                stability=stability, first_seen=time.time(), last_seen=time.time())
            logger.info(f"[Emergence] НОВЫЙ КОНЦЕПТ: {concept_id} "
                        f"({key[0]} ⊕ {key[1]}, stability={stability:.3f})")

    def _already_exists(self, vector: np.ndarray) -> bool:
        for concept in self.emergent.values():
            sim = np.dot(vector, concept.state_vector) / (
                np.linalg.norm(vector) * np.linalg.norm(concept.state_vector) + 1e-8)
            if sim > self.similarity_threshold:
                return True
        return False

    def reinforce(self, concept_id: str, z_new: np.ndarray):
        if concept_id in self.emergent:
            c = self.emergent[concept_id]
            alpha = 1.0 / (c.composition_count + 1)
            c.state_vector = (1 - alpha) * c.state_vector + alpha * z_new
            c.state_vector /= np.linalg.norm(c.state_vector) + 1e-8
            c.composition_count += 1
            import time
            c.last_seen = time.time()


# ======================== 8. TRANSFORM DISTANCE ========================

class TransformDistance:
    """Метрика: d(A, B) = минимальное число применений правил A → B."""

    def __init__(self):
        self.transform_graph: Dict[int, Dict[int, float]] = defaultdict(dict)

    def add_transform(self, from_id: int, to_id: int, rule: TransformRule):
        costs = {TransformRule.COMPOSE: 1.0, TransformRule.SPECIFY: 0.5,
                 TransformRule.GENERALIZE: 0.5, TransformRule.ANALOGIZE: 2.0,
                 TransformRule.CONTRAST: 2.0, TransformRule.SEQUENCE: 1.5,
                 TransformRule.NEGATE: 3.0}
        self.transform_graph[from_id][to_id] = min(
            self.transform_graph[from_id].get(to_id, float('inf')),
            costs.get(rule, 1.0))

    def distance(self, from_id: int, to_id: int) -> float:
        """Алгоритм Дейкстры по графу трансформаций."""
        if from_id == to_id: return 0.0
        if from_id not in self.transform_graph: return float('inf')
        dist = {from_id: 0.0}; visited = set()
        import heapq
        pq = [(0.0, from_id)]
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u)
            if u == to_id: return d
            for v, w in self.transform_graph.get(u, {}).items():
                if v not in visited:
                    nd = d + w
                    if nd < dist.get(v, float('inf')):
                        dist[v] = nd; heapq.heappush(pq, (nd, v))
        return float('inf')

    def similarity_from_distance(self, from_id: int, to_id: int,
                                  max_dist: float = 10.0) -> float:
        d = self.distance(from_id, to_id)
        if d == float('inf'): return 0.0
        return float(np.exp(-d / max_dist))


# ======================== 9. CONSERVATION LAWS ========================

class ConservationLaws:
    """Инварианты: какие измерения НЕ меняются при композиции."""

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon
        self.invariants: Dict[TransformRule, Dict[str, np.ndarray]] = {}

    def discover(self, rule: TransformRule, before_states: List[np.ndarray],
                 after_states: List[np.ndarray]) -> np.ndarray:
        """Находит инвариантные измерения для правила."""
        diffs = np.array([a - b for a, b in zip(after_states, before_states)])
        mean_diff = np.mean(np.abs(diffs), axis=0)
        invariant_mask = mean_diff < self.epsilon
        self.invariants[rule] = {
            "mask": invariant_mask,
            "num_invariant": int(np.sum(invariant_mask)),
            "fraction": float(np.mean(invariant_mask)),
        }
        return invariant_mask

    def is_conserved(self, rule: TransformRule, dim_idx: int) -> bool:
        if rule not in self.invariants:
            return False
        return bool(self.invariants[rule]["mask"][dim_idx])

    def conservation_ratio(self, rule: TransformRule) -> float:
        if rule not in self.invariants:
            return 0.0
        return self.invariants[rule]["fraction"]


# ======================== 10. SELF-REFERENCE ========================

class SelfReference:
    """Неподвижные точки z = f(z) и степень самореферентности."""

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6):
        self.max_iterations = max_iterations; self.tolerance = tolerance

    def find_fixed_point(self, f, z_init: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, int, bool]:
        """Ищет неподвижную точку: z* = f(z*)."""
        z = z_init.copy()
        for i in range(self.max_iterations):
            z_new = f(z)
            diff = np.linalg.norm(z_new - z)
            if verbose and i % 10 == 0:
                logger.debug(f"[FixedPoint] iter={i}, diff={diff:.6f}")
            if diff < self.tolerance:
                return z_new, i, True
            z = z_new
        return z, self.max_iterations, False

    def self_reference_degree(self, z: np.ndarray, f) -> float:
        """cos(z, f(z)) — насколько состояние ссылается на себя."""
        z_f = f(z)
        sim = np.dot(z.flatten(), z_f.flatten()) / (
            np.linalg.norm(z) * np.linalg.norm(z_f) + 1e-8)
        return float((sim + 1.0) / 2.0)

    def is_paradoxical(self, z: np.ndarray, f, threshold: float = 0.95) -> bool:
        """Парадокс: высокая самореферентность + расходимость итераций."""
        deg = self.self_reference_degree(z, f)
        if deg < threshold: return False
        z1 = f(z); z2 = f(z1)
        divergence = np.linalg.norm(z2 - z1) / (np.linalg.norm(z1 - z) + 1e-8)
        return divergence > 2.0 and deg > threshold


# ======================== 11. INFORMATION ENTROPY ========================

class InformationEntropy:
    """ΔI = H(A) + H(B) - H(A⊕B) — прирост информации при композиции."""

    def __init__(self, dim: int = 2560, num_bins: int = 50):
        self.dim = dim; self.num_bins = num_bins
        self.global_distribution: Optional[np.ndarray] = None
        self._samples: List[np.ndarray] = []

    def update_global(self, vectors: List[np.ndarray]):
        self._samples.extend(vectors)
        if len(self._samples) > 10000:
            self._samples = self._samples[-10000:]
        if len(self._samples) >= 100:
            all_vecs = np.stack(self._samples)
            means = np.mean(all_vecs, axis=0)
            stds = np.std(all_vecs, axis=0) + 1e-8
            self.global_distribution = (means, stds)

    def estimate_entropy(self, z: np.ndarray) -> float:
        """Оценка энтропии состояния через дисперсию компонент."""
        z = z.flatten()
        probs = np.abs(z) / (np.sum(np.abs(z)) + 1e-10)
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / np.sum(probs)
        return float(-np.sum(probs * np.log2(probs)))

    def composition_surprise(self, z_a: np.ndarray, z_b: np.ndarray,
                             z_result: np.ndarray) -> float:
        """ΔI = H(A) + H(B) - H(A⊕B). Большое ΔI = неожиданная композиция."""
        h_a = self.estimate_entropy(z_a)
        h_b = self.estimate_entropy(z_b)
        h_result = self.estimate_entropy(z_result)
        delta_I = h_a + h_b - h_result
        if self.global_distribution is not None:
            means, stds = self.global_distribution
            z_score = np.mean(np.abs(z_result - means) / stds)
            delta_I *= (1.0 + 0.1 * z_score)
        return float(delta_I)

    def is_interesting(self, delta_I: float, threshold: float = 2.0) -> bool:
        return delta_I > threshold

    def curiosity_drive(self, existing_compositions: List[Tuple[np.ndarray, np.ndarray]],
                        candidate_pairs: List[Tuple[np.ndarray, np.ndarray]],
                        n_select: int = 5) -> List[int]:
        """Выбирает N самых интересных пар для исследования."""
        existing_surprises = [self.composition_surprise(a, b, a + b)
                             for a, b in existing_compositions]
        baseline = np.mean(existing_surprises) if existing_surprises else 0.0

        scores = []
        for i, (a, b) in enumerate(candidate_pairs):
            di = self.composition_surprise(a, b, a + b)
            scores.append((i, di - baseline))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:n_select]]


# ======================== ОБЪЕДИНЯЮЩИЙ КЛАСС ========================

class StateGrammar:
    """Полная грамматика состояний со всеми 11 механизмами."""

    def __init__(self, dim: int = 2560):
        self.dim = dim

        self.valence = StateValence(dim)
        self.temporal = TemporalChain(dim)
        self.negation = NegationAlgebra(dim)
        self.superposition = SuperpositionCollapse(dim)
        self.validator = CompositionalValidator(dim)
        self.inheritance = StateInheritanceGraph()
        self.emergence = EmergentGenesis()
        self.distance = TransformDistance()
        self.conservation = ConservationLaws()
        self.self_ref = SelfReference()
        self.info_entropy = InformationEntropy(dim)

        self.contextual_composer = ContextualComposer(dim)
        self.rules: Dict[TransformRule, CompositionRule] = {
            rt: CompositionRule(rt, dim) for rt in TransformRule}

    def compose(self, z_a: np.ndarray, z_b: np.ndarray,
                z_context: np.ndarray, alpha_a: float = 1.0,
                alpha_b: float = 1.0,
                concept_a: str = "", concept_b: str = "") -> Dict[str, any]:
        """Полная композиция с использованием всех механизмов."""

        za = torch.from_numpy(z_a).float().unsqueeze(0)
        zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zc = torch.from_numpy(z_context).float().unsqueeze(0)

        za_eff, _ = self.valence(za, alpha_a)
        zb_eff, _ = self.valence(zb, alpha_b)

        validity = self.validator.validate(z_a, z_b, z_context)

        z_ctx_aware = self.contextual_composer(za_eff, zb_eff, zc)
        result = z_ctx_aware.squeeze(0).detach().numpy()

        delta_I = self.info_entropy.composition_surprise(z_a, z_b, result)

        if concept_a and concept_b:
            self.emergence.record_composition(concept_a, concept_b, result)

        z_result_z = torch.from_numpy(result).float().unsqueeze(0)
        collapsed = self.superposition.collapse(result, z_context)
        uncertainty = self.superposition.collapse_uncertainty(z_result_z)

        return {
            "result": result, "validity": validity,
            "delta_I": delta_I, "interesting": self.info_entropy.is_interesting(delta_I),
            "uncertainty": uncertainty, "collapsed": collapsed,
            "alpha_a": float(alpha_a), "alpha_b": float(alpha_b),
        }

    def analyze(self, state_sequence: List[np.ndarray],
                context: np.ndarray) -> Dict[str, any]:
        """Полный анализ последовательности состояний."""
        if len(state_sequence) < 2:
            return {}

        stacked = torch.from_numpy(np.stack(state_sequence)).float().unsqueeze(0)
        transition = self.temporal(stacked)
        next_pred = self.temporal.predict_next(state_sequence)

        coherence = float(1.0 / (1.0 + torch.norm(
            transition[:, 1:] - stacked[:, 1:], dim=-1).mean()))

        negations = []
        for i in range(len(state_sequence) - 1):
            za = torch.from_numpy(state_sequence[i]).float().unsqueeze(0)
            sim = F.cosine_similarity(za, self.negation.negate(
                torch.from_numpy(state_sequence[i + 1]).float().unsqueeze(0)))
            negations.append(float(sim.item()))

        return {
            "coherence": float(coherence),
            "next_prediction": next_pred,
            "negation_score": float(np.mean(negations)) if negations else 0.0,
            "sequence_length": len(state_sequence),
        }

    def formalize(self) -> Dict[str, str]:
        """Формализует все правила в читаемые описания."""
        desc = {}
        for rule_type, rule in self.rules.items():
            params = sum(p.numel() for p in rule.parameters())
            desc[rule_type.value] = (
                f"Rule({rule_type.value}): {params} params, "
                f"conservation={self.conservation.conservation_ratio(rule_type):.2f}")
        desc["emergent"] = f"{len(self.emergence.emergent)} concepts discovered"
        return desc


class ContextualComposer(nn.Module):
    """Контекстно-зависимая композиция: z_{A⊕B|C} = f(z_A, z_B, z_C, θ)."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.dim = dim
        self.context_encoder = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.SiLU(), nn.Linear(dim, dim // 2),
            nn.SiLU(), nn.Linear(dim // 2, dim))
        self.attention = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, z_A: torch.Tensor, z_B: torch.Tensor,
                z_context: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z_A, z_B, z_context], dim=-1)
        ctx_encoded = self.context_encoder(combined)
        stacked = torch.stack([z_A, z_B, ctx_encoded], dim=1)
        attn_out, _ = self.attention(stacked, stacked, stacked)
        return self.layer_norm(attn_out.mean(dim=1))


class CompositionRule(nn.Module):
    """Обучаемое правило композиции."""

    def __init__(self, rule_type: TransformRule, dim: int = 2560):
        super().__init__()
        self.rule_type = rule_type; self.dim = dim
        if rule_type == TransformRule.ANALOGIZE:
            self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
            self.proj = nn.Linear(dim * 2, dim)
        elif rule_type == TransformRule.SEQUENCE:
            self.rnn = nn.GRU(dim, dim, 1, batch_first=True)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.rule_type == TransformRule.ANALOGIZE:
            s = torch.stack([z_a, z_b], dim=1)
            a, _ = self.attn(s, s, s)
            return self.proj(torch.cat([a[:, 0], a[:, 1]], dim=-1))
        if self.rule_type == TransformRule.SEQUENCE:
            s = torch.stack([z_a, z_b], dim=1); _, h = self.rnn(s); return h[-1]
        return self.net(torch.cat([z_a, z_b], dim=-1))
