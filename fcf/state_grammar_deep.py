"""
StateGrammar — глубинное расширение (32–41).

32. CulturalRelativity     — Sapir-Whorf для состояний
33. DreamRecombination     — сюрреалистическая рекомбинация
34. EthicalCalculus        — полный этический ландшафт
35. StateEconomy           — стоимость, ценность, амортизация
36. EvolutionaryPressure   — генетический алгоритм состояний
37. GameTheoretic          — конкуренция, Nash equilibrium
38. AttentionEconomy       — селективный фокус
39. MetaphorGeneration     — Lakoff концептуальная метафора
40. RecursiveIntrospection — глубина самосознания
41. EntropySeeking         — любопытство как фундаментальная сила
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time as _time
import math, random
from loguru import logger


# ============================================================
# 32. CULTURAL RELATIVITY — Sapir-Whorf для состояний
# ============================================================

class CulturalRelativity(nn.Module):
    """Пространство смыслов зависит от культурного контекста."""

    def __init__(self, dim: int = 2560, num_cultures: int = 8):
        super().__init__()
        self.dim = dim; self.num_cultures = num_cultures
        self.culture_embeddings = nn.Parameter(torch.randn(num_cultures, dim) * 0.02)
        self.culture_translator = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.relativity_measure = nn.Sequential(
            nn.Linear(dim * 2, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())

    def translate(self, z: np.ndarray, from_culture: int, to_culture: int) -> np.ndarray:
        zt = torch.from_numpy(z).float().unsqueeze(0)
        cf = self.culture_embeddings[from_culture:from_culture+1]
        ct = self.culture_embeddings[to_culture:to_culture+1]
        with torch.no_grad(): return self.culture_translator(torch.cat([zt + cf, ct], dim=-1)).squeeze(0).numpy()

    def untranslatability(self, z: np.ndarray, culture_a: int, culture_b: int) -> float:
        """Насколько смысл теряется при переводе между культурами."""
        z_ab = self.translate(z, culture_a, culture_b)
        z_ba = self.translate(z_ab, culture_b, culture_a)
        return float(1.0 - np.dot(z.flatten(), z_ba.flatten()) / (np.linalg.norm(z)*np.linalg.norm(z_ba)+1e-8))

    def relativity_field(self, z: np.ndarray) -> np.ndarray:
        """Как одно состояние выглядит в разных культурах."""
        variants = []
        for c in range(min(self.num_cultures, 4)):
            variants.append(self.translate(z, 0, c))
        return np.stack(variants)

    def cultural_distance(self, culture_a: int, culture_b: int) -> float:
        ca = self.culture_embeddings[culture_a]; cb = self.culture_embeddings[culture_b]
        return float(1.0 - F.cosine_similarity(ca.unsqueeze(0), cb.unsqueeze(0)))


# ============================================================
# 33. DREAM RECOMBINATION — сюрреалистическая рекомбинация
# ============================================================

class DreamRecombination(nn.Module):
    """Рекомбинация без ограничений валидности — источник креативности."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.surreal_composer = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.SiLU(),
                                              nn.Linear(dim * 2, dim), nn.Tanh())
        self._output_dim = dim
        self.dream_coherence = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.waking_validator = nn.Sequential(nn.Linear(dim * 2, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())

    def dream(self, z_a: np.ndarray, z_b: np.ndarray, context: np.ndarray,
              surrealism_level: float = 0.5) -> Tuple[np.ndarray, float]:
        za = torch.from_numpy(z_a).float().unsqueeze(0); zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zc = torch.from_numpy(context).float().unsqueeze(0)
        noise = torch.randn(1, self._output_dim) * surrealism_level
        with torch.no_grad():
            dream_state = self.surreal_composer(torch.cat([za, zb, zc], dim=-1)) + noise
            coherence = float(self.dream_coherence(dream_state))
        return dream_state.squeeze(0).numpy(), coherence

    def is_reality_compatible(self, dream_state: np.ndarray, waking_context: np.ndarray) -> float:
        """Может ли сюрреалистическая композиция существовать в реальности."""
        ds = torch.from_numpy(dream_state).float().unsqueeze(0)
        wc = torch.from_numpy(waking_context).float().unsqueeze(0)
        with torch.no_grad(): return float(self.waking_validator(torch.cat([ds, wc], dim=-1)))

    def dream_sequence(self, seed: np.ndarray, context: np.ndarray,
                       length: int = 5, surrealism: float = 0.7) -> List[Tuple[np.ndarray, float]]:
        """Последовательность сюрреалистических трансформаций."""
        current = seed.copy(); sequence = []
        for i in range(length):
            dream, coh = self.dream(current, current + np.random.randn(*current.shape).astype(np.float32)*0.1,
                                    context, surrealism * (1 + i * 0.2))
            sequence.append((dream, coh)); current = dream
        return sequence

    def creative_potential(self, surreal_states: List[Tuple[np.ndarray, float]]) -> float:
        """Насколько сюрреалистическая последовательность креативна."""
        if len(surreal_states) < 2: return 0.0
        diversities = [np.linalg.norm(surreal_states[i][0] - surreal_states[i-1][0])
                       for i in range(1, len(surreal_states))]
        coherences = [s[1] for s in surreal_states]
        return float(np.mean(diversities) * (1.0 - np.mean(coherences)))


# ============================================================
# 34. ETHICAL CALCULUS — полный этический ландшафт
# ============================================================

class EthicalCalculus(nn.Module):
    """Утилитаризм vs деонтология. Trolley problems на состояниях."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.utility_function = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 1))
        self.deontological_score = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 6))
        self.ethical_dilemma = nn.Sequential(nn.Linear(dim * 2, 128), nn.SiLU(), nn.Linear(128, 2))

    PRINCIPLES = ["non_harm", "autonomy", "justice", "beneficence", "dignity", "honesty"]

    def utility(self, z_state: np.ndarray) -> float:
        zt = torch.from_numpy(z_state).float().unsqueeze(0)
        with torch.no_grad(): return float(self.utility_function(zt))

    def deontological_profile(self, z_action: np.ndarray) -> Dict[str, float]:
        zt = torch.from_numpy(z_action).float().unsqueeze(0)
        with torch.no_grad():
            scores = torch.sigmoid(self.deontological_score(zt))
        return {p: float(scores[0, i]) for i, p in enumerate(self.PRINCIPLES)}

    def trolley_problem(self, action_a: np.ndarray, action_b: np.ndarray) -> Dict:
        """Этическая дилемма: действие A спасает X но вредит Y, B наоборот."""
        za = torch.from_numpy(action_a).float().unsqueeze(0)
        zb = torch.from_numpy(action_b).float().unsqueeze(0)
        with torch.no_grad():
            out = torch.softmax(self.ethical_dilemma(torch.cat([za, zb], dim=-1)), dim=-1)
        u_a = self.utility(action_a); u_b = self.utility(action_b)
        d_a = self.deontological_profile(action_a); d_b = self.deontological_profile(action_b)
        return {"choice_A": float(out[0, 0]), "choice_B": float(out[0, 1]),
                "utility_A": u_a, "utility_B": u_b,
                "utilitarian_winner": "A" if u_a > u_b else "B",
                "deontology_A": sum(d_a.values())/6, "deontology_B": sum(d_b.values())/6}

    def ethical_landscape(self, z: np.ndarray, variations: List[np.ndarray]) -> List[Dict]:
        """Этический ландшафт вокруг состояния."""
        landscape = []
        for var in variations:
            u = self.utility(var); d = self.deontological_profile(var)
            landscape.append({"utility": u, "deontology_mean": sum(d.values())/6, "state": var})
        return sorted(landscape, key=lambda x: x["utility"], reverse=True)

    def pareto_frontier(self, states: List[np.ndarray]) -> List[int]:
        """Pareto-оптимальные состояния (utility × deontology)."""
        scores = [(self.utility(s), sum(self.deontological_profile(s).values())/6)
                  for s in states]
        pareto = []
        for i in range(len(states)):
            dominated = False
            for j in range(len(states)):
                if i != j and scores[j][0] >= scores[i][0] and scores[j][1] >= scores[i][1]:
                    if scores[j][0] > scores[i][0] or scores[j][1] > scores[i][1]:
                        dominated = True; break
            if not dominated: pareto.append(i)
        return pareto


# ============================================================
# 35. STATE ECONOMY — стоимость, ценность, амортизация
# ============================================================

class StateEconomy:
    """Внутренний рынок состояний: cost, value, amortization, ROI."""

    def __init__(self):
        self.costs: Dict[int, float] = {}
        self.values: Dict[int, float] = {}
        self.usage: Dict[int, int] = defaultdict(int)
        self.creation_time: Dict[int, float] = {}
        self.depreciation_rate = 0.01

    def creation_cost(self, z: np.ndarray, complexity_weight: float = 1.0) -> float:
        return np.linalg.norm(z.flatten()) * complexity_weight * 0.001

    def use_value(self, z: np.ndarray, context_relevance: float = 1.0) -> float:
        novelty = 1.0 / (np.linalg.norm(z.flatten()) + 1e-8)
        return novelty * context_relevance

    def register(self, state_id: int, z: np.ndarray):
        self.costs[state_id] = self.creation_cost(z)
        self.values[state_id] = self.use_value(z)
        self.creation_time[state_id] = _time.time()

    def use(self, state_id: int, context: np.ndarray):
        self.usage[state_id] += 1
        relevance = float(np.linalg.norm(context) / (self.values.get(state_id, 1.0) + 1e-8))
        self.values[state_id] = self.values.get(state_id, 0.0) * 0.9 + 0.1 * relevance

    def amortized_value(self, state_id: int, current_time: float = 0.0) -> float:
        if state_id not in self.creation_time: return 0.0
        age = current_time - self.creation_time[state_id] if current_time > 0 else 0.0
        base = self.values.get(state_id, 0.0)
        return base * math.exp(-self.depreciation_rate * age)

    def roi(self, state_id: int) -> float:
        cost = self.costs.get(state_id, 1.0)
        value = self.values.get(state_id, 0.0) * self.usage.get(state_id, 1)
        return (value - cost) / (cost + 1e-8)

    def garbage_collect(self, min_roi: float = -0.5) -> List[int]:
        """Удалить состояния с отрицательным ROI."""
        return [sid for sid in self.costs if self.roi(sid) < min_roi]

    def most_valuable(self, n: int = 5) -> List[int]:
        return sorted(self.values.keys(), key=lambda sid: self.values[sid], reverse=True)[:n]

    def portfolio_balance(self) -> Dict[str, float]:
        """Баланс портфеля состояний."""
        total_cost = sum(self.costs.values())
        total_value = sum(v * self.usage.get(k, 1) for k, v in self.values.items())
        return {"total_cost": total_cost, "total_value": total_value,
                "roi": (total_value - total_cost) / (total_cost + 1e-8),
                "states": len(self.costs)}


# ============================================================
# 36. EVOLUTIONARY PRESSURE — генетический алгоритм
# ============================================================

class EvolutionaryPressure:
    """Генетический алгоритм: crossover, mutation, tournament selection."""

    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7, elitism: int = 5):
        self.population_size = population_size; self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate; self.elitism = elitism
        self.population: List[Tuple[np.ndarray, float]] = []
        self.generation = 0

    def initialize(self, seed_fn: Callable[[], np.ndarray], fitness_fn: Callable):
        self.population = []
        for _ in range(self.population_size):
            z = seed_fn()
            self.population.append((z, fitness_fn(z)))
        self.generation = 0

    def tournament_select(self, k: int = 3) -> int:
        candidates = random.sample(range(len(self.population)), min(k, len(self.population)))
        return max(candidates, key=lambda i: self.population[i][1])

    def crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        mask = np.random.rand(*parent_a.shape) < 0.5
        child = np.where(mask, parent_a, parent_b)
        return child / (np.linalg.norm(child) + 1e-8)

    def mutate(self, z: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*z.shape).astype(np.float32) * self.mutation_rate
        mutated = z + noise
        return mutated / (np.linalg.norm(mutated) + 1e-8)

    def evolve(self, fitness_fn: Callable) -> List[Tuple[np.ndarray, float]]:
        self.population.sort(key=lambda x: x[1], reverse=True)
        new_pop = self.population[:self.elitism]

        while len(new_pop) < self.population_size:
            p1_idx = self.tournament_select(3); p2_idx = self.tournament_select(3)
            p1, _ = self.population[p1_idx]; p2, _ = self.population[p2_idx]

            if random.random() < self.crossover_rate:
                child = self.crossover(p1, p2)
            else:
                child = p1.copy()

            child = self.mutate(child)
            new_pop.append((child, fitness_fn(child)))

        self.population = new_pop; self.generation += 1
        return self.population

    def best(self) -> Tuple[np.ndarray, float]:
        return max(self.population, key=lambda x: x[1])

    def diversity(self) -> float:
        if len(self.population) < 2: return 0.0
        centroids = np.mean([p[0] for p in self.population], axis=0)
        return float(np.mean([np.linalg.norm(p[0] - centroids) for p in self.population]))

    def fitness_landscape(self) -> Dict[str, float]:
        if not self.population: return {}
        fits = [f for _, f in self.population]
        return {"max": max(fits), "mean": np.mean(fits), "std": np.std(fits),
                "min": min(fits), "generation": self.generation}


# ============================================================
# 37. GAME THEORETIC — Nash equilibrium
# ============================================================

class GameTheoretic:
    """Конкуренция состояний за контекстный слот."""

    def __init__(self, dim: int = 2560):
        self.dim = dim
        self.payoff_history: Dict[Tuple[int, int], float] = {}

    def payoff_matrix(self, states: List[np.ndarray], context: np.ndarray) -> np.ndarray:
        """Платёжная матрица: насколько состояние i выигрывает у j в контексте."""
        n = len(states); M = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim_i = np.dot(states[i].flatten(), context.flatten())
                    sim_j = np.dot(states[j].flatten(), context.flatten())
                    M[i, j] = float(sim_i - sim_j)
                else:
                    M[i, j] = 0.0
        return M

    def nash_equilibrium(self, payoff: np.ndarray, iterations: int = 1000) -> np.ndarray:
        """Итеративное нахождение смешанной стратегии Нэша."""
        n = payoff.shape[0]; strategy = np.ones(n) / n
        for _ in range(iterations):
            expected = payoff @ strategy
            strategy = strategy * np.exp(expected * 0.1)
            strategy = strategy / (np.sum(strategy) + 1e-8)
        return strategy

    def dominant_state(self, states: List[np.ndarray], context: np.ndarray) -> int:
        """Строго доминирующее состояние."""
        payoff = self.payoff_matrix(states, context)
        for i in range(len(states)):
            if all(payoff[i, j] > 0 for j in range(len(states)) if j != i):
                return i
        return -1

    def mixed_strategy_sample(self, states: List[np.ndarray],
                              context: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Смешанная стратегия: вероятности выбора каждого состояния."""
        payoff = self.payoff_matrix(states, context)
        strategy = self.nash_equilibrium(payoff)
        return list(range(len(states))), strategy

    def coalition_value(self, coalition: List[int], states: List[np.ndarray],
                        context: np.ndarray) -> float:
        """Ценность коалиции состояний."""
        if not coalition: return 0.0
        combined = np.mean([states[i] for i in coalition], axis=0)
        return float(np.dot(combined.flatten(), context.flatten()))


# ============================================================
# 38. ATTENTION ECONOMY — селективный фокус
# ============================================================

class AttentionEconomy(nn.Module):
    """Ограниченный бюджет внимания. Механизм селективного фокуса."""

    def __init__(self, dim: int = 2560, attention_budget: float = 1.0):
        super().__init__()
        self.dim = dim; self.attention_budget = attention_budget
        self.salience_detector = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.focus_projector = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.attention_allocated = 0.0

    def salience(self, z: np.ndarray) -> float:
        zt = torch.from_numpy(z).float().unsqueeze(0)
        with torch.no_grad(): return float(self.salience_detector(zt))

    def allocate(self, candidates: List[np.ndarray]) -> List[Tuple[int, float, float]]:
        """Распределяет бюджет внимания между кандидатами."""
        saliences = [self.salience(z) for z in candidates]
        total = sum(saliences) + 1e-8
        allocations = []
        remaining = self.attention_budget
        ranked = sorted(enumerate(saliences), key=lambda x: x[1], reverse=True)

        for idx, sal in ranked:
            alloc = min(sal / total * self.attention_budget, remaining)
            remaining -= alloc
            allocations.append((idx, alloc, sal))
            if remaining <= 0: break

        self.attention_allocated = self.attention_budget - remaining
        return allocations

    def focus(self, z: np.ndarray, attention_weight: float = 1.0) -> np.ndarray:
        """Усиливает состояние пропорционально выделенному вниманию."""
        zt = torch.from_numpy(z).float().unsqueeze(0)
        with torch.no_grad():
            focused = self.focus_projector(zt) * attention_weight + zt * (1 - attention_weight)
        return focused.squeeze(0).numpy()

    def attention_remaining(self) -> float:
        return self.attention_budget - self.attention_allocated

    def reset_budget(self):
        self.attention_allocated = 0.0


# ============================================================
# 39. METAPHOR GENERATION — Lakoff концептуальная метафора
# ============================================================

class MetaphorGeneration(nn.Module):
    """'Спор — это война' — отождествление доменов."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.source_encoder = nn.Sequential(nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, dim // 2))
        self.target_encoder = nn.Sequential(nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, dim // 2))
        self.mapping_net = nn.Sequential(nn.Linear(dim // 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.metaphor_scorer = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())

    def generate_metaphor(self, source_domain: np.ndarray,
                          target_domain: np.ndarray) -> Tuple[np.ndarray, float]:
        """Порождает метафору: понимание target через source."""
        sd = torch.from_numpy(source_domain).float().unsqueeze(0)
        td = torch.from_numpy(target_domain).float().unsqueeze(0)
        with torch.no_grad():
            se = self.source_encoder(sd); te = self.target_encoder(td)
            mapped = self.mapping_net(se - te + te)
            score = float(self.metaphor_scorer(mapped))
        return mapped.squeeze(0).numpy(), score

    def metaphor_strength(self, source: np.ndarray, target: np.ndarray) -> float:
        """Насколько сильна метафорическая связь между доменами."""
        _, score = self.generate_metaphor(source, target)
        return score

    def find_best_metaphor(self, target: np.ndarray,
                           source_candidates: List[np.ndarray]) -> int:
        """Находит лучший source domain для метафоры target."""
        scores = [self.metaphor_strength(src, target) for src in source_candidates]
        return int(np.argmax(scores))

    def metaphor_chain(self, source: np.ndarray, depth: int = 3) -> List[np.ndarray]:
        """Цепочка метафор: A → B → C → ..."""
        chain = [source]
        current = source
        for _ in range(depth - 1):
            next_m, _ = self.generate_metaphor(current, current + np.random.randn(*current.shape).astype(np.float32)*0.1)
            chain.append(next_m); current = next_m
        return chain


# ============================================================
# 40. RECURSIVE INTROSPECTION — глубина самосознания
# ============================================================

class RecursiveIntrospection(nn.Module):
    """'Я знаю что я знаю что X' — глубина самосознания."""

    def __init__(self, dim: int = 2560, max_depth: int = 10):
        super().__init__()
        self.dim = dim; self.max_depth = max_depth
        self.introspection_layer = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.meta_confidence = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())

    def introspect(self, z_base: np.ndarray, depth: int = 1) -> np.ndarray:
        """K^n(z) — знание о знании о ... знании о z."""
        current = z_base.copy()
        for _ in range(depth):
            zt = torch.from_numpy(current).float().unsqueeze(0)
            zb = torch.from_numpy(z_base).float().unsqueeze(0)
            with torch.no_grad():
                current = self.introspection_layer(torch.cat([zt, zb], dim=-1)).squeeze(0).numpy()
        return current

    def depth_of_self_awareness(self, z: np.ndarray) -> int:
        """Определяет максимальную глубину самосознания до вырождения."""
        prev = z.copy()
        for d in range(1, self.max_depth + 1):
            curr = self.introspect(prev, 1)
            sim = np.dot(prev.flatten(), curr.flatten()) / (np.linalg.norm(prev)*np.linalg.norm(curr)+1e-8)
            if sim > 0.95: return d - 1
            prev = curr
        return self.max_depth

    def introspection_chain(self, z: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Цепочка интроспекции с мета-уверенностью."""
        chain = [(z.copy(), 1.0)]
        for d in range(1, self.max_depth):
            next_z = self.introspect(chain[-1][0], 1)
            zt = torch.from_numpy(next_z).float().unsqueeze(0)
            with torch.no_grad(): conf = float(self.meta_confidence(zt))
            chain.append((next_z, conf))
            if conf < 0.1: break
        return chain

    def self_model_accuracy(self, z_actual: np.ndarray,
                            z_self_model: np.ndarray) -> float:
        """Насколько self-model соответствует реальности."""
        return float(np.dot(z_actual.flatten(), z_self_model.flatten()) /
                     (np.linalg.norm(z_actual)*np.linalg.norm(z_self_model)+1e-8))


# ============================================================
# 41. ENTROPY SEEKING — любопытство как фундаментальная сила
# ============================================================

class EntropySeeking(nn.Module):
    """Активный поиск состояний с максимальным приростом энтропии."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.dim = dim
        self.curiosity_net = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 1), nn.Tanh())
        self.exploration_noise = nn.Parameter(torch.tensor(0.1))
        self.known_territory: Dict[str, float] = {}

    def curiosity_score(self, z: np.ndarray) -> float:
        """Насколько состояние возбуждает любопытство. Положительное = интересно."""
        zt = torch.from_numpy(z).float().unsqueeze(0)
        with torch.no_grad(): return float(self.curiosity_net(zt))

    def entropy_gradient(self, z: np.ndarray, step_size: float = 0.01) -> np.ndarray:
        """Градиент в направлении максимального прироста энтропии."""
        zt = torch.from_numpy(z).float().unsqueeze(0).requires_grad_(True)
        curiosity = self.curiosity_net(zt)
        grad = torch.autograd.grad(curiosity, zt)[0]
        z_new = zt + step_size * grad / (torch.norm(grad) + 1e-8)
        return z_new.detach().squeeze(0).numpy()

    def explore(self, z_start: np.ndarray, steps: int = 20,
                step_size: float = 0.02) -> List[Tuple[np.ndarray, float]]:
        """Исследует пространство состояний, движимое любопытством."""
        current = z_start.copy(); trajectory = []
        for _ in range(steps):
            score = self.curiosity_score(current)
            trajectory.append((current.copy(), score))
            current = self.entropy_gradient(current, step_size)
        return trajectory

    def novelty_score(self, z: np.ndarray) -> float:
        """Насколько состояние ново относительно известной территории."""
        z_key = hash(z.tobytes()) if hasattr(z, 'tobytes') else hash(str(z.shape))
        if z_key in self.known_territory: return 1.0 - self.known_territory[z_key]
        return 1.0

    def mark_known(self, z: np.ndarray, familiarity: float = 0.5):
        z_key = hash(z.tobytes()) if hasattr(z, 'tobytes') else hash(str(z.shape))
        prev = self.known_territory.get(z_key, 0.0)
        self.known_territory[z_key] = prev * 0.9 + familiarity * 0.1

    def curiosity_driven_exploration(self, known_states: List[np.ndarray],
                                     steps: int = 10) -> List[np.ndarray]:
        """Отправляется от известных состояний в неизведанное."""
        for z in known_states: self.mark_known(z, 1.0)

        start = random.choice(known_states) if known_states else np.random.randn(self.dim).astype(np.float32)
        trajectory = self.explore(start, steps)

        discoveries = []
        for z, score in trajectory:
            if self.novelty_score(z) > 0.7 and score > 0.0:
                discoveries.append(z)
                self.mark_known(z, 0.3)

        return discoveries

    def curiosity_saturation(self, trajectory: List[float]) -> float:
        """Насколько быстро насыщается любопытство вдоль траектории."""
        if len(trajectory) < 2: return 1.0
        deltas = [trajectory[i] - trajectory[i-1] for i in range(1, len(trajectory))]
        return float(np.mean(deltas) / (np.std(deltas) + 1e-8))
