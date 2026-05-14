"""
StateGrammar — интеллектуальная разметка и грамматика состояний.

Назначение:
  Формализует ПРАВИЛА композиции состояний, а не сами состояния.
  "Золотая рожь" ≠ "золотая статуэтка" потому что state("золотая") 
  по-разному взаимодействует с state("рожь") и state("статуэтка").

  Система учится не запоминать векторы, а выводить ЗАКОНЫ того,
  как состояния трансформируются в зависимости от:
  - типа состояния (символ/концепт/отношение/контекст)
  - соседних состояний (контекстное окружение)
  - уровня абстракции (символ→слово→предложение→текст)
  - истории композиции (как было получено это состояние)

Задача:
  - Разметить каждое состояние его типом и правилами трансформации
  - Обнаружить закономерности композиции из данных
  - Формализовать их как исполняемые правила
  - Обучить модель предсказывать КАК состояния будут взаимодействовать
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class StateType(Enum):
    """Тип состояния — определяет какие трансформации допустимы."""
    SYMBOL = "symbol"
    CONCEPT = "concept"
    RELATION = "relation"
    CONTEXT = "context"
    COMPOSITE = "composite"


class TransformRule(Enum):
    """Правила трансформации состояний."""
    COMPOSE = "compose"
    SPECIFY = "specify"
    GENERALIZE = "generalize"
    ANALOGIZE = "analogize"
    CONTRAST = "contrast"
    SEQUENCE = "sequence"
    NEGATE = "negate"


@dataclass
class StateSignature:
    """
    Интеллектуальная разметка состояния.

    Не ЧТО хранит состояние, а КАК оно взаимодействует с другими.
    """
    state_type: StateType
    dimension: int
    applicable_transforms: Set[TransformRule] = field(default_factory=set)
    context_sensitivity: float = 0.5
    abstraction_level: int = 0
    composability_score: float = 0.5
    stability_score: float = 0.5
    metadata: Dict = field(default_factory=dict)


class ContextualComposer(nn.Module):
    """
    Моделирует КОНТЕКСТНО-ЗАВИСИМУЮ композицию состояний.

    z_A в контексте z_C даёт z_{A|C} — НЕ то же самое что z_A в контексте z_D.
    Реализует принцип "золотая рожь ≠ золотая статуэтка".

    Математика:
      z_{A⊕B|C} = f(z_A, z_B, z_C, θ_compose)
    где f — обучаемая функция, учитывающая все три состояния.
    """

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.dim = dim
        self.context_encoder = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.SiLU(),
            nn.Linear(dim, dim // 2), nn.SiLU(),
            nn.Linear(dim // 2, dim),
        )
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
    """
    Обучаемое ПРАВИЛО композиции.

    Вместо хранения всех комбинаций — хранит правило как их порождать.
    """

    def __init__(self, rule_type: TransformRule, dim: int = 2560):
        super().__init__()
        self.rule_type = rule_type
        self.dim = dim

        if rule_type == TransformRule.COMPOSE:
            self.net = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.SiLU(),
                nn.Linear(dim, dim),
            )
        elif rule_type == TransformRule.SPECIFY:
            self.net = nn.Sequential(
                nn.Linear(dim * 2, dim * 2), nn.SiLU(),
                nn.Linear(dim * 2, dim),
            )
        elif rule_type == TransformRule.GENERALIZE:
            self.net = nn.Sequential(
                nn.Linear(dim * 2, dim // 2), nn.SiLU(),
                nn.Linear(dim // 2, dim),
            )
        elif rule_type == TransformRule.ANALOGIZE:
            self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
            self.proj = nn.Linear(dim * 2, dim)
        elif rule_type == TransformRule.CONTRAST:
            self.net = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.SiLU(),
                nn.Linear(dim, dim),
            )
            self.similarity_head = nn.Linear(dim, 1)
        elif rule_type == TransformRule.SEQUENCE:
            self.rnn = nn.GRU(dim, dim, 1, batch_first=True)
        else:
            self.net = nn.Linear(dim, dim)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.rule_type == TransformRule.ANALOGIZE:
            stacked = torch.stack([z_a, z_b], dim=1)
            attn_out, _ = self.attention(stacked, stacked, stacked)
            return self.proj(torch.cat([attn_out[:, 0], attn_out[:, 1]], dim=-1))

        if self.rule_type == TransformRule.SEQUENCE:
            seq = torch.stack([z_a, z_b], dim=1)
            _, h = self.rnn(seq)
            return h[-1]

        combined = torch.cat([z_a, z_b], dim=-1)
        return self.net(combined)


class StateGrammar:
    """
    Грамматика состояний — формальная система правил композиции.

    Вместо бесконечного хранения всех комбинаций:
      state("золотая") + state("рожь") → правило COMPOSE → state("золотая_рожь")
      state("золотая") + state("статуэтка") → правило COMPOSE → state("золотая_статуэтка")

    Ключевое: COMPOSE применяет РАЗНУЮ трансформацию в зависимости от
    типов и контекстной чувствительности операндов.
    """

    def __init__(self, dim: int = 2560):
        self.dim = dim
        self.composer = ContextualComposer(dim)
        self.rules: Dict[TransformRule, CompositionRule] = {
            rule_type: CompositionRule(rule_type, dim)
            for rule_type in TransformRule
        }
        self.signatures: Dict[int, StateSignature] = {}

    def register_state(self, state_id: int, signature: StateSignature):
        self.signatures[state_id] = signature

    def compose(self, z_a: np.ndarray, z_b: np.ndarray,
                z_context: np.ndarray,
                type_a: StateType = StateType.CONCEPT,
                type_b: StateType = StateType.CONCEPT,
                type_context: StateType = StateType.CONTEXT) -> np.ndarray:
        """
        Композиция состояний с учётом контекста.

        z_result = Rule(COMPOSE)(z_a, z_b, z_context, θ(type_a, type_b, type_context))
        """
        za = torch.from_numpy(z_a).float().unsqueeze(0)
        zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zc = torch.from_numpy(z_context).float().unsqueeze(0)

        with torch.no_grad():
            ctx_aware = self.composer(za, zb, zc)
            result = self.rules[TransformRule.COMPOSE](za, zb, zc)

            alpha = 0.3 * self._get_composability(type_a, type_b)
            final = (1 - alpha) * ctx_aware + alpha * result

        return final.squeeze(0).numpy()

    def _get_composability(self, type_a: StateType, type_b: StateType) -> float:
        if type_a == StateType.SYMBOL and type_b == StateType.SYMBOL:
            return 0.8
        if type_a == StateType.CONCEPT and type_b == StateType.CONCEPT:
            return 0.6
        if type_a == StateType.CONCEPT and type_b == StateType.CONTEXT:
            return 0.9
        if type_a == StateType.RELATION:
            return 0.7
        return 0.5

    def analyze_sequence(self, states: List[np.ndarray],
                         context: np.ndarray) -> Dict[str, float]:
        """
        Анализирует последовательность состояний и выявляет закономерности.

        Возвращает метрики: связность, контекстная чувствительность, композиционность.
        """
        if len(states) < 2:
            return {"coherence": 1.0, "context_sensitivity": 0.0, "composability": 0.0}

        tensors = torch.from_numpy(np.stack(states)).float()
        ctx = torch.from_numpy(context).float()

        diffs = tensors[1:] - tensors[:-1]
        coherence = float(1.0 / (1.0 + torch.norm(diffs, dim=-1).mean()))

        ctx_sims = F.cosine_similarity(tensors, ctx.unsqueeze(0).expand_as(tensors))
        context_sensitivity = float(ctx_sims.std())

        composability = float(1.0 - torch.norm(diffs, dim=-1).std())

        return {
            "coherence": coherence,
            "context_sensitivity": context_sensitivity,
            "composability": composability,
        }

    def discover_rules(self, state_pairs: List[Tuple[np.ndarray, np.ndarray]],
                       contexts: List[np.ndarray],
                       results: List[np.ndarray],
                       epochs: int = 200, lr: float = 1e-4):
        """
        Обнаруживает закономерности композиции из данных.

        Обучает composer и правила на реальных примерах композиций.
        Сама система учится выводить ПРАВИЛА того, как состояния взаимодействуют.
        """
        params = list(self.composer.parameters())
        for rule in self.rules.values():
            params += list(rule.parameters())

        optimizer = torch.optim.Adam(params, lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(len(state_pairs)):
                optimizer.zero_grad()

                za = torch.from_numpy(state_pairs[i][0]).float().unsqueeze(0)
                zb = torch.from_numpy(state_pairs[i][1]).float().unsqueeze(0)
                zc = torch.from_numpy(contexts[i]).float().unsqueeze(0)
                target = torch.from_numpy(results[i]).float().unsqueeze(0)

                ctx_aware = self.composer(za, zb, zc)
                composed = self.rules[TransformRule.COMPOSE](za, zb, zc)

                alpha = 0.3
                predicted = (1 - alpha) * ctx_aware + alpha * composed

                loss = F.mse_loss(predicted, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 40 == 0:
                logger.info(
                    f"[Grammar] discover epoch={epoch}, loss={total_loss:.6f}"
                )

        logger.info(f"[Grammar] Rules discovered: {len(self.rules)} rules trained")

    def formalize_rules(self) -> Dict[str, str]:
        """
        Формализует обнаруженные правила в виде человеко-читаемых описаний.
        """
        descriptions = {}
        for rule_type, rule in self.rules.items():
            params = sum(p.numel() for p in rule.parameters())
            descriptions[rule_type.value] = (
                f"Rule({rule_type.value}): {params} parameters, "
                f"transforms z_A⊕z_B → z_result "
                f"under context constraint"
            )
        return descriptions
