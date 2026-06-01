"""
H2K Pipeline — Hypothesis → Knowledge for EVA Symbolic v3.

Три компонента:
1. HypothesisBuffer — хранит K candidate trajectories (состояния h + выходные токены).
2. HypothesisValidator — оценивает каждую гипотезу по SRG + concept + contradiction.
3. EWC — Elastic Weight Consolidation для непрерывного обучения без забывания.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import math, json


# ============================================================
# Hypothesis Dataclass
# ============================================================

@dataclass
class Hypothesis:
    tokens: List[int]
    hidden_states: torch.Tensor  # [L, D]
    scores: Dict[str, float] = field(default_factory=dict)
    srg_val: float = 0.0
    concept_val: float = 0.0
    contra_val: float = 0.0
    combined_score: float = 0.0

    def to_dict(self):
        return {
            'tokens': self.tokens,
            'srg_val': self.srg_val,
            'concept_val': self.concept_val,
            'contra_val': self.contra_val,
            'combined_score': self.combined_score,
        }


# ============================================================
# 1. HypothesisBuffer
# ============================================================

class HypothesisBuffer:
    """
    Хранит top-K гипотез (траекторий).
    
    Добавление: prune если > K.
    Выбор: best по combined_score.
    """
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.hypotheses: List[Hypothesis] = []
    
    def add(self, hyp: Hypothesis):
        self.hypotheses.append(hyp)
        self._prune()
    
    def add_batch(self, hyps: List[Hypothesis]):
        self.hypotheses.extend(hyps)
        self._prune()
    
    def _prune(self):
        if len(self.hypotheses) > self.max_size:
            self.hypotheses.sort(key=lambda h: h.combined_score, reverse=True)
            self.hypotheses = self.hypotheses[:self.max_size]
    
    def best(self) -> Optional[Hypothesis]:
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.combined_score)
    
    def clear(self):
        self.hypotheses = []
    
    def __len__(self):
        return len(self.hypotheses)
    
    def __repr__(self):
        return f"HypothesisBuffer({len(self)} hyps, best={self.best().combined_score if self.best() else 'N/A'})"


# ============================================================
# 2. HypothesisValidator
# ============================================================

class HypothesisValidator:
    """
    Оценивает гипотезы по трём метрикам:
    - SRG: насколько траектория топологически связна
    - Concept: средняя плотность кластера (из ConceptHead)
    - Contradiction: средняя неопределённость (из ContradictionHead)
    
    Итог: combined = w_srg * srg + w_conc * concept - w_contra * contra
    """
    def __init__(self, w_srg: float = 1.0, w_conc: float = 1.0, w_contra: float = 2.0):
        self.w_srg = w_srg
        self.w_conc = w_conc
        self.w_contra = w_contra
    
    def score(self, hyp: Hypothesis) -> float:
        return (self.w_srg * hyp.srg_val +
                self.w_conc * hyp.concept_val -
                self.w_contra * hyp.contra_val)
    
    def batch_score(self, hyps: List[Hypothesis]) -> List[float]:
        return [self.score(h) for h in hyps]
    
    def compute_srg(self, hidden_states: torch.Tensor) -> float:
        """
        SRG = topological connectivity.
        Среднее exp(-distance) между соседними точками траектории.
        """
        if hidden_states.shape[0] < 2:
            return 0.0
        diffs = hidden_states[1:] - hidden_states[:-1]
        dists = torch.norm(diffs, dim=-1)
        connectivity = torch.exp(-dists).mean().item()
        return connectivity
    
    def compute_concept(self, concept_head: nn.Module, hidden_states: torch.Tensor) -> float:
        """
        Concept = средняя плотность кластера из ConceptHead.
        """
        if hidden_states.shape[0] == 0:
            return 0.0
        with torch.no_grad():
            conc = concept_head(hidden_states.unsqueeze(0)).squeeze(0)
        return conc.mean().item()
    
    def compute_contra(self, contra_head: nn.Module, hidden_states: torch.Tensor) -> float:
        """
        Contradiction = средняя неопределённость из ContradictionHead.
        """
        if hidden_states.shape[0] == 0:
            return 1.0
        with torch.no_grad():
            contra = contra_head(hidden_states.unsqueeze(0)).squeeze(0)
        return contra.mean().item()


# ============================================================
# 3. EWC — Elastic Weight Consolidation
# ============================================================

class EWC:
    """
    Elastic Weight Consolidation.
    
    Предотвращает catastrophic forgetting при continuous learning.
    Сохраняет Fisher Information Matrix (диагональную аппроксимацию)
    и оптимальные веса после каждой задачи.
    
    Loss = L_new + lambda * sum_i F_i * (theta_i - theta_star_i)^2
    """
    def __init__(self, model: nn.Module, fisher_multiplier: float = 0.1):
        self.model = model
        self.fisher_multiplier = fisher_multiplier
        self.fisher: Dict[str, torch.Tensor] = {}
        self.opt_params: Dict[str, torch.Tensor] = {}
        self._initialized = False
    
    def compute_fisher(self, dataloader, num_samples: int = 100):
        """
        Вычисляет диагональную Fisher Information Matrix.
        
        Fisher_i = E[(d log p(y|x) / d theta_i)^2]
        
        Используем градиенты по всем параметрам, усреднённые по sample.
        """
        self.model.eval()
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)
                self.opt_params[name] = param.data.clone()
        
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            input_ids = batch['input_ids']
            
            self.model.zero_grad()
            _, _, _, heads = self.model.forward(input_ids, return_heads=True)
            
            # Loss = log p(y|x) — мы максимизируем правдоподобие
            loss = (heads['concept'].mean() + heads['contradiction'].mean() +
                    heads['uncertainty'].mean())
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad ** 2) / num_samples
            count += 1
        
        for name in fisher:
            fisher[name] = fisher[name].clamp(min=1e-8)
        
        self.fisher = fisher
        self._initialized = True
        self.model.train()
        
        return fisher
    
    def ewc_loss(self) -> torch.Tensor:
        """
        EWC regularization loss:
        L_ewc = sum_i F_i * (theta_i - theta_star_i)^2
        """
        if not self._initialized:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher:
                diff = param - self.opt_params[name]
                loss += (self.fisher[name] * diff ** 2).sum()
        
        return self.fisher_multiplier * loss


# ============================================================
# 4. HypothesisWriter — сохраняет лучшие гипотезы для H2K
# ============================================================

class HypothesisWriter:
    """
    Сохраняет гипотезы в H2K базу (JSON-файл) для последующей загрузки
    в TrajectoryStore.
    """
    def __init__(self, path: str):
        self.path = path
    
    def write(self, hypothesis: Hypothesis):
        data = hypothesis.to_dict()
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def write_batch(self, hypotheses: List[Hypothesis]):
        with open(self.path, 'a', encoding='utf-8') as f:
            for hyp in hypotheses:
                f.write(json.dumps(hyp.to_dict(), ensure_ascii=False) + '\n')
    
    def read_all(self) -> List[dict]:
        entries = []
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except FileNotFoundError:
            pass
        return entries
