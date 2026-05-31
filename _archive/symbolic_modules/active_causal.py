"""
EVA — ActiveLearner + CausalDiscovery.

ActiveLearner: запрашивает больше данных когда модель неуверенна.
CausalDiscovery: находит причинно-следственные связи в паттернах траекторий.
"""

import numpy as np, torch
from collections import defaultdict
from typing import List, Tuple


class ActiveLearner:
    """
    Определяет когда модели нужны дополнительные данные.
    
    Признаки неуверенности:
    1. Высокая энтропия предсказаний (много равновероятных вариантов)
    2. Низкая confidence в SelfReflection
    3. Траектория проходит через зону противоречий
    4. Сильное расхождение между гипотезами
    """
    
    def __init__(self, entropy_threshold=3.0, confidence_threshold=0.3):
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
        self.uncertainty_log = []
    
    def should_query(self, logits=None, diagnostic=None,
                     hypotheses_divergence=None) -> Tuple[bool, float, str]:
        """
        Решить: нужны ли дополнительные данные?
        
        Returns: (should_query, urgency, reason)
        """
        reasons = []
        urgency = 0.0
        
        # 1. Entropy check
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            if entropy > self.entropy_threshold:
                reasons.append(f"High entropy: {entropy:.2f}")
                urgency += (entropy - self.entropy_threshold) / 5.0
        
        # 2. Confidence check
        if diagnostic is not None:
            if diagnostic.confidence < self.confidence_threshold:
                reasons.append(f"Low confidence: {diagnostic.confidence:.2f}")
                urgency += (self.confidence_threshold - diagnostic.confidence) * 2
        
        # 3. Contradiction check
        if diagnostic is not None and diagnostic.n_contradictions > 0:
            reasons.append(f"Contradictions: {diagnostic.n_contradictions}")
            urgency += diagnostic.n_contradictions * 0.1
        
        # 4. Hypothesis divergence
        if hypotheses_divergence is not None and hypotheses_divergence > 0.5:
            reasons.append(f"Divergence: {hypotheses_divergence:.2f}")
            urgency += hypotheses_divergence * 0.5
        
        should = urgency > 0.3
        reason = "; ".join(reasons) if reasons else "confident"
        
        self.uncertainty_log.append({
            'should_query': should,
            'urgency': urgency,
            'reason': reason,
        })
        
        return should, urgency, reason
    
    def generate_query(self, uncertainty_reason: str, current_ids: List[int],
                       char_vocab=None) -> str:
        """
        Сгенерировать запрос на дополнительные данные.
        
        Например: "Неуверен после 'человек идёт по'. Покажи больше примеров."
        """
        prefix = char_vocab.decode(current_ids[-10:]) if char_vocab else str(current_ids[-10:])
        return f"UNCERTAIN after '{prefix}': {uncertainty_reason}. Need more examples."


class CausalDiscovery:
    """
    Поиск причинно-следственных связей в траекториях.
    
    Метод: Granger causality через сравнение предиктивной силы.
    Если траектория A предсказывает траекторию B лучше чем наоборот — A → B.
    """
    
    def __init__(self, trajectory_store=None):
        self.store = trajectory_store
        self.causal_graph = defaultdict(list)  # A → [B1, B2, ...]
        self.causal_strength = {}  # (A, B) → strength
    
    def discover_from_pair(self, traj_a: np.ndarray, traj_b: np.ndarray) -> float:
        """
        Проверить: предсказывает ли A → B?
        
        Метод: корреляция между последним шагом A и первым шагом B.
        """
        if len(traj_a) < 2 or len(traj_b) < 2:
            return 0.0
        
        # Step vector of A's last transition
        step_a = traj_a[-1] - traj_a[-2]
        
        # First step of B
        step_b = traj_b[1] - traj_b[0]
        
        # Cosine similarity
        na, nb = np.linalg.norm(step_a), np.linalg.norm(step_b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        
        cos_sim = float(np.dot(step_a, step_b) / (na * nb))
        return max(0.0, cos_sim)  # only positive (causal, not anticausal)
    
    def discover_from_store(self, query_ids: List[int], top_k=10) -> List[Tuple[str, float]]:
        """
        Найти причинно-следственные связи для запроса из TrajectoryStore.
        """
        if self.store is None or self.store.total_stored == 0:
            return []
        
        # Find trajectories starting with query
        matches = self.store.find_by_prefix(query_ids, top_k=top_k * 3)
        
        causal_links = []
        for match in matches:
            ids = match['ids']
            traj = match['trajectory']
            
            # Split into context + continuation
            split = len(query_ids)
            if split < len(ids) - 2:
                context_traj = traj[:split]
                continuation_traj = traj[split:]
                
                strength = self.discover_from_pair(context_traj, continuation_traj)
                if strength > 0.3:
                    continuation_text = self._ids_to_text(ids[split:split+5])
                    causal_links.append((continuation_text, strength))
        
        causal_links.sort(key=lambda x: x[1], reverse=True)
        return causal_links[:top_k]
    
    def _ids_to_text(self, ids):
        """Decode IDs to text (uses CharacterVocab if imported)."""
        try:
            from eva.symbolic.char_vocab import CharacterVocab
            cv = CharacterVocab()
            return cv.decode([int(i) for i in ids if 0 < i < 157])
        except:
            return str(ids)
