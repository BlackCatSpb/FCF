"""
Continuous Learning Runtime for EVA v3.

Цикл: read → generate → measure → H2K → learn.

Без внешнего датасета: EVA читает свой собственный выход,
оценивает его качество (concept, contradiction, residual),
и учится на лучших гипотезах.

Компоненты:
- DataSource: читает тексты (корпус или стрим)
- Evaluator: оценивает качество генерации по головам
- Cycle: read → generate → evaluate → H2K → train_step
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import json, os, math, time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field

from .h2k_pipeline import (
    Hypothesis, HypothesisBuffer, HypothesisValidator,
    EWC, HypothesisWriter
)
from .train_v3 import MultiTaskLoss, TrainingConfig
from .thought_loop import generate_with_thought, ThoughtLoopConfig, analyze_thought_trace
from .trajectory_store import TrajectoryStore, HierarchicalTrajectory


@dataclass
class RuntimeConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_tokens_per_cycle: int = 128
    max_generations: int = 5          # сколько гипотез генерировать за цикл
    h2k_buffer_size: int = 10
    train_batch_size: int = 4
    train_steps_per_cycle: int = 5
    corpus_path: str = ""
    h2k_path: str = "h2k_hypotheses.jsonl"
    trajectory_store_path: str = ""
    log_every: int = 1


class Evaluator:
    """
    Оценивает качество генерации по головам EVA.

    Метрики:
    - concept: среднее ConceptHead по всей траектории
    - contradiction: среднее ContradictionHead
    - uncertainty: средняя UncertaintyHead
    - residual_error: средняя ResidualHead error
    - srg: topological smoothness (среднее exp(-dist) между соседями)
    - fluency: доля уникальных токенов / общее число токенов

    Return: dict с метриками, float от 0 до 1 (1 = best).
    """
    def __init__(self):
        pass

    def evaluate(self, tokens: List[int], h: torch.Tensor,
                 heads_out: dict) -> Dict[str, float]:
        B, L, D = h.shape
        L_eff = min(len(tokens), L)
        h_seq = h[0, :L_eff]
        metrics = {}

        # Concept
        if 'concept' in heads_out:
            metrics['concept'] = heads_out['concept'][0, :L_eff].mean().item()

        # Contradiction
        if 'contradiction' in heads_out:
            metrics['contradiction'] = heads_out['contradiction'][0, :L_eff].mean().item()

        # Uncertainty (mean variance)
        if 'uncertainty' in heads_out:
            metrics['uncertainty'] = heads_out['uncertainty'][0, :L_eff].mean().item()

        # Residual error
        if 'residual_error' in heads_out:
            metrics['residual_error'] = heads_out['residual_error'][0, :L_eff].mean().item()

        # SRG (topological smoothness)
        if h_seq.shape[0] >= 2:
            diffs = h_seq[1:] - h_seq[:-1]
            dists = torch.norm(diffs, dim=-1)
            metrics['srg'] = torch.exp(-dists).mean().item()
        else:
            metrics['srg'] = 0.0

        # Fluency: доля уникальных в окне 50
        window = tokens[-min(50, len(tokens)):]
        metrics['fluency'] = len(set(window)) / max(len(window), 1)

        # Composite score (higher = better)
        w_conc = 1.0; w_contra = 0.5; w_res = 0.3; w_srg = 0.5; w_fl = 0.2
        metrics['composite'] = (
            w_conc * metrics.get('concept', 0.0) +
            w_srg * metrics.get('srg', 0.0) +
            w_fl * metrics.get('fluency', 0.0) -
            w_contra * metrics.get('contradiction', 0.0) -
            w_res * metrics.get('residual_error', 0.0)
        )

        return metrics


class DataSource:
    """
    Читает тексты из корпуса для затравки генерации.
    """
    def __init__(self, path: str):
        self.path = path
        self.texts = []
        self._load()

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            self.texts = [""]  # fallback: пустой промпт
            return
        with open(self.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.texts = [l.strip() for l in lines if l.strip()][:200]

    def sample_prompt(self, max_chars: int = 32):
        text = np.random.choice(self.texts)
        return text[:max_chars]


def _batch_from_tokens(tokens, config: TrainingConfig, device):
    """Create training batch from a single trajectory (overlapping windows)."""
    B, L = config.batch_size, config.seq_len
    if len(tokens) < L + 1:
        return None
    starts = np.random.randint(0, len(tokens) - L - 1, size=B)
    batch_ids = np.stack([tokens[s:s+L+1] for s in starts])
    input_ids = torch.tensor(batch_ids[:, :L], dtype=torch.long, device=device)
    target_ids = torch.tensor(batch_ids[:, 1:L+1], dtype=torch.long, device=device)
    return {'input_ids': input_ids, 'target_ids': target_ids}


def run_cycle(model, cv, optimizer, loss_fn,
              runtime_cfg: RuntimeConfig = None,
              thought_cfg: ThoughtLoopConfig = None,
              train_cfg: TrainingConfig = None,
              knn_retriever=None,
              data_source=None):
    """
    Один цикл continuous learning:

    1. Sample prompt from corpus (или пустой)
    2. Generate K гипотез с thought loop
    3. Evaluate каждую гипотезу (Evaluator)
    4. Сохранить в H2K лучшие
    5. Train несколько шагов на лучших гипотезах

    Args:
        model: EVA model
        cv: CharacterVocab
        optimizer: torch optimizer
        loss_fn: MultiTaskLoss
        knn_retriever: опционально KNNRetriever
        data_source: DataSource (если None, использует пустой промпт)

    Returns:
        cycle_report: dict с метриками цикла
    """
    if runtime_cfg is None:
        runtime_cfg = RuntimeConfig()
    if thought_cfg is None:
        thought_cfg = ThoughtLoopConfig()
    if train_cfg is None:
        train_cfg = TrainingConfig()

    device = runtime_cfg.device
    evaluator = Evaluator()
    hypothesis_buffer = HypothesisBuffer(runtime_cfg.h2k_buffer_size)
    hypothesis_writer = HypothesisWriter(runtime_cfg.h2k_path)

    trajectory_store = None
    if runtime_cfg.trajectory_store_path:
        trajectory_store = TrajectoryStore(max_trajectories=100000)
        if os.path.exists(runtime_cfg.trajectory_store_path):
            try:
                trajectory_store.load(runtime_cfg.trajectory_store_path)
                print(f'[Runtime] Loaded {trajectory_store.total_stored} trajectories')
            except Exception:
                pass

    device = torch.device(device)
    cycle_metrics = {
        'prompt': '',
        'generations': 0,
        'best_composite': -999,
        'avg_composite': 0,
        'train_loss': 0.0,
        'thought_iters': 0.0,
        'best_text': '',
        'trajectories': 0,
    }

    # === 1. Sample prompt ===
    if data_source is not None:
        prompt_raw = data_source.sample_prompt(48)
    else:
        prompt_raw = ""

    # Encode prompt
    prompt_ids_inner = [cv.SENT_OPEN_IDX]
    if prompt_raw:
        for ch in prompt_raw:
            prompt_ids_inner.append(cv.char_to_idx(ch))
    cycle_metrics['prompt'] = prompt_raw

    # === 2. Generate K гипотез ===
    for g in range(runtime_cfg.max_generations):
        text_out, thought_trace = generate_with_thought(
            model, prompt_ids_inner, cv,
            config=thought_cfg,
            max_new=runtime_cfg.max_tokens_per_cycle,
            temperature=0.8 + g * 0.05,  # explore with higher temp
            knn_retriever=knn_retriever,
            skip_decoder=True,  # head-only: no decoder logits
        )

        # Encode the FULL generated text (not just prompt)
        gen_ids = cv.encode_with_boundaries(text_out)
        with torch.no_grad():
            inp = torch.tensor([gen_ids], dtype=torch.long, device=device)
            h, _, _, heads_out = model.forward(inp, return_heads=True, capture_attn=True)

        # Evaluate on full generation
        metrics = evaluator.evaluate(gen_ids, h, heads_out)

        # Store as Hypothesis
        hyp = Hypothesis(
            tokens=gen_ids,
            hidden_states=h[0].cpu(),
            concept_val=metrics.get('concept', 0.0),
            contra_val=metrics.get('contradiction', 0.0),
            srg_val=metrics.get('srg', 0.0),
        )
        hyp.combined_score = metrics.get('composite', 0.0)
        hypothesis_buffer.add(hyp)

        # Store trajectory in TrajectoryStore
        if trajectory_store is not None and len(h[0]) > 0:
            traj_np = h[0].cpu().numpy().astype(np.float32)
            trajectory_store.store(text_out, gen_ids, traj_np)
            cycle_metrics['trajectories'] = trajectory_store.total_stored

        trace_analysis = analyze_thought_trace(thought_trace)
        cycle_metrics['thought_iters'] += trace_analysis.get('avg_iterations', 0)

        if metrics.get('composite', -999) > cycle_metrics['best_composite']:
            cycle_metrics['best_composite'] = metrics.get('composite', 0)
            cycle_metrics['best_text'] = text_out[:100]

    cycle_metrics['generations'] = runtime_cfg.max_generations
    cycle_metrics['avg_composite'] = np.mean(
        [h.combined_score for h in hypothesis_buffer.hypotheses]
    ) if hypothesis_buffer.hypotheses else 0.0
    cycle_metrics['thought_iters'] /= max(runtime_cfg.max_generations, 1)

    # === 3. Save best to H2K ===
    best_hyp = hypothesis_buffer.best()
    if best_hyp is not None:
        hypothesis_writer.write(best_hyp)

    # === 4. Train on best hypotheses ===
    model.train()
    total_train_loss = 0.0
    for train_step in range(runtime_cfg.train_steps_per_cycle):
        best_hyp = hypothesis_buffer.best()
        if best_hyp is None:
            break
        tokens = best_hyp.tokens
        if len(tokens) < train_cfg.seq_len + 1:
            continue

        # Create batch from best hypothesis trajectory (overlapping windows)
        batch = _batch_from_tokens(tokens, train_cfg, device)
        if batch is None:
            continue

        optimizer.zero_grad()
        model_out = model.forward(
            batch['input_ids'],
            return_scores=False,
            return_heads=True,
            capture_attn=True,
            use_weight=train_cfg.use_weight_context,
        )
        h_out, _, _, heads_out = model_out

        # Intrinsic labels
        contra_labels = model._intrinsic_contra_labels(h_out, batch['input_ids'])
        conc_labels = model._intrinsic_concept_labels(h_out, batch['input_ids'])
        intrinsic = {'contra': contra_labels, 'concept': conc_labels}

        # Core losses
        loss_dict = loss_fn(model_out, batch, intrinsic)

        # Head-only losses
        loss_dict['srg'] = model.temporal_smoothness_loss(h_out)
        loss_dict['srg_real'] = model.srg_loss(h_out)
        loss_dict['attn_entropy'] = model.attention_entropy()
        loss_dict['head_consistency'] = model.head_consistency_loss(heads_out)
        try:
            loss_dict['self_distill'] = model.self_distill_thought(heads_out, h_out)
        except Exception:
            loss_dict['self_distill'] = torch.tensor(0.0, device=h_out.device)

        # nxt_all_loss: coordinate delta prediction on EVERY position
        nxt = heads_out.get('boundary_next')
        if nxt is not None and h_out.shape[1] > 1:
            delta_actual = h_out[:, 1:] - h_out[:, :-1]
            nxt_pred = nxt[:, :-1]
            loss_dict['nxt'] = F.mse_loss(nxt_pred, delta_actual)
        else:
            loss_dict['nxt'] = torch.tensor(0.0, device=h_out.device)

        # KCA auxiliary loss
        if train_cfg.w_kca > 0:
            loss_dict['kca'] = model.kca_aux_loss(h_out, heads_out)
        else:
            loss_dict['kca'] = torch.tensor(0.0, device=h_out.device)

        # MetaWeighter KL divergence
        context = h_out.mean(dim=1)
        loss_dict['meta_kl'] = model.meta_weighter.kl_loss(context)

        # Distillation
        distill_val = model.distill_loss(h_out, heads_out)
        loss_dict['distill'] = distill_val

        # Weighted sum
        head_only_w = {
            'srg': train_cfg.w_srg,
            'srg_real': train_cfg.w_srg_real,
            'attn_entropy': train_cfg.w_attn_entropy,
            'head_consistency': train_cfg.w_head_consistency,
            'self_distill': train_cfg.w_self_distill,
        }
        total = (train_cfg.w_concept * loss_dict['concept'] +
                 train_cfg.w_contra * loss_dict['contradiction'] +
                 train_cfg.w_uncertainty * loss_dict['uncertainty'] +
                 train_cfg.w_boundary * loss_dict['boundary'] +
                 train_cfg.w_boundary_valid * loss_dict['boundary_valid'] +
                 train_cfg.w_residual * loss_dict['residual'] +
                 train_cfg.w_distill * distill_val +
                 train_cfg.w_nxt * loss_dict['nxt'] +
                 train_cfg.w_kca * loss_dict['kca'] +
                 train_cfg.w_meta_kl * loss_dict['meta_kl'])
        for k, w in head_only_w.items():
            total = total + w * loss_dict.get(k, torch.tensor(0.0, device=h_out.device))

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
        optimizer.step()

        if train_cfg.use_weight_context and train_step % train_cfg.update_weight_every == 0:
            model.update_weight_token()

        total_train_loss += total.item()

    cycle_metrics['train_loss'] = total_train_loss / max(runtime_cfg.train_steps_per_cycle, 1)

    # Save TrajectoryStore
    if trajectory_store is not None and runtime_cfg.trajectory_store_path:
        trajectory_store.save(runtime_cfg.trajectory_store_path)

    model.eval()
    cycle_metrics['trajectories'] = trajectory_store.total_stored if trajectory_store else 0

    return cycle_metrics


# === Convenience: multi-cycle runner ===

def continuous_learning_loop(model, cv, optimizer, loss_fn, n_cycles: int = 100,
                             runtime_cfg: RuntimeConfig = None,
                             thought_cfg: ThoughtLoopConfig = None,
                             train_cfg: TrainingConfig = None,
                             knn_retriever=None,
                             data_source=None,
                             log_fn=print,
                             trajectory_store_path: str = ""):
    """
    Запускает N циклов continuous learning.
    """
    if runtime_cfg is None:
        runtime_cfg = RuntimeConfig()
    if trajectory_store_path:
        runtime_cfg.trajectory_store_path = trajectory_store_path

    results = []
    for cycle in range(n_cycles):
        t0 = time.time()
        report = run_cycle(
            model, cv, optimizer, loss_fn,
            runtime_cfg, thought_cfg, train_cfg,
            knn_retriever, data_source,
        )
        elapsed = time.time() - t0

        if cycle % runtime_cfg.log_every == 0 and log_fn:
            log_fn(
                f"[Cycle {cycle+1}/{n_cycles}] "
                f"prompt={report['prompt'][:30]} "
                f"best={report['best_composite']:.3f} "
                f"avg={report['avg_composite']:.3f} "
                f"train_loss={report['train_loss']:.4f} "
                f"thought_iters={report['thought_iters']:.1f} "
                f"traj={report['trajectories']} "
                f"({elapsed:.1f}s)"
            )

        results.append(report)

    return results
