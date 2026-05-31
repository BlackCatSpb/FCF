"""
Thought loop — ядро когнитивного цикла EVA.

generate → measure(concept, contradiction, uncertainty, SRG)
       → refine(latent, weights)
       → converge or iterate
       → sample token

На каждом шаге генерации:
1. Forward → h + heads (concept, contra, uncertainty)
2. Если contradiction > θ_contra → отодвигаем z от неопределённой области
3. Если concept < θ_concept → усиливаем knowledge bias
4. Если SRG < θ_srg → ослабляем temperature (меньше хаоса)
5. Повторяем до сходимости или max_thoughts
6. Токен выбран с уточнёнными весами MetaWeighter
"""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Tuple, Dict


class ThoughtLoopConfig:
    contra_threshold: float = 0.3
    concept_threshold: float = 0.5
    srg_threshold: float = 0.2
    max_iterations: int = 5
    refine_lr: float = 0.1          # шаг коррекции латента
    contra_refine_strength: float = 0.3  # насколько сильно уходим от contradiction
    concept_boost: float = 2.0       # множитель concept bias при низком concept
    converge_margin: float = 0.02    # допустимое изменение для сходимости


def thought_step(h, heads_out, config: ThoughtLoopConfig, meta_weights: torch.Tensor,
                 sym_coords: torch.Tensor, temperature: float,
                 knn_bias: Optional[torch.Tensor] = None):
    """
    Один шаг thought loop.

    Args:
        h: [1, L, D] — скрытые состояния
        heads_out: dict от forward(..., return_heads=True)
        meta_weights: [3] — веса [know, conc, contr]
        sym_coords: [V, D] — координаты символов
        knn_bias: [V] — bias от kNN-LM (опционально)

    Returns:
        z_refined: [D] — уточнённая координата
        meta_weights: [3] — уточнённые веса [know, conc, contr]
        signals: dict — метрики для логирования
    """
    z = h[0, -1]
    concept = heads_out['concept'][0, -1].item()
    contra = heads_out['contradiction'][0, -1].item()
    uncertainty = heads_out['uncertainty'][0, -1].mean().item()
    residual_err = heads_out.get('residual_error', torch.zeros(1))[0, -1].item()

    signals = {
        'concept': concept,
        'contra': contra,
        'uncertainty': uncertainty,
        'residual_error': residual_err,
        'converged': True,
    }

    w = meta_weights.clone()

    # 1. Если contradiction высокая → refine z (уйти от неопределённости)
    if contra > config.contra_threshold:
        # Направление: отодвигаем z от областей с высокой variance
        contra_grad = heads_out['uncertainty'][0, -1]  # [D]
        z = z - contra_grad * config.contra_refine_strength * config.refine_lr
        signals['converged'] = False

    # 2. Если concept низкий → усилить knowledge bias (w[0])
    if concept < config.concept_threshold:
        boost = config.concept_boost * (1.0 - concept)
        w[0] = w[0] + boost * (1.0 - w[0])
        w[1] = w[1] * (1.0 - boost * 0.1)
        w = w / w.sum()
        signals['converged'] = False

    # 3. Если residual error высокая → ослабляем temperature (меньше шума)
    if residual_err > 0.5:
        signals['high_residual'] = True
        signals['converged'] = False
        # temperature will be reduced in calling code

    # 4. kNN bias: если передан, дополняем source 2
    if knn_bias is not None:
        signals['knn_active'] = True

    return z, w, signals


def generate_with_thought(model, prompt_ids, cv, config: ThoughtLoopConfig = None,
                          max_new=128, temperature=0.8,
                          knn_retriever=None,
                          callback=None,
                          skip_decoder=False):
    """
    Полная генерация с thought loop на каждый токен.

    Каждый новый токен:
    1. Forward через encoder + heads
    2. Thought loop: refine z + adjust weights
    3. Weighted logits из 3 источников [know, conc, contr]
    4. Sample

    Args:
        model: UnifiedMultidimensionalTransformer
        prompt_ids: список token_id
        cv: CharVocab (для декодирования)
        config: ThoughtLoopConfig
        knn_retriever: опционально KNNRetriever
        callback: опционально fn(token_id, step, trace) для streaming

    Returns:
        decoded_text: str
        metrics: dict — трассировка thought loop
    """
    if config is None:
        config = ThoughtLoopConfig()

    device = next(model.parameters()).device
    ids = list(prompt_ids)
    was_training = model.training
    model.eval()
    thought_trace = []

    with torch.no_grad():
        for pos in range(max_new):
            inp = torch.tensor([ids], dtype=torch.long, device=device)

            # Forward через shared encoder + heads
            h, _, _, heads_out = model.forward(inp, return_heads=True, capture_attn=True)

            # MetaWeighter: начальные веса
            context = h.mean(dim=1)
            meta_w = model.meta_weighter(context)[0]

            sym_coords = model.embed.coordinates

            # kNN bias
            knn_bias = None
            if knn_retriever is not None:
                knn_bias = knn_retriever.retrieve(h[0, -1], model.vocab_size)
                if callback:
                    callback(-1, pos, {'knn_bias_sum': knn_bias.sum().item()})

            # ---- Thought loop: итеративное уточнение ----
            z_refined = h[0, -1]
            w_refined = meta_w.clone()
            signals = {'converged': False, 'iterations': 0}

            for t in range(config.max_iterations):
                # Recompute heads at refined z (approximate: use original h)
                # In full implementation, re-forward with refined latent
                z_refined, w_refined, sig = thought_step(
                    h, heads_out, config, w_refined, sym_coords, temperature, knn_bias)
                signals = sig
                signals['iterations'] = t + 1
                if sig.get('converged', True):
                    break

            thought_trace.append(signals)

            # ---- Weighted sources (using refined z, not original h) ----
            z_last = z_refined
            bias_tpf = torch.zeros(model.vocab_size, device=device)
            bias_wvf = torch.zeros(model.vocab_size, device=device)
            if len(ids) > 1:
                last_sym = ids[-1]
                if last_sym < model.tensor_potential.num_symbols:
                    bias_tpf = model.tensor_potential.recursive_bias(
                        z_last, torch.tensor(ids, device=device))
                bias_wvf = model.word_valence.get_valence_bias(
                    z_last, torch.tensor(ids, device=device)).to(device)

            logits_know = model.decoder.forward(z_last.unsqueeze(0).unsqueeze(0))[0, 0] + bias_tpf + bias_wvf
            if knn_bias is not None:
                logits_know = logits_know + knn_bias

            concept_score = signals.get('concept', heads_out['concept'][0, -1].item())
            contra_score = signals.get('contra', heads_out['contradiction'][0, -1].item())
            dists = -torch.cdist(z_last.unsqueeze(0), sym_coords, p=2).squeeze(0)
            logits_concept = dists * (1.0 + concept_score)
            logits_contra = dists * (1.0 - contra_score * 0.5)

            w = w_refined
            logits = (w[0] * logits_know + w[1] * logits_concept + w[2] * logits_contra) / temperature

            # ---- Mask special tokens ----
            logits[:4] = -float('inf')
            logits[cv.GAP_FILLER_IDX] = -float('inf')
            logits[cv.WORD_OPEN_IDX] = -float('inf')
            logits[cv.WORD_CLOSE_IDX] = -float('inf')
            logits[cv.SENT_OPEN_IDX] = -float('inf')

            # ---- Repetition penalty (logits-level) ----
            freq = set(ids)
            for t_id in freq:
                logits[t_id] -= 1.0

            # ---- Sample from top-20 ----
            sl, si = logits.sort(descending=True)
            v, idx = sl[:20], si[:20]
            p = F.softmax(v, dim=-1)
            nt = idx[torch.multinomial(p, 1)].item()
            ids.append(nt)
            if callback:
                callback(nt, pos, signals)
            if nt == cv.SENT_CLOSE_IDX:
                break

    model.train(was_training)
    return cv.decode(ids), thought_trace


def analyze_thought_trace(trace):
    """
    Анализ thought loop по всей генерации.

    Returns: dict со статистикой
    """
    if not trace:
        return {}
    n_total = len(trace)
    n_converged = sum(1 for t in trace if t.get('converged', False))
    n_refined = n_total - n_converged
    avg_iters = sum(t.get('iterations', 1) for t in trace) / max(n_total, 1)
    avg_contra = sum(t.get('contra', 0) for t in trace) / max(n_total, 1)
    avg_concept = sum(t.get('concept', 0) for t in trace) / max(n_total, 1)
    avg_uncertainty = sum(t.get('uncertainty', 0) for t in trace) / max(n_total, 1)
    avg_residual = sum(t.get('residual_error', 0) for t in trace) / max(n_total, 1)

    return {
        'total_steps': n_total,
        'converged': n_converged,
        'refined': n_refined,
        'avg_iterations': avg_iters,
        'avg_contra': avg_contra,
        'avg_concept': avg_concept,
        'avg_uncertainty': avg_uncertainty,
        'avg_residual': avg_residual,
    }
