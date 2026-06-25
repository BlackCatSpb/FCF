"""STDPTrainer — training methods separated from CrystalGenerator.
Extracts STDP, negative sampling, contrastive, centroid pull, evaluate.
"""

import math
import numpy as np
import torch
import json, os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Set

# G-48: torch.compile support
_HAS_COMPILE = hasattr(torch, 'compile')

_META_I = 0
_META_J = 1
_META_PMI = 2
_META_DW = 3
_META_FW = 4
_META_FIELD_W = 5
_META_SLOW = 6
_META_PREV_CID = 7
_META_NEXT_CID = 8
_META_ANTONYM = 9


from eva.symbolic.fcf_config import EnvironmentResolver, FCFConfig

# P1.8: Антоним-словарь из JSON с fallback на хардкод
_ANTONYM_PATH = EnvironmentResolver().antonym_path

def _load_antonym_map(path=_ANTONYM_PATH):
    """Загрузить антоним-словарь из JSON. При отсутствии — минимальный fallback."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k.lower(): [v.lower() for v in vals] for k, vals in data.items()}
        except Exception:
            pass
    return {
        'быстрый': ['медленный'], 'медленный': ['быстрый'],
        'хороший': ['плохой'], 'плохой': ['хороший'],
        'высокий': ['низкий'], 'низкий': ['высокий'],
        'большой': ['маленький'], 'маленький': ['большой'],
        'да': ['нет'], 'нет': ['да'],
    }

_ANTONYM_MAP = _load_antonym_map()
_ANTONYM_RELOAD_COUNTER = 0
_ANTONYM_RELOAD_EVERY = 100  # reload every 100 calls to _build_pairs

def _reload_antonym_map():
    """Periodically reload antonym map from JSON (P1.6)."""
    global _ANTONYM_MAP
    if os.path.exists(_ANTONYM_PATH):
        fresh = _load_antonym_map()
        if len(fresh) > len(_ANTONYM_MAP):
            _ANTONYM_MAP = fresh


def _update_hdc_ngrams(cs, ids, max_n=3):
    """P1.5: Batch HDC deduplication — group by prefix, update only unique prefixes."""
    if not hasattr(cs.fractal, 'hdc_memory') or not cs.fractal.hdc_memory_max:
        return
    codes = {}
    for cid in ids:
        code = cs.fractal.codes.get(cid)
        if code is not None:
            codes[cid] = code
    if len(codes) < 2:
        return
    # Deduplicate: collect all next_codes per prefix
    updates = {}
    for n in range(2, max_n + 1):
        for i in range(len(ids) - n + 1):
            ngram = ids[i:i + n]
            if not all(cid in codes for cid in ngram):
                continue
            prefix = tuple(ngram[:-1])
            if prefix not in updates:
                updates[prefix] = []
            updates[prefix].append(codes[ngram[-1]])
    # Update each prefix once with averaged next_code
    memory_counts = cs.fractal.hdc_memory_counts if hasattr(cs.fractal, 'hdc_memory_counts') else {}
    for prefix, next_codes in updates.items():
        count = memory_counts.get(prefix, 0)
        if count > 50:
            continue  # skip well-learned prefixes
        avg_code = np.mean(next_codes, axis=0)
        avg_norm = np.linalg.norm(avg_code)
        if avg_norm > 1e-10:
            avg_code /= avg_norm
        cs.fractal.hdc_update_ngram(list(prefix), avg_code)


class STDPTrainer:
    """STDP training logic for CrystalGenerator.
    Operates on gen.cs, gen.lattice, gen._vecs_t, gen.hormones, gen.concept_error.
    """
    def __init__(self, gen, subspace_lr=None):
        self.gen = gen
        self.subspace_lr = subspace_lr
        from eva.symbolic.fcf_config import FCFConfig
        from eva.symbolic.transition_manifold import TransitionManifold
        _c = FCFConfig()
        if _c.beam_buffer_size > 0:
            self.manifold = TransitionManifold(
                dim=_c.beam_dim or gen.cs.dim,
                buffer_size=_c.beam_buffer_size,
                cos_threshold=_c.beam_cos_threshold,
                max_beams=_c.beam_max,
                rebuild_interval=_c.beam_rebuild_interval,
                eps=_c.beam_eps,
            )
        else:
            self.manifold = None


    # ═══════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════

    def _unwrap(self, val, key, fallback):
        if val is not None:
            return val
        from eva.symbolic.fcf_config import FCFConfig
        return getattr(FCFConfig(), key, fallback)

    def train_from_text(self, text, base_lr=None, context_window=None, pmi_strength=None,
                         pmi_gate_min=None, neg_samples=None, inh_strength=None, inh_threshold=None,
                         neg_lr_ratio=None, field_gate=True, use_torch=None, destab_scale=None,
                         momentum_mu=None, gradient_noise_scale=None, fluctuation_amp=None):
        cf = lambda k, f: self._unwrap(None, k, f)  # noqa
        context_window = cf('context_window', 2) if context_window is None else context_window
        pmi_strength = cf('pmi_strength', 1.0) if pmi_strength is None else pmi_strength
        pmi_gate_min = cf('pmi_gate_min', 0.20) if pmi_gate_min is None else pmi_gate_min
        neg_samples = cf('neg_samples', 1) if neg_samples is None else neg_samples
        inh_strength = cf('inh_strength', 0.05) if inh_strength is None else inh_strength
        inh_threshold = cf('inh_threshold', 0.10) if inh_threshold is None else inh_threshold
        neg_lr_ratio = cf('neg_lr_ratio', 0.5) if neg_lr_ratio is None else neg_lr_ratio
        destab_scale = cf('destab_scale', 0.0) if destab_scale is None else destab_scale
        momentum_mu = cf('momentum_mu', 0.9) if momentum_mu is None else momentum_mu
        fluctuation_amp = cf('fluctuation_amp', 0.003) if fluctuation_amp is None else fluctuation_amp
        n = self._train(text, base_lr, context_window, pmi_strength, pmi_gate_min,
                        neg_samples, inh_strength, inh_threshold, neg_lr_ratio,
                        field_gate, use_torch, destab_scale, batch=False,
                        momentum_mu=momentum_mu, gradient_noise_scale=gradient_noise_scale)
        self.gen._sync_dirty_cpu()
        return n

    def train_batch(self, texts, base_lr=None, context_window=None, pmi_strength=None,
                     pmi_gate_min=None, neg_samples=None, inh_strength=None, inh_threshold=None,
                     neg_lr_ratio=None, field_gate=True, use_torch=None, destab_scale=None,
                     momentum_mu=None, gradient_noise_scale=None, fluctuation_amp=None):
        cf = lambda k, f: self._unwrap(None, k, f)
        context_window = context_window if context_window is not None else cf('context_window', 2)
        pmi_strength = pmi_strength if pmi_strength is not None else cf('pmi_strength', 1.0)
        pmi_gate_min = pmi_gate_min if pmi_gate_min is not None else cf('pmi_gate_min', 0.20)
        neg_samples = neg_samples if neg_samples is not None else cf('neg_samples', 1)
        inh_strength = inh_strength if inh_strength is not None else cf('inh_strength', 0.05)
        inh_threshold = inh_threshold if inh_threshold is not None else cf('inh_threshold', 0.10)
        neg_lr_ratio = neg_lr_ratio if neg_lr_ratio is not None else cf('neg_lr_ratio', 0.5)
        destab_scale = destab_scale if destab_scale is not None else cf('destab_scale', 0.0)
        momentum_mu = momentum_mu if momentum_mu is not None else cf('momentum_mu', 0.9)
        fluctuation_amp = fluctuation_amp if fluctuation_amp is not None else cf('fluctuation_amp', 0.003)
        """Train on a batch of texts (GPU batched). Returns total pairs."""
        n = self._train(texts, base_lr, context_window, pmi_strength, pmi_gate_min,
                        neg_samples, inh_strength, inh_threshold, neg_lr_ratio,
                        field_gate, use_torch, destab_scale, batch=True,
                        momentum_mu=momentum_mu, gradient_noise_scale=gradient_noise_scale)
        self.gen._sync_dirty_cpu()
        return n

    def evaluate(self, corpus_path, max_lines=500, use_torch=None):
        """Evaluate on a corpus: perplexity, accuracy, vector metrics."""
        return self._evaluate(corpus_path, max_lines, use_torch)

    # ═══════════════════════════════════════════════════
    # Internal: train
    # ═══════════════════════════════════════════════════

    def _train(self, inputs, base_lr, context_window, pmi_strength, pmi_gate_min,
               neg_samples, inh_strength, inh_threshold, neg_lr_ratio, field_gate,
               use_torch, destab_scale, batch, momentum_mu=0.9, gradient_noise_scale=0.0):
        gen = self.gen
        cs = gen.cs
        if base_lr is None:
            base_lr = gen.base_learning_rate
        if use_torch is None:
            use_torch = gen._use_torch

        if use_torch:
            gen._ensure_torch()

        if not batch:
            inputs = [inputs]

        # Shared pair building structures
        gen_updates = {}
        gpu_ctx_l = []
        gpu_tgt_l = []
        gpu_meta_l = []
        gpu_cid_ctx = []
        gpu_cid_gen = []
        cid_to_idx = {}
        total_pairs = 0
        total_freq = gen._get_total_freq()
        all_ids = []  # cached encoded ids for centroid pull + lattice update

        for text in inputs:
            ids = gen._encode_input(text)
            if len(ids) < 2:
                continue
            all_ids.append(ids)

            if use_torch and gen._vecs_t is not None:
                for cid in ids:
                    if cid not in cid_to_idx and cid < gen._vecs_t.shape[0]:
                        cid_to_idx[cid] = cid

            for cid in ids:
                gen_updates.setdefault(cid, [])

            total_pairs += self._build_pairs(ids, context_window, total_freq,
                pmi_strength, pmi_gate_min, field_gate, base_lr,
                use_torch, cid_to_idx, gen_updates,
                gpu_ctx_l, gpu_tgt_l, gpu_meta_l, gpu_cid_ctx, gpu_cid_gen)

        if total_pairs == 0:
            return 0

        # ── GPU STDP / CPU STDP ──
        if use_torch and gpu_ctx_l:
            unique_gen = self._gpu_stdp_apply(gpu_ctx_l, gpu_tgt_l, gpu_meta_l, gpu_cid_gen,
                base_lr, field_gate, inh_strength, inh_threshold, destab_scale,
                gradient_noise_scale=gradient_noise_scale, momentum_mu=momentum_mu)
        else:
            self._cpu_stdp_apply(gen_updates, base_lr, destab_scale, inh_strength, inh_threshold)

        # ── Negative sampling + Contrastive (G-52: fused into single GPU pass) ──
        if use_torch and gpu_ctx_l:
            self._gpu_poststdp_fused(gpu_ctx_l, gpu_meta_l, gpu_cid_ctx, gpu_cid_gen,
                gen_updates, field_gate, base_lr, neg_lr_ratio, neg_samples)
        else:
            if neg_samples > 0:
                self._negative_sampling_cpu(gen_updates, neg_lr_ratio, field_gate, neg_samples)
            self._contrastive_objective(gen_updates, field_gate)

        # ── Centroid pull + lattice update (reuses cached ids from first loop) ──
        self._centroid_pull_batch(all_ids, base_lr)
        self._cluster_centroid_pull(all_ids, base_lr, pull_strength=0.05)
        for ids in all_ids:
            gen.lattice.update(ids)
            gen._graph_cache.clear()
            # HDC n-gram memory update
            if hasattr(cs.fractal, 'hdc_memory') and cs.fractal.W_proj is not None:
                _update_hdc_ngrams(cs, ids, max_n=3)

        # _torch_dirty is NOT set here — would force full tensor rebuild every batch
        # (_build_torch_tensors iterates all 146K codes, O(V·D) CPU + 636MB PCIe xfer)
        # Only _invalidate_torch() (after fluctuate) should set it.

        # ── Morphological harmonization (Phase 3) ──
        self._harmonize_batch(gen, cs, all_ids)

        # Minesweeper (inverted): periodic cluster potential refresh
        if gen._cluster_update_counter > 0 and gen._cluster_update_counter % gen._cluster_update_every == 0:
            gen._update_cluster_potential()
        gen._cluster_update_counter += 1

        return total_pairs

    def _harmonize_batch(self, gen, cs, all_ids):
        if not hasattr(cs, 'harmonizer') or not cs.harmonizer.word_morphs:
            return
        harm = cs.harmonizer
        ef = getattr(cs, 'entity_field', None)
        if ef is None:
            return

        focus_cids = set()
        for ids in all_ids:
            focus_cids.update(ids)
        if gen._dirty_cids:
            focus_cids.update(gen._dirty_cids)

        morph_cids = [c for c in focus_cids if c in harm.word_morphs]
        if not morph_cids and not focus_cids:
            return
        ef.clear_bind_cache()
        # P1.7: periodic entity field cleanup
        ef._entity_batch_counter += 1
        if ef._entity_batch_counter % 100 == 0 and len(ef.entities) > ef._max_entities:
            ef.cleanup()

        # ── 1. Sync GPU→CPU + sync word vectors into entity_field ──
        all_cids = list(focus_cids | set(morph_cids))
        updated_cids = []
        if gen._use_torch and gen._vecs_t is not None:
            cids_t = torch.tensor(all_cids, dtype=torch.long, device=gen._torch_device)
            vecs_cpu = gen._vecs_t[cids_t].cpu().numpy()
            gen._skip_gpu_sync = True
            try:
                for cid, v_new in zip(all_cids, vecs_cpu):
                    cs._apply_vector_update(cid, v_new)
                    ef.sync_word(cid, v_new)
                    updated_cids.append(cid)
            finally:
                gen._skip_gpu_sync = False
                # ══ P1.13: batched GPU write instead of individual _on_vector_update ══
                if gen._vecs_t is not None and updated_cids:
                    batch_v = np.stack([cs.concept_vectors[cid] for cid in updated_cids])
                    batch_t = torch.from_numpy(batch_v).to(device=gen._vecs_t.device, dtype=gen._vecs_t.dtype, non_blocking=True)
                    gen._vecs_t[torch.tensor(updated_cids, device=gen._vecs_t.device)] = batch_t
        else:
            for cid in all_cids:
                v = cs.concept_vectors.get(cid)
                if v is not None:
                    ef.sync_word(cid, v)

        for cid in morph_cids:
            harm.mark_word_dirty(cid)

        # ── 2. Cross-level bindings: char↔word, word↔sent, sent↔para ──
        # P2.6: precompute per-sentence vectors for reuse in step 3
        sent_vec_cache = {}
        for ids in all_ids:
            if not ids:
                continue

            # ---- 2a. char ↔ word ----
            for cid in ids:
                word_text = None
                if hasattr(gen, 'sp') and gen.sp is not None:
                    try:
                        word_text = gen.sp.IdToPiece(int(cid)).replace('\u2581', ' ').strip()
                    except Exception:
                        pass
                if word_text and len(word_text) >= 1:
                    for ch in word_text:
                        cp = ord(ch)
                        ef.bind('c', cp, 'w', cid, lr=0.05)
                        ef.bind('w', cid, 'c', cp, lr=0.05)

            # ---- 2b. word → sent ----
            sent_key = hash(tuple(ids))
            skey = ef.key_sent(sent_key)
            sv = ef.get(skey)
            if sv is None:
                codes = []
                for cid in ids:
                    wv = ef.get(ef.key_word(cid))
                    if wv is not None:
                        codes.append(wv)
                if len(codes) >= 2:
                    sv = cs.fractal.hdc_ngram_repr(codes)
                    if sv is not None:
                        ef.set(skey, sv)
                else:
                    sv = ef.ensure(skey)

            if sv is not None:
                for cid in ids:
                    ef.bind('w', cid, 's', sent_key, lr=0.03)
                ef.bind('s', sent_key, 'w', ids[0], lr=0.01)

            # ---- 2c. sent → para (P2.3) ----
            para_key = getattr(gen, '_current_para_key', None)
            if para_key is not None:
                pkey = ef.key_para(para_key)
                pv = ef.get(pkey)
                if pv is None:
                    pv = ef.ensure(pkey)
                ef.bind('s', sent_key, 'p', para_key, lr=0.02)
                ef.bind('p', para_key, 's', sent_key, lr=0.01)

            # P2.6: cache sent_vec built from concept vectors (768D) for morpheme harmonisation
            sv_768 = cs.fractal.hdc_ngram_repr([cs.concept_vectors.get(c) for c in ids if cs.concept_vectors.get(c) is not None])
            if sv_768 is not None:
                sent_vec_cache[sent_key] = sv_768

        # ── 3. Morpheme harmonisation ──
        dirty_words = list(harm.word_dirty)
        for cid in dirty_words:
            v = cs.concept_vectors.get(cid)
            if v is not None:
                sv = None
                for ids in all_ids:
                    if cid in ids:
                        sent_key = hash(tuple(ids))
                        sv = sent_vec_cache.get(sent_key)
                        break
                new_v, delta = harm.harmonize(cid, v, sent_vec=sv)
                if new_v is not None:
                    cs._apply_vector_update(cid, new_v)
                    ef.sync_word(cid, new_v)
                    updated_cids.append(cid)

        # ── 3b. P1.3: EntityField → STDP feedback (error_clip + momentum) ──
        if morph_cids:
            proj = getattr(ef, '_proj', None)
            for cid in morph_cids:
                wkey = ef.key_word(cid)
                v_word = ef.get(wkey)
                if v_word is not None:
                    char_query = ef.query('w', cid)
                    if char_query is not None and proj is not None:
                        # Project 2048D query → 768D concept space
                        char_query_768 = proj.T @ char_query
                        cqn = np.linalg.norm(char_query_768)
                        if cqn > 1e-10:
                            char_query_768 /= cqn
                            v_cs = cs.concept_vectors.get(cid)
                            if v_cs is not None:
                                sim = float(v_cs @ char_query_768)
                                error = char_query_768 - sim * v_cs
                                error_clipped = np.clip(error, -0.1, 0.1)
                                ce = gen.concept_error.get(cid, 0.0)
                                pull_strength = 0.001 * max(0.1, 1.0 - ce * 2.0)
                                pull = error_clipped * pull_strength
                                v_new = v_cs + pull
                                nv = np.linalg.norm(v_new)
                                if nv > 1e-10:
                                    v_new /= nv
                                cs._apply_vector_update(cid, v_new)
                                updated_cids.append(cid)
                                # Clear char cache after feedback
                                ef._char_word_cache.clear()

        # ── 4. Batched GPU sync for harmonize updates (P1.13) ──
        if gen._use_torch and gen._vecs_t is not None and updated_cids:
            harmonize_cids = [c for c in dirty_words if c in updated_cids]
            if harmonize_cids:
                batch_v = np.stack([cs.concept_vectors[cid] for cid in harmonize_cids])
                batch_t = torch.from_numpy(batch_v).to(device=gen._vecs_t.device, dtype=gen._vecs_t.dtype, non_blocking=True)
                gen._vecs_t[torch.tensor(harmonize_cids, device=gen._vecs_t.device)] = batch_t

        # ── 5. Cleanup ──
        if gen._dirty_cids:
            gen._dirty_cids.difference_update(harm.word_dirty)
        harm.clear_dirty()

    # ═══════════════════════════════════════════════════
    # Pair building
    # ═══════════════════════════════════════════════════

    def _build_pairs(self, ids, context_window, total_freq, pmi_strength, pmi_gate_min,
                     field_gate, base_lr, use_torch, cid_to_idx,
                     gen_updates, gpu_ctx_l, gpu_tgt_l, gpu_meta_l,
                     gpu_cid_ctx, gpu_cid_gen):
        """Build STDP pairs for one sentence. Shared CPU/GPU pair generation."""
        global _ANTONYM_RELOAD_COUNTER
        _ANTONYM_RELOAD_COUNTER += 1
        if _ANTONYM_RELOAD_COUNTER % _ANTONYM_RELOAD_EVERY == 0:
            _reload_antonym_map()
        gen = self.gen
        cs = gen.cs
        T = len(ids)
        n_pairs = 0
        _fc = FCFConfig().formula

        # GPU path: pre-gather frequency/error tensors for O(1) per-pair lookups
        use_gpu_freq = use_torch and gen._cf_t is not None
        if use_gpu_freq:
            ids_t = torch.tensor(ids, dtype=torch.long, device=gen._torch_device)
            _cf_arr = gen._cf_t[ids_t].cpu().numpy()
            _pt2_arr = gen._pt2_t[ids_t].cpu().numpy()
            _skip2_arr = gen._skip2_t[ids_t].cpu().numpy()
            _ce_arr = gen._ce_t[ids_t].cpu().numpy()
            _ngrams2_dict = gen.lattice.ngrams[2]
            _skip2_dict = gen.lattice.skip2
            _total_freq_gpu = gen._total_freq_t.item()
        else:
            _cf_arr = None

        # G-65/SN-48: Pre-compute field overlap matrix on GPU (one batch, no per-pair .item())
        _overlap_lookup = None
        if use_torch and field_gate > 0 and gen._fb_t is not None:
            ids_t = torch.tensor(ids, dtype=torch.long, device=gen._torch_device)
            fb_t = gen._fb_t[ids_t]
            overlap_mat = (fb_t.unsqueeze(1) & fb_t.unsqueeze(0)).sum(dim=-1).cpu().numpy()
            _overlap_lookup = lambda i, j: int(overlap_mat[i, j])

        for i in range(T):
            start = max(0, i - context_window)
            end = min(T, i + context_window + 1)
            for j in range(start, end):
                if j <= i:
                    continue
                dist = abs(j - i)
                dist_weight = math.exp(-dist / 2.0)

                if use_gpu_freq:
                    fa = _cf_arr[i]
                    fb = _cf_arr[j]
                    freq_weight = 1.0 / (1.0 + math.log(max(max(fa, fb), 1)) * _fc.freq_weight_log_scale)
                    # Inline PMI with pre-gathered GPU tensors + minimal CPU dict for sparse ngrams
                    if dist == 1:
                        counter = _ngrams2_dict.get((ids[i],))
                        count_pair = counter.get(ids[j], 0) if counter else 0
                        count_prev = _pt2_arr[i]
                    elif dist == 2:
                        inner = _skip2_dict.get(ids[i])
                        count_pair = inner.get(ids[j], 0) if inner else 0
                        count_prev = _skip2_arr[i]
                    else:
                        count_pair = 0
                        count_prev = 0
                    count_next = _cf_arr[j]
                    if count_pair > 0 and count_prev > 0 and count_next > 0:
                        p_next_given_prev = count_pair / count_prev
                        p_next = count_next / _total_freq_gpu
                        pmi_w_raw = math.log(max(p_next_given_prev, 1e-10) / max(p_next, 1e-10))
                        pmi_w_raw = max(min(pmi_w_raw / 2.0 + 0.2, 2.0), pmi_gate_min)
                    else:
                        pmi_w_raw = 0.1
                    # Inline PMI gate with GPU _ce_arr
                    if pmi_strength >= 0.01:
                        _use_min = min(pmi_gate_min * pmi_strength,
                                       pmi_gate_min * max(0.25, 1.0 - _ce_arr[j] * 0.75))
                        if pmi_w_raw <= _use_min:
                            continue
                        pmi_w = 1.0 + (pmi_w_raw - 1.0) * pmi_strength
                    else:
                        pmi_w = 1.0
                else:
                    fa = gen.lattice.concept_freq.get(ids[i], 0)
                    fb = gen.lattice.concept_freq.get(ids[j], 0)
                    freq_weight = 1.0 / (1.0 + math.log(max(max(fa, fb), 1)) * _fc.freq_weight_log_scale)
                    pmi_w_raw = gen._pmi_weight(ids[i], ids[j], distance=dist,
                                                  min_weight=pmi_gate_min)
                    _skip, pmi_w = gen._apply_pmi_gate(pmi_w_raw, pmi_strength, pmi_gate_min, ids[j])
                    if _skip:
                        continue

                field_weight = 1.0
                if field_gate > 0:
                    if _overlap_lookup is not None:
                        overlap = _overlap_lookup(i, j)
                    elif hasattr(cs.fractal, 'field_bits') and len(cs.fractal.field_bits) > 0:
                        overlap = cs.fractal.field_overlap(ids[i], ids[j])
                    else:
                        overlap = 0
                    fw = min(1.0 + math.log(overlap + 1) * _fc.field_weight_log_scale, _fc.field_weight_cap) if overlap > 0 else _fc.field_weight_floor
                    field_weight = 1.0 + (fw - 1.0) * field_gate

                lr = base_lr * max(freq_weight, _fc.freq_weight_min) * pmi_w * field_weight
                lr *= (_fc.hormonal_mod_baseline + gen.hormones.acetylcholine * _fc.hormonal_mod_scale) * (_fc.hormonal_mod_baseline + gen.hormones.dopamine * _fc.hormonal_mod_scale)
                theta_gate = math.exp(-min(abs(j-i), 5) / max(gen.theta_tau, 1.0))
                gen_updates[ids[j]].append((ids[i], lr * max(theta_gate, _fc.theta_fast_min)))
                n_pairs += 1
                theta_slow = math.exp(-min(abs(j-i), 10) / max(gen.theta_tau * _fc.theta_tau_slow_mult, 1.0))
                slow_lr = lr * max(theta_slow, _fc.theta_slow_min) * _fc.theta_slow_scale
                if slow_lr > 1e-6:
                    gen_updates[ids[j]].append((ids[i], slow_lr))
                    n_pairs += 1

                if use_torch:
                    ci = cid_to_idx.get(ids[i])
                    cj = cid_to_idx.get(ids[j])
                    if ci is None or cj is None:
                        continue
                    gpu_ctx_l.append(ci)
                    gpu_tgt_l.append(cj)

                    # Antonym check: decode BPE tokens, compare against ANTONYM_MAP
                    antonym_flag = 0.0
                    if hasattr(gen.sp, 'IdToPiece'):
                        if not hasattr(self, '_cid_text_cache'):
                            self._cid_text_cache = {}
                        if ids[i] not in self._cid_text_cache:
                            try:
                                self._cid_text_cache[ids[i]] = gen.sp.IdToPiece(ids[i]).replace('\u2581', ' ').strip().lower()
                            except Exception:
                                self._cid_text_cache[ids[i]] = ''
                        if ids[j] not in self._cid_text_cache:
                            try:
                                self._cid_text_cache[ids[j]] = gen.sp.IdToPiece(ids[j]).replace('\u2581', ' ').strip().lower()
                            except Exception:
                                self._cid_text_cache[ids[j]] = ''
                        ti = self._cid_text_cache[ids[i]]
                        tj = self._cid_text_cache[ids[j]]
                        if ti and tj:
                            if ti in _ANTONYM_MAP:
                                if any(ant in tj for ant in _ANTONYM_MAP[ti]):
                                    antonym_flag = 1.0
                            elif tj in _ANTONYM_MAP:
                                if any(ant in ti for ant in _ANTONYM_MAP[tj]):
                                    antonym_flag = 1.0

                    gpu_meta_l.append((i, j, pmi_w, dist_weight, freq_weight, field_weight, 0.0, ids[i], ids[j], antonym_flag))
                    gpu_cid_ctx.append(ids[i])
                    gpu_cid_gen.append(ids[j])
                    # SN-25: Add slow STDP pair to GPU lists (matches CPU slow_lr > 1e-6 gate)
                    theta_slow = math.exp(-min(abs(j-i), 10) / max(gen.theta_tau * 3.0, 1.0))
                    slow_lr = lr * max(theta_slow, 0.02) * 0.3
                    if slow_lr > 1e-6:
                        gpu_ctx_l.append(ci)
                        gpu_tgt_l.append(cj)
                        # Slow STDP also inherits antonym flag
                        gpu_meta_l.append((i, j, pmi_w, dist_weight, freq_weight, field_weight, 1.0, ids[i], ids[j], antonym_flag))
                        gpu_cid_ctx.append(ids[i])
                        gpu_cid_gen.append(ids[j])

        return n_pairs

    # ═══════════════════════════════════════════════════
    # Lattice-based semantic bootstrap
    # ═══════════════════════════════════════════════════
    def _semantic_bootstrap(self, cs, lattice, base_lr=0.05, k_pos=5, k_neg=10):
        """Use lattice PMI connections to bootstrap semantic vector space.

        For each seen token:
          - Pull toward top-PPMI neighbors (distributional similarity)
          - Push away from random unconnected tokens

        Returns number of updated tokens.
        """
        gen = self.gen
        from eva.symbolic.seed_registry import DEFAULT_REGISTRY as _R
        bs = getattr(self, '_bootstrap_seed', 0)
        rng = _R.rng('semantic_bootstrap' + '_' + str(bs))
        self._bootstrap_seed = bs + 1

        seen_cids = [int(c) for c, u in cs.concept_usage.items() if u > 0]
        if len(seen_cids) < 10:
            return 0

        rng.shuffle(seen_cids)
        updated = 0
        for cid in seen_cids[:max(200, len(seen_cids) // 2)]:
            v_anc = cs.concept_vectors.get(cid)
            if v_anc is None:
                continue

            # Positive: top PPMI neighbors from lattice connections
            conns = lattice.connections_of(cid, top_k=k_pos * 2, use_ppmi=True)
            conns = [(c, info) for c, info in conns if cs.concept_vectors.get(c) is not None]
            if not conns:
                continue

            pos_vecs = [cs.concept_vectors.get(c) for c, _ in conns[:k_pos]]
            pos_vecs = [v for v in pos_vecs if v is not None]
            if not pos_vecs:
                continue
            pos_mean = np.mean(pos_vecs, axis=0)
            pn = np.linalg.norm(pos_mean)
            if pn > 1e-10:
                pos_mean /= pn

            # Negative: random seen tokens not in connections
            conn_set = {c for c, _ in conns}
            neg_pool = [c for c in seen_cids if c not in conn_set and c != cid]
            if len(neg_pool) < k_neg:
                continue
            neg_sel = rng.choice(neg_pool, k_neg, replace=False).tolist()
            neg_vecs = [cs.concept_vectors.get(c) for c in neg_sel]
            neg_vecs = [v for v in neg_vecs if v is not None]
            if not neg_vecs:
                continue
            neg_mean = np.mean(neg_vecs, axis=0)
            nn = np.linalg.norm(neg_mean)
            if nn > 1e-10:
                neg_mean /= nn

            # Contrastive gradient (Riemannian tangent)
            cos_pos = float(v_anc @ pos_mean)
            cos_neg = max(float(v_anc @ neg_mean), -1.0)

            # Pull toward positives, push from negatives
            pull = pos_mean - cos_pos * v_anc        # tangent toward pos
            push = (v_anc - neg_mean * cos_neg) * 0.5  # tangent away from neg
            grad = pull * base_lr + push * base_lr
            gn = float(np.linalg.norm(grad))
            if gn > 0.3:
                grad = grad / gn * 0.3

            v_new = v_anc + grad
            nv = np.linalg.norm(v_new)
            if nv > 1e-10:
                v_new /= nv
            cs._apply_vector_update(cid, v_new, max_shift=0.3)
            updated += 1

        return updated

    # ═══════════════════════════════════════════════════
    # CPU STDP (AM-25: legacy, GPU preferred)
    # ═══════════════════════════════════════════════════

    # AM-25: CPU path — fallback when GPU unavailable. Kept for backward compat.
    def _cpu_stdp_apply(self, gen_updates, base_lr_val, destab_scale, inh_strength, inh_threshold):
        gen = self.gen
        cs = gen.cs
        for gen_cid, updates in gen_updates.items():
            v_gen = cs.concept_vectors.get(gen_cid)
            if v_gen is None or not updates:
                continue

            n_updates = len(updates)
            ctx_cids, elrs = zip(*updates)
            valid_ctx = []
            valid_elr = []
            for cid, elr in zip(ctx_cids, elrs):
                v = cs.concept_vectors.get(cid)
                if v is not None:
                    valid_ctx.append(v)
                    valid_elr.append(elr)

            if not valid_ctx:
                continue

            ctx_mat = np.array(valid_ctx, dtype=np.float32)
            elr_arr = np.array(valid_elr, dtype=np.float32)
            total_elr = float(elr_arr.sum())

            y = np.maximum(v_gen @ ctx_mat.T, 0.05)

            err = 1.0 - float(np.mean(y))
            gen.concept_error.update(gen_cid, err)

            total_delta = ((ctx_mat * elr_arr[:, None]).sum(axis=0) -
                          v_gen * (y * elr_arr).sum())

            # Transition Manifold: push переходы для каждой контекстной пары
            if self.manifold is not None:
                _eps_m = self.manifold._eps
                for vc, elr in zip(valid_ctx, valid_elr):
                    if elr > _eps_m:
                        T = self.manifold._vsa_transition(v_gen, vc)
                        if np.linalg.norm(T) > _eps_m:
                            self.manifold.push(T)

            if n_updates > 0 and total_elr > 0:
                grad = total_delta / max(total_elr, 1e-10)
                gn = float(np.linalg.norm(grad))
                if gn > gen.max_grad_norm > 0:
                    grad = grad / gn * gen.max_grad_norm

                if destab_scale > 0 and gen.main_rng.random() < destab_scale * 0.3:
                    ppmi_candidates = gen.lattice.connections_of(
                        gen_cid, top_k=20, use_ppmi=True)
                    if ppmi_candidates:
                        ppmi_cid = ppmi_candidates[gen.main_rng.randint(0, len(ppmi_candidates) - 1)][0]
                        v_ppmi = cs.concept_vectors.get(ppmi_cid)
                    else:
                        v_ppmi = gen._destab_field_fallback(gen_cid, v_gen)
                    if v_ppmi is not None:
                        y_ppmi = max(float(np.dot(v_gen, v_ppmi)), 0.05)
                        noise = (v_ppmi - y_ppmi * v_gen)
                        nlen = float(np.linalg.norm(noise))
                        if nlen > 1e-10:
                            noise /= nlen
                            mix = min(destab_scale, 0.5)
                            grad = grad * (1 - mix) + noise * mix

                if self.subspace_lr is not None and cs.fractal.basis is not None and cs.fractal.codes.get(gen_cid) is not None:
                    cs._apply_subspace_update(gen_cid, grad, base_lr_val, self.subspace_lr)
                else:
                    v_new = v_gen + grad * base_lr_val
                    # Beam pull: притяжение к ближайшему лучу
                    if self.manifold is not None:
                        cent, sim, _cnt = self.manifold.nearest_beam(v_new)
                        from eva.symbolic.fcf_config import FCFConfig as _FCfg
                        if cent is not None and sim > self.manifold.cos_threshold * _FCfg().beam_pull_sim_ratio:
                            v_new = v_new + cent * _FCfg().beam_pull_strength
                    nv = np.linalg.norm(v_new)
                    if nv > _FCfg().beam_eps:
                        v_new /= nv
                    cs._apply_vector_update(gen_cid, v_new)

        # Lateral inhibition
        if inh_strength > 0:
            self._lateral_inhibition_cpu(gen_updates, inh_strength, inh_threshold, base_lr_val)

    def _lateral_inhibition_cpu(self, gen_updates, inh_strength, inh_threshold, base_lr_val):
        gen = self.gen
        cs = gen.cs
        gen_cids = [gc for gc, upd in gen_updates.items() if sum(elr for _, elr in upd) > inh_threshold]
        if len(gen_cids) < 2:
            return
        sims = cs._batch_cosine(gen_cids)
        n = len(gen_cids)
        for gi in range(n):
            if not (sims[gi] > 0).any():
                continue
            mask = sims[gi] > inh_threshold * 2
            mask[gi] = False
            if not mask.any():
                continue
            inhibition = sims[gi][mask].sum()
            v = cs.concept_vectors.get(gen_cids[gi])
            if v is None:
                continue
            inhibit_vec = np.zeros_like(v)
            for gj in np.where(mask)[0]:
                v_other = cs.concept_vectors.get(gen_cids[gj])
                if v_other is not None:
                    inhibit_vec += (sims[gi][gj] * v_other - sims[gi][gj]**2 * v)
            norm = float(np.linalg.norm(inhibit_vec))
            if norm > 1e-10:
                inhibit_vec /= norm
                v_new = v + inhibit_vec * inh_strength * base_lr_val
                nv = np.linalg.norm(v_new)
                if nv > 1e-10:
                    v_new /= nv
                cs._apply_vector_update(gen_cids[gi], v_new)

    # ═══════════════════════════════════════════════════
    # GPU STDP
    # ═══════════════════════════════════════════════════

    # G-48/G-66: torch.compile candidate — pure-tensor core for CUDA Graph
    # Falls back to eager on unsupported hardware (2GB GPU, no Triton, etc.)
    def _gpu_stdp_core(self, ctx_t, tgt_t, meta_t, unique_gen, inv_t, gen, cs,
                       gradient_noise_scale=0.0):
        """Pure-tensor core of _gpu_stdp_apply. torch.compile-friendly."""
        _fc = FCFConfig().formula
        D = cs.dim
        N = len(ctx_t)
        ng = len(unique_gen)
        device = gen._torch_device

        i_pos = meta_t[:, _META_I]
        j_pos = meta_t[:, _META_J]
        dist = j_pos - i_pos

        if meta_t.shape[1] > _META_NEXT_CID and gen._cf_t is not None:
            prev_cid_t = meta_t[:, _META_PREV_CID].long()
            next_cid_t = meta_t[:, _META_NEXT_CID].long()
            fa_t = gen._cf_t[prev_cid_t]; fb_t = gen._cf_t[next_cid_t]
            fw_t = 1.0 / (1.0 + torch.log(torch.max(torch.stack([fa_t, fb_t]), dim=0).values.clamp(min=1)) * _fc.freq_weight_log_scale)
            pmi_w_t = meta_t[:, _META_PMI]; field_w_t = meta_t[:, _META_FIELD_W]
        else:
            pmi_w_t = meta_t[:, _META_PMI]; fw_t = meta_t[:, _META_FW]; field_w_t = meta_t[:, _META_FIELD_W]
        dw_t = torch.exp(-dist / 2.0)

        lr = torch.clamp(fw_t, min=_fc.freq_weight_min) * dw_t * pmi_w_t * field_w_t
        lr *= (_fc.hormonal_mod_baseline + gen.hormones.acetylcholine * _fc.hormonal_mod_scale) * (_fc.hormonal_mod_baseline + gen.hormones.dopamine * _fc.hormonal_mod_scale)
        if gen._cluster_potential is not None and gen._cluster_map is not None:
            lr *= gen._cluster_potential[gen._cluster_map[tgt_t]]
        dist_clamped = torch.clamp(dist, max=10.0)
        theta_fast = torch.exp(-dist_clamped.clamp(max=5.0) / max(gen.theta_tau, 1.0))
        theta_slow = torch.exp(-dist_clamped / max(gen.theta_tau * _fc.theta_tau_slow_mult, 1.0))
        slow_mask = meta_t[:, _META_SLOW] if meta_t.shape[1] > _META_SLOW else torch.zeros(N, device=device)
        theta = (1 - slow_mask) * torch.clamp(theta_fast, min=_fc.theta_fast_min) + slow_mask * torch.clamp(theta_slow, min=_fc.theta_slow_min) * _fc.theta_slow_scale
        effective_lr = lr * theta

        vc = gen._vecs_t[ctx_t].float(); vg = gen._vecs_t[tgt_t].float()
        y = torch.clamp((vg * vc).sum(dim=1), min=0.05)
        pair_delta = vc * effective_lr[:, None] - vg * (y * effective_lr)[:, None]

        # Antonym repel: если пара — антонимы, разворачиваем градиент (push apart)
        if meta_t.shape[1] > _META_ANTONYM:
            antonym_mask = meta_t[:, _META_ANTONYM] > 0.5  # (N,) bool
            if antonym_mask.any():
                # Удвоенная сила отталкивания: -2 * pair_delta
                repel_factor = torch.where(antonym_mask, -2.0, 1.0)
                pair_delta = pair_delta * repel_factor[:, None]

        fused_src = torch.cat([pair_delta, effective_lr[:, None]], dim=1)

        if gen._fused_buf.shape[0] < ng:
            gen._fused_buf = torch.zeros(ng * 2, D + 1, device=device, dtype=torch.float32)
        buf = gen._fused_buf[:ng]; buf.zero_()
        buf.scatter_add_(0, inv_t[:, None].expand(-1, D + 1), fused_src)
        acc = buf[:, :D]; elr_grouped = buf[:, D]
        cnt = torch.bincount(inv_t, minlength=ng).float()

        if gradient_noise_scale > 0:
            acc += torch.randn_like(acc) * gradient_noise_scale * (elr_grouped[:, None] / elr_grouped.max().clamp(min=1))

        err_per_pair = 1.0 - y
        err_grouped = torch.zeros(ng, device=device)
        err_grouped.scatter_add_(0, inv_t, err_per_pair)
        cnt_err = torch.zeros(ng, device=device)
        cnt_err.scatter_add_(0, inv_t, torch.ones(N, device=device))
        avg_err = err_grouped / cnt_err.clamp(min=1)

        if not hasattr(gen, '_ce_t') or gen._ce_t is None:
            gen._ce_t = torch.zeros(gen._vecs_t.shape[0], device=device)
        gen._ce_t[unique_gen] = gen.concept_error_decay * gen._ce_t[unique_gen] + (1 - gen.concept_error_decay) * avg_err

        gen._gpu_elr_avg = elr_grouped / cnt.clamp(min=1)
        gen._gpu_unique_gen = unique_gen

        avg_err_cpu = avg_err.cpu().numpy()
        for gi, gen_cid in enumerate(unique_gen):
            gen.concept_error.update(gen_cid, float(avg_err_cpu[gi]))

        return acc, elr_grouped, cnt, D, ng

    # G-48: torch.compile candidate — _gpu_stdp_apply uses dynamic shapes (variable N).
    # To enable: extract pure-tensor core into a standalone function with
    # @torch.compile(mode='reduce-overhead', fullgraph=True)
    def _gpu_stdp_apply(self, gpu_ctx_l, gpu_tgt_l, gpu_meta_l, gpu_cid_gen,
                        base_lr_val, field_gate, inh_strength, inh_threshold, destab_scale,
                        gradient_noise_scale=0.0, momentum_mu=0.0, nesterov=False):
        gen = self.gen
        cs = gen.cs
        device = gen._torch_device
        ctx_t = torch.tensor(gpu_ctx_l, dtype=torch.long, device=device)
        tgt_t = torch.tensor(gpu_tgt_l, dtype=torch.long, device=device)
        N = len(gpu_ctx_l)

        with torch.no_grad():
            meta_t = torch.tensor(gpu_meta_l, dtype=torch.float32, device=device)
            gen_cids_arr = np.array(gpu_cid_gen, dtype=np.int32)
            unique_gen, inv_idx = np.unique(gen_cids_arr, return_inverse=True)
            inv_t = torch.from_numpy(inv_idx).to(device, non_blocking=True)
            ng = len(unique_gen)
            D = cs.dim

            # G-66: compiled pure-tensor core handles LR, scatter_add, CE, gradient noise
            acc, elr_grouped, cnt, _, _ = self._gpu_stdp_core(
                ctx_t, tgt_t, meta_t, unique_gen, inv_t, gen, cs, gradient_noise_scale)

            # Transition Manifold: push переходы ctx→tgt (семплируем для скорости)
            if self.manifold is not None and N > 0:
                from eva.symbolic.fcf_config import FCFConfig as _FCfg
                max_push = min(N, _FCfg().beam_batch_push_max)
                idxs = torch.randperm(N, device=device)[:max_push]
                vc = gen._vecs_t[ctx_t[idxs]].float()
                vg = gen._vecs_t[tgt_t[idxs]].float()
                cos = (vg * vc).sum(dim=1, keepdim=True).clamp(min=-1, max=1)
                T_dir = vg - cos * vc
                T_norm = T_dir.norm(dim=1, keepdim=True).clamp(min=_FCfg().beam_eps)
                T_dir /= T_norm
                T_cpu = T_dir.cpu().numpy().astype(np.float32)
                self.manifold.push_batch(T_cpu)

        # G-46: Persistent _mom_t tensor (replace CPU dict)
        if momentum_mu > 0:
            if gen._mom_t is None:
                gen._mom_t = torch.zeros(gen._vecs_t.shape[0], D, device=device, dtype=torch.bfloat16)
            avg_grad = acc / cnt[:, None].clamp(min=1)
            mom_new = momentum_mu * gen._mom_t[unique_gen] + (1 - momentum_mu) * avg_grad
            gen._mom_t[unique_gen] = mom_new.to(torch.bfloat16)

        # G-60/SN-45: GPU destabilization (replaces CPU per-element loop with tensor ops)
        if destab_scale > 0:
            destab_p = torch.clamp(gen._ce_t[unique_gen] * 0.5 * destab_scale, max=0.5)
            destab_mask = torch.rand(ng, device=device) < destab_p
            if destab_mask.any():
                n_v = gen._vecs_t.shape[0]
                unique_gen_t = torch.tensor(unique_gen, dtype=torch.long, device=device)
                rand_idx = torch.randint(1, n_v, (ng,), device=device)
                rand_idx = torch.where(rand_idx == unique_gen_t, (rand_idx + 1) % n_v, rand_idx)
                v_ppmi = gen._vecs_t[rand_idx].float()
                v_self = gen._vecs_t[unique_gen].float()
                y_ppmi = torch.clamp((v_self * v_ppmi).sum(dim=1), min=0.05)
                noise_gpu = v_ppmi - y_ppmi[:, None] * v_self
                nlen = noise_gpu.norm(dim=1, keepdim=True).clamp(min=1e-10)
                noise_gpu = noise_gpu / nlen
                mix_gpu = torch.clamp(gen._ce_t[unique_gen] * 0.5, max=0.5)
                destab_update = mix_gpu[:, None] * noise_gpu * elr_grouped[:, None]
                acc = torch.where(destab_mask[:, None],
                                  acc * (1 - mix_gpu[:, None]) + destab_update,
                                  acc)

        # Pure GPU per-concept loop: no CPU syncs, no .item(), no .numpy()
        valid_mask = cnt > 0.5
        elr_clamped = elr_grouped.clamp(min=1e-10)
        grad_gpu = acc / elr_clamped[:, None]
        gn_all = grad_gpu.norm(dim=1)
        if gen.max_grad_norm > 0:
            clip_mask = (gn_all > gen.max_grad_norm) & valid_mask
            if clip_mask.any():
                grad_gpu[clip_mask] = grad_gpu[clip_mask] / gn_all[clip_mask, None] * gen.max_grad_norm
        # SN-7: momentum already applied on GPU; _mom_t IS the smoothed gradient
        if momentum_mu > 0 and gen._mom_t is not None:
            grad_gpu = gen._mom_t[unique_gen]

        _subspace_cids = []
        _subspace_grads = []
        _deferred_updates = []
        for gi, gen_cid in enumerate(unique_gen):
            if not valid_mask[gi] or elr_grouped[gi] <= 0:
                continue
            if self.subspace_lr is not None and cs.fractal.basis is not None and gen._codes_t is not None:
                _subspace_cids.append(gen_cid)
                _subspace_grads.append(grad_gpu[gi].cpu().numpy())
            else:
                v_gpu = gen._vecs_t[gen_cid].float()
                v_new_gpu = v_gpu + grad_gpu[gi] * base_lr_val
                # Beam pull: притяжение к ближайшему лучу
                from eva.symbolic.fcf_config import FCFConfig as _FCfg2
                if self.manifold is not None and self.manifold.n_beams() >= _FCfg2().beam_pull_min_beams:
                    v_cpu = v_new_gpu.cpu().numpy().astype(np.float32)
                    cent, sim, _cnt = self.manifold.nearest_beam(v_cpu)
                    if cent is not None and sim > self.manifold.cos_threshold * _FCfg2().beam_pull_sim_ratio:
                        v_new_gpu = v_new_gpu + torch.from_numpy(cent).to(v_new_gpu.device) * _FCfg2().beam_pull_strength
                nv = v_new_gpu.norm()
                if nv > _FCfg2().beam_eps:
                    v_new_gpu /= nv
                _deferred_updates.append((gen_cid, v_new_gpu))

        if _subspace_cids:
            cs._apply_subspace_update_batch(_subspace_cids, np.array(_subspace_grads, dtype=np.float32), base_lr_val, self.subspace_lr, gen)

        # G-51/G-62: Batched _vecs_t write + dirty tracking
        if _deferred_updates:
            cids_batch = [d[0] for d in _deferred_updates]
            vecs_batch = torch.stack([d[1] for d in _deferred_updates]).to(gen._vecs_t.dtype)
            gen._vecs_t[cids_batch] = vecs_batch
            gen._dirty_cids.update(cids_batch)

        # AM-30: Batched EMA update outside per-concept loop
        if gen._vecs_t is not None and gen._ema_vecs_t is not None and gen._ema_steps >= 0:
            ema_updated = torch.lerp(gen._ema_vecs_t[unique_gen].float(),
                                     gen._vecs_t[unique_gen].float(),
                                     1.0 - gen._ema_decay)
            gen._ema_vecs_t[unique_gen] = ema_updated.to(gen._ema_vecs_t.dtype)
            gen._ema_steps += len(unique_gen)

        if inh_strength > 0 and len(unique_gen) >= 2:
            self._lateral_inhibition_gpu(unique_gen, inh_strength, inh_threshold, base_lr_val)

        return unique_gen

    # G-52: Fused post-STDP pass — negative sampling + contrastive in one GPU-enabled call.
    # Reuses gpu_ctx_l / gpu_meta_l to avoid re-reading _vecs_t between separate methods.
    def _gpu_poststdp_fused(self, gpu_ctx_l, gpu_meta_l, gpu_cid_ctx, gpu_cid_gen,
                            gen_updates, field_gate, base_lr_val, neg_lr_ratio, neg_samples):
        gen = self.gen
        if neg_samples > 0 and gen._vecs_t is not None:
            device = gen._torch_device
            self._negative_sampling_gpu(gpu_ctx_l, gpu_meta_l, gpu_cid_ctx, gpu_cid_gen,
                device, field_gate, base_lr_val, neg_lr_ratio, neg_samples)
        if gen_updates:
            self._contrastive_objective_gpu(gen_updates, field_gate)

    def _lateral_inhibition_gpu(self, gen_cids, inh_strength, inh_threshold, base_lr_val):
        gen = self.gen
        cs = gen.cs
        device = gen._torch_device
        n = len(gen_cids)
        if n == 0:
            return
        idxs = torch.tensor(gen_cids, dtype=torch.long, device=device)
        gv = gen._vecs_t[idxs].float()
        sim = gv @ gv.T
        mask_all = sim > inh_threshold * 2
        mask_all.fill_diagonal_(False)
        mask_sim = mask_all * sim
        has_inh = mask_all.any(dim=1)  # (n,) — which gen_cids have inhibition targets
        if not has_inh.any():
            return
        # AM-100: batched inhibition — all concepts in one matmul, no Python loop
        sum_sim_gv = mask_sim @ gv  # (n, D): vectorized sum of sim_ij * gv_j
        sum_sim2 = (mask_all * sim ** 2).sum(dim=1, keepdim=True)  # (n, 1): sum of sim_ij^2
        inhibit_vec = sum_sim_gv - sum_sim2 * gv  # (n, D)
        inh_norm = inhibit_vec.norm(dim=1, keepdim=True)
        inh_ok = inh_norm.squeeze() > 1e-10
        inhibit_vec[inh_ok] /= inh_norm[inh_ok]
        v_old = gen._vecs_t[idxs].float()
        v_new = v_old + inhibit_vec * inh_strength * base_lr_val
        nn = v_new.norm(dim=1, keepdim=True)
        nn_ok = nn.squeeze() > 1e-10
        v_new[nn_ok] /= nn[nn_ok]
        # Only update rows that had active inhibition
        update_mask = has_inh & inh_ok & nn_ok
        if update_mask.any():
            cids_batch = [gen_cids[i] for i in range(n) if update_mask[i]]
            vecs_batch = v_new[update_mask].to(gen._vecs_t.dtype)
            gen._vecs_t[cids_batch] = vecs_batch
            gen._dirty_cids.update(cids_batch)

    # ═══════════════════════════════════════════════════
    # Negative sampling
    # ═══════════════════════════════════════════════════

    def _negative_sampling_cpu(self, gen_updates, neg_lr_ratio, field_gate, neg_samples):
        gen = self.gen
        cs = gen.cs
        total_vocab = list(cs.concept_vectors.keys())
        if not total_vocab:
            return
        for gen_cid, updates in gen_updates.items():
            v_gen = cs.concept_vectors.get(gen_cid)
            if v_gen is None:
                continue
            avg_elr = sum(elr for _, elr in updates) / max(len(updates), 1)
            neg_lr = avg_elr * neg_lr_ratio * 0.3
            # SN-57: concept_error reweighting with multiplicative field_gate strength
            ce = gen.concept_error.get(gen_cid, 0.0)
            neg_lr *= (1.0 + ce * 2.0 * field_gate)
            neg_candidates = gen.main_rng.choices(total_vocab, k=min(neg_samples, len(total_vocab)))
            for neg_cid in neg_candidates:
                if neg_cid == gen_cid:
                    continue
                v_neg = cs.concept_vectors.get(neg_cid)
                if v_neg is None:
                    continue
                sim = float(np.dot(v_gen, v_neg))
                if sim > 0.1:
                    grad = v_neg - sim * v_gen
                    gn = float(np.linalg.norm(grad))
                    if gn > 1e-10:
                        grad = grad / gn * min(gn, 1.0)
                        v_new = v_gen - grad * neg_lr
                        nv = np.linalg.norm(v_new)
                        if nv > 1e-10:
                            v_new /= nv
                        cs._apply_vector_update(gen_cid, v_new)
                        v_gen = cs.concept_vectors.get(gen_cid)

    def _negative_sampling_gpu(self, gpu_ctx_l, gpu_meta_l, gpu_cid_ctx, gpu_cid_gen,
                               device, field_gate, base_lr_val, neg_lr_ratio, neg_samples):
        gen = self.gen
        cs = gen.cs
        if gen._vecs_t is None:
            return
        n_v = gen._vecs_t.shape[0]
        n_neg = min(neg_samples, n_v - 1)
        if n_neg < 1:
            return

        gen_cids_arr = np.array(gpu_cid_gen, dtype=np.int32)
        unique_gen, inv_idx = np.unique(gen_cids_arr, return_inverse=True)
        inv_t = torch.from_numpy(inv_idx).to(device, non_blocking=True)
        gen_t = torch.tensor(unique_gen, dtype=torch.long, device=device)
        gv = gen._vecs_t[gen_t].float()

        noise = torch.randint(0, n_v, (len(unique_gen), n_neg), device=device)
        ngv = gen._vecs_t[noise].float()
        sim = (gv[:, None, :] * ngv).sum(dim=-1)
        mask = sim > 0.1

        # Use avg_elr stored by _gpu_stdp_apply (avoids recomputing from meta)
        _gpu_avg = getattr(gen, '_gpu_elr_avg', None)
        if _gpu_avg is not None and len(_gpu_avg) == len(unique_gen):
            avg_elr_per_gen = _gpu_avg
        else:
            device_t = torch.tensor(gpu_meta_l, dtype=torch.float32, device=device)
            pair_elr = torch.clamp(device_t[:, _META_FW], min=0.05) * device_t[:, _META_DW] * device_t[:, _META_PMI] * device_t[:, _META_FIELD_W]
            elr_per_gen = torch.zeros(len(unique_gen), device=device)
            elr_per_gen.scatter_add_(0, inv_t, pair_elr)
            cnt_per_gen = torch.zeros(len(unique_gen), device=device)
            cnt_per_gen.scatter_add_(0, inv_t, torch.ones(len(gpu_cid_gen), device=device))
            avg_elr_per_gen = elr_per_gen / cnt_per_gen.clamp(min=1)

        neg_lr = avg_elr_per_gen * neg_lr_ratio * 0.3
        neg_lr *= (1.0 + gen._ce_t[unique_gen] * 2.0 * field_gate)

        # AM-100: batched negative sampling — all concepts in one tensor op, no Python loop
        noise_vecs = gen._vecs_t[noise].float()  # (ng, n_neg, D)
        mask_3d = mask.unsqueeze(-1)  # (ng, n_neg, 1)
        has_valid = mask.any(dim=1) & (neg_lr > 0)  # (ng,)
        if not has_valid.any():
            return
        sum_noise = (noise_vecs * mask_3d).sum(dim=1)  # (ng, D) — zero for rows w/o valid
        sum_sim = (sim * mask).sum(dim=1, keepdim=True)  # (ng, 1)
        grad = sum_noise - sum_sim * gv  # (ng, D)
        gn = grad.norm(dim=1, keepdim=True)
        grad_ok = gn.squeeze() > 1e-10
        grad[grad_ok] = grad[grad_ok] / gn[grad_ok] * gn[grad_ok].clamp(max=1.0)
        v_new = gv - grad * neg_lr.unsqueeze(-1)  # (ng, D)
        nn = v_new.norm(dim=1, keepdim=True)
        nn_ok = nn.squeeze() > 1e-10
        v_new[nn_ok] /= nn[nn_ok]
        um = (has_valid & grad_ok & nn_ok).cpu()
        if um.any():
            cids_batch = unique_gen[um.cpu().numpy()]
            vecs_batch = v_new[um].to(gen._vecs_t.dtype)
            gen._vecs_t[cids_batch] = vecs_batch
            gen._dirty_cids.update(cids_batch.tolist())

    # ═══════════════════════════════════════════════════
    # Contrastive objective
    # ═══════════════════════════════════════════════════

    def _contrastive_objective(self, gen_updates, field_gate=True):
        gen = self.gen
        if gen._vecs_t is not None and gen._use_torch:
            self._contrastive_objective_gpu(gen_updates, field_gate)
        else:
            self._contrastive_objective_cpu(gen_updates, field_gate)

    def _contrastive_objective_cpu(self, gen_updates, field_gate=True):
        gen = self.gen
        cs = gen.cs
        for gen_cid, updates in gen_updates.items():
            v_gen = cs.concept_vectors.get(gen_cid)
            if v_gen is None:
                continue
            avg_elr = sum(elr for _, elr in updates) / max(len(updates), 1)
            contr_lr = avg_elr * 0.3
            contr_lr *= (1.0 + gen.concept_error.get(gen_cid, 0.0) * 2.0 * field_gate)

            neighbours = cs.topk_similar_concepts(gen_cid, k=100, sample_size=2000)
            cooc_set = {ctx_cid for ctx_cid, _ in updates}
            hard_negatives = []
            for neg_cid, cos_val in neighbours:
                if neg_cid == gen_cid or neg_cid in cooc_set:
                    continue
                if gen.lattice.connection_strength(gen_cid, neg_cid) > 0.1:
                    continue
                if cos_val > 0.05 and cos_val < 0.5:
                    hard_negatives.append((neg_cid, cos_val))

            hard_negatives.sort(key=lambda x: -x[1])
            for neg_cid, cos_val in hard_negatives[:5]:
                v_neg = cs.concept_vectors.get(neg_cid)
                if v_neg is None:
                    continue
                # SN-16: Field-Aware Contrastive Decoupling
                overlap = 0
                fb_gen = gen.cs.fractal.field_bits.get(gen_cid)
                fb_neg = gen.cs.fractal.field_bits.get(neg_cid)
                if fb_gen is not None and fb_neg is not None:
                    overlap = int((fb_gen & fb_neg).sum())
                if overlap == 0 and cos_val > 0.05:
                    pass  # aggressive push for cross-field
                elif overlap > 0 and cos_val > 0.3:
                    continue  # skip gentle push for same-field beyond threshold
                contr_grad = cos_val * v_neg - v_gen
                gn = float(np.linalg.norm(contr_grad))
                if gn > gen.max_grad_norm > 0:
                    contr_grad = contr_grad / gn * gen.max_grad_norm
                push = contr_grad * contr_lr
                v_new = v_gen + push
                nv = np.linalg.norm(v_new)
                if nv > 1e-10:
                    v_new /= nv
                cs._apply_vector_update(gen_cid, v_new)
                v_gen = cs.concept_vectors.get(gen_cid)

    def _contrastive_objective_gpu(self, gen_updates, field_gate=True):
        gen = self.gen
        cs = gen.cs
        d = gen._torch_device

        # GPU path: use avg_elr stored by _gpu_stdp_apply (avoids iterating gen_updates on CPU)
        _gpu_avg = getattr(gen, '_gpu_elr_avg', None)
        _gpu_ug = getattr(gen, '_gpu_unique_gen', None)
        if _gpu_avg is not None and _gpu_ug is not None:
            gen_cids = list(_gpu_ug)
            avg_elrs = _gpu_avg
        else:
            gen_cids = list(gen_updates.keys())
            if not gen_cids:
                return
            avg_elrs = torch.tensor([
                sum(elr for _, elr in gen_updates[c]) / max(len(gen_updates[c]), 1)
                for c in gen_cids
            ], device=d)

        ng = len(gen_cids)
        if ng == 0:
            return
        gen_idxs = torch.tensor(gen_cids, dtype=torch.long, device=d)
        gen_idxs_l = gen_cids[:]

        contr_lrs = avg_elrs * 0.3
        contr_lrs *= (1.0 + gen._ce_t[gen_idxs] * 2.0 * field_gate)

        n_v = gen._vecs_t.shape[0]
        g_vecs = gen._vecs_t[gen_idxs].float()  # fp32 — used in gradient loop below
        all_vecs = gen._vecs_t[:n_v]            # fp16 — saves 224MB vs .float()
        sim = (g_vecs.half() @ all_vecs.T).float()  # fp16 matmul → fp32 result

        # AM-100: build cooc masks as sparse scatter (one batch scatter, no per-gen loop)
        cooc_masks = torch.zeros(ng, n_v, dtype=torch.bool, device=d)
        ctx_scatter_gi = []
        ctx_scatter_cid = []
        for i, gen_cid in enumerate(gen_idxs_l):
            for ctx_cid, _ in gen_updates[gen_cid]:
                ctx_scatter_gi.append(i)
                ctx_scatter_cid.append(ctx_cid)
        if ctx_scatter_gi:
            ctx_gi_t = torch.tensor(ctx_scatter_gi, dtype=torch.long, device=d)
            ctx_cid_t = torch.tensor(ctx_scatter_cid, dtype=torch.long, device=d)
            cooc_masks[ctx_gi_t, ctx_cid_t] = True

        # AM-100: batched field overlaps via chunked scatter (avoids O(ng*V*fb_bytes) RAM)
        if gen._fb_t is not None:
            fb_gen_all = gen._fb_t[gen_idxs]  # (ng, fb_bytes)
            fb_overlaps = torch.zeros(ng, n_v, device=d, dtype=torch.int32)
            fb_chunk_size = 4096
            for chunk_start in range(0, n_v, fb_chunk_size):
                chunk_end = min(chunk_start + fb_chunk_size, n_v)
                fb_chunk = gen._fb_t[chunk_start:chunk_end]  # (chunk, fb_bytes)
                chunk_overlap = (fb_gen_all.unsqueeze(1) & fb_chunk.unsqueeze(0)).sum(dim=-1).to(torch.int32)
                # chunk_overlap: (ng, chunk)
                fb_overlaps[:, chunk_start:chunk_end] = chunk_overlap
        else:
            fb_overlaps = None

        with torch.no_grad():
            topk = sim.topk(min(2000, n_v), dim=-1)
            topk_idx = topk.indices
            topk_val = topk.values

            mask_self = topk_idx == torch.arange(ng, device=d)[:, None]
            valid = ~mask_self

            if not valid.any():
                return

            max_hard = min(5, topk_idx.shape[1])
            best_idx = topk_idx[:, :max_hard]
            best_val = topk_val[:, :max_hard]

            # SN-44: Pre-compute all boolean masks on GPU (no .item() in loops)
            self_hn = best_idx == gen_idxs[:, None]
            cooc_hn = cooc_masks.gather(1, best_idx)

            if fb_overlaps is not None:
                fb_hn = fb_overlaps.gather(1, best_idx)
                cos_upper = torch.where(fb_hn > 0, 0.3, 0.999)
                # TN-14: cross-field regularization masks
                reg_n = min(50, topk_idx.shape[1])
                reg_idx = topk_idx[:, :reg_n]
                reg_val = topk_val[:, :reg_n]
                reg_self = reg_idx == gen_idxs[:, None]
                reg_cooc = cooc_masks.gather(1, reg_idx)
                reg_fb = fb_overlaps.gather(1, reg_idx)
                valid_reg = ~reg_self & ~reg_cooc & (reg_fb == 0) & (reg_val > 0.2)
            else:
                cos_upper = torch.full((ng, max_hard), 0.999, device=d)

            valid_hn = ~self_hn & ~cooc_hn & (best_val > 0.05) & (best_val < cos_upper)

            # AM-100: vectorized contrastive — all concepts in batched tensor ops
            v_local = g_vecs.clone()  # (ng, D)

            if fb_overlaps is not None:
                reg_mask = valid_reg.unsqueeze(-1)  # (ng, reg_n, 1)
                v_reg_all = gen._vecs_t[reg_idx].float()  # (ng, reg_n, D)
                cos_reg_all = reg_val.unsqueeze(-1)  # (ng, reg_n, 1)
                sum_reg = (cos_reg_all * v_reg_all * reg_mask).sum(dim=1)  # (ng, D)
                count_reg = valid_reg.sum(dim=1, keepdim=True).clamp(min=1)
                rep_grad = sum_reg / count_reg - v_local  # (ng, D)
                rep_gn = rep_grad.norm(dim=1, keepdim=True)
                if gen.max_grad_norm > 0:
                    rep_grad = rep_grad / rep_gn.clamp(min=1e-10) * rep_gn.clamp(max=gen.max_grad_norm)
                v_local = v_local + rep_grad * contr_lrs.unsqueeze(-1) * 0.05

            hn_mask = valid_hn.unsqueeze(-1)  # (ng, max_hard, 1)
            v_neg_all = gen._vecs_t[best_idx].float()  # (ng, max_hard, D)
            cos_neg_all = best_val.unsqueeze(-1)  # (ng, max_hard, 1)
            sum_weighted = (cos_neg_all * v_neg_all * hn_mask).sum(dim=1)  # (ng, D)
            count_hn = valid_hn.sum(dim=1, keepdim=True).clamp(min=1)
            has_hn = valid_hn.any(dim=1)
            grad_hn = sum_weighted / count_hn - v_local  # (ng, D)
            gn_hn = grad_hn.norm(dim=1, keepdim=True)
            if gen.max_grad_norm > 0:
                grad_hn = grad_hn / gn_hn.clamp(min=1e-10) * gn_hn.clamp(max=gen.max_grad_norm)
            v_new_hn = v_local + grad_hn * contr_lrs.unsqueeze(-1)
            nv_hn = v_new_hn.norm(dim=1, keepdim=True)
            nv_ok_hn = nv_hn.squeeze() > 1e-10
            v_new_hn[nv_ok_hn] /= nv_hn[nv_ok_hn]

            v_local_changed = (v_local != g_vecs).any(dim=1)  # was modified by rep grad
            update_hn = has_hn & nv_ok_hn
            update_rep = ~has_hn & v_local_changed
            update_mask = update_hn | update_rep
            um_cpu = update_mask.cpu()
            if um_cpu.any():
                v_new = torch.where(update_hn.unsqueeze(-1), v_new_hn, v_local)
                nv_all = v_new.norm(dim=1, keepdim=True)
                nv_ok_all = nv_all.squeeze() > 1e-10
                v_new[nv_ok_all] /= nv_all[nv_ok_all]
                cids_batch = [gen_idxs_l[i] for i in range(ng) if um_cpu[i]]
                vecs_batch = v_new[um_cpu].to(gen._vecs_t.dtype)
                gen._vecs_t[cids_batch] = vecs_batch
                gen._dirty_cids.update(cids_batch)

    # ═══════════════════════════════════════════════════
    # Centroid pull
    # ═══════════════════════════════════════════════════

    # G-42: GPU _centroid_pull_batch (CPU fallback when GPU tensors absent)
    def _centroid_pull_batch(self, all_ids, base_lr_val):
        gen = self.gen
        cs = gen.cs
        if gen._vecs_t is None or gen._torch_device is None:
            for ids in all_ids:
                if len(ids) < 3:
                    continue
                sent_vecs = [cs.concept_vectors.get(c) for c in ids]
                sent_vecs = [v for v in sent_vecs if v is not None]
                if len(sent_vecs) < 3:
                    continue
                centroid = np.mean(sent_vecs, axis=0).astype(np.float32)
                n_cent = np.linalg.norm(centroid)
                if n_cent > 1e-10:
                    centroid /= n_cent
                    sent_lr = base_lr_val * 0.3
                    for cid in ids:
                        v = cs.concept_vectors.get(cid)
                        if v is None:
                            continue
                        sim = float(np.dot(v, centroid))
                        shift = (centroid - sim * v) * sent_lr
                        v_new = v + shift
                        nv = np.linalg.norm(v_new)
                        if nv > 1e-10:
                            v_new /= nv
                        cs._apply_vector_update(cid, v_new)
            return
        device = gen._torch_device
        _centroid_updates = []
        for ids in all_ids:
            if len(ids) < 3:
                continue
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            valid_mask = ids_t < gen._vecs_t.shape[0]
            ids_t = ids_t[valid_mask]
            if len(ids_t) < 3:
                continue
            vecs = gen._vecs_t[ids_t].float()
            centroid = vecs.mean(dim=0)
            cn = centroid / centroid.norm().clamp(min=1e-10)
            sims = (vecs * cn).sum(dim=1)
            pulls = (cn - sims[:, None] * vecs)
            v_new = vecs + pulls * base_lr_val * 0.3
            nv = v_new.norm(dim=1, keepdim=True).clamp(min=1e-10)
            v_new /= nv
            for i, cid in enumerate(ids_t.tolist()):
                _centroid_updates.append((cid, v_new[i]))
        if _centroid_updates:
            cids_batch = [d[0] for d in _centroid_updates]
            vecs_batch = torch.stack([d[1] for d in _centroid_updates]).to(gen._vecs_t.dtype)
            gen._vecs_t[cids_batch] = vecs_batch
            gen._dirty_cids.update(cids_batch)

    # ═══════════════════════════════════════════════════
    # Cluster centroid pull (octree cluster, not sentence)
    # ═══════════════════════════════════════════════════

    def _cluster_centroid_pull(self, all_ids, base_lr_val, pull_strength=0.1):
        """Pull concepts toward their octree cluster centroid.

        Uses _cluster_map (anchor per CID) to group into clusters,
        computes centroid per cluster, pulls members toward it.
        Prevents embedding sparsity within semantic fields.
        """
        gen = self.gen
        cs = gen.cs
        if gen._cluster_map is None or gen._vecs_t is None:
            return

        device = gen._torch_device
        # Collect unique CIDs from this batch
        batch_cids = set()
        for ids in all_ids:
            batch_cids.update(ids)
        if len(batch_cids) < 2:
            return

        cid_list = sorted(batch_cids)
        cid_t = torch.tensor(cid_list, dtype=torch.long, device=device)
        cluster_ids = gen._cluster_map[cid_t]  # (M,) anchor per CID

        # Group by cluster
        unique_clusters = torch.unique(cluster_ids)
        vecs = gen._vecs_t[cid_t].float()  # (M, D)

        updates = []
        for cl in unique_clusters:
            mask = cluster_ids == cl
            members = cid_t[mask]
            if len(members) < 2:
                continue
            member_vecs = vecs[mask]  # (K, D)
            centroid = member_vecs.mean(dim=0)
            cn = centroid / centroid.norm().clamp(min=1e-10)
            sims = (member_vecs * cn).sum(dim=1)
            pulls = (cn - sims[:, None] * member_vecs)
            v_new = member_vecs + pulls * base_lr_val * pull_strength
            nv = v_new.norm(dim=1, keepdim=True).clamp(min=1e-10)
            v_new /= nv
            for i, cid in enumerate(members.tolist()):
                updates.append((cid, v_new[i]))

        if updates:
            cids_batch = [d[0] for d in updates]
            vecs_batch = torch.stack([d[1] for d in updates]).to(gen._vecs_t.dtype)
            gen._vecs_t[cids_batch] = vecs_batch
            gen._dirty_cids.update(cids_batch)

    # ═══════════════════════════════════════════════════
    # Evaluate
    # ═══════════════════════════════════════════════════

    def _evaluate(self, corpus_path, max_lines=500, use_torch=None):
        import time
        gen = self.gen
        cs = gen.cs
        gen._sync_dirty_cpu()
        if use_torch is None:
            use_torch = gen._use_torch
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
        except FileNotFoundError:
            return None
        lines = lines[:max_lines]
        if len(lines) < 5:
            return None

        total_tokens = 0
        total_pairs = 0
        n_correct_1 = 0
        n_correct_v1 = 0
        log_probs = []
        vec_log_probs = []
        start_t = time.time()

        for line in lines:
            ids = gen._encode_input(line)
            if len(ids) < 3:
                continue
            for i in range(1, len(ids)):
                ctx = ids[max(0, i - 5):i]
                target = ids[i]
                similarities = cs.batch_dot(ctx, target)
                if not similarities:
                    continue
                sims = np.array(similarities, dtype=np.float32)
                max_sim = float(sims.max())
                log_prob = max_sim - math.log(max(len(ctx), 2))
                log_probs.append(log_prob)
                total_tokens += 1
                pred = ctx[int(np.argmax(sims))]
                if pred == target:
                    n_correct_1 += 1
                vec_probs = []
                for cid in ctx:
                    v_c = cs.concept_vectors.get(cid)
                    v_t = cs.concept_vectors.get(target)
                    if v_c is None or v_t is None:
                        continue
                    vec_probs.append(float(np.dot(v_c, v_t)))
                if vec_probs:
                    v_ppl = -float(np.mean(vec_probs))
                    vec_log_probs.append(v_ppl)
                    v_pred = ctx[int(np.argmax(vec_probs))]
                    if v_pred == target:
                        n_correct_v1 += 1

        elapsed = time.time() - start_t
        if total_tokens == 0:
            return {'perplexity': float('inf'), 'vec_perplexity': float('inf'),
                    'accuracy_top1': 0.0, 'vec_accuracy_top1': 0.0, 'total_tokens': 0,
                    'eval_time_s': elapsed}
        perplexity = math.exp(sum(log_probs) / max(len(log_probs), 1)) if log_probs else float('inf')
        v_perplexity = math.exp(sum(vec_log_probs) / max(len(vec_log_probs), 1)) if vec_log_probs else float('inf')
        return {
            'perplexity': perplexity,
            'vec_perplexity': v_perplexity,
            'accuracy_top1': n_correct_1 / max(total_tokens, 1),
            'vec_accuracy_top1': n_correct_v1 / max(total_tokens, 1),
            'total_tokens': total_tokens,
            'eval_time_s': elapsed,
        }


# G-66: patch _gpu_stdp_core with torch.compile on CUDA-capable hardware
# Only when CUDA is truly available (not CPU test mode) and has enough VRAM
if (_HAS_COMPILE and torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (7, 0)  # Volta+ for efficient graph
        and torch.cuda.get_device_properties(0).total_memory >= 3 * 1024**3):  # ≥3GB
    try:
        STDPTrainer._gpu_stdp_core = torch.compile(
            STDPTrainer._gpu_stdp_core, mode='reduce-overhead', fullgraph=False)
    except Exception:
        pass
