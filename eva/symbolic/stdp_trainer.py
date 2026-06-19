"""STDPTrainer — training methods separated from CrystalGenerator.
Extracts STDP, negative sampling, contrastive, centroid pull, evaluate.
"""

import math
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Set

_META_I = 0
_META_J = 1
_META_PMI = 2
_META_DW = 3
_META_FW = 4
_META_FIELD_W = 5
_META_SLOW = 6


class STDPTrainer:
    """STDP training logic for CrystalGenerator.
    Operates on gen.cs, gen.lattice, gen._vecs_t, gen.hormones, gen.concept_error.
    """
    def __init__(self, gen, subspace_lr=None):
        self.gen = gen
        self.subspace_lr = subspace_lr  # (lr_c, lr_a, lr_m) or None for uniform LR
        # G-45: Persistent CUDA events (created once, reused across batches)
        if torch.cuda.is_available():
            self._prof_start = torch.cuda.Event(enable_timing=True)
            self._prof_end = torch.cuda.Event(enable_timing=True)
        else:
            self._prof_start = self._prof_end = None

    # ═══════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════

    def train_from_text(self, text, base_lr=None, context_window=2, pmi_strength=1.0,
                         pmi_gate_min=0.20, neg_samples=1, inh_strength=0.05, inh_threshold=0.10,
                         neg_lr_ratio=0.5, field_gate=True, use_torch=None, destab_scale=0.0,
                         momentum_mu=0.9, noise_scale=0.0):
        """Train on one text line. Returns number of STDP pairs built."""
        return self._train(text, base_lr, context_window, pmi_strength, pmi_gate_min,
                           neg_samples, inh_strength, inh_threshold, neg_lr_ratio,
                           field_gate, use_torch, destab_scale, batch=False,
                           momentum_mu=momentum_mu, noise_scale=noise_scale)

    def train_batch(self, texts, base_lr=None, context_window=2, pmi_strength=1.0,
                     pmi_gate_min=0.20, neg_samples=1, inh_strength=0.05, inh_threshold=0.10,
                     neg_lr_ratio=0.5, field_gate=True, use_torch=None, destab_scale=0.0,
                     momentum_mu=0.9, noise_scale=0.0):
        """Train on a batch of texts (GPU batched). Returns total pairs."""
        return self._train(texts, base_lr, context_window, pmi_strength, pmi_gate_min,
                           neg_samples, inh_strength, inh_threshold, neg_lr_ratio,
                           field_gate, use_torch, destab_scale, batch=True,
                           momentum_mu=momentum_mu, noise_scale=noise_scale)

    def evaluate(self, corpus_path, max_lines=500, use_torch=None):
        """Evaluate on a corpus: perplexity, accuracy, vector metrics."""
        return self._evaluate(corpus_path, max_lines, use_torch)

    # ═══════════════════════════════════════════════════
    # Internal: train
    # ═══════════════════════════════════════════════════

    def _train(self, inputs, base_lr, context_window, pmi_strength, pmi_gate_min,
               neg_samples, inh_strength, inh_threshold, neg_lr_ratio, field_gate,
               use_torch, destab_scale, batch, momentum_mu=0.9, noise_scale=0.0):
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

        for text in inputs:
            ids = gen._encode_input(text)
            if len(ids) < 2:
                continue

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
                noise_scale=noise_scale, momentum_mu=momentum_mu)
        else:
            self._cpu_stdp_apply(gen_updates, base_lr, destab_scale, inh_strength, inh_threshold)

        # ── Negative sampling ──
        if neg_samples > 0 and use_torch and gen._vecs_t is not None and gpu_ctx_l:
            device = gen._torch_device
            self._negative_sampling_gpu(gpu_ctx_l, gpu_meta_l, gpu_cid_ctx, gpu_cid_gen,
                device, field_gate, base_lr, neg_lr_ratio, neg_samples)
        elif neg_samples > 0:
            self._negative_sampling_cpu(gen_updates, neg_lr_ratio, field_gate, neg_samples)

        # ── Contrastive objective ──
        self._contrastive_objective(gen_updates)

        # ── Centroid pull + lattice update ──
        all_ids = []
        for text in inputs:
            ids = gen._encode_input(text)
            if len(ids) >= 2:
                all_ids.append(ids)
        self._centroid_pull_batch(all_ids, base_lr)
        for ids in all_ids:
            gen.lattice.update(ids)
            gen._graph_cache.clear()

        # Prune concept_error cache (redundant — AdaptiveErrorTracker auto-prunes on update)
        if use_torch:
            gen._torch_dirty = True

        return total_pairs

    # ═══════════════════════════════════════════════════
    # Pair building
    # ═══════════════════════════════════════════════════

    def _build_pairs(self, ids, context_window, total_freq, pmi_strength, pmi_gate_min,
                     field_gate, base_lr, use_torch, cid_to_idx,
                     gen_updates, gpu_ctx_l, gpu_tgt_l, gpu_meta_l,
                     gpu_cid_ctx, gpu_cid_gen):
        """Build STDP pairs for one sentence. Shared CPU/GPU pair generation."""
        gen = self.gen
        cs = gen.cs
        T = len(ids)
        n_pairs = 0
        for i in range(T):
            start = max(0, i - context_window)
            end = min(T, i + context_window + 1)
            for j in range(start, end):
                if j <= i:
                    continue
                dist = abs(j - i)
                dist_weight = math.exp(-dist / 2.0)

                fa = gen.lattice.concept_freq.get(ids[i], 0)
                fb = gen.lattice.concept_freq.get(ids[j], 0)
                freq_weight = 1.0 / (1.0 + math.log(max(max(fa, fb), 1)) * 0.15)

                pmi_w_raw = gen._pmi_weight(ids[i], ids[j], distance=dist,
                                              min_weight=pmi_gate_min)
                _skip, pmi_w = gen._apply_pmi_gate(pmi_w_raw, pmi_strength, pmi_gate_min, ids[j])
                if _skip:
                    continue

                field_weight = 1.0
                if field_gate:
                    if use_torch:
                        if gen._fb_t is None:
                            gen._ensure_fb_tensor(gen._torch_device)
                    if use_torch and gen._fb_t is not None:
                        overlap = int(torch.bitwise_and(gen._fb_t[ids[i]], gen._fb_t[ids[j]]).sum().item())
                    elif hasattr(cs.fractal, 'field_bits') and len(cs.fractal.field_bits) > 0:
                        overlap = cs.fractal.field_overlap(ids[i], ids[j])
                    else:
                        overlap = 0
                    field_weight = min(1.0 + math.log(overlap + 1) * 2.0, 3.0) if overlap > 0 else 0.1

                lr = base_lr * max(freq_weight, 0.05) * pmi_w * field_weight
                lr *= (0.5 + gen.hormones.acetylcholine * 0.5) * (0.5 + gen.hormones.dopamine * 0.5)
                theta_gate = math.exp(-min(abs(j-i), 5) / max(gen.theta_tau, 1.0))
                gen_updates[ids[j]].append((ids[i], lr * max(theta_gate, 0.1)))
                n_pairs += 1
                theta_slow = math.exp(-min(abs(j-i), 10) / max(gen.theta_tau * 3.0, 1.0))
                slow_lr = lr * max(theta_slow, 0.02) * 0.3
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
                    gpu_meta_l.append((i, j, pmi_w, dist_weight, freq_weight, field_weight, 0.0))
                    gpu_cid_ctx.append(ids[i])
                    gpu_cid_gen.append(ids[j])
                    # SN-25: Add slow STDP pair to GPU lists (matches CPU slow_lr > 1e-6 gate)
                    theta_slow = math.exp(-min(abs(j-i), 10) / max(gen.theta_tau * 3.0, 1.0))
                    slow_lr = lr * max(theta_slow, 0.02) * 0.3
                    if slow_lr > 1e-6:
                        gpu_ctx_l.append(ci)
                        gpu_tgt_l.append(cj)
                        gpu_meta_l.append((i, j, pmi_w, dist_weight, freq_weight, field_weight, 1.0))
                        gpu_cid_ctx.append(ids[i])
                        gpu_cid_gen.append(ids[j])

        return n_pairs

    # ═══════════════════════════════════════════════════
    # CPU STDP
    # ═══════════════════════════════════════════════════

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
                    nv = np.linalg.norm(v_new)
                    if nv > 1e-10:
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

    def _gpu_stdp_apply(self, gpu_ctx_l, gpu_tgt_l, gpu_meta_l, gpu_cid_gen,
                        base_lr_val, field_gate, inh_strength, inh_threshold, destab_scale,
                        noise_scale=0.0, momentum_mu=0.0, nesterov=False):
        gen = self.gen
        cs = gen.cs
        device = gen._torch_device
        ctx_t = torch.tensor(gpu_ctx_l, dtype=torch.long, device=device)
        tgt_t = torch.tensor(gpu_tgt_l, dtype=torch.long, device=device)
        N = len(gpu_ctx_l)

        if self._prof_start is not None:
            self._prof_start.record()

        with torch.no_grad():
            meta_t = torch.tensor(gpu_meta_l, dtype=torch.float32, device=device)
            i_pos = meta_t[:, _META_I]
            j_pos = meta_t[:, _META_J]
            dist = j_pos - i_pos
            pmi_w_t = meta_t[:, _META_PMI]
            dw_t = meta_t[:, _META_DW]
            fw_t = meta_t[:, _META_FW]
            field_w_t = meta_t[:, _META_FIELD_W]

            lr = torch.clamp(fw_t, min=0.05) * dw_t * pmi_w_t * field_w_t
            lr *= (0.5 + gen.hormones.acetylcholine * 0.5) * (0.5 + gen.hormones.dopamine * 0.5)
            # SN-25: Handle slow vs fast STDP theta
            dist_clamped = torch.clamp(dist, max=10.0)
            theta_fast = torch.exp(-dist_clamped.clamp(max=5.0) / max(gen.theta_tau, 1.0))
            theta_slow = torch.exp(-dist_clamped / max(gen.theta_tau * 3.0, 1.0))
            slow_mask = meta_t[:, _META_SLOW] if meta_t.shape[1] > _META_SLOW else torch.zeros(N, device=device)
            theta = (1 - slow_mask) * torch.clamp(theta_fast, min=0.1) + slow_mask * torch.clamp(theta_slow, min=0.02) * 0.3
            effective_lr = lr * theta

            gen_cids_arr = np.array(gpu_cid_gen, dtype=np.int32)
            unique_gen, inv_idx = np.unique(gen_cids_arr, return_inverse=True)
            inv_t = torch.from_numpy(inv_idx).to(device, non_blocking=True)
            D = cs.dim

            # G-12: Fused scatter-add — acc (D) and elr_grouped (1) in one tensor
            vc = gen._vecs_t[ctx_t].float()
            vg = gen._vecs_t[tgt_t].float()
            y = torch.clamp((vg * vc).sum(dim=1), min=0.05)
            pair_delta = vc * effective_lr[:, None] - vg * (y * effective_lr)[:, None]
            fused_src = torch.cat([pair_delta, effective_lr[:, None]], dim=1)  # (N, D+1)
            fused = torch.zeros(len(unique_gen), D + 1, dtype=torch.float32, device=device)
            fused.scatter_add_(0, inv_t[:, None].expand(-1, D + 1), fused_src)
            acc = fused[:, :D]
            elr_grouped = fused[:, D]
            cnt = torch.bincount(inv_t, minlength=len(unique_gen)).float()

            # TN-6: Gradient Noise Injection
            if noise_scale > 0:
                acc += torch.randn_like(acc) * noise_scale * (elr_grouped[:, None] / elr_grouped.max().clamp(min=1))

            err_per_pair = 1.0 - y
            err_grouped = torch.zeros(len(unique_gen), device=device)
            err_grouped.scatter_add_(0, inv_t, err_per_pair)
            cnt_err = torch.zeros(len(unique_gen), device=device)
            cnt_err.scatter_add_(0, inv_t, torch.ones(N, device=device))
            avg_err = err_grouped / cnt_err.clamp(min=1)
            # G-16: In-place Concept Error EMA on GPU
            if not hasattr(gen, '_ce_t') or gen._ce_t is None:
                gen._ce_t = torch.zeros(gen._vecs_t.shape[0], device=device)
            ce_decay = gen.concept_error_decay
            gen._ce_t[unique_gen] = ce_decay * gen._ce_t[unique_gen] + (1 - ce_decay) * avg_err
            avg_err_cpu = avg_err.cpu().numpy()
            for gi, gen_cid in enumerate(unique_gen):
                gen.concept_error.update(gen_cid, float(avg_err_cpu[gi]))

        # SN-7: Momentum-Accumulated STDP
        mom_cpu = None
        if momentum_mu > 0:
            if not hasattr(gen, '_mom_buf'):
                gen._mom_buf = {}
            avg_grad = acc / cnt[:, None].clamp(min=1)
            mom_gpu = torch.zeros(len(unique_gen), D, dtype=torch.float32, device=device)
            for gi, cid in enumerate(unique_gen):
                cid_i = int(cid)
                prev = gen._mom_buf.get(cid_i, None)
                if prev is not None:
                    prev_t = torch.from_numpy(prev).to(device, non_blocking=True)
                    mom_gpu[gi] = momentum_mu * prev_t + (1 - momentum_mu) * avg_grad[gi]
                else:
                    mom_gpu[gi] = avg_grad[gi]
            for gi, cid in enumerate(unique_gen):
                gen._mom_buf[int(cid)] = mom_gpu[gi].cpu().numpy().astype(np.float32)
            mom_cpu = mom_gpu.cpu().numpy()

        acc_cpu = acc.cpu().numpy()
        cnt_cpu = cnt.cpu().numpy()
        elr_cpu = elr_grouped.cpu().numpy()

        for gi, gen_cid in enumerate(unique_gen):
            v = cs.concept_vectors.get(gen_cid)
            if v is None or cnt_cpu[gi] < 0.5:
                continue
            # SN-8: Concept-Error Adaptive Destabilization
            ce = gen.concept_error.get(gen_cid, 0.0)
            _destab_p = min(ce * 0.5 * max(destab_scale, 0.0), 0.5)
            if destab_scale > 0 and gen.main_rng.random() < _destab_p:
                ppmi_candidates = gen.lattice.connections_of(
                    gen_cid, top_k=20, use_ppmi=True)
                if ppmi_candidates:
                    ppmi_cid = ppmi_candidates[gen.main_rng.randint(0, len(ppmi_candidates) - 1)][0]
                    v_ppmi = cs.concept_vectors.get(ppmi_cid)
                else:
                    v_ppmi = gen._destab_field_fallback(gen_cid, v)
                if v_ppmi is not None:
                    y_ppmi = max(float(np.dot(v, v_ppmi)), 0.05)
                    noise_vec = (v_ppmi - y_ppmi * v)
                    nlen = float(np.linalg.norm(noise_vec))
                    if nlen > 1e-10:
                        noise_vec /= nlen
                        mix = min(ce * 0.5, 0.5)
                        acc_cpu[gi] = acc_cpu[gi] * (1 - mix) + noise_vec * mix * elr_cpu[gi]

            if cnt_cpu[gi] > 0 and elr_cpu[gi] > 0:
                grad = acc_cpu[gi] / max(elr_cpu[gi], 1e-10)
                gn = float(np.linalg.norm(grad))
                if gn > gen.max_grad_norm > 0:
                    grad = grad / gn * gen.max_grad_norm
                # SN-7: use momentum-smoothed gradient when available
                if mom_cpu is not None:
                    grad = momentum_mu * mom_cpu[gi] + (1 - momentum_mu) * grad
                # T-B3/SN-B1: EMA BEFORE any update (uses old _vecs_t)
                if gen._vecs_t is not None and gen._ema_vecs_t is not None and gen._ema_steps >= 0:
                    gen._ema_vecs_t[gen_cid].lerp_(gen._vecs_t[gen_cid].float(), 1.0 - gen._ema_decay)
                    gen._ema_steps += 1
                if self.subspace_lr is not None and cs.fractal.basis is not None and cs.fractal.codes.get(gen_cid) is not None:
                    cs._apply_subspace_update(gen_cid, grad, base_lr_val, self.subspace_lr)
                else:
                    v_new = v + grad * base_lr_val
                    nv = np.linalg.norm(v_new)
                    if nv > 1e-10:
                        v_new /= nv
                    cs._apply_vector_update(gen_cid, v_new)
        
        if inh_strength > 0 and len(unique_gen) >= 2:
            self._lateral_inhibition_gpu(unique_gen, inh_strength, inh_threshold, base_lr_val)

        if self._prof_end is not None:
            self._prof_end.record()
            self._prof_end.synchronize()
            self.gen._prof_ms = self._prof_start.elapsed_time(self._prof_end)

        return unique_gen

    def _lateral_inhibition_gpu(self, gen_cids, inh_strength, inh_threshold, base_lr_val):
        gen = self.gen
        cs = gen.cs
        device = gen._torch_device
        idxs = torch.tensor(gen_cids, dtype=torch.long, device=device)
        gv = gen._vecs_t[idxs].float()
        sim = gv @ gv.T
        n = len(gen_cids)
        for gi in range(n):
            mask = sim[gi] > inh_threshold * 2
            mask[gi] = False
            if not mask.any():
                continue
            inhibition = sim[gi][mask].sum().item()
            inhibit_vec = (sim[gi][mask] * gv[mask] - (sim[gi][mask]**2) * gv[gi]).sum(dim=0)
            nv = inhibit_vec.norm()
            if nv > 1e-10:
                inhibit_vec /= nv
                v_np = cs.concept_vectors.get(gen_cids[gi])
                if v_np is None:
                    continue
                v_new = v_np + inhibit_vec.cpu().numpy() * inh_strength * base_lr_val
                nn = np.linalg.norm(v_new)
                if nn > 1e-10:
                    v_new /= nn
                cs._apply_vector_update(gen_cids[gi], v_new)

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
            # SN-B2: concept_error reweighting (applied only when field_gate=True, matching original intent)
            if field_gate:
                ce = gen.concept_error.get(gen_cid, 0.0)
                neg_lr *= (1.0 + ce * 2.0)
            neg_candidates = gen.main_rng.sample(total_vocab, min(neg_samples, len(total_vocab)))
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
        unique_gen = np.unique(gen_cids_arr)
        gen_t = torch.tensor(unique_gen, dtype=torch.long, device=device)
        gv = gen._vecs_t[gen_t].float()

        noise = torch.randint(0, n_v, (len(unique_gen), n_neg), device=device)
        ngv = gen._vecs_t[noise].float()
        sim = (gv[:, None, :] * ngv).sum(dim=-1)
        mask = sim > 0.1
        device_t = torch.tensor(gpu_meta_l, dtype=torch.float32, device=device)
        elr_sum = (torch.clamp(device_t[:, _META_FW], min=0.05) *
                   device_t[:, _META_DW] * device_t[:, _META_PMI] * device_t[:, _META_FIELD_W]).sum().item()
        neg_lr = (elr_sum / max(len(gpu_ctx_l), 1)) * neg_lr_ratio * 0.3

        for gi, gen_cid in enumerate(unique_gen):
            neg_lr_i = neg_lr
            # SN-22.2: Guard concept_error reweighting with field_gate (matching CPU parity)
            if field_gate:
                ce = gen.concept_error.get(gen_cid, 0.0)
                neg_lr_i *= (1.0 + ce * 2.0)
            neg_mask = mask[gi]
            if not neg_mask.any():
                continue
            valid_idx = noise[gi][neg_mask]
            vg_i = gv[gi]
            vg_i_2d = vg_i.unsqueeze(0)
            grad = (gen._vecs_t[valid_idx].float() - sim[gi][neg_mask][:, None] * vg_i_2d).sum(dim=0)
            gn = grad.norm()
            if gn > 1e-10:
                grad = grad / gn * min(gn, 1.0)
                v_np = cs.concept_vectors.get(gen_cid)
                if v_np is None:
                    continue
                v_new = v_np - grad.cpu().numpy() * neg_lr_i
                nn = np.linalg.norm(v_new)
                if nn > 1e-10:
                    v_new /= nn
                cs._apply_vector_update(gen_cid, v_new)

    # ═══════════════════════════════════════════════════
    # Contrastive objective
    # ═══════════════════════════════════════════════════

    def _contrastive_objective(self, gen_updates):
        gen = self.gen
        if gen._vecs_t is not None and gen._use_torch:
            self._contrastive_objective_gpu(gen_updates)
        else:
            self._contrastive_objective_cpu(gen_updates)

    def _contrastive_objective_cpu(self, gen_updates):
        gen = self.gen
        cs = gen.cs
        for gen_cid, updates in gen_updates.items():
            v_gen = cs.concept_vectors.get(gen_cid)
            if v_gen is None:
                continue
            avg_elr = sum(elr for _, elr in updates) / max(len(updates), 1)
            contr_lr = avg_elr * 0.3
            contr_lr *= (1.0 + gen.concept_error.get(gen_cid, 0.0) * 2.0)

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

    def _contrastive_objective_gpu(self, gen_updates):
        gen = self.gen
        cs = gen.cs
        d = gen._torch_device
        gen_cids = list(gen_updates.keys())
        if not gen_cids:
            return
        n_v = gen._vecs_t.shape[0]
        ng = len(gen_cids)

        contr_lrs = torch.zeros(ng, device=d)
        gen_idxs = gen_cids[:]
        for i, gen_cid in enumerate(gen_cids):
            updates = gen_updates[gen_cid]
            avg_elr = sum(elr for _, elr in updates) / max(len(updates), 1)
            contr_lrs[i] = avg_elr * 0.3 * (1.0 + gen.concept_error.get(gen_cid, 0.0) * 2.0)

        g_vecs = gen._vecs_t[gen_idxs].float()
        all_vecs = gen._vecs_t[:n_v].float()
        sim = g_vecs @ all_vecs.T

        with torch.no_grad():
            topk = sim.topk(min(2000, n_v), dim=-1)
            topk_idx = topk.indices
            topk_val = topk.values

            mask_self = topk_idx == torch.arange(ng, device=d)[:, None]
            valid = ~mask_self

            if valid.any():
                max_hard = min(5, topk_idx.shape[1])
                best_idx = topk_idx[:, :max_hard]
                best_val = topk_val[:, :max_hard]

                push_total = torch.zeros(ng, g_vecs.shape[1], device=d)
                lr_scale = torch.zeros(ng, device=d)
                for i in range(ng):
                    hn = []
                    for j in range(max_hard):
                        neg_cid = int(best_idx[i, j].item())
                        if neg_cid == gen_idxs[i]:
                            continue
                        cooc_set = {ctx for ctx, _ in gen_updates[gen_idxs[i]]}
                        if neg_cid in cooc_set:
                            continue
                        if gen.lattice.connection_strength(gen_idxs[i], neg_cid) > 0.1:
                            continue
                        cos_val = float(best_val[i, j].item())
                        # SN-16: Field-Aware Contrastive Decoupling
                        overlap = 0
                        if gen._fb_t is not None:
                            fb_gen = gen._fb_t[gen_idxs[i]]
                            fb_neg = gen._fb_t[neg_cid]
                            overlap = int(torch.bitwise_and(fb_gen, fb_neg).sum().item())
                        max_cos_upper = 0.3 if overlap > 0 else 0.999
                        if cos_val > 0.05 and cos_val < max_cos_upper:
                            hn.append(neg_cid)
                            if len(hn) >= 5:
                                break
                    # TN-14: Field-Aware Contrastive Regularization (cross-field penalty)
                    v_local = g_vecs[i].clone()
                    if gen._fb_t is not None:
                        reg_lam = 0.05
                        reg_thresh = 0.2
                        for j in range(min(50, topk_idx.shape[1])):
                            rcid = int(topk_idx[i, j].item())
                            if rcid == gen_idxs[i]:
                                continue
                            rc = set(ctx for ctx, _ in gen_updates.get(gen_idxs[i], []))
                            if rcid in rc:
                                continue
                            rcos = float(topk_val[i, j].item())
                            if rcos > reg_thresh:
                                fb_gn = gen._fb_t[gen_idxs[i]]
                                fb_rn = gen._fb_t[rcid]
                                ro = int(torch.bitwise_and(fb_gn, fb_rn).sum().item())
                                if ro == 0:
                                    v_repel = gen._vecs_t[rcid].float()
                                    rep_grad = rcos * v_repel - v_local
                                    rn = rep_grad.norm()
                                    if rn > gen.max_grad_norm > 0:
                                        rep_grad = rep_grad / rn * gen.max_grad_norm
                                    v_local = v_local + rep_grad * contr_lrs[i] * reg_lam
                    if hn:
                        v_neg = gen._vecs_t[hn].float()
                        cos_v = best_val[i, :len(hn)]
                        grad = (cos_v[:, None] * v_neg).mean(dim=0) - v_local
                        gn = grad.norm()
                        if gn > gen.max_grad_norm > 0:
                            grad = grad / gn * gen.max_grad_norm
                        push = grad * contr_lrs[i]
                        v_new = v_local + push
                        nv = v_new.norm()
                        if nv > 1e-10:
                            v_new /= nv
                        cs._apply_vector_update(gen_idxs[i], v_new.cpu().numpy())
                    elif (v_local != g_vecs[i]).any():
                        nv = v_local.norm()
                        if nv > 1e-10:
                            v_local /= nv
                        cs._apply_vector_update(gen_idxs[i], v_local.cpu().numpy())

    # ═══════════════════════════════════════════════════
    # Centroid pull
    # ═══════════════════════════════════════════════════

    def _centroid_pull_batch(self, all_ids, base_lr_val):
        gen = self.gen
        cs = gen.cs
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

    # ═══════════════════════════════════════════════════
    # Evaluate
    # ═══════════════════════════════════════════════════

    def _evaluate(self, corpus_path, max_lines=500, use_torch=None):
        import time
        gen = self.gen
        cs = gen.cs
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
