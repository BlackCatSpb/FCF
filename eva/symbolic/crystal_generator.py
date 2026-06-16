"""CrystalGenerator — BPE-token concept navigation generator.

Generation as semantic navigation through a concept space where
each BPE token IS a concept. Input text is tokenized via SentencePiece,
producing a sequence of concept IDs (0..vocab_size-1). Generation
picks the next token ID via STDP-guided beam search over the fractal
concept field. Output is decoded back to text via SentencePiece.

Key simplifications vs old architecture:
  - No word->CID resolution: BPE tokens ARE concepts
  - No word-form selection: each CID has exactly one BPE token text
  - No special token stream: raw token IDs with SentencePiece BOS/EOS
"""

import math, os, random
import numpy as np
from collections import Counter
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.hormonal_system import HormonalSystem
from eva.symbolic.fcf_config import FCFConfig

CFG = FCFConfig()


_BOS_ID = 1
_EOS_ID = 2


class CrystalGenerator:
    """Generation as semantic navigation through BPE-token concept space."""

    def __init__(self, cs, sp, lattice, config=None):
        self.cs = cs
        self.sp = sp
        self.lattice = lattice
        self.config = config or {}

        self.max_words = self.config.get('max_words', 30)
        self.min_words = self.config.get('min_words', 3)
        self._graph_cache = {}
        self.base_concept_temp = self.config.get('concept_temp', 0.5)
        self.theta_tau = self.config.get('theta_tau', 12.0)
        self.base_learning_rate = self.config.get('learning_rate', 0.1)

        # Token diversity
        self.top_p = self.config.get('top_p', 0.9)
        self.len_norm_alpha = self.config.get('len_norm_alpha', 0.7)
        self.block_ngram = self.config.get('block_ngram', 4)
        self.mmi_lambda = self.config.get('mmi_lambda', 0.2)

        self.main_rng = random.Random(42)
        if not self.cs.concept_usage:
            self.cs.init_homeostasis()
        self.branch_rngs = {}
        self.hormones = HormonalSystem()

        # Per-concept prediction error EMA (Level 2: error-based PMI gate)
        self.concept_error = {}          # cid → EMA of (1 - cos(v_gen, v_ctx))
        self.concept_error_decay = 0.9  # EMA decay

        # Torch state (lazy init, invalidated by fluctuate_fractal)
        self._torch_device = None
        self._torch_cid_order = []
        self._torch_cid_to_idx = {}
        self._vecs_t = None
        self._fb_t = None
        self._basis_t = None
        self._torch_dirty = False  # set True after fluctuate → trigger rebuild

    def _ensure_torch(self, device=None):
        """Precompute GPU tensors for batched training. Rebuilds if dirty."""
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for GPU training.")
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dev = torch.device(device)
        if self._torch_device == dev and self._vecs_t is not None and not self._torch_dirty:
            return

        cs = self.cs
        V = cs.vocab_size
        D = cs.dim

        # Build tensors for ALL CIDs — zero-fill for those without codes
        cids = list(range(V))
        self._torch_cid_order = cids
        self._torch_cid_to_idx = {cid: i for i, cid in enumerate(cids)}
        self._torch_device = dev

        vecs = np.zeros((V, D), dtype=np.float32)  # ~225MB for V=146K, D=384
        if getattr(cs.fractal, 'codes', None) is not None and cs.fractal.basis is not None:
            basis = cs.fractal.basis
            for cid, code in cs.fractal.codes.items():
                v = code @ basis
                n = np.linalg.norm(v)
                if n > 1e-10:
                    vecs[cid] = v / n
        if hasattr(cs, 'concept_vectors') and cs.concept_vectors:
            for cid, v in cs.concept_vectors.items():
                if np.all(vecs[cid] == 0):
                    vecs[cid] = v

        self._vecs_t = torch.from_numpy(vecs).to(device)
        self._basis_t = torch.from_numpy(cs.fractal.basis.astype(np.float32)).to(device) if cs.fractal.basis is not None else None

        # Field bits for ALL CIDs — detect actual byte width from first entry
        if hasattr(cs.fractal, 'field_bits') and cs.fractal.field_bits:
            sample_fb = next(iter(cs.fractal.field_bits.values()))
            fb_bytes = len(np.frombuffer(sample_fb, dtype=np.uint8)) if isinstance(sample_fb, bytes) else len(sample_fb)
        else:
            fb_bytes = (getattr(cs, 'n_anchors', 1024) + 7) // 8
        fb_arr = np.zeros((V, fb_bytes), dtype=np.uint8)
        if hasattr(cs.fractal, 'field_bits'):
            for cid, fb in cs.fractal.field_bits.items():
                fb_arr[cid] = np.frombuffer(fb, dtype=np.uint8) if isinstance(fb, bytes) else fb
        self._fb_t = torch.from_numpy(fb_arr).to(device)

        self._torch_dirty = False
        if dev.type == 'cuda':
            torch.cuda.synchronize()

    def _invalidate_torch(self):
        """Mark GPU tensors as stale; triggers rebuild on next _ensure_torch.
        Call after fluctuate_fractal() or any code-level change."""
        self._torch_dirty = True

    # ── Temperature ────────────────────────────────────────────

    def _theta_temp(self, word_num):
        t = self.base_concept_temp * math.exp(-word_num / max(self.theta_tau, 1.0))
        return max(t, self.base_concept_temp * 0.15)

    # ── Encode / Decode ────────────────────────────────────────

    def _encode_input(self, text):
        return self.sp.encode(text, add_bos=True, add_eos=True)

    def _decode_tokens(self, token_ids):
        return self.sp.decode(token_ids)

    def _token_text(self, cid):
        try:
            return self.sp.IdToPiece(cid)
        except IndexError:
            return f'[CID{cid}]'

    def _is_semantic_token(self, cid):
        """Filter function words and punctuation that dominate graph connections."""
        text = self._token_text(cid).strip()
        if not text:
            return False
        # Punctuation and single non-letter characters
        if len(text) == 1 and not ('а' <= text.lower() <= 'я' or text.lower() == 'ё' or text.isalpha()):
            return False
        # Pure punctuation tokens
        if all(c in '.,!?;:()[]{}""''…—–«»' for c in text):
            return False
        return True

    # ── Generation ─────────────────────────────────────────────

    def generate(self, seed_word=None, seed_cid=None, target_text=None,
                 query_words=None, max_words=None, beam_width=3):
        """Generate a token sequence via beam search over concept IDs.

        Args:
            seed_word: starting word -> tokenized to seed CID
            seed_cid: starting concept ID (overrides seed_word)
            target_text: target for supervised training
            query_words: list of words from the query

        Returns:
            dict with response text, concept path, score, etc.
        """
        # Encode target if provided
        target_ids = self._encode_input(target_text) if target_text else []

        # Determine seed CID
        if seed_cid is None:
            if seed_word:
                token_ids = self._encode_input(seed_word)
                seed_cid = token_ids[0] if token_ids else _BOS_ID
            else:
                seed_cid = _BOS_ID

        # Encode query words to centroid vector
        src_ids = self._encode_input(' '.join(query_words)) if query_words else [seed_cid]
        query_vecs = [self.cs.concept_vector(cid)
                      for cid in src_ids if self.cs.concept_vector(cid) is not None]
        centroid = np.mean(query_vecs, axis=0).astype(np.float32) if query_vecs else None
        if centroid is not None:
            n = np.linalg.norm(centroid)
            if n > 1e-10:
                centroid /= n

        effective_max = max_words or self.max_words
        total_freq = max(sum(self.lattice.concept_freq.values()), 1)

        # Beam: list of (concept_sequence, score, branch_id)
        beam = [([seed_cid], 0.0, 0)]
        all_chains = []
        finished = []
        next_branch_id = 1

        for wn in range(effective_max):
            new_beam = []

            theta_temp = self._theta_temp(wn)
            h_temp = self.hormones.modulate_temperature(theta_temp)
            h_lr = self.hormones.modulate_stdp_lr(self.base_learning_rate)
            effective_beam = max(1, beam_width)

            for seq, score, branch_id in beam:
                prev_cid = seq[-1]
                expected_cid = target_ids[wn] if wn < len(target_ids) else None

                candidates = self._branch(seq, wn, h_temp, expected_cid, centroid)
                if not candidates:
                    self.hormones.update(confidence=0.0, is_match=False,
                        novelty=0.0, surprise=0.5, expected_cid=expected_cid)
                    finished.append((seq, score, wn))
                    continue

                for ci, (cid, cand_score) in enumerate(candidates):
                    new_seq = seq + [cid]
                    new_score = score + cand_score

                    # Anti-repetition
                    recent = seq[-6:] if len(seq) >= 6 else seq
                    count = recent.count(cid)
                    if count > 0:
                        new_score += -0.3 * count

                    # MMI: penalize high-frequency (generic) continuations
                    if self.mmi_lambda > 0:
                        p_cid = max(self.lattice.concept_freq.get(cid, 0) / total_freq, 1e-10)
                        new_score -= self.mmi_lambda * math.log(p_cid)

                    conf = 1.0 / (1.0 + ci * 0.5)
                    is_match = (expected_cid is not None and cid == expected_cid)

                    self.cs.update_usage(cid)

                    novelty = 1.0 - min(self.lattice.concept_freq.get(cid, 0) / 50, 1.0)
                    surprise = 0.1 if is_match else 0.5
                    self.hormones.update(confidence=conf, is_match=is_match,
                        novelty=novelty, surprise=surprise,
                        expected_cid=expected_cid, gen_cid=cid)

                    new_beam.append((new_seq, new_score, next_branch_id))
                    next_branch_id += 1

            new_beam.sort(key=lambda x: -x[1] / (len(x[0]) ** self.len_norm_alpha))
            beam = new_beam[:effective_beam]
            all_chains.extend([(s, sc) for s, sc, _ in new_beam])

            # EOS
            for item in list(beam):
                seq, score, bid = item
                if wn >= self.min_words:
                    token_text = self._token_text(seq[-1])
                    if token_text in ('.', '!', '?', '…', '...'):
                        finished.append((seq, score, wn))
                        beam.remove(item)

            if not beam:
                break

        if finished:
            best_seq, best_score, wn = max(finished, key=lambda x: x[1] / (len(x[0]) ** self.len_norm_alpha))
        elif beam:
            best_seq, best_score, _ = beam[0]
        else:
            return {'text': '', 'chains': all_chains}

        text = self._decode_tokens(best_seq)

        semantic_delta = 0.0
        if len(best_seq) >= 2:
            deltas = []
            for i in range(len(best_seq) - 1):
                v1 = self.cs.concept_vector(best_seq[i])
                v2 = self.cs.concept_vector(best_seq[i+1])
                if v1 is not None and v2 is not None:
                    deltas.append(1.0 - float(v1 @ v2))
            if deltas:
                semantic_delta = sum(deltas) / len(deltas)

        return {
            'text': text,
            'concept_path': best_seq,
            'score': best_score,
            'word_count': len(best_seq),
            'max_words': effective_max,
            'chains': all_chains,
            'semantic_delta': semantic_delta,
        }

    # ── Graph-based semantic search ──────────────────────────────

    def _graph_search(self, sources, B=2.0, max_candidates=30, max_depth=5):
        """BMSSP-EVA: single multi-source BFS for semantic paths.

        Args:
            sources: seed concept IDs
            B: distance budget (path cost threshold)
            max_candidates: max results to return
            max_depth: max BFS steps (safety bound, B is the primary limiter)
        """
        if not sources:
            return {}
        sources = list(set(sources))
        # Keep only semantic sources (no punctuation / function words)
        sources = [s for s in sources if self._is_semantic_token(s)]
        if not sources:
            return {}

        d = {}
        visited = set()
        # Track which source(s) reached each node
        origins = {}
        frontier = []

        for src_idx, s in enumerate(sources):
            d[s] = 0.0
            visited.add(s)
            origins[s] = {src_idx}
            frontier.append(s)

        step = 0
        while frontier and step < max_depth:
            step += 1
            next_frontier = []
            for u in frontier:
                conns = self.lattice.connections_of(u, top_k=8, use_ppmi=True)
                for v, conn_info in conns:
                    if not self._is_semantic_token(v):
                        continue
                    # Edge weight from PPMI: high PPMI = specific connection = low weight (short path)
                    ppmi = conn_info.get('ppmi', 0.0)
                    w = max(0.20, 1.0 - min(ppmi / 8.0, 1.0) * 0.7)
                    dv = d[u] + w
                    if dv >= B:
                        continue
                    if v not in visited:
                        d[v] = dv
                        visited.add(v)
                        origins[v] = origins.get(u, set())
                        next_frontier.append(v)
                    elif dv < d[v] - 0.01:
                        d[v] = dv
                        origins[v] |= origins.get(u, set())
            frontier = next_frontier

        for s in sources:
            d.pop(s, None)

        if not d:
            return {}

        # RRF: for each candidate, sum over unique sources reached
        total_src = len(sources)
        for cid, dist in d.items():
            n_src = len(origins.get(cid, {1}))
            rrf = (n_src / total_src) / (B + dist)
            d[cid] = rrf

        ranked = sorted(d.items(), key=lambda x: -x[1])
        return dict(ranked[:max_candidates])

    # ── Branch ─────────────────────────────────────────────────

    def _branch(self, seq, word_num, theta_temp=0.3, target_cid=None, centroid=None):
        """Generate diverse branching candidates via RRF over multiple signals."""
        prev_cid = seq[-1]
        cids = seq[-3:] if len(seq) >= 3 else seq
        K = 3

        # 1. Graph-based semantic paths (BMSSP-EVA, replaces single-hop connections)
        sources = list(set(cids))  # unique context tokens
        sources_key = tuple(sorted(set(sources)))
        if sources_key not in self._graph_cache:
            self._graph_cache[sources_key] = self._graph_search(sources, B=1.2, max_candidates=30)
        graph_candidates = self._graph_cache[sources_key]

        # 2. N-gram syntax (filter to semantic tokens only)
        syn_preds = self.lattice.predict(cids)
        syn_ranked = {cid: i + 1 for i, (cid, _) in enumerate(syn_preds[:80])
                      if self._is_semantic_token(cid)}

        # 3. All candidates from learned signals
        all_cids = set(graph_candidates.keys()) | set(syn_ranked.keys())

        # 4. Vector similarity fallback
        v_prev = self.cs.concept_vector(prev_cid)
        vector_sim = {}
        if v_prev is not None:
            sim_candidates = self.cs.topk_similar_concepts(prev_cid, k=20, sample_size=500)
            for cid, sim in sim_candidates:
                if cid not in all_cids and sim > 0.05:
                    all_cids.add(cid)
                vector_sim[cid] = sim

        if not all_cids:
            return []

        # 5. RRF scoring
        combined = {}
        for cid in all_cids:
            rrf = 0.0
            if cid in graph_candidates:
                rrf += 0.7 * graph_candidates[cid]
            if cid in syn_ranked:
                rrf += 0.15 / (K + syn_ranked[cid])
            if cid in vector_sim:
                rrf += 0.15 * vector_sim[cid] / (K + 1)
            freq = self.lattice.concept_freq.get(cid, 0)
            prior = 0.02 / (K + 1) * (1.0 - min(freq / 1000, 1.0))
            rrf += prior
            combined[cid] = rrf

        # 5. Homeostatic boost
        for cid in list(combined.keys()):
            h_boost = self.cs.homeostatic_boost(cid)
            combined[cid] *= (1.0 + h_boost * 0.3)

        # 6. Intent centroid bonus: prefer candidates near the query centroid
        if centroid is not None and np.linalg.norm(centroid) > 1e-10:
            cn = centroid / np.linalg.norm(centroid)
            for cid in list(combined.keys()):
                v = self.cs.concept_vector(cid)
                if v is not None:
                    sim_to_query = float(np.dot(v, cn))
                    # ideal: not too close (parroting), not too far (drift)
                    # bonus = 0.0 at sim=0, peaking at sim=0.5, then decays
                    intent_bonus = max(0, sim_to_query * (1.0 - sim_to_query)) * 0.3
                    combined[cid] *= (1.0 + intent_bonus)

        # 7. Anti-repetition + n-gram blocking
        recent = seq[-6:] if len(seq) >= 6 else seq
        ngram_set = set()
        if len(seq) >= self.block_ngram - 1:
            for i in range(len(seq) - (self.block_ngram - 2)):
                ngram_set.add(tuple(seq[i:i + self.block_ngram - 1]))
        for cid in list(combined.keys()):
            count = recent.count(cid)
            if count > 0:
                combined[cid] *= math.exp(-0.3 * count)
            if len(seq) >= self.block_ngram - 2:
                candidate_ngram = tuple(seq[-(self.block_ngram - 2):] + [cid])
                if candidate_ngram in ngram_set:
                    combined.pop(cid, None)
            if len(seq) >= 4 and cid == seq[-2] and seq[-1] == seq[-3]:
                combined.pop(cid, None)

        # 8. Field mask bonus: prefer candidates sharing field bits with context
        if hasattr(self.cs.fractal, 'field_bits') and len(self.cs.fractal.field_bits) > 0:
            ctx_cids = seq[-3:] if len(seq) >= 3 else seq
            ctx_field = None
            for cc in ctx_cids:
                fb = self.cs.fractal.get_field_bits(cc)
                if fb is not None:
                    if ctx_field is None:
                        ctx_field = fb.copy()
                    else:
                        ctx_field = np.bitwise_or(ctx_field, fb)
            if ctx_field is not None:
                for cid in list(combined.keys()):
                    fb_c = self.cs.fractal.get_field_bits(cid)
                    if fb_c is not None:
                        overlap = int(np.bitwise_and(ctx_field, fb_c).sum())
                        field_bonus = 1.0 + math.log(overlap + 1) * 0.1
                        combined[cid] *= field_bonus

        if not combined:
            return []

        # 7. Temperature softmax
        result = [(cid, max(s, 1e-10)) for cid, s in combined.items()]
        result.sort(key=lambda x: -x[1])
        scores = np.array([s for _, s in result], dtype=np.float64)
        scores = scores - scores.max()
        scores = np.clip(scores, -50, 50)
        temp = max(theta_temp, 0.01)
        probs = np.exp(scores / temp)
        probs /= probs.sum()

        # Target boosting
        if target_cid is not None and target_cid in self.cs.concept_vectors:
            for i, (cid, _) in enumerate(result):
                if cid == target_cid:
                    boost = 5.0 * (1.0 - theta_temp * 0.5)
                    probs[i] *= boost
                    break
            probs /= probs.sum()

        # Top-p (nucleus) sampling
        if self.top_p < 1.0 and theta_temp > 0.05:
            order = np.argsort(probs)[::-1]
            sorted_probs = probs[order]
            cumsum = np.cumsum(sorted_probs)
            cut = min(int((cumsum <= self.top_p).sum()) + 1, len(sorted_probs))
            cut = max(1, cut)
            truncated = sorted_probs[:cut].copy()
            truncated /= truncated.sum()
            n_candidates = min(15 + int(15 * theta_temp), cut)
            idx = np.random.choice(cut, size=n_candidates, replace=False, p=truncated)
            scored = [(result[order[i]][0], math.log(sorted_probs[i] + 1e-10)) for i in idx]
        else:
            n_candidates = min(15 + int(15 * theta_temp), len(result))
            scored = [(result[i][0], math.log(probs[i] + 1e-10))
                      for i in range(n_candidates)]
        return scored

    # ── PMI-gated STDP ─────────────────────────────────────────

    def _pmi_weight(self, prev_cid, next_cid, distance=1, total_freq=None, min_weight=0.05):
        """Pointwise Mutual Information weight for STDP pull strength.

        PMI = log(P(next|prev) / P(next))
        High PMI = specific, statistically surprising pair (e.g. князь→великий)
        Low PMI  = generic transition (e.g. а→также→в→качестве)
        Negative PMI = they avoid each other

        Uses adjacent ngrams for |i-j|=1, skip2 dict for |i-j|=2.

        Maps to [min_weight, 2.0] multiplier on learning rate.

        Args:
            total_freq: cached sum(concept_freq.values()), computed once per line
            min_weight: floor for the PMI multiplier (tunable via pmi_gate_min)
        """
        if total_freq is None:
            total_freq = sum(self.lattice.concept_freq.values())
        if total_freq < 1:
            return 0.1

        if distance == 1:
            prefix_counter = self.lattice.ngrams[2].get((prev_cid,))
            if not prefix_counter:
                return 0.1
            count_pair = prefix_counter.get(next_cid, 0)
            count_prev = self.lattice._prefix_total.get((prev_cid,), 0)
        elif distance == 2:
            skip2 = self.lattice.skip2.get(prev_cid)
            if not skip2:
                return 0.1
            count_pair = skip2.get(next_cid, 0)
            count_prev = self.lattice._skip2_total.get(prev_cid, 0)
        else:
            return 0.1

        count_next = self.lattice.concept_freq.get(next_cid, 0)
        if count_pair < 1 or count_prev < 1 or count_next < 1:
            return 0.1

        p_next_given_prev = count_pair / count_prev
        p_next = count_next / total_freq
        pmi = math.log(max(p_next_given_prev, 1e-10) / max(p_next, 1e-10))

        # PMI=0 → 0.2, PMI=2 → 1.0, PMI=5 → 2.0, negative → min_weight
        return max(min(pmi / 2.0 + 0.2, 2.0), min_weight)

    def _apply_pmi_gate(self, pmi_w_raw, pmi_strength, pmi_gate_min, cid):
        """Apply PMI gate: returns (skip: bool, pmi_w: float)."""
        if pmi_strength >= 0.01:
            effective_min = pmi_gate_min * pmi_strength
            per_cid_factor = max(0.25, 1.0 - self.concept_error.get(cid, 0.0) * 0.75)
            per_cid_min = pmi_gate_min * per_cid_factor
            use_min = min(effective_min, per_cid_min)
            if pmi_w_raw <= use_min:
                return True, 0.0
            pmi_w = 1.0 + (pmi_w_raw - 1.0) * pmi_strength
        else:
            pmi_w = 1.0
        return False, pmi_w

    # ── Training ───────────────────────────────────────────────

    def _gpu_stdp_apply(self, gpu_ctx_l, gpu_tgt_l, gpu_meta_l, gpu_cid_gen, base_lr_val,
                         field_gate, inh_strength, inh_threshold, destab_scale):
        """GPU batched STDP: field overlaps, LR compute, scatter_add, updates, lateral inhibition."""
        cs = self.cs
        device = self._torch_device
        ctx_t = torch.tensor(gpu_ctx_l, dtype=torch.long, device=device)
        tgt_t = torch.tensor(gpu_tgt_l, dtype=torch.long, device=device)
        N = len(gpu_ctx_l)

        with torch.no_grad():
            if field_gate and self._fb_t is not None:
                ctx_fb = self._fb_t[ctx_t]
                tgt_fb = self._fb_t[tgt_t]
                ovs = (ctx_fb & tgt_fb).sum(dim=1).float()
            else:
                ovs = torch.zeros(N, device=device)

            meta_t = torch.tensor(gpu_meta_l, dtype=torch.float32, device=device)
            i_pos = meta_t[:, 0]
            j_pos = meta_t[:, 1]
            dist = j_pos - i_pos
            pmi_w_t = meta_t[:, 2]
            dw_t = meta_t[:, 3]
            fw_t = meta_t[:, 4]

            field_w = torch.where(
                ovs > 0,
                1.0 + torch.log(ovs + 1.0) * 2.0,
                torch.full_like(ovs, 0.1))
            lr = torch.clamp(fw_t, min=0.05) * dw_t * pmi_w_t * field_w
            theta = torch.exp(-torch.clamp(dist, max=5.0) / max(self.theta_tau, 1.0))
            effective_lr = lr * torch.clamp(theta, min=0.1)

            gen_cids_arr = np.array(gpu_cid_gen, dtype=np.int32)
            unique_gen, inv_idx = np.unique(gen_cids_arr, return_inverse=True)
            inv_t = torch.from_numpy(inv_idx).to(device)

            elr_grouped = torch.zeros(len(unique_gen), device=device)
            elr_grouped.scatter_add_(0, inv_t, effective_lr)

            vc = self._vecs_t[ctx_t]
            vg = self._vecs_t[tgt_t]
            y = torch.clamp((vg * vc).sum(dim=1), min=0.05)
            pair_delta = vc * effective_lr[:, None] - vg * (y * effective_lr)[:, None]

            D = cs.dim
            acc = torch.zeros(len(unique_gen), D, dtype=torch.float32, device=device)
            acc.scatter_add_(0, inv_t[:, None].expand(-1, D), pair_delta)
            cnt = torch.zeros(len(unique_gen), device=device)
            cnt.scatter_add_(0, inv_t, torch.ones(N, device=device))

            # Per-concept prediction error tracking (Level 2, deduplicated)
            err_per_pair = 1.0 - y
            err_grouped = torch.zeros(len(unique_gen), device=device)
            err_grouped.scatter_add_(0, inv_t, err_per_pair)
            cnt_err = torch.zeros(len(unique_gen), device=device)
            cnt_err.scatter_add_(0, inv_t, torch.ones(N, device=device))
            avg_err_cpu = (err_grouped / cnt_err.clamp(min=1)).cpu().numpy()
            for gi, gen_cid in enumerate(unique_gen):
                err = float(avg_err_cpu[gi])
                old = self.concept_error.get(gen_cid, err)
                self.concept_error[gen_cid] = self.concept_error_decay * old + (1 - self.concept_error_decay) * err

        acc_cpu = acc.cpu().numpy()
        cnt_cpu = cnt.cpu().numpy()
        elr_cpu = elr_grouped.cpu().numpy()

        # Apply updates + lateral inhibition (batched on GPU)
        gen_vecs = []
        gen_cids_list = []
        for gi, gen_cid in enumerate(unique_gen):
            v = cs.concept_vectors.get(gen_cid)
            if v is None or cnt_cpu[gi] < 0.5:
                continue
            if destab_scale > 0 and self.main_rng.random() < destab_scale * 0.3:
                ppmi_candidates = self.lattice.connections_of(
                    gen_cid, top_k=20, use_ppmi=True)
                if ppmi_candidates:
                    ppmi_cid = ppmi_candidates[self.main_rng.randint(0, len(ppmi_candidates) - 1)][0]
                    v_ppmi = cs.concept_vectors.get(ppmi_cid)
                    if v_ppmi is not None:
                        y_ppmi = max(float(np.dot(v, v_ppmi)), 0.05)
                        noise = (v_ppmi - y_ppmi * v)
                        nlen = float(np.linalg.norm(noise))
                        if nlen > 1e-10:
                            noise /= nlen
                            mix = min(destab_scale, 0.5)
                            acc_cpu[gi] = acc_cpu[gi] * (1 - mix) + noise * mix

            grad = acc_cpu[gi] / max(elr_cpu[gi], 1e-10)
            v_new = v + grad * base_lr_val
            nv = np.linalg.norm(v_new)
            if nv > 1e-10:
                v_new /= nv
            cs._apply_vector_update(gen_cid, v_new)
            gen_vecs.append(v_new)
            gen_cids_list.append(gen_cid)

        # Lateral inhibition: CPU (sampling) for small batches (<50 gen), GPU (full-V) for large
        if gen_vecs and inh_strength > 0:
            if len(gen_vecs) < 50:
                for gi, gen_cid in enumerate(gen_cids_list):
                    total_elr = float(elr_cpu[gi])
                    str_val = inh_strength * total_elr
                    if str_val < 1e-8:
                        continue
                    cs._lateral_inhibition_fractal(
                        gen_cid,
                        strength=str_val,
                        threshold=max(inh_threshold, 0.01),
                        sample_size=min(100, len(cs.concept_vectors)),
                    )
            else:
                gv_t = torch.from_numpy(np.array(gen_vecs, dtype=np.float32)).to(device)
                gv_all = self._vecs_t
                sims = gv_t @ gv_all.T
                for gi, gen_cid in enumerate(gen_cids_list):
                    threshold = max(inh_threshold, 0.01)
                    total_elr = float(elr_cpu[gi])
                    str_val = inh_strength * total_elr
                    if str_val < 1e-8:
                        continue
                    below = sims[gi] < -0.5
                    above = sims[gi] > threshold
                    inhibit = above & ~below
                    if inhibit.sum() < 1:
                        continue
                    targets = torch.where(inhibit)[0]
                    if len(targets) > 100:
                        top_sim, top_idx = torch.topk(sims[gi], k=100)
                        targets = top_idx[top_sim > threshold]
                    if len(targets) < 1:
                        continue
                    v_self = gv_t[gi]
                    v_opp = gv_all[targets]
                    dot = (v_self * v_opp).sum(dim=1)
                    delta = str_val * (dot[:, None] * v_opp - v_self[None, :])
                    delta_cpu = delta.cpu().numpy()
                    target_cids = targets.cpu().numpy()
                    unique_cids, inv_idx = np.unique(target_cids, return_inverse=True)
                    delta_sum = np.zeros((len(unique_cids), delta_cpu.shape[1]), dtype=np.float32)
                    np.add.at(delta_sum, inv_idx, delta_cpu)
                    for ui, ucid in enumerate(unique_cids):
                        v_t = cs.concept_vectors.get(int(ucid))
                        if v_t is not None:
                            v_new_t = v_t + delta_sum[ui]
                            nt = np.linalg.norm(v_new_t)
                            if nt > 1e-10:
                                v_new_t /= nt
                            cs._apply_vector_update(int(ucid), v_new_t)

        return unique_gen

    def _cpu_stdp_apply(self, gen_updates, base_lr_val, destab_scale, inh_strength, inh_threshold):
        """One combined STDP update per unique gen_cid (batched numpy) with destab support."""
        cs = self.cs
        best_gen_cid = None
        best_total_elr = 0.0
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

            # Per-concept prediction error tracking (Level 2, deduplicated)
            err = 1.0 - float(np.mean(y))
            old = self.concept_error.get(gen_cid, err)
            self.concept_error[gen_cid] = self.concept_error_decay * old + (1 - self.concept_error_decay) * err

            total_delta = ((ctx_mat * elr_arr[:, None]).sum(axis=0) -
                          v_gen * (y * elr_arr).sum())

            if n_updates > 0 and total_elr > 0:
                grad = total_delta / max(total_elr, 1e-10)

                if destab_scale > 0 and self.main_rng.random() < destab_scale * 0.3:
                    ppmi_candidates = self.lattice.connections_of(
                        gen_cid, top_k=20, use_ppmi=True)
                    if ppmi_candidates:
                        ppmi_cid = ppmi_candidates[self.main_rng.randint(0, len(ppmi_candidates) - 1)][0]
                        v_ppmi = cs.concept_vectors.get(ppmi_cid)
                        if v_ppmi is not None:
                            y_ppmi = max(float(np.dot(v_gen, v_ppmi)), 0.05)
                            noise = (v_ppmi - y_ppmi * v_gen)
                            nlen = float(np.linalg.norm(noise))
                            if nlen > 1e-10:
                                noise /= nlen
                                mix = min(destab_scale, 0.5)
                                grad = grad * (1 - mix) + noise * mix

                v_new = v_gen + grad * base_lr_val
            else:
                v_new = v_gen

            nv = np.linalg.norm(v_new)
            if nv > 1e-10:
                v_new /= nv

            cs._apply_vector_update(gen_cid, v_new)

            if total_elr > best_total_elr:
                best_total_elr = total_elr
                best_gen_cid = gen_cid

        # ── Lateral inhibition (single call, not per gen_cid) ──
        if best_gen_cid is not None and inh_strength > 0:
            str_val = inh_strength * best_total_elr
            if str_val >= 1e-8:
                cs._lateral_inhibition_fractal(
                    best_gen_cid,
                    strength=str_val,
                    threshold=max(inh_threshold, 0.01),
                    sample_size=min(100, len(cs.concept_vectors)),
                )

    def _negative_sampling_gpu(self, gpu_ctx_l, gpu_meta_l, gpu_cid_ctx, device, field_gate, base_lr_val, neg_lr_ratio, neg_samples):
        """GPU-vectorized negative sampling (fully batched)."""
        cs = self.cs
        n_pairs = len(gpu_ctx_l)
        if n_pairs == 0:
            return

        ctx_t = torch.tensor(gpu_ctx_l, dtype=torch.long, device=device)

        with torch.no_grad():
            neg_idxs = torch.randint(0, len(self._torch_cid_order),
                                     (n_pairs, neg_samples), device=device)

            # Field overlap filter
            # When _fb_t is None (no field_bits loaded), all negatives pass as valid
            if field_gate and self._fb_t is not None:
                ctx_fb = self._fb_t[ctx_t]
                neg_fb = self._fb_t[neg_idxs]
                neg_ovs = (ctx_fb.unsqueeze(1) & neg_fb).sum(dim=2)
                valid_mask = neg_ovs == 0
            else:
                valid_mask = torch.ones(n_pairs, neg_samples, dtype=torch.bool, device=device)

            # Flatten: collect all valid (pair, neg) combos
            valid_idx = torch.nonzero(valid_mask)  # (N_valid, 2) — [pi, ni]
            if valid_idx.numel() == 0:
                return
            valid_pi = valid_idx[:, 0]
            valid_ni = valid_idx[:, 1]
            n_valid = len(valid_pi)

            # LR per valid sample (vectorized from meta)
            meta_arr = np.array(gpu_meta_l, dtype=np.float32)
            i_arr = meta_arr[:, 0].astype(np.int32)
            j_arr = meta_arr[:, 1].astype(np.int32)
            dist_arr = np.abs(j_arr - i_arr)
            theta_gate_arr = np.exp(-np.minimum(dist_arr, 5.0) / max(self.theta_tau, 1.0))
            neg_elr_arr = np.maximum(meta_arr[:, 4], 0.05) * meta_arr[:, 3] * meta_arr[:, 2] * neg_lr_ratio * np.maximum(theta_gate_arr, 0.1)
            valid_elr_t = torch.from_numpy(neg_elr_arr[valid_pi.cpu().numpy()]).to(device)

            # Gather neg CIDs and vectors from _vecs_t
            neg_cids = torch.tensor(self._torch_cid_order, device=device)[neg_idxs]  # (n_pairs, neg_samples)
            valid_neg_cids = neg_cids[valid_pi, valid_ni]  # (n_valid,)
            valid_ctx_cids = torch.tensor(gpu_cid_ctx, device=device)[valid_pi]  # (n_valid,)

            v_neg = self._vecs_t[valid_neg_cids]   # (n_valid, D)
            v_ctx = self._vecs_t[valid_ctx_cids]    # (n_valid, D)

            # Vectorized push-away: shift = (y * v_neg - v_ctx) * elr
            sims = (v_neg * v_ctx).sum(dim=1, keepdim=True).clamp(min=0.05)  # (n_valid, 1)
            shifts = (sims * v_neg - v_ctx) * valid_elr_t[:, None]  # (n_valid, D)

            # Accumulate shifts + total_elr by unique neg CID (scatter_add_), normalize, apply base_lr_val
            unique_neg_cids, inverse = torch.unique(valid_neg_cids, return_inverse=True)
            n_unique = len(unique_neg_cids)
            D = v_neg.shape[1]
            acc_shifts = torch.zeros(n_unique, D, device=device)
            acc_shifts.scatter_add_(0, inverse[:, None].expand(-1, D), shifts)
            acc_elr = torch.zeros(n_unique, device=device)
            acc_elr.scatter_add_(0, inverse, valid_elr_t)

            # Normalize by total_elr, then apply base_lr_val as step size (matching CPU STDP pattern)
            v_cur = self._vecs_t[unique_neg_cids]  # (n_unique, D)
            grad = acc_shifts / acc_elr[:, None].clamp(min=1e-10)
            v_new = v_cur + grad * base_lr_val
            norms = torch.norm(v_new, dim=1, keepdim=True).clamp(min=1e-10)
            v_new = v_new / norms

            # Apply — one _apply_vector_update per unique neg CID
            for ui in range(n_unique):
                cs._apply_vector_update(int(unique_neg_cids[ui]), v_new[ui].cpu().numpy())

    def _negative_sampling_cpu(self, gen_updates, neg_lr_ratio, field_gate, neg_samples):
        """Original negative sampling (non-torch path)."""
        cs = self.cs
        for gen_cid, updates in gen_updates.items():
            for prev_cid, elr in updates:
                neg_elr = elr * neg_lr_ratio
                v_ctx = cs.get_vec(prev_cid)
                if v_ctx is None:
                    continue
                for _ in range(neg_samples):
                    neg_cid = cs.rng.randint(0, cs.vocab_size)
                    v_neg = cs.get_vec(neg_cid)
                    if v_neg is None:
                        continue
                    if field_gate and hasattr(cs.fractal, 'field_bits') and len(cs.fractal.field_bits) > 0:
                        neg_overlap = cs.fractal.field_overlap(prev_cid, neg_cid)
                        if neg_overlap > 0:
                            continue
                    y = max(float(np.dot(v_neg, v_ctx)), 0.05)
                    shift = (y * v_neg - v_ctx) * neg_elr
                    v_new = v_neg + shift
                    nv = np.linalg.norm(v_new)
                    if nv > 1e-10:
                        v_new /= nv
                    cs._apply_vector_update(neg_cid, v_new)

    def _contrastive_objective(self, gen_updates):
        """Hard-negative push for similar non-co-occurring pairs."""
        cs = self.cs
        for gen_cid, updates in gen_updates.items():
            v_gen = cs.concept_vectors.get(gen_cid)
            if v_gen is None:
                continue
            avg_elr = sum(elr for _, elr in updates) / max(len(updates), 1)
            contr_lr = avg_elr * 0.3

            n_candidates = min(80, cs.vocab_size)
            candidates = cs.rng.randint(0, cs.vocab_size, size=n_candidates)

            best_cos = 0.05
            best_neg = None
            best_v_neg = None
            for neg_cid in candidates:
                if neg_cid == gen_cid:
                    continue
                if neg_cid in (ctx_cid for ctx_cid, _ in updates):
                    continue
                if self.lattice.connection_strength(gen_cid, neg_cid) > 0.1:
                    continue

                v_neg = cs.concept_vectors.get(neg_cid)
                if v_neg is None:
                    continue
                cos_val = float(np.dot(v_gen, v_neg))
                if cos_val > best_cos and cos_val < 0.5:
                    best_cos = cos_val
                    best_neg = neg_cid
                    best_v_neg = v_neg

            if best_neg is not None:
                push = (best_cos * best_v_neg - v_gen) * contr_lr
                v_new = v_gen + push
                nv = np.linalg.norm(v_new)
                if nv > 1e-10:
                    v_new /= nv
                cs._apply_vector_update(gen_cid, v_new)

    def _centroid_pull(self, ids, base_lr_val):
        """Sentence-level centroid pull (CBOW-like, raw vectors only)."""
        cs = self.cs
        if len(ids) >= 3:
            sent_vecs = [cs.concept_vectors.get(c) for c in ids]
            sent_vecs = [v for v in sent_vecs if v is not None]
            if len(sent_vecs) >= 3:
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

    def train_from_text(self, text, base_lr=None, context_window=2, pmi_strength=1.0, pmi_gate_min=0.20, neg_samples=1,
                        inh_strength=0.05, inh_threshold=0.10, neg_lr_ratio=0.5, field_gate=True, use_torch=None,
                        destab_scale=0.0):
        """Train via PMI-gated context-window STDP with optional GPU batching.

        Same STDP logic as train_from_text, but with GPU batched compute
        for the hot path (dot products, field overlaps, negative sampling checks).
        """
        if use_torch is None:
            use_torch = CFG.use_torch
        if use_torch:
            self._ensure_torch()
            if self._vecs_t is None:
                use_torch = False  # fallback to numpy

        ids = self._encode_input(text)
        if len(ids) < 2:
            return 0

        base_lr = base_lr if base_lr is not None else getattr(self, 'train_lr', 0.01)
        base_lr_val = base_lr
        cs = self.cs
        total_freq = max(sum(self.lattice.concept_freq.values()), 1)
        T = len(ids)
        device = self._torch_device if use_torch else None
        cid_to_idx = self._torch_cid_to_idx if use_torch else {}

        # ── Build pairs, group by gen_cid ──
        from collections import defaultdict
        gen_updates = defaultdict(list)

        # Collect indices for GPU batch
        gpu_ctx_l = []   # tensor indices for GPU
        gpu_tgt_l = []
        gpu_meta_l = []  # (i, j_pos, pmi_w, dist_weight, freq_weight) for STDP
        gpu_cid_ctx = []  # raw CID for context
        gpu_cid_gen = []  # raw CID for generated

        for i in range(T):
            start = max(0, i - context_window)
            end = min(T, i + context_window + 1)
            for j in range(start, end):
                if j <= i:
                    continue
                dist = abs(j - i)
                dist_weight = math.exp(-dist / 2.0)

                fa = self.lattice.concept_freq.get(ids[i], 0)
                fb = self.lattice.concept_freq.get(ids[j], 0)
                freq_weight = 1.0 / (1.0 + math.log(max(max(fa, fb), 1)) * 0.15)

                # ── Continuous PMI gate (Level 1) + per-concept error threshold (Level 2) ──
                pmi_w_raw = self._pmi_weight(ids[i], ids[j], distance=dist, total_freq=total_freq,
                                              min_weight=pmi_gate_min)
                _skip, pmi_w = self._apply_pmi_gate(pmi_w_raw, pmi_strength, pmi_gate_min, ids[j])
                if _skip:
                    continue

                # Field gate (compute unconditionally for contrastive objective)
                field_weight = 1.0
                if field_gate and hasattr(cs.fractal, 'field_bits') and len(cs.fractal.field_bits) > 0:
                    overlap = cs.fractal.field_overlap(ids[i], ids[j])
                    field_weight = 1.0 + math.log(overlap + 1) * 2.0 if overlap > 0 else 0.1

                lr = base_lr * max(freq_weight, 0.05) * dist_weight * pmi_w * field_weight
                theta_gate = math.exp(-min(abs(j-i), 5) / max(self.theta_tau, 1.0))
                gen_updates[ids[j]].append((ids[i], lr * max(theta_gate, 0.1)))

                if use_torch:
                    ci = cid_to_idx.get(ids[i])
                    cj = cid_to_idx.get(ids[j])
                    if ci is None or cj is None:
                        continue
                    gpu_ctx_l.append(ci)
                    gpu_tgt_l.append(cj)
                    gpu_meta_l.append((i, j, pmi_w, dist_weight, freq_weight))
                    gpu_cid_ctx.append(ids[i])
                    gpu_cid_gen.append(ids[j])

        # ── GPU STDP / CPU STDP ──
        if use_torch and gpu_ctx_l:
            unique_gen = self._gpu_stdp_apply(gpu_ctx_l, gpu_tgt_l, gpu_meta_l, gpu_cid_gen, base_lr_val,
                                               field_gate, inh_strength, inh_threshold, destab_scale)
        else:
            self._cpu_stdp_apply(gen_updates, base_lr_val, destab_scale, inh_strength, inh_threshold)

        # ── Negative sampling ──
        if neg_samples > 0 and use_torch and self._vecs_t is not None and gpu_ctx_l:
            self._negative_sampling_gpu(gpu_ctx_l, gpu_meta_l, gpu_cid_ctx, device, field_gate,
                                         base_lr_val, neg_lr_ratio, neg_samples)
        elif neg_samples > 0:
            self._negative_sampling_cpu(gen_updates, neg_lr_ratio, field_gate, neg_samples)

        # ── Mark torch as dirty (vectors modified via _apply_vector_update) ──
        if use_torch:
            self._torch_dirty = True

        # ── Contrastive objective ──
        self._contrastive_objective(gen_updates)
        self._centroid_pull(ids, base_lr_val)

        self.lattice.update(ids)
        self._graph_cache.clear()

        # Prune concept_error cache
        if len(self.concept_error) > 50000:
            pruned = dict(sorted(self.concept_error.items(), key=lambda x: -x[1])[:30000])
            self.concept_error = pruned

        return 1

    def train_batch(self, texts, base_lr=None, context_window=2, pmi_strength=1.0, pmi_gate_min=0.20,
                    neg_samples=1, inh_strength=0.05, inh_threshold=0.10, neg_lr_ratio=0.5,
                    field_gate=True, use_torch=None, destab_scale=0.0):
        """Process multiple texts in one GPU batch for higher throughput.

        Builds pairs from all texts, does a single GPU STDP call,
        then per-text centroid pull + lattice update.
        """
        if use_torch is None:
            use_torch = CFG.use_torch
        if use_torch:
            self._ensure_torch()
            if self._vecs_t is None:
                use_torch = False

        all_ids = []
        for text in texts:
            ids = self._encode_input(text)
            if len(ids) >= 2:
                all_ids.append(ids)
        if not all_ids:
            return 0

        base_lr = base_lr if base_lr is not None else getattr(self, 'train_lr', 0.01)
        base_lr_val = base_lr
        cs = self.cs
        total_freq = max(sum(self.lattice.concept_freq.values()), 1)
        device = self._torch_device if use_torch else None
        cid_to_idx = self._torch_cid_to_idx if use_torch else {}

        from collections import defaultdict
        gen_updates = defaultdict(list)

        gpu_ctx_l = []
        gpu_tgt_l = []
        gpu_meta_l = []
        gpu_cid_ctx = []
        gpu_cid_gen = []

        for ids in all_ids:
            T = len(ids)
            for i in range(T):
                start = max(0, i - context_window)
                end = min(T, i + context_window + 1)
                for j in range(start, end):
                    if j <= i:
                        continue
                    dist = abs(j - i)
                    dist_weight = math.exp(-dist / 2.0)

                    fa = self.lattice.concept_freq.get(ids[i], 0)
                    fb = self.lattice.concept_freq.get(ids[j], 0)
                    freq_weight = 1.0 / (1.0 + math.log(max(max(fa, fb), 1)) * 0.15)

                    pmi_w_raw = self._pmi_weight(ids[i], ids[j], distance=dist, total_freq=total_freq,
                                                  min_weight=pmi_gate_min)
                    _skip, pmi_w = self._apply_pmi_gate(pmi_w_raw, pmi_strength, pmi_gate_min, ids[j])
                    if _skip:
                        continue

                    # Field gate (compute unconditionally for contrastive objective)
                    field_weight = 1.0
                    if field_gate and hasattr(cs.fractal, 'field_bits') and len(cs.fractal.field_bits) > 0:
                        overlap = cs.fractal.field_overlap(ids[i], ids[j])
                        field_weight = 1.0 + math.log(overlap + 1) * 2.0 if overlap > 0 else 0.1

                    lr = base_lr * max(freq_weight, 0.05) * dist_weight * pmi_w * field_weight
                    theta_gate = math.exp(-min(abs(j-i), 5) / max(self.theta_tau, 1.0))
                    gen_updates[ids[j]].append((ids[i], lr * max(theta_gate, 0.1)))

                    if use_torch:
                        ci = cid_to_idx.get(ids[i])
                        cj = cid_to_idx.get(ids[j])
                        if ci is None or cj is None:
                            continue
                        gpu_ctx_l.append(ci)
                        gpu_tgt_l.append(cj)
                        gpu_meta_l.append((i, j, pmi_w, dist_weight, freq_weight))
                        gpu_cid_ctx.append(ids[i])
                        gpu_cid_gen.append(ids[j])

        # ── GPU STDP (single call for all pairs) ──
        if use_torch and gpu_ctx_l:
            unique_gen = self._gpu_stdp_apply(gpu_ctx_l, gpu_tgt_l, gpu_meta_l, gpu_cid_gen, base_lr_val,
                                               field_gate, inh_strength, inh_threshold, destab_scale)
        else:
            self._cpu_stdp_apply(gen_updates, base_lr_val, destab_scale, inh_strength, inh_threshold)

        # ── Negative sampling (GPU, single call) ──
        if neg_samples > 0 and use_torch and self._vecs_t is not None and gpu_ctx_l:
            self._negative_sampling_gpu(gpu_ctx_l, gpu_meta_l, gpu_cid_ctx, device, field_gate,
                                         base_lr_val, neg_lr_ratio, neg_samples)
        elif neg_samples > 0:
            self._negative_sampling_cpu(gen_updates, neg_lr_ratio, field_gate, neg_samples)

        # ── Mark torch as dirty (vectors modified via _apply_vector_update) ──
        if use_torch:
            self._torch_dirty = True

        # ── Contrastive objective (run ONCE, not per-text) ──
        self._contrastive_objective(gen_updates)

        # ── Per-text centroid pull + lattice update ──
        for ids in all_ids:
            self._centroid_pull(ids, base_lr_val)
            self.lattice.update(ids)
            self._graph_cache.clear()

        # Prune concept_error cache
        if len(self.concept_error) > 50000:
            pruned = dict(sorted(self.concept_error.items(), key=lambda x: -x[1])[:30000])
            self.concept_error = pruned
        if len(self._graph_cache) > 1000:
            self._graph_cache.clear()

        return len(all_ids)

    # ── Evaluation ────────────────────────────────────────────

    def evaluate(self, corpus_path, max_lines=None, batch_size=500, use_gpu=True):
        """Compute perplexity and accuracy on held-out corpus.

        GPU-accelerated: uses _vecs_t for batch matmul when use_gpu=True.
        """
        import time

        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        if max_lines:
            lines = lines[:max_lines]

        all_ids = []
        for line in lines:
            ids = self._encode_input(line)
            if len(ids) >= 2:
                all_ids.extend(ids)

        n_positions = len(all_ids) - 1
        if n_positions < 1:
            return {'perplexity': float('inf'), 'accuracy_top1': 0.0,
                    'accuracy_top5': 0.0, 'n_tokens': 0}

        cids = sorted(self.cs.concept_vectors.keys())
        cid_to_idx = {c: i for i, c in enumerate(cids)}
        vocab_size = len(cids)
        K = 3

        # CPU path: build V matrix, prior, ngram_boost (same as before)
        if not use_gpu or self._vecs_t is None:
            V = np.array([self.cs.concept_vectors[c] for c in cids], dtype=np.float32)
            norms = np.linalg.norm(V, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            V /= norms

        total_freq = sum(self.lattice.concept_freq.values()) or 1.0
        prior_arr = np.zeros(vocab_size, dtype=np.float32)
        for i, c in enumerate(cids):
            freq = self.lattice.concept_freq.get(c, 0)
            prior_arr[i] = 0.02 / (K + 1) * (1.0 - min(freq / 1000, 1.0))

        # Ngram+PMI boost
        ngram_boost = {}
        for (prev_cid,), counter in self.lattice.ngrams[2].items():
            total_ng = sum(counter.values())
            if total_ng < 1:
                continue
            boost_map = {}
            for ncid, ncount in counter.items():
                idx = cid_to_idx.get(ncid)
                if idx is None:
                    continue
                prob = ncount / total_ng
                count_next = self.lattice.concept_freq.get(ncid, 0)
                if count_next < 1:
                    pmi_w = 0.1
                else:
                    p_next_given_prev = ncount / total_ng
                    p_next = count_next / total_freq
                    pmi = math.log(max(p_next_given_prev, 1e-10) / max(p_next, 1e-10))
                    pmi_w = max(min(pmi / 2.0 + 0.2, 2.0), 0.05)
                boost_map[ncid] = (0.25 * prob + 0.5 * pmi_w) / (K + 1)
            ngram_boost[prev_cid] = boost_map

        total_log_prob = 0.0
        vec_log_prob = 0.0
        correct_top1 = 0
        correct_top5 = 0
        vec_correct_top1 = 0
        n_eval = 0
        t0 = time.time()

        if use_gpu and self._torch_dirty:
            self._ensure_torch()
        use_cuda = use_gpu and self._vecs_t is not None
        if use_cuda:
            device = self._torch_device
            V_gpu = self._vecs_t

        for start in range(0, n_positions, batch_size):
            end = min(start + batch_size, n_positions)
            batch_prev = all_ids[start:end]
            batch_next = all_ids[start + 1:end + 1]
            batch_n = len(batch_prev)

            if use_cuda:
                # GPU matmul: build prev_vecs on CPU, transfer, compute sims, bring back
                prev_vecs = np.array([
                    self.cs.concept_vectors.get(c, np.zeros(self.cs.dim, dtype=np.float32))
                    for c in batch_prev
                ], dtype=np.float32)
                pn = np.linalg.norm(prev_vecs, axis=1, keepdims=True)
                pn[pn < 1e-10] = 1.0
                prev_vecs /= pn
                with torch.no_grad():
                    pv_t = torch.from_numpy(prev_vecs).to(device, non_blocking=True)
                    sims_t = pv_t @ V_gpu.T  # (batch, vocab_size)
                    sims_t = torch.clamp(sims_t, min=0)
                    sims = sims_t.cpu().numpy()
            else:
                # CPU matmul (original path)
                prev_vecs = np.array([
                    self.cs.concept_vectors.get(c, np.zeros(self.cs.dim, dtype=np.float32))
                    for c in batch_prev
                ], dtype=np.float32)
                pn = np.linalg.norm(prev_vecs, axis=1, keepdims=True)
                pn[pn < 1e-10] = 1.0
                prev_vecs /= pn
                sims = prev_vecs @ V.T
                sims = np.maximum(sims, 0)

            for pos in range(batch_n):
                prev_cid = batch_prev[pos]
                next_cid = batch_next[pos]

                scores = prior_arr.copy()
                scores += 0.15 * sims[pos] / (K + 1)

                boost = ngram_boost.get(prev_cid)
                if boost:
                    for ncid, bval in boost.items():
                        idx = cid_to_idx.get(ncid)
                        if idx is not None:
                            scores[idx] += bval

                scores -= scores.max()
                scores = np.clip(scores, -50, 50)
                exp_s = np.exp(scores)
                probs = exp_s / exp_s.sum()

                actual_idx = cid_to_idx.get(next_cid)
                if actual_idx is not None:
                    lp = np.log(max(probs[actual_idx], 1e-30))
                    total_log_prob += lp
                    if cids[np.argmax(scores)] == next_cid:
                        correct_top1 += 1
                    if next_cid in {cids[i] for i in np.argsort(-scores)[:5]}:
                        correct_top5 += 1

                    vec_scores = sims[pos]
                    vec_scores -= vec_scores.max()
                    vec_scores = np.clip(vec_scores, -50, 50)
                    exp_v = np.exp(vec_scores)
                    vprobs = exp_v / exp_v.sum()
                    vlp = np.log(max(vprobs[actual_idx], 1e-30))
                    vec_log_prob += vlp
                    if cids[np.argmax(vec_scores)] == next_cid:
                        vec_correct_top1 += 1

                    n_eval += 1

            if start % 500 == 0 and n_eval > 0:
                elapsed = time.time() - t0
                rate = end / max(elapsed, 1)
                ppl = np.exp(-total_log_prob / n_eval)
                vppl = np.exp(-vec_log_prob / n_eval)
                acc1 = correct_top1 / n_eval
                vacc1 = vec_correct_top1 / n_eval
                print(f"  eval {end}/{n_positions} | {rate:.0f} tok/s | "
                      f"PPL={ppl:.1f} acc@1={acc1:.3f} | "
                      f"vecPPL={vppl:.1f} vacc@1={vacc1:.3f}")

        elapsed = time.time() - t0
        perplexity = np.exp(-total_log_prob / max(n_eval, 1))
        vec_perplexity = np.exp(-vec_log_prob / max(n_eval, 1))
        return {
            'perplexity': float(perplexity),
            'vec_perplexity': float(vec_perplexity),
            'accuracy_top1': correct_top1 / max(n_eval, 1),
            'accuracy_top5': correct_top5 / max(n_eval, 1),
            'vec_accuracy_top1': vec_correct_top1 / max(n_eval, 1),
            'n_tokens': n_eval,
            'total_log_prob': float(total_log_prob),
            'vec_log_prob': float(vec_log_prob),
            'elapsed_s': float(elapsed),
        }


if __name__ == '__main__':
    import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    import sentencepiece as spm
    from eva.symbolic.concept_space import ConceptSpace
    from eva.symbolic.syntax_lattice import SyntaxLattice

    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'bpe_ru_146k.model'))

    print("Initializing ConceptSpace (146K)...")
    cs = ConceptSpace(vocab_size=sp.vocab_size(), dim=384)
    cs.init_concepts()
    cs.init_homeostasis()

    print("Initializing lattice...")
    lattice = SyntaxLattice()
    gen = CrystalGenerator(cs, sp, lattice)

    print("\n--- Generation tests ---")
    for seed in ['князь', 'человек', 'война']:
        result = gen.generate(seed_word=seed)
        print(f"  [{seed}] path_len={len(result['concept_path'])} score={result['score']:.2f}")

    print("\n--- Training on sample ---")
    for sent in ["Князь Андрей вышел на крыльцо.", "Человек должен быть свободен."]:
        n = gen.train_from_text(sent)
        print(f"  trained: {n}")

    print("\n--- After training ---")
    for seed in ['князь', 'человек', 'война']:
        result = gen.generate(seed_word=seed)
        print(f"  [{seed}] path_len={len(result['concept_path'])} score={result['score']:.2f}")

    print("OK")
