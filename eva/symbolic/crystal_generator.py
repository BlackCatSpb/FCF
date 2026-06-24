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
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from eva.symbolic.adaptive_error_tracker import AdaptiveErrorTracker
from eva.symbolic.rng_registry import RNGRegistry
from eva.symbolic.stdp_trainer import STDPTrainer

# meta_t column indices (gpu_meta_l tuple order)
_META_I = 0
_META_J = 1
_META_PMI = 2
_META_DW = 3
_META_FW = 4
_META_FIELD_W = 5
_META_SLOW = 6
_META_PREV_CID = 7  # GPU-only: raw prev_cid for on-GPU PMI
_META_NEXT_CID = 8  # GPU-only: raw next_cid for on-GPU PMI
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.hormonal_system import HormonalSystem
from eva.symbolic.fcf_config import FormulaCoefficients


_BOS_ID = 1
_EOS_ID = 2


@dataclass
class GenerationResult:
    text: str = ""
    concept_path: list = field(default_factory=list)
    score: float = 0.0
    word_count: int = 0
    max_words: int = 0
    chains: list = field(default_factory=list)
    semantic_delta: float = 0.0
    time: float = 0.0


class CrystalGenerator:
    """Generation as semantic navigation through BPE-token concept space."""

    def __init__(self, cs, sp, lattice, config=None, qwen_knowledge=None):
        self.cs = cs
        self.sp = sp
        self.lattice = lattice
        self.config = config or {}
        cs._after_update_hook = self._on_vector_update

        self.max_words = self.config.get('max_words', 30)
        self.min_words = self.config.get('min_words', 3)
        self._graph_cache = OrderedDict()
        self._graph_cache_max = 5000
        self.base_concept_temp = self.config.get('concept_temp', 0.5)
        self.temperature = self.config.get('temperature', 1.0)
        self.theta_tau = self.config.get('theta_tau', 12.0)
        self.base_learning_rate = self.config.get('learning_rate', 0.1)

        # Token diversity
        self.top_p = self.config.get('top_p', 0.9)
        self.len_norm_alpha = self.config.get('len_norm_alpha', 0.7)
        self.block_ngram = self.config.get('block_ngram', 4)
        self.mmi_lambda = self.config.get('mmi_lambda', 0.2)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)

        self.rng_registry = RNGRegistry(master_seed=42)
        self.main_rng = self.rng_registry.get('main')
        if not self.cs.concept_usage:
            self.cs.init_homeostasis()
        self.branch_rngs = {}
        self.hormones = HormonalSystem()

        # Per-concept prediction error EMA (Level 2: error-based PMI gate)
        ce_max = min(3 * self.cs.vocab_size // 4, 100000)
        self.concept_error_decay = 0.9  # EMA decay (kept for backward compat)
        self.concept_error = AdaptiveErrorTracker(
            decay=self.concept_error_decay, max_size=ce_max)

        self.train_lr = 0.01

        # STDPTrainer — delegated training methods (created eagerly for AM-24)
        self._trainer = STDPTrainer(self)
        self._use_torch = _HAS_TORCH and torch.cuda.is_available()

        # Torch state (lazy init, invalidated by fluctuate_fractal)
        self._torch_device = None
        self._torch_cid_order = []
        self._torch_cid_to_idx = {}
        self._vecs_t = None
        self._fb_t = None
        self._ce_t = None
        self._mom_t = None
        self._basis_t = None
        self._codes_t = None
        self._codes_master_t = None  # fp32 master for STDP gradient precision
        self._ema_vecs_t = None  # EMA copy for stable eval/generation (TN-2)
        self._ema_decay = 0.999
        self._ema_steps = 0
        self._torch_dirty = False  # set True after fluctuate → trigger rebuild
        self._skip_gpu_sync = False  # B4: suppress GPU copy in hook after batched write
        self._cluster_map = None  # (V,) int32: primary anchor index per CID
        self._cluster_potential = None  # (n_anchors,) float32: minesweeper potential per cluster
        self._cluster_update_counter = 0
        self._cluster_update_every = 50  # recompute potential every N batches
        self._dirty_cids: set = set()  # G-72: CIDs modified on GPU, pending CPU sync
        self._total_freq_cache = None
        self._fused_buf = None  # G-49: pre-allocated fused buffer for scatter_add

        # GPU frequency/ngram tensors for on-GPU PMI (lazy init, synced incrementally)
        self._cf_t = None      # concept_freq [V] float32
        self._pt2_t = None     # _prefix_total for 2-grams [V] float32
        self._skip2_t = None   # _skip2_total [V] float32
        self._total_freq_t = None  # scalar GPU tensor

        # Hook lattice mutations to invalidate total_freq cache + sync GPU tensors
        _orig_update = self.lattice.update
        def _cached_update(concept_sequence):
            _orig_update(concept_sequence)
            self._total_freq_cache = None
            if self._cf_t is not None:
                self._sync_freq_tensors(concept_sequence)
        self.lattice.update = _cached_update

        _orig_decay = self.lattice.decay_all
        def _cached_decay(min_freq=0.01, **kwargs):
            _orig_decay(min_freq, **kwargs)
            self._total_freq_cache = None
            if self._cf_t is not None:
                self._rebuild_freq_tensors()
        self.lattice.decay_all = _cached_decay

    def _get_total_freq(self):
        if self._total_freq_cache is None:
            self._total_freq_cache = max(sum(self.lattice.concept_freq.values()), 1)
        return self._total_freq_cache

    def _sync_freq_tensors(self, concept_sequence):
        """Incremental CPU→GPU sync of frequency tensors after lattice.update()."""
        device = self._cf_t.device
        seen = list(set(concept_sequence))
        cf_vals = [self.lattice.concept_freq.get(c, 0.0) for c in seen]
        self._cf_t[seen] = torch.tensor(cf_vals, dtype=torch.float32, device=device)

        # _prefix_total for 2-grams (single-CID prefixes)
        changed_pt2 = set()
        for i in range(len(concept_sequence) - 1):
            prefix = (concept_sequence[i],)
            if prefix in self.lattice._prefix_total:
                changed_pt2.add(concept_sequence[i])
        if changed_pt2:
            pt2_l = list(changed_pt2)
            pt2_v = [self.lattice._prefix_total.get((c,), 0) for c in pt2_l]
            self._pt2_t[pt2_l] = torch.tensor(pt2_v, dtype=torch.float32, device=device)

        # _skip2_total
        changed_sk2 = set()
        for i in range(len(concept_sequence) - 2):
            cid = concept_sequence[i]
            if cid in self.lattice._skip2_total:
                changed_sk2.add(cid)
        if changed_sk2:
            sk2_l = list(changed_sk2)
            sk2_v = [self.lattice._skip2_total.get(c, 0) for c in sk2_l]
            self._skip2_t[sk2_l] = torch.tensor(sk2_v, dtype=torch.float32, device=device)

        self._total_freq_t = torch.tensor(self._get_total_freq(), dtype=torch.float32, device=device)

    def _rebuild_freq_tensors(self):
        """Full rebuild of GPU frequency tensors from CPU dicts (after decay_all)."""
        V = self._cf_t.shape[0]
        device = self._cf_t.device
        cf_arr = np.zeros(V, dtype=np.float32)
        for cid, freq in self.lattice.concept_freq.items():
            cf_arr[int(cid)] = freq
        self._cf_t.copy_(torch.from_numpy(cf_arr).to(device, non_blocking=True))
        pt2_arr = np.zeros(V, dtype=np.float32)
        for prefix, total in self.lattice._prefix_total.items():
            if len(prefix) == 1:
                pt2_arr[prefix[0]] = total
        self._pt2_t.copy_(torch.from_numpy(pt2_arr).to(device, non_blocking=True))
        sk2_arr = np.zeros(V, dtype=np.float32)
        for cid, total in self.lattice._skip2_total.items():
            sk2_arr[int(cid)] = total
        self._skip2_t.copy_(torch.from_numpy(sk2_arr).to(device, non_blocking=True))
        self._total_freq_t = torch.tensor(self._get_total_freq(), dtype=torch.float32, device=device)

    def _destab_field_fallback(self, gen_cid, v_gen):
        """Field-based destab fallback: pick a random concept with overlapping field."""
        cs = self.cs
        if not hasattr(cs.fractal, 'field_bits') or len(cs.fractal.field_bits) < 2:
            return None
        gen_fb = cs.fractal.get_field_bits(gen_cid)
        if gen_fb is None:
            return None
        candidates = [cid for cid, fb in cs.fractal.field_bits.items()
                      if cid != gen_cid and np.bitwise_and(gen_fb, fb).any()]
        if not candidates:
            return None
        neg_cid = candidates[self.main_rng.randint(0, len(candidates) - 1)]
        v_neg = cs.concept_vectors.get(neg_cid)
        if v_neg is None:
            return None
        return v_neg

    def _on_vector_update(self, cid, v_new):
        """Hook called by ConceptSpace._apply_vector_update to keep _vecs_t in sync."""
        if self._vecs_t is not None and not self._skip_gpu_sync:
            self._vecs_t[cid].copy_(torch.from_numpy(v_new).to(device=self._vecs_t.device, dtype=self._vecs_t.dtype, non_blocking=True))

    def _ensure_torch(self, device=None):
        """Precompute GPU tensors for batched training. Rebuilds if dirty."""
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for GPU training.")
        # Persistent CPU fallback: if OOM was hit once, stay on CPU
        if getattr(self, '_torch_fallback', False):
            device = 'cpu'
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dev = torch.device(device)
        if self._torch_device == dev and self._vecs_t is not None and not self._torch_dirty and not self.cs.fractal._fb_dirty:
            return

        # OOM fallback: try GPU, fall back to CPU on CUDA out of memory
        if dev.type == 'cuda':
            try:
                return self._build_torch_tensors(dev)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e):
                    print(f"[WARN] CUDA OOM ({torch.cuda.max_memory_allocated()/1024**2:.0f}MB) — falling back to CPU")
                    self._torch_fallback = True
                    torch.cuda.empty_cache()
                    dev = torch.device('cpu')
                else:
                    raise
        self._build_torch_tensors(dev)

    def _build_torch_tensors(self, dev):
        """Build GPU/CPU tensors (shared by _ensure_torch and OOM fallback)."""
        cs = self.cs
        V = cs.vocab_size
        D = cs.dim

        cids = list(range(V))
        self._torch_cid_order = cids
        self._torch_cid_to_idx = {cid: i for i, cid in enumerate(cids)}
        self._torch_device = dev

        # Build on CPU, copy to pre-allocated GPU buffer
        vecs = np.zeros((V, D), dtype=np.float32)
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

        if self._vecs_t is None or self._vecs_t.shape[0] != V or self._vecs_t.device != dev:
            self._vecs_t = torch.empty(V, D, device=dev, dtype=torch.float16)
        self._vecs_t.copy_(torch.from_numpy(vecs), non_blocking=True)

        if self._basis_t is None or self._basis_t.device != dev:
            self._basis_t = torch.from_numpy(cs.fractal.basis.astype(np.float32)).to(dev, non_blocking=True) if cs.fractal.basis is not None else None

        # G-40: Latent codes tensor for batched subspace update
        latent_dim = cs.fractal.latent_dim
        codes_arr = np.zeros((V, latent_dim), dtype=np.float32)
        for cid, code in cs.fractal.codes.items():
            codes_arr[cid] = code
        if self._codes_t is None or self._codes_t.shape[0] != V or self._codes_t.device != dev:
            self._codes_t = torch.empty(V, latent_dim, device=dev, dtype=torch.bfloat16)
            self._codes_master_t = torch.empty(V, latent_dim, device=dev, dtype=torch.float32)
        self._codes_t.copy_(torch.from_numpy(codes_arr), non_blocking=True)
        self._codes_master_t.copy_(torch.from_numpy(codes_arr), non_blocking=True)

        # Concept error tensor for vectorized GPU negative sampling
        ce_arr = np.zeros(V, dtype=np.float32)
        for cid, err in self.concept_error.items():
            ce_arr[int(cid)] = err
        if self._ce_t is None or self._ce_t.shape[0] != V or self._ce_t.device != dev:
            self._ce_t = torch.empty(V, device=dev, dtype=torch.float32)
        self._ce_t.copy_(torch.from_numpy(ce_arr), non_blocking=True)

        # Frequency/ngram tensors for on-GPU PMI
        if self._cf_t is None or self._cf_t.shape[0] != V or self._cf_t.device != dev:
            self._cf_t = torch.zeros(V, device=dev, dtype=torch.float32)
            self._pt2_t = torch.zeros(V, device=dev, dtype=torch.float32)
            self._skip2_t = torch.zeros(V, device=dev, dtype=torch.float32)
        self._rebuild_freq_tensors()

        # Initialize EMA as a copy of vecs_t (fp16 to save 112MB)
        if self._ema_vecs_t is None or self._ema_vecs_t.shape[0] != V or self._ema_vecs_t.device != dev:
            self._ema_vecs_t = self._vecs_t.clone().to(torch.bfloat16)
        else:
            self._ema_vecs_t.copy_(self._vecs_t.to(torch.bfloat16))
        self._ema_steps = 0

        if self._mom_t is None or self._mom_t.shape[0] != V or self._mom_t.device != dev:
            self._mom_t = torch.zeros(V, D, device=dev, dtype=torch.bfloat16)

        # G-49: pre-allocate fused buffer for scatter_add (grows on demand, not full V)
        if self._fused_buf is None or self._fused_buf.shape[1] != D + 1 or self._fused_buf.device != dev:
            init_rows = min(V, 4096)
            self._fused_buf = torch.zeros(init_rows, D + 1, device=dev, dtype=torch.float32)

        # Minesweeper: build cluster map from field_bits
        self._ensure_cluster_map(dev)
        self._cluster_potential = None  # reset, will be updated periodically

        self._torch_dirty = False
        self.cs.fractal._fb_dirty = False
        if dev.type == 'cuda':
            torch.cuda.synchronize()
            alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
            if alloc_mb > 1500:
                print(f"[INFO] GPU VRAM: {alloc_mb:.0f}MB used (limit ~2048MB)")

    def _invalidate_torch(self):
        """Mark GPU tensors as stale; triggers rebuild on next _ensure_torch.
        Call after fluctuate_fractal() or any code-level change."""
        self._mom_t = None
        self._codes_t = None
        self._codes_master_t = None
        self._torch_dirty = True
        self._cluster_potential = None  # stale after vector flush

    def _sync_after_fluctuate(self):
        """SN-54: Incremental GPU sync after fluctuate — no full O(V·D) rebuild.

        Reads CPU codes → copies to GPU _codes_t → recomputes _vecs_t via
        batched matmul (_codes_t @ _basis_t), avoiding the per-concept
        CPU loop + PCIe transfer of _build_torch_tensors.
        """
        cs = self.cs
        if self._torch_device is None or self._vecs_t is None or not hasattr(cs.fractal, 'codes'):
            self._invalidate_torch()
            return
        dev = self._torch_device
        V = cs.vocab_size
        latent_dim = cs.fractal.latent_dim

        codes_arr = np.zeros((V, latent_dim), dtype=np.float32)
        for cid, code in cs.fractal.codes.items():
            codes_arr[cid] = code

        if self._codes_t is None or self._codes_t.shape[0] != V:
            self._codes_t = torch.empty(V, latent_dim, device=dev, dtype=torch.bfloat16)
            self._codes_master_t = torch.empty(V, latent_dim, device=dev, dtype=torch.float32)
        self._codes_t.copy_(torch.from_numpy(codes_arr), non_blocking=True)
        self._codes_master_t.copy_(torch.from_numpy(codes_arr), non_blocking=True)

        basis_t = self._basis_t
        # Recompute _vecs_t on GPU: (V, latent_dim) @ (latent_dim, D) → (V, D)
        vecs_gpu = self._codes_master_t @ basis_t.to(dev, non_blocking=True)
        nv = vecs_gpu.norm(dim=1, keepdim=True).clamp(min=1e-10)
        vecs_gpu /= nv
        if self._vecs_t.shape[0] != V:
            self._vecs_t = torch.empty(V, vecs_gpu.shape[1], device=dev, dtype=torch.float16)
        self._vecs_t.copy_(vecs_gpu.to(torch.float16), non_blocking=True)

        # Refresh basis_t (may have changed after fluctuate)
        if cs.fractal.basis is not None:
            self._basis_t = torch.from_numpy(cs.fractal.basis.astype(np.float32)).to(dev, non_blocking=True)

        # Reset momentum (codes changed — old momentum is stale)
        if self._mom_t is not None:
            self._mom_t.zero_()
        # SN-58: refresh EMA from new vectors after fluctuate
        if self._ema_vecs_t is not None and self._vecs_t is not None:
            self._ema_vecs_t.copy_(self._vecs_t.to(torch.bfloat16))
        self._torch_dirty = False

    def _sync_ema(self):
        """SN-18: Copy EMA vectors → _vecs_t for stable eval/generation.
        Saves original _vecs_t backup internally. Call restore_vectors() after eval."""
        if self._ema_vecs_t is None or self._vecs_t is None:
            return
        self._eval_backup = self._vecs_t.float().clone()
        self._vecs_t.copy_(self._ema_vecs_t.to(self._vecs_t.dtype))

    def _restore_vectors(self):
        """SN-18: Restore _vecs_t from backup saved by _sync_ema()."""
        if hasattr(self, '_eval_backup') and self._eval_backup is not None and self._vecs_t is not None:
            self._vecs_t.copy_(self._eval_backup.to(self._vecs_t.dtype))
            self._eval_backup = None

    def _sync_dirty_cpu(self):
        """G-72: Batch-sync dirty CIDs from GPU _vecs_t to CPU concept_vectors."""
        if not self._dirty_cids or self._vecs_t is None or self._torch_device is None:
            return
        cids = list(self._dirty_cids)
        cids_t = torch.tensor(cids, dtype=torch.long, device=self._torch_device)
        vecs_cpu = self._vecs_t[cids_t].cpu().numpy()
        self._skip_gpu_sync = True
        for cid, v_new in zip(cids, vecs_cpu):
            self.cs._apply_vector_update(cid, v_new)
        self._skip_gpu_sync = False
        self._dirty_cids.clear()
        if hasattr(self.cs.fractal, '_matrix_dirty'):
            self.cs.fractal._matrix_dirty = True

    def _ensure_fb_tensor(self, dev=None):
        """Lazy-build _fb_t field bit tensor (only if field_gate is needed)."""
        if not _HAS_TORCH:
            return
        if dev is None:
            dev = self._torch_device
        if dev is None:
            return
        if self._fb_t is not None and not self._torch_dirty and not self.cs.fractal._fb_dirty:
            return
        cs = self.cs
        V = cs.vocab_size
        if hasattr(cs.fractal, 'field_bits') and cs.fractal.field_bits:
            sample_fb = next(iter(cs.fractal.field_bits.values()))
            fb_bytes = len(np.asarray(sample_fb, dtype=np.uint8).ravel())
        else:
            fb_bytes = (getattr(cs, 'n_anchors', 1024) + 7) // 8
        if fb_bytes == 0:
            fb_bytes = (getattr(cs, 'n_anchors', 1024) + 7) // 8
        fb_arr = np.zeros((V, fb_bytes), dtype=np.uint8)
        if hasattr(cs.fractal, 'field_bits'):
            for cid, fb in cs.fractal.field_bits.items():
                fb_arr[cid] = np.asarray(fb, dtype=np.uint8).ravel()
        if self._fb_t is None or self._fb_t.shape[0] != V or self._fb_t.device != dev:
            self._fb_t = torch.empty(V, fb_bytes, device=dev, dtype=torch.uint8)
        self._fb_t.copy_(torch.from_numpy(fb_arr), non_blocking=True)

    def _ensure_cluster_map(self, dev=None):
        """Lazy-build _cluster_map from field_bits — primary anchor per CID.
        Cluster = index of first set bit in field_bits[cid]."""
        if self._cluster_map is not None:
            return
        cs = self.cs
        if dev is None:
            dev = self._torch_device
        if not hasattr(cs.fractal, 'field_bits') or not cs.fractal.field_bits:
            return
        V = cs.vocab_size
        n_anchors = getattr(cs, 'n_anchors', 2048)
        n_bytes = (n_anchors + 7) // 8
        cluster_arr = np.zeros(V, dtype=np.int32)
        for cid, fb in cs.fractal.field_bits.items():
            fb_arr = np.asarray(fb, dtype=np.uint8).ravel()
            if len(fb_arr) < n_bytes:
                fb_arr = np.pad(fb_arr, (0, n_bytes - len(fb_arr)))
            # Find first set bit
            mask = fb_arr.view(np.uint64) if n_bytes >= 8 else fb_arr.view(np.uint32)
            bits = int(mask[0]) if len(mask) > 0 else 0
            if bits:
                cluster_arr[cid] = (bits & -bits).bit_length() - 1  # ctz
        self._cluster_map = torch.tensor(cluster_arr, device=dev, dtype=torch.long)

    def _update_cluster_potential(self):
        """Minesweeper: update potential per cluster based on _ce_t of members.
        Called every N lines (not every batch). Potential decays toward 1.0,
        boosted for clusters with low CE (stable concepts), reduced for high CE."""
        if self._cluster_map is None or self._ce_t is None:
            return
        if self._cluster_potential is None:
            cs = self.cs
            n_anchors = getattr(cs, 'n_anchors', 2048)
            self._cluster_potential = torch.ones(n_anchors, device=self._torch_device, dtype=torch.float32)
        dev = self._torch_device
        cm = self._cluster_map.to(dev)
        ce = self._ce_t  # (V,) float32
        # Mean CE per cluster via scatter_add
        sum_ce = torch.zeros(len(self._cluster_potential), device=dev)
        cnt = torch.zeros(len(self._cluster_potential), device=dev)
        sum_ce.scatter_add_(0, cm, ce)
        cnt.scatter_add_(0, cm, torch.ones_like(cm, dtype=torch.float32))
        mean_ce = sum_ce / cnt.clamp(min=1)
        # Minesweeper inverted: high CE → boost (up to 1.2), low CE → reduce (down to 0.8)
        # Rare/struggling concepts get MORE learning signal instead of less
        target = 1.0 + (mean_ce - 0.5) * 0.4  # ce=0 → 0.8, ce=0.5 → 1.0, ce=1.0 → 1.2
        self._cluster_potential = self._cluster_potential * 0.9 + target * 0.1

    # ── Temperature ────────────────────────────────────────────

    def _theta_temp(self, word_num):
        t = self.base_concept_temp * math.exp(-word_num / max(self.theta_tau, 1.0))
        t *= self.temperature
        return max(t, self.base_concept_temp * 0.15)

    # ── Encode / Decode ────────────────────────────────────────

    def _encode_input(self, text):
        return self.sp.encode(text, add_bos=False, add_eos=False)

    def _decode_tokens(self, token_ids):
        return self.sp.decode(token_ids)

    def _token_text(self, cid):
        try:
            return self.sp.IdToPiece(int(cid))
        except IndexError:
            return f'[CID{cid}]'

    def _is_semantic_token(self, cid):
        """Filter function words and punctuation that dominate graph connections."""
        if self.sp is None:
            return True
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

    def _adaptive_beam_width(self, probs, base_width):
        """Scale beam width by entropy ratio."""
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = float(np.log(len(probs)))
        ratio = entropy / max_entropy if max_entropy > 0 else 1.0
        return max(1, int(base_width * (0.5 + ratio)))

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
        total_freq = self._get_total_freq()

        # Beam: list of (concept_sequence, score, branch_id)
        beam = [([seed_cid], 0.0, 0)]
        all_chains = []
        finished = []
        next_branch_id = 1

        for wn in range(effective_max):
            new_beam = []

            theta_temp = self._theta_temp(wn)
            h_temp = self.hormones.modulate_temperature(theta_temp)

            # Adaptive beam width from previous step's entropy (first step uses base)
            entropy_ratio = getattr(self, '_last_branch_entropy', 1.0)
            base_bw = self.hormones.modulate_beam_width(beam_width)
            effective_beam = max(1, int(base_bw * (0.5 + entropy_ratio)))

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

                    _f = FormulaCoefficients()
                    novelty = 1.0 - min(self.lattice.concept_freq.get(cid, 0) / _f.novelty_freq_cap, 1.0)
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
                    token_text = self._token_text(seq[-1]).lstrip('▁')
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
            return GenerationResult(chains=all_chains)

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

        return GenerationResult(
            text=text,
            concept_path=best_seq,
            score=best_score,
            word_count=len(best_seq),
            max_words=effective_max,
            chains=all_chains,
            semantic_delta=semantic_delta,
        )

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
                    _f = FormulaCoefficients()
                    w = max(_f.edge_weight_min, 1.0 - min(ppmi / _f.edge_ppmi_cap, 1.0) * _f.edge_weight_strength)
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

    def _branch(self, seq: List[int], word_num: int, theta_temp: float = 0.3, target_cid: Optional[int] = None, centroid: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        """Generate diverse branching candidates via RRF over multiple signals."""
        if not seq:
            return []
        prev_cid = seq[-1]
        cids = seq[-3:] if len(seq) >= 3 else seq
        K = 3

        # 1. Graph-based semantic paths (BMSSP-EVA, replaces single-hop connections)
        sources = list(set(cids))  # unique context tokens
        sources_key = tuple(sorted(set(sources)))
        if sources_key not in self._graph_cache:
            if len(self._graph_cache) >= self._graph_cache_max:
                self._graph_cache.popitem(last=False)
            self._graph_cache[sources_key] = self._graph_search(sources, B=1.2, max_candidates=30)
        else:
            self._graph_cache.move_to_end(sources_key)
        graph_candidates = self._graph_cache[sources_key]

        # 2. N-gram syntax (filter to semantic tokens only)
        syn_preds = self.lattice.predict(cids)
        syn_ranked = {cid: i + 1 for i, (cid, _) in enumerate(syn_preds[:80])
                      if self._is_semantic_token(cid)}

        # 2b. HDC n-gram fallback (always participates in RRF for stability)
        hdc_candidates = {}
        if len(cids) >= 2:
            ctx_cids = list(reversed(cids[-2:]))
            if hasattr(self.cs.fractal, 'hdc_memory'):
                hdc_preds = self.cs.fractal.hdc_predict(
                    ctx_cids, self.cs.fractal.codes, k=30)
                for hcid, hscore in hdc_preds:
                    if self._is_semantic_token(hcid) and hscore > 0.05:
                        hdc_candidates[hcid] = hscore

        # 3. All candidates from learned signals
        all_cids = set(graph_candidates.keys()) | set(syn_ranked.keys()) | set(hdc_candidates.keys())

        # 4. Vector similarity fallback (sector search if available, else full)
        v_prev = self.cs.concept_vector(prev_cid)
        vector_sim = {}
        if v_prev is not None:
            if hasattr(self.cs.fractal, '_sector_index') and self.cs.fractal._sector_index:
                sim_candidates = self.cs.fractal.search_in_sector(prev_cid, depth=1, k=40)
                if len(sim_candidates) < 5:
                    sim_candidates = self.cs.fractal.focal_refine(prev_cid, start_depth=0, target_k=20)
            else:
                sim_candidates = self.cs.topk_similar_concepts(prev_cid, k=20, sample_size=500)
            for cid, sim in sim_candidates:
                if cid not in all_cids and sim > 0.05:
                    all_cids.add(cid)
                vector_sim[cid] = sim

        if not all_cids:
            return []

        # 5. RRF scoring
        _fc = FormulaCoefficients()
        combined = {}
        for cid in all_cids:
            rrf = 0.0
            if cid in graph_candidates:
                rrf += _fc.rrf_graph * graph_candidates[cid]
            if cid in syn_ranked:
                rrf += _fc.rrf_syntax / (K + syn_ranked[cid])
            if cid in hdc_candidates:
                rrf += _fc.rrf_hdc * hdc_candidates[cid] / (K + 1)
            if cid in vector_sim:
                rrf += _fc.rrf_vector * vector_sim[cid] / (K + 1)
            freq = self.lattice.concept_freq.get(cid, 0)
            prior = _fc.rrf_prior / (K + 1) * (1.0 - min(freq / _fc.rrf_prior_freq_cap, 1.0))
            rrf += prior
            combined[cid] = rrf

        # 5. Homeostatic boost
        for cid in list(combined.keys()):
            h_boost = self.cs.homeostatic_boost(cid)
            combined[cid] *= (1.0 + h_boost * _fc.homeostatic_rrf_mult)

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

        # 8. Field mask filter + bonus: exclude candidates with zero field overlap
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
                        if overlap == 0:
                            combined.pop(cid, None)
                        else:
                            combined[cid] *= (1.0 + math.log(overlap + 1) * 0.1)

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
                    boost = max(0.0, 5.0 * (1.0 - theta_temp * 0.5))
                    probs[i] *= boost
                    break
            probs /= probs.sum()

        # Adaptive entropy for beam width modulation
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        self._last_branch_entropy = entropy / max(float(np.log(len(probs))), 1e-10)

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
            total_freq = self._get_total_freq()
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


    def train_from_text(self, text, base_lr=None, context_window=2, pmi_strength=1.0, pmi_gate_min=0.20, neg_samples=1,
                        inh_strength=0.05, inh_threshold=0.10, neg_lr_ratio=0.5, field_gate=True, use_torch=None,
                        destab_scale=0.0, momentum_mu=0.9, gradient_noise_scale=0.0, fluctuation_amp=0.003):
        """Delegate to STDPTrainer for STDP training on one text."""
        return self._trainer.train_from_text(text, base_lr, context_window, pmi_strength, pmi_gate_min,
                                              neg_samples, inh_strength, inh_threshold, neg_lr_ratio,
                                              field_gate, use_torch, destab_scale,
                                              momentum_mu=momentum_mu, gradient_noise_scale=gradient_noise_scale)

    def train_batch(self, texts, base_lr=None, context_window=2, pmi_strength=1.0, pmi_gate_min=0.20,
                    neg_samples=1, inh_strength=0.05, inh_threshold=0.10, neg_lr_ratio=0.5,
                    field_gate=True, use_torch=None, destab_scale=0.0, momentum_mu=0.9, gradient_noise_scale=0.0, fluctuation_amp=0.003):
        """Delegate to STDPTrainer for batched STDP training."""
        return self._trainer.train_batch(texts, base_lr, context_window, pmi_strength, pmi_gate_min,
                                          neg_samples, inh_strength, inh_threshold, neg_lr_ratio,
                                          field_gate, use_torch, destab_scale,
                                          momentum_mu=momentum_mu, gradient_noise_scale=gradient_noise_scale)

    # ── Evaluation ────────────────────────────────────────────

    def evaluate(self, corpus_path, max_lines=None, batch_size=500, use_gpu=True):
        """Delegate to STDPTrainer for evaluation."""
        return self._trainer.evaluate(corpus_path, max_lines=max_lines)


if __name__ == '__main__':
    from eva.symbolic.fcf_config import EnvironmentResolver
    _env = EnvironmentResolver()
    import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    import sentencepiece as spm
    from eva.symbolic.concept_space import ConceptSpace
    from eva.symbolic.syntax_lattice import SyntaxLattice

    sp = spm.SentencePieceProcessor(model_file=_env.bpe_model_path)

    print("Initializing ConceptSpace (146K)...")
    cs = ConceptSpace(vocab_size=sp.vocab_size())
    cs.init_concepts()
    cs.init_homeostasis()

    print("Initializing lattice...")
    lattice = SyntaxLattice()
    gen = CrystalGenerator(cs, sp, lattice)

    print("\n--- Generation tests ---")
    for seed in ['князь', 'человек', 'война']:
        result = gen.generate(seed_word=seed)
        print(f"  [{seed}] path_len={len(result.concept_path)} score={result.score:.2f}")

    print("\n--- Training on sample ---")
    for sent in ["Князь Андрей вышел на крыльцо.", "Человек должен быть свободен."]:
        n = gen.train_from_text(sent)
        print(f"  trained: {n}")

    print("\n--- After training ---")
    for seed in ['князь', 'человек', 'война']:
        result = gen.generate(seed_word=seed)
        print(f"  [{seed}] path_len={len(result.concept_path)} score={result.score:.2f}")

    print("OK")
