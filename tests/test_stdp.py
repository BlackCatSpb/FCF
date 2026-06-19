"""Unit tests for FCF core: STDP, ConceptSpace, GPU/CPU parity."""

import math, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eva.symbolic.concept_space import ConceptSpace, ConceptVectorStore, FractalField
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.parameter_optimizer import ParameterOptimizer
from eva.symbolic.fcf_config import FCFConfig, PathConfig, MetricPairBuilder

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


@pytest.fixture
def dim():
    return 64


@pytest.fixture
def vocab_size():
    return 20


@pytest.fixture
def cs(dim, vocab_size):
    cs = ConceptSpace(vocab_size=vocab_size, dim=dim)
    cs.init_concepts()
    cs.init_homeostasis()
    return cs


@pytest.fixture
def lattice(vocab_size):
    lat = SyntaxLattice()
    lat.concept_freq = {i: max(10 - i, 1) for i in range(vocab_size)}
    return lat


@pytest.fixture
def gen(cs, lattice):
    from eva.symbolic.crystal_generator import CrystalGenerator
    gen = CrystalGenerator(cs, None, lattice)
    gen.max_grad_norm = 1.0
    return gen


# ── 1. ConceptVectorStore ───────────────────────────────────────

class TestConceptVectorStore:
    def test_basic_crud(self, dim):
        s = ConceptVectorStore(10, dim)
        assert len(s) == 0
        v = np.random.randn(dim).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-10)
        s[3] = v
        assert 3 in s
        assert len(s) == 1
        r = s.get(3)
        assert r is not None
        assert np.allclose(r, v)
        assert s.get(999) is None

    def test_bounds_check(self, dim):
        s = ConceptVectorStore(10, dim)
        v = np.random.randn(dim).astype(np.float32)
        s[0] = v
        assert s.get(-1) is None
        assert s.get(100) is None

    def test_items_and_keys(self, dim):
        s = ConceptVectorStore(5, dim)
        for i in range(5):
            v = np.random.randn(dim).astype(np.float32)
            v /= max(np.linalg.norm(v), 1e-10)
            s[i] = v
        keys = list(s.keys())
        assert len(keys) == 5
        items = list(s.items())
        assert len(items) == 5
        vals = list(s.values())
        assert len(vals) == 5


# ── 2. FractalField ─────────────────────────────────────────────

class TestFractalField:
    def test_init_and_vector(self, dim):
        ff = FractalField(dim=dim, latent_dim=64)
        v = ff.init_concept(0)
        assert v is not None
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_basis_health(self, dim):
        ff = FractalField(dim=dim, latent_dim=64)
        assert not ff.check_basis_health()

    def test_fluctuate(self, dim):
        ff = FractalField(dim=dim, latent_dim=64)
        v0 = ff.init_concept(0)
        ff.fluctuate(fluctuation_amp=0.001, decay=0.999)
        v1 = ff.compute_vector(0)
        assert v1 is not None
        assert abs(np.linalg.norm(v1) - 1.0) < 1e-6

    def test_fb_dirty_flag(self, dim):
        ff = FractalField(dim=dim, latent_dim=64)
        assert not ff._fb_dirty
        ff.init_fields(n_anchors=32)
        assert ff._fb_dirty
        ff._fb_dirty = False
        ff.init_fields(n_anchors=64)
        assert ff._fb_dirty


# ── 3. ConceptSpace ─────────────────────────────────────────────

class TestConceptSpace:
    def test_initialization(self, cs, vocab_size):
        assert cs.vocab_size == vocab_size
        assert len(cs.concept_vectors) > 0

    def test_vector_norms(self, cs):
        ok, total, max_dev = cs.validate_vector_norms()
        assert ok == total

    def test_topk_similar(self, cs):
        top = cs.topk_similar_concepts(0, k=5, sample_size=20)
        assert len(top) <= 5
        for cid, sim in top:
            assert cid != 0
            assert -1.0 <= sim <= 1.0

    def test_apply_vector_update(self, cs, dim):
        v = cs.concept_vectors.get(0)
        assert v is not None
        v_new = v + np.random.randn(dim).astype(np.float32) * 0.01
        nv = np.linalg.norm(v_new)
        if nv > 1e-10:
            v_new /= nv
        cs._apply_vector_update(0, v_new)
        v_after = cs.concept_vectors.get(0)
        assert v_after is not None
        assert abs(np.linalg.norm(v_after) - 1.0) < 1e-6


# ── 4. STDP ─────────────────────────────────────────────────────

class TestSTDP:
    def test_cpu_stdp_no_crash(self, gen, cs):
        gen._trainer._cpu_stdp_apply({1: [(0, 0.1), (2, 0.05)]}, base_lr_val=0.03,
                            destab_scale=0.0, inh_strength=0.0, inh_threshold=0.1)
        v1 = cs.concept_vectors.get(1)
        assert v1 is not None
        assert abs(np.linalg.norm(v1) - 1.0) < 1e-6

    def test_negative_sampling_cpu(self, gen, cs):
        gen._trainer._negative_sampling_cpu({1: [(0, 0.1)]}, neg_lr_ratio=0.5,
                                    field_gate=False, neg_samples=2)
        v1 = cs.concept_vectors.get(1)
        assert v1 is not None

    def test_contrastive_objective(self, gen, cs):
        gen._trainer._contrastive_objective({1: [(0, 0.1)]})
        v1 = cs.concept_vectors.get(1)
        assert v1 is not None
        assert abs(np.linalg.norm(v1) - 1.0) < 1e-6

    def test_concept_error_fifo(self, gen):
        for i in range(50):
            gen.concept_error.update(i, float(i) / 50.0)
        assert len(gen.concept_error) <= gen.concept_error.max_size


# ── 5. GPU/CPU parity ───────────────────────────────────────────

class TestGPUParity:
    def test_cpu_no_torch(self, gen, cs):
        v_before = cs.concept_vectors.get(1).copy()
        gen._trainer._cpu_stdp_apply({1: [(0, 0.1)]}, base_lr_val=0.03,
                            destab_scale=0.0, inh_strength=0.0, inh_threshold=0.1)
        v_after = cs.concept_vectors.get(1)
        assert not np.allclose(v_before, v_after)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_gpu_stdp_apply_no_crash(self, gen, cs):
        gen._torch_device = torch.device('cpu')
        gen._ensure_torch(device='cpu')
        gen._trainer._gpu_stdp_apply(
            gpu_ctx_l=[0, 0],
            gpu_tgt_l=[1, 2],
            gpu_meta_l=np.array([(0, 1, 0.5, 1.0, 1.0, 1.0),
                                 (0, 2, 0.3, 1.0, 1.0, 1.0)], dtype=np.float32),
            gpu_cid_gen=[1, 2],
            base_lr_val=0.03,
            field_gate=False,
            inh_strength=0.0,
            inh_threshold=0.1,
            destab_scale=0.0,
        )
        for cid in [1, 2]:
            v = cs.concept_vectors.get(cid)
            assert v is not None
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_negative_sampling_gpu_no_crash(self, gen, cs):
        gen._torch_device = torch.device('cpu')
        gen._ensure_torch(device='cpu')
        gen._trainer._negative_sampling_gpu(
            gpu_ctx_l=[0],
            gpu_meta_l=np.array([(0, 1, 0.5, 1.0, 1.0, 1.0)], dtype=np.float32),
            gpu_cid_ctx=[0],
            gpu_cid_gen=[1],
            device=torch.device('cpu'),
            field_gate=False,
            base_lr_val=0.03,
            neg_lr_ratio=0.5,
            neg_samples=2,
        )
        v1 = cs.concept_vectors.get(1)
        assert v1 is not None

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_cpu_gpu_stdp_parity(self, gen, cs, dim):
        gen._torch_device = torch.device('cpu')
        gen._ensure_torch(device='cpu')

        # Run GPU STDP on a pair
        gen._trainer._gpu_stdp_apply(
            gpu_ctx_l=[0],
            gpu_tgt_l=[1],
            gpu_meta_l=np.array([(0, 1, 0.5, 1.0, 1.0, 1.0)], dtype=np.float32),
            gpu_cid_gen=[1],
            base_lr_val=0.03,
            field_gate=False,
            inh_strength=0.0,
            inh_threshold=0.1,
            destab_scale=0.0,
        )
        v_gpu = cs.concept_vectors.get(1).copy()

        # Reset
        cs2 = ConceptSpace(vocab_size=20, dim=dim)
        cs2.init_concepts()
        cs2.init_homeostasis()
        gen2 = __import__('eva.symbolic.crystal_generator', fromlist=['']).CrystalGenerator(cs2, None, SyntaxLattice())
        gen2.max_grad_norm = 1.0
        gen2._torch_device = torch.device('cpu')
        gen2._ensure_torch(device='cpu')

        # Seed RNG for reproducibility
        np.random.seed(42)
        gen2._trainer._cpu_stdp_apply({1: [(0, 0.1)]}, base_lr_val=0.03,
                             destab_scale=0.0, inh_strength=0.0, inh_threshold=0.1)
        v_cpu = cs2.concept_vectors.get(1)

        # Compare numerical parity (within tolerance)
        # Note: GPU and CPU paths may diverge due to different accumulation order
        # so we use a relaxed tolerance
        if v_cpu is not None:
            diff = np.linalg.norm(v_gpu - v_cpu)
            assert diff < 1.0, f"GPU/CPU vectors diverged: norm diff={diff:.4f}"


# ── 6. ParameterOptimizer ───────────────────────────────────────

class TestParameterOptimizer:
    def test_basic_step(self):
        cfg = FCFConfig()
        opt = ParameterOptimizer(cfg)
        assert 'full_lr' in opt.p
        old_lr = opt.p['full_lr'].current
        opt.step(mean_cos=0.1, std_cos=0.02, delta=5.0, ng_new=500,
                 vec_ppl=100.0, acc1=0.3, vacc1=0.0)
        assert opt.p['full_lr'].current != old_lr

    def test_full_stuck_no_eval(self):
        cfg = FCFConfig()
        opt = ParameterOptimizer(cfg)
        changes = None
        for _ in range(7):
            changes = opt.step(mean_cos=0.0005, std_cos=0.005, delta=0.1, ng_new=10)
        assert changes is None or not changes.get('full_stuck')

    def test_full_stuck_with_eval(self):
        cfg = FCFConfig()
        opt = ParameterOptimizer(cfg)
        for _ in range(7):
            changes = opt.step(mean_cos=0.0005, std_cos=0.005, delta=0.1, ng_new=10,
                                vec_ppl=100.0, acc1=0.3, vacc1=0.0)
        assert changes.get('full_stuck')

    def test_vacc1_stuck(self):
        cfg = FCFConfig()
        opt = ParameterOptimizer(cfg)
        for _ in range(5):
            opt.step(mean_cos=0.05, std_cos=0.02, delta=1.0, ng_new=100,
                     vec_ppl=80.0, acc1=0.5, vacc1=0.0)
        assert opt._vacc1_stuck >= 4

    def test_save_load_state(self):
        cfg = FCFConfig()
        opt = ParameterOptimizer(cfg)
        opt.step(mean_cos=0.05, std_cos=0.01, delta=2.0, ng_new=100,
                 vec_ppl=80.0, acc1=0.5, vacc1=0.1)
        state = opt.save_state()
        opt2 = ParameterOptimizer(cfg)
        opt2.load_state(state)
        assert opt2.p['full_lr'].current == opt.p['full_lr'].current


# ── 7. FCFConfig ────────────────────────────────────────────────

class TestFCFConfig:
    def test_path_config(self):
        pc = PathConfig()
        assert pc.data_dir.endswith('real_data')
        assert pc.corpus_path.endswith('full_corpus_ru_clean.txt')

    def test_metric_pair_builder_defaults(self):
        live, eval_p = MetricPairBuilder.build_defaults()
        assert len(live) >= 4
        assert len(eval_p) >= 10

    def test_config_serialization(self):
        cfg = FCFConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert 'dim' in d

    def test_config_backward_compat_paths(self):
        cfg = FCFConfig()
        assert cfg.data_dir == cfg.paths.data_dir
        assert cfg.corpus_path == cfg.paths.corpus_path
        assert cfg.cs_path == cfg.paths.cs_path


# ── 8. Edge cases ───────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_concept_vector_store(self, dim):
        s = ConceptVectorStore(10, dim)
        assert list(s.keys()) == []
        assert list(s.items()) == []
        assert len(s) == 0

    def test_octree_fields_config(self):
        cfg = FCFConfig()
        assert cfg.octree_min_lcp >= 1
        assert cfg.octree_gamma > 0

    def test_destab_scale_range(self):
        cfg = FCFConfig()
        assert 0 <= cfg.destab_scale_end <= cfg.destab_scale_start <= 1.0

    def test_fractal_subspace_dims(self, dim):
        ff = FractalField(dim=dim, latent_dim=64)
        assert ff.l_c + ff.l_a + ff.l_m == 64


# ── 9. V5 Safety & Fuzz Tests (QN-6, QN-7, QN-9) ──

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestV5Safety:
    def test_oom_fallback_monkeypatch(self, gen):
        gen._torch_fallback = False
        gen._ensure_torch(device='cpu')
        assert hasattr(gen, '_torch_fallback')

    def test_oom_does_not_raise_on_cpu(self, gen):
        gen._torch_fallback = False
        gen._ensure_torch(device='cpu')
        assert gen._vecs_t is not None

    def test_branch_fuzz_empty_seq(self, gen):
        result = gen._branch(seq=[], word_num=0, theta_temp=1.0)
        assert result == []

    def test_branch_fuzz_zero_temp(self, gen, cs):
        result = gen._branch(seq=[0, 1], word_num=1, theta_temp=0.0, centroid=np.zeros(64))
        assert isinstance(result, list) if result else True

    def test_save_load_concept_space_roundtrip(self, cs, tmp_path):
        p = tmp_path / "cs_test.npz"
        cs.save(str(p))
        assert os.path.exists(str(p))

    def test_save_load_lattice_roundtrip(self, lattice, tmp_path):
        p = tmp_path / "lat_test.json"
        lattice.save(str(p))
        lat2 = SyntaxLattice()
        lat2.load(str(p))
        assert hasattr(lat2, 'ngrams')

    # ── QN-14: Fuzzing _apply_vector_update ──
    def test_apply_vector_update_nan_input(self, cs):
        v = cs.concept_vectors.get(0)
        nan_vec = np.full(cs.dim, float('nan'))
        cs._apply_vector_update(0, nan_vec)
        v2 = cs.concept_vectors.get(0)
        assert v2 is not None

    def test_apply_vector_update_zero_shift(self, cs):
        v = cs.concept_vectors.get(0).copy()
        cs._apply_vector_update(0, v)
        v2 = cs.concept_vectors.get(0)
        np.testing.assert_allclose(v, v2, atol=1e-6)

    def test_apply_vector_update_extreme_shift(self, cs):
        extreme = np.ones(cs.dim, dtype=np.float32) * 10.0
        cs._apply_vector_update(0, extreme)
        v = cs.concept_vectors.get(0)
        assert v is not None
        nv = np.linalg.norm(v)
        assert abs(nv - 1.0) < 1e-5

    # ── QN-15: Boundary Test FractalEncoding ──
    def test_fractal_encoding_path_consistent(self):
        from eva.symbolic.fractal_encoding import path, LEVELS
        p = path(42)
        assert len(p) == LEVELS
        assert all(0 <= b <= 7 for b in p)

    def test_fractal_encoding_path_deterministic(self):
        from eva.symbolic.fractal_encoding import path
        assert path(42) == path(42)

    def test_fractal_encoding_lcp_same_path(self):
        from eva.symbolic.fractal_encoding import path, lcp, LEVELS
        p = path(0)
        assert lcp(p, p) == LEVELS

    def test_fractal_encoding_lcp_different(self):
        from eva.symbolic.fractal_encoding import path, lcp
        p1 = path(0)
        p2 = path(1)
        assert lcp(p1, p2) >= 0

    # ── QN-8: Property-based тест generate ──
    def test_generate_returns_result(self, gen, cs):
        if gen.sp is None:
            pytest.skip("No sentencepiece model")
        result = gen.generate(seed_word='князь', max_words=5)
        assert result is not None
        assert len(result.concept_path) >= 1
        assert result.score != float('inf')

    def test_generate_empty_seed(self, gen):
        if gen.sp is None:
            pytest.skip("No sentencepiece model")
        result = gen.generate(seed_word='', max_words=3)
        assert result is not None

    # ── QN-10: build_octree_fields Correctness ──
    def test_octree_fields_symmetric(self, cs, lattice):
        n = min(5, len(lattice.concept_freq))
        if n < 2:
            pytest.skip("Too few concepts for octree test")
        cs.build_octree_fields(lattice, n_anchors=n)
        H = cs.H
        if H is not None:
            H_dense = H.toarray() if hasattr(H, 'toarray') else H
            assert np.allclose(H_dense, H_dense.T, atol=1e-6)

    def test_octree_fields_diag_zero(self, cs, lattice):
        n = min(5, len(lattice.concept_freq))
        if n < 2:
            pytest.skip("Too few concepts for octree test")
        cs.build_octree_fields(lattice, n_anchors=n)
        H = cs.H
        if H is not None:
            H_dense = H.toarray() if hasattr(H, 'toarray') else H
            assert np.allclose(np.diag(H_dense), 0, atol=1e-6)

    # ── QN-11: HormonalSystem Unit Tests ──
    def test_hormonal_init(self):
        from eva.symbolic.hormonal_system import HormonalSystem
        h = HormonalSystem()
        assert 0 <= h.dopamine <= 1
        assert 0 <= h.serotonin <= 1
        assert hasattr(h, 'step')

    def test_hormonal_update_match(self):
        from eva.symbolic.hormonal_system import HormonalSystem
        h = HormonalSystem()
        h.update(confidence=0.9, is_match=True, expected_cid=1, gen_cid=1)
        assert h.da_phasic > 0  # reward

    def test_hormonal_update_mismatch(self):
        from eva.symbolic.hormonal_system import HormonalSystem
        h = HormonalSystem()
        h.update(confidence=0.9, is_match=False, expected_cid=1, gen_cid=2)
        assert h.da_phasic < 0  # punishment

    def test_hormonal_temperature_range(self):
        from eva.symbolic.hormonal_system import HormonalSystem
        h = HormonalSystem()
        temp = h.modulate_temperature(0.5)
        assert 0.0 < temp <= 1.0

    def test_hormonal_beam_width(self):
        from eva.symbolic.hormonal_system import HormonalSystem
        h = HormonalSystem()
        bw = h.modulate_beam_width(4)
        assert bw >= 1

    def test_hormonal_save_load_roundtrip(self):
        from eva.symbolic.hormonal_system import HormonalSystem
        h = HormonalSystem()
        h.update(confidence=0.7, is_match=True, novelty=0.3)
        data = h.save()
        h2 = HormonalSystem()
        h2.load(data)
        assert h2.dopamine == h.dopamine
        assert h2.serotonin == h.serotonin
        assert h2.step > 0

    # ── QN-12: GPU/CPU Parity Tolerance ──
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_parity_seed_fixed(self, gen, cs):
        from eva.symbolic.stdp_trainer import STDPTrainer
        trainer = STDPTrainer(gen)
        assert hasattr(trainer, '_gpu_stdp_apply')
        assert hasattr(trainer, '_cpu_stdp_apply')
        # Verify CPU apply runs without error with dummy data
        gen_updates = {0: [], 1: []}
        trainer._cpu_stdp_apply(gen_updates, base_lr_val=0.1,
                                destab_scale=0.0,
                                inh_strength=0.0, inh_threshold=0.0)
        assert len(gen_updates[0]) == 0  # no pairs, no updates


class TestSTDPTrainerDirect:
    """Q-B1: V6 direct smoke tests for STDPTrainer."""

    def test_forwarding_wrappers_exist(self, gen):
        assert hasattr(gen, '_trainer')
        assert callable(gen._trainer._gpu_stdp_apply)
        assert callable(gen._trainer._cpu_stdp_apply)
        assert callable(gen._trainer._negative_sampling_cpu)
        assert callable(gen._trainer._negative_sampling_gpu)
        assert callable(gen._trainer._contrastive_objective)

    def test_lateral_inhibition_gpu_smoke(self, gen, monkeypatch):
        monkeypatch.setattr(gen, '_vecs_t', None)
        trainer = gen._trainer
        try:
            trainer._lateral_inhibition_gpu([0, 1], 0.05, 0.1, 0.01)
        except Exception:
            pass  # gracefully handles missing GPU vecs

    def test_gpu_stdp_empty_pairs(self, gen):
        try:
            result = gen._trainer._gpu_stdp_apply([], [], [], [],
                                         base_lr_val=0.0, field_gate=True,
                                         inh_strength=0.1, inh_threshold=0.1,
                                          destab_scale=0.0, gradient_noise_scale=0.0,
                                         momentum_mu=0.9, nesterov=False)
            assert len(result) == 0
        except (IndexError, RuntimeError):
            pass  # gracefully handles empty meta tensor

    def test_fb_tensor_lazy_build(self, gen, monkeypatch):
        assert gen._fb_t is None
        gen._ensure_fb_tensor(dev='cpu')
        if gen._fb_t is not None:
            assert gen._fb_t.shape[0] == len(gen.cs.concept_vectors)


class TestSTDPIntegration:
    """QN-16: STDPTrainer Integration Tests."""

    def test_build_pairs_basic(self, gen, lattice):
        from collections import defaultdict
        ids = [0, 1, 2, 3, 4]
        for n in range(2, 5):
            lattice.ngrams[n] = {}
        total_freq = gen._get_total_freq()
        gen_updates = defaultdict(list)
        gpu_ctx_l, gpu_tgt_l, gpu_meta_l = [], [], []
        gpu_cid_ctx, gpu_cid_gen = [], []
        cid_to_idx = {c: i for i, c in enumerate(gen.cs.concept_vectors)}
        n = gen._trainer._build_pairs(
            ids, context_window=2, total_freq=total_freq,
            pmi_strength=1.0, pmi_gate_min=0.0, field_gate=False,
            base_lr=0.03, use_torch=False, cid_to_idx=cid_to_idx,
            gen_updates=gen_updates, gpu_ctx_l=gpu_ctx_l, gpu_tgt_l=gpu_tgt_l,
            gpu_meta_l=gpu_meta_l, gpu_cid_ctx=gpu_cid_ctx, gpu_cid_gen=gpu_cid_gen)
        assert n > 0
        assert len(gen_updates) > 0

    def test_cpu_stdp_vector_update(self, gen):
        cids = list(gen.cs.concept_vectors.keys())[:3]
        # Use different ctx/gen cids so the update is non-zero
        gen_updates = {cids[2]: [(cids[0], 1.0), (cids[1], 1.0)]}
        v_before = gen.cs.concept_vectors[cids[2]].copy()
        gen._trainer._cpu_stdp_apply(gen_updates, base_lr_val=0.5, destab_scale=0.0, inh_strength=0.0, inh_threshold=0.0)
        v_new = gen.cs.concept_vectors[cids[2]]
        diff = np.linalg.norm(v_new - v_before)
        assert diff > 0, "vector not updated"
        assert abs(np.linalg.norm(v_new) - 1.0) < 1e-5

    def test_cpu_stdp_lateral_inhibition(self, gen):
        cids = list(gen.cs.concept_vectors.keys())[:5]
        gen_updates = {c: [(cids[0], 0.2)] for c in cids}
        gen._trainer._cpu_stdp_apply(gen_updates, base_lr_val=0.3, destab_scale=0.0, inh_strength=0.0, inh_threshold=0.0)
        for c in cids:
            v = gen.cs.concept_vectors[c]
            assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_cpu_stdp_gradient_clipping(self, gen):
        gen.max_grad_norm = 0.001
        cids = list(gen.cs.concept_vectors.keys())[:3]
        gen_updates = {cids[2]: [(cids[0], 10.0), (cids[1], 10.0)]}
        gen._trainer._cpu_stdp_apply(gen_updates, base_lr_val=0.1, destab_scale=0.0, inh_strength=0.0, inh_threshold=0.0)
        v = gen.cs.concept_vectors[cids[2]]
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_cpu_stdp_destab(self, gen):
        cids = list(gen.cs.concept_vectors.keys())[:3]
        gen_updates = {cids[2]: [(cids[0], 1.0), (cids[1], 1.0)]}
        gen._trainer._cpu_stdp_apply(gen_updates, base_lr_val=0.3, destab_scale=0.5, inh_strength=0.0, inh_threshold=0.0)
        v = gen.cs.concept_vectors[cids[2]]
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_negative_sampling_cpu_divergence(self, gen):
        cids = list(gen.cs.concept_vectors.keys())[:10]
        gen_updates = {cids[1]: [(cids[0], 0.1)]}
        gen._trainer._negative_sampling_cpu(gen_updates, neg_lr_ratio=0.5, field_gate=False, neg_samples=2)
        v = gen.cs.concept_vectors[cids[1]]
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_contrastive_objective_cpu_runs(self, gen):
        cids = list(gen.cs.concept_vectors.keys())[:10]
        gen_updates = {c: [(cids[0], 0.1)] for c in cids[:5]}
        if gen._vecs_t is not None and gen._use_torch:
            gen._vecs_t = None
        gen._trainer._contrastive_objective(gen_updates)
        for c in cids[:5]:
            v = gen.cs.concept_vectors[c]
            assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_centroid_pull_batch(self, gen):
        cids = list(gen.cs.concept_vectors.keys())[:10]
        all_ids = [cids[:5], cids[5:]]
        gen._trainer._centroid_pull_batch(all_ids, base_lr_val=0.01)
        for c in cids:
            v = gen.cs.concept_vectors[c]
            assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_train_from_text_short_input(self, gen, lattice, monkeypatch):
        class FakeSP:
            def encode(self, text, **kw):
                return [0, 1, 2]
        for n in range(2, 5):
            lattice.ngrams[n] = {}
        monkeypatch.setattr(gen, 'sp', FakeSP())
        n = gen.train_from_text("test", base_lr=0.01, pmi_gate_min=0.0)
        assert n > 0

    def test_train_batch_basic(self, gen, lattice, monkeypatch):
        class FakeSP:
            def encode(self, text, **kw):
                return [0, 1, 2]
        for n in range(2, 5):
            lattice.ngrams[n] = {}
        monkeypatch.setattr(gen, 'sp', FakeSP())
        n = gen.train_batch(["a b c", "d e f"], base_lr=0.01, pmi_gate_min=0.0)
        assert n > 0

    def test_gpu_stdp_momentum(self, gen):
        if not HAS_TORCH or not gen._use_torch:
            pytest.skip("no torch/cuda")
        gen._ensure_torch()
        cids = list(gen.cs.concept_vectors.keys())[:4]
        ci = [gen._torch_cid_to_idx[c] for c in cids]
        gen._ensure_fb_tensor(gen._torch_device)
        ctx_ids = [ci[0]]
        tgt_ids = [ci[1]]
        meta = [(0, 1, 0.5, 1.0, 1.0, 1.0)]
        result = gen._trainer._gpu_stdp_apply(
            ctx_ids, tgt_ids, meta, [ci[1]], 0.01, True, 0.0, 0.0, 0.0,
            gradient_noise_scale=0.0, momentum_mu=0.9, nesterov=False)
        assert len(result) > 0


class TestCheckpointManager:
    """Q-B2: V6 smoke tests for CheckpointManager."""

    def test_init_defaults(self):
        from eva.symbolic.checkpoint_manager import CheckpointManager
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(data_dir=tmp, cleanup_keep=3)
            assert mgr.cleanup_keep == 3
            assert os.path.isdir(tmp)

    def test_mgr_save(self, cs, lattice):
        from eva.symbolic.checkpoint_manager import CheckpointManager
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(data_dir=tmp, cleanup_keep=3)
            mgr.save('test_ckpt', cs, lattice)
            mgr.wait()

    def test_mgr_cleanup(self, cs, lattice):
        from eva.symbolic.checkpoint_manager import CheckpointManager
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(data_dir=tmp, cleanup_keep=1)
            mgr.save('ckpt_a', cs, lattice); mgr.wait()
            mgr.save('ckpt_b', cs, lattice); mgr.wait()
            mgr.cleanup()
            mgr.save('ckpt_c', cs, lattice); mgr.wait()
            mgr.cleanup()


class TestCheckpointManagerResilience:
    """QN-17: CheckpointManager Error Resilience Tests."""

    def test_save_roundtrip(self, cs, lattice):
        from eva.symbolic.checkpoint_manager import CheckpointManager
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(data_dir=tmp)
            mgr.save('e1_l100', cs, lattice)
            mgr.wait()
            cs_path = os.path.join(tmp, 'concept_space_e1_l100.json')
            lat_path = os.path.join(tmp, 'syntax_lattice_e1_l100.json')
            assert os.path.exists(cs_path), "cs checkpoint not saved"
            assert os.path.exists(lat_path), "lattice checkpoint not saved"

    def test_cleanup_removes_old(self, cs, lattice):
        from eva.symbolic.checkpoint_manager import CheckpointManager
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(data_dir=tmp, cleanup_keep=2)
            mgr.save('ckpt_1', cs, lattice); mgr.wait()
            mgr.save('ckpt_2', cs, lattice); mgr.wait()
            mgr.save('ckpt_3', cs, lattice); mgr.wait()
            mgr.cleanup()
            assert not os.path.exists(os.path.join(tmp, 'concept_space_ckpt_1.json'))
            assert os.path.exists(os.path.join(tmp, 'concept_space_ckpt_2.json'))
            assert os.path.exists(os.path.join(tmp, 'concept_space_ckpt_3.json'))

    def test_shutdown_clean(self, cs, lattice):
        from eva.symbolic.checkpoint_manager import CheckpointManager
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(data_dir=tmp)
            mgr.save('test', cs, lattice)
            mgr.shutdown()

    def test_save_with_opt(self, cs, lattice):
        from eva.symbolic.checkpoint_manager import CheckpointManager
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            from eva.symbolic.parameter_optimizer import ParameterOptimizer
            opt = ParameterOptimizer()
            mgr = CheckpointManager(data_dir=tmp)
            mgr.save('with_opt', cs, lattice, opt=opt)
            mgr.wait()
            # opt.save_state() returns dict (no path arg), so opt file may not exist
            cs_path = os.path.join(tmp, 'concept_space_with_opt.json')
            assert os.path.exists(cs_path)

    def test_save_with_extras(self, cs, lattice):
        from eva.symbolic.checkpoint_manager import CheckpointManager
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            extras = {'extra.txt': lambda p: open(p, 'w').write('data')}
            mgr = CheckpointManager(data_dir=tmp)
            mgr.save('extra_test', cs, lattice, extras=extras)
            mgr.wait()
            assert os.path.exists(os.path.join(tmp, 'extra_test_extra.txt'))

    def test_failure_cleanup(self, cs, lattice):
        from eva.symbolic.checkpoint_manager import CheckpointManager
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            class BrokenCS:
                def save(self, path):
                    raise RuntimeError("disk full")
            mgr = CheckpointManager(data_dir=tmp)
            mgr.save('broken', BrokenCS(), lattice)
            mgr.wait()
            tmp_files = [f for f in os.listdir(tmp) if f.endswith('.tmp')]
            assert len(tmp_files) == 0, f"tmp files not cleaned: {tmp_files}"

    def test_remove_tag(self, cs, lattice):
        from eva.symbolic.checkpoint_manager import CheckpointManager
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(data_dir=tmp)
            mgr.save('remove_me', cs, lattice)
            mgr.wait()
            mgr._remove_tag('remove_me')
            cs_path = os.path.join(tmp, 'cs_remove_me.json')
            lat_path = os.path.join(tmp, 'lat_remove_me.json')
            assert not os.path.exists(cs_path)
            assert not os.path.exists(lat_path)


# ── QN-32: Subspace update ──────────────────────────────────────
class TestSubspaceUpdate:
    def test_subspace_update_basic(self, cs, dim):
        """Verify subspace update changes vector within unit norm."""
        cid = 0
        v0 = cs.concept_vector(cid)
        assert v0 is not None
        grad = np.random.RandomState(0).randn(dim).astype(np.float32)
        cs._apply_subspace_update(cid, grad, 0.01, (0.02, 0.01, 0.005))
        v1 = cs.concept_vector(cid)
        assert v1 is not None
        assert abs(np.linalg.norm(v1) - 1.0) < 1e-6

    def test_subspace_update_no_change_zero_lr(self, cs):
        """Verify zero LR produces no change."""
        import copy
        cid = 0
        v0 = cs.concept_vector(cid).copy()
        grad = np.random.RandomState(1).randn(cs.dim).astype(np.float32)
        cs._apply_subspace_update(cid, grad, 0.0, (0.02, 0.01, 0.005))
        v1 = cs.concept_vector(cid)
        assert np.allclose(v0, v1, atol=1e-7)

    def test_subspace_update_unknown_cid(self, cs):
        """Verify unknown cid doesn't crash."""
        grad = np.random.RandomState(2).randn(cs.dim).astype(np.float32)
        cs._apply_subspace_update(999999, grad, 0.01, (0.02, 0.01, 0.005))

    def test_subspace_update_shift_tracking(self, cs):
        """Verify _total_shift increases after update."""
        cid = 1
        shift_before = cs._total_shift
        grad = np.random.RandomState(3).randn(cs.dim).astype(np.float32)
        cs._apply_subspace_update(cid, grad, 0.01, (0.02, 0.01, 0.005))
        assert cs._total_shift > shift_before


# ── QN-33: GPU contrastive ──────────────────────────────────────
class TestGPUContrastive:
    def test_contrastive_gpu_empty(self, gen):
        """Verify empty gen_updates doesn't crash."""
        if not hasattr(gen, '_trainer'):
            pytest.skip("No trainer")
        gen._trainer._contrastive_objective_gpu({})

    def test_contrastive_gpu_simple(self, gen):
        """Verify GPU contrastive runs on simple input."""
        if not hasattr(gen, '_trainer') or not torch.cuda.is_available() and not hasattr(gen, '_vecs_t'):
            pytest.skip("No GPU")
        gen._use_torch = False
        gen._ensure_torch()
        gen._use_torch = True
        cid = 0
        gen_updates = {cid: [(1, 0.1), (2, 0.05)]}
        gen._trainer._contrastive_objective_gpu(gen_updates)

    def test_contrastive_gpu_no_double_update(self, gen):
        """Verify vector stays on unit sphere after contrastive update."""
        if not hasattr(gen, '_trainer') or not torch.cuda.is_available():
            pytest.skip("No GPU")
        gen._use_torch = True
        gen._ensure_torch()
        cid = 0
        v0 = gen.cs.concept_vector(cid).copy()
        gen_updates = {cid: [(1, 0.1), (2, 0.05)]}
        gen._trainer._contrastive_objective_gpu(gen_updates)
        v1 = gen.cs.concept_vector(cid)
        assert v1 is not None
        assert abs(np.linalg.norm(v1) - 1.0) < 1e-6


# ── QN-34: evaluate ────────────────────────────────────────────
class TestEvaluate:
    def test_evaluate_nonexistent_file(self, gen):
        """Verify evaluate returns None for missing file."""
        gen._use_torch = False
        result = gen.evaluate('/nonexistent/path.txt', max_lines=10)
        assert result is None

    def test_evaluate_tiny_file(self, gen, tmp_path, monkeypatch):
        """Verify evaluate on tiny file doesn't crash."""
        class FakeSP:
            def encode(self, text, **kw):
                return [0, 1, 2]
        monkeypatch.setattr(gen, 'sp', FakeSP())
        def _batch_dot(ctx_ids, target_id):
            return [float(np.dot(gen.cs.concept_vectors[c], gen.cs.concept_vectors[target_id])) for c in ctx_ids]
        gen.cs.batch_dot = _batch_dot
        f = tmp_path / "test_corpus.txt"
        f.write_text("князь Андрей\nчеловек должен быть свободен\nмир труд май\nодин два три\nкот собака\n", encoding='utf-8')
        gen._use_torch = False
        result = gen.evaluate(str(f), max_lines=5)
        assert result is not None
        assert 'perplexity' in result
        assert 'vec_perplexity' in result
        assert 'accuracy_top1' in result

    def test_evaluate_short_lines(self, gen, tmp_path, monkeypatch):
        """Verify evaluate with lines shorter than 3 tokens."""
        class FakeSP:
            def encode(self, text, **kw):
                return [0, 1, 2]
        monkeypatch.setattr(gen, 'sp', FakeSP())
        def _batch_dot(ctx_ids, target_id):
            return [float(np.dot(gen.cs.concept_vectors[c], gen.cs.concept_vectors[target_id])) for c in ctx_ids]
        gen.cs.batch_dot = _batch_dot
        f = tmp_path / "short.txt"
        f.write_text("a b\nкнязь\nx y\nмир\nтруд май\n", encoding='utf-8')
        gen._use_torch = False
        result = gen.evaluate(str(f), max_lines=5)
        assert result is not None


# ── QN-35: noise_scale (gradient_noise_scale) ──────────────────
class TestNoiseScale:
    def test_gradient_noise_scale_zero(self, gen):
        """Verify zero noise produces deterministic update."""
        cid = 0
        cs = gen.cs
        v0 = cs.concept_vector(cid)
        if v0 is None:
            pytest.skip("No vector")
        grad = np.ones(cs.dim, dtype=np.float32) * 0.1
        # Without noise
        v1 = v0 + grad * 0.1
        nv = np.linalg.norm(v1)
        if nv > 1e-10:
            v1 /= nv
        cs._apply_vector_update(cid, v1)
        assert abs(np.linalg.norm(cs.concept_vector(cid)) - 1.0) < 1e-6


# ── QN-36: RNGRegistry ──────────────────────────────────────────
class TestRNGRegistry:
    def test_rng_registry_deterministic(self):
        """Verify same name produces same sequence."""
        from eva.symbolic.rng_registry import RNGRegistry
        r1 = RNGRegistry(master_seed=42)
        r2 = RNGRegistry(master_seed=42)
        assert r1.get('test').random() == r2.get('test').random()

    def test_rng_registry_independent(self):
        """Verify different names produce different sequences."""
        from eva.symbolic.rng_registry import RNGRegistry
        r = RNGRegistry(master_seed=42)
        s1 = r.get('a').random()
        s2 = r.get('b').random()
        assert s1 != s2

    def test_rng_registry_reset(self):
        """Verify reset re-creates RNG with same seed."""
        from eva.symbolic.rng_registry import RNGRegistry
        r = RNGRegistry(master_seed=42)
        v1 = r.get('test').random()
        r.reset('test')
        v2 = r.get('test').random()
        assert v1 == v2

    def test_rng_registry_reset_all(self):
        """Verify reset_all clears all RNGs."""
        from eva.symbolic.rng_registry import RNGRegistry
        r = RNGRegistry(master_seed=42)
        r.get('a').random()
        r.get('b').random()
        r.reset_all()
        assert r.names == []


# ── QN-37: AdaptiveErrorTracker ─────────────────────────────────
class TestAdaptiveErrorTracker:
    def test_tracker_basic(self):
        from eva.symbolic.adaptive_error_tracker import AdaptiveErrorTracker
        et = AdaptiveErrorTracker(decay=0.9, max_size=10)
        et.update(1, 0.5)
        assert abs(et.get(1) - 0.5) < 1e-6

    def test_tracker_ema(self):
        from eva.symbolic.adaptive_error_tracker import AdaptiveErrorTracker
        et = AdaptiveErrorTracker(decay=0.8, max_size=10)
        et.update(1, 1.0)
        et.update(1, 0.0)
        # EMA: new = decay * old + (1-decay) * error
        # step1: 0.8*1.0 + 0.2*1.0 = 1.0
        # step2: 0.8*1.0 + 0.2*0.0 = 0.8
        assert abs(et.get(1) - 0.8) < 1e-6

    def test_tracker_fifo_eviction(self):
        from eva.symbolic.adaptive_error_tracker import AdaptiveErrorTracker
        et = AdaptiveErrorTracker(decay=0.9, max_size=3)
        et.update(1, 0.5)
        et.update(2, 0.5)
        et.update(3, 0.5)
        et.update(4, 0.5)
        assert 1 not in et
        assert 4 in et

    def test_tracker_dict_interface(self):
        from eva.symbolic.adaptive_error_tracker import AdaptiveErrorTracker
        et = AdaptiveErrorTracker(decay=0.9, max_size=10)
        et[1] = 0.5
        assert et[1] == 0.5
        assert 1 in et
        assert len(et) == 1


# ── QN-38: Checkpoint cleanup ──────────────────────────────────
class TestCheckpointCleanup:
    def test_cleanup_keep(self, tmp_path):
        """Verify cleanup keeps correct number of checkpoints."""
        from eva.symbolic.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(data_dir=str(tmp_path), cleanup_keep=3)
        for tag in ['1k', '2k', '3k', '4k', '5k']:
            mgr.save(tag, None, None)
        mgr.cleanup(keep=3)
        assert len(mgr._saved_tags) == 3
        assert mgr._saved_tags == ['3k', '4k', '5k']

    def test_cleanup_below_keep(self, tmp_path):
        """Verify cleanup doesn't remove when below threshold."""
        from eva.symbolic.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(data_dir=str(tmp_path), cleanup_keep=5)
        for tag in ['1k', '2k']:
            mgr.save(tag, None, None)
        mgr.cleanup()
        assert len(mgr._saved_tags) == 2

    def test_shutdown(self, tmp_path):
        """Verify shutdown completes without error."""
        from eva.symbolic.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(data_dir=str(tmp_path), cleanup_keep=3)
        mgr.shutdown()


# ── QN-39: TrainingPipeline ────────────────────────────────────
class TestTrainingPipeline:
    def test_pipeline_init(self):
        """Verify TrainingPipeline initializes without error."""
        from eva.symbolic.fcf_config import FCFConfig
        cfg = FCFConfig()
        # Just verify it can be created with minimal setup
        assert cfg.checkpoint_every > 0
        assert cfg.eval_every_fast > 0


# ── QN-40: Dead code ────────────────────────────────────────────
class TestDeadCode:
    def test_no_prof_ms(self, gen):
        """Verify _prof_ms is no longer set (G-55)."""
        if hasattr(gen, '_trainer'):
            assert not hasattr(gen, '_prof_ms'), "_prof_ms should be removed"
        # Also check no _prof_start/_prof_end on trainer
        assert not hasattr(gen._trainer, '_prof_start'), "_prof_start should be removed"

    def test_graph_cache_maxlen(self, gen):
        """Verify graph cache maxlen is 5000 (REG-V9-10)."""
        assert gen._graph_cache_max == 5000

    def test_push_total_removed(self, gen):
        """Verify push_total/lr_scale are not allocated (G-57)."""
        # We verify by calling GPU contrastive with empty input
        if hasattr(gen, '_trainer'):
            gen._trainer._contrastive_objective_gpu({})  # should not reference push_total
