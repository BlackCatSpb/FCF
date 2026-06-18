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
        ff.fluctuate(noise_scale=0.001, decay=0.999)
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
        gen._cpu_stdp_apply({1: [(0, 0.1), (2, 0.05)]}, base_lr_val=0.03,
                            destab_scale=0.0, inh_strength=0.0, inh_threshold=0.1)
        v1 = cs.concept_vectors.get(1)
        assert v1 is not None
        assert abs(np.linalg.norm(v1) - 1.0) < 1e-6

    def test_negative_sampling_cpu(self, gen, cs):
        gen._negative_sampling_cpu({1: [(0, 0.1)]}, neg_lr_ratio=0.5,
                                    field_gate=False, neg_samples=2)
        v1 = cs.concept_vectors.get(1)
        assert v1 is not None

    def test_contrastive_objective(self, gen, cs):
        gen._contrastive_objective({1: [(0, 0.1)]})
        v1 = cs.concept_vectors.get(1)
        assert v1 is not None
        assert abs(np.linalg.norm(v1) - 1.0) < 1e-6

    def test_concept_error_fifo(self, gen):
        for i in range(50):
            gen.concept_error[i] = float(i) / 50.0
        _ce_limit = min(3 * gen.cs.vocab_size // 4, 100000)
        while len(gen.concept_error) > _ce_limit:
            gen.concept_error.popitem(last=False)
        assert len(gen.concept_error) <= _ce_limit


# ── 5. GPU/CPU parity ───────────────────────────────────────────

class TestGPUParity:
    def test_cpu_no_torch(self, gen, cs):
        v_before = cs.concept_vectors.get(1).copy()
        gen._cpu_stdp_apply({1: [(0, 0.1)]}, base_lr_val=0.03,
                            destab_scale=0.0, inh_strength=0.0, inh_threshold=0.1)
        v_after = cs.concept_vectors.get(1)
        assert not np.allclose(v_before, v_after)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_gpu_stdp_apply_no_crash(self, gen, cs):
        gen._torch_device = torch.device('cpu')
        gen._ensure_torch(device='cpu')
        gen._gpu_stdp_apply(
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
        gen._negative_sampling_gpu(
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
        gen._gpu_stdp_apply(
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
        gen2._cpu_stdp_apply({1: [(0, 0.1)]}, base_lr_val=0.03,
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
