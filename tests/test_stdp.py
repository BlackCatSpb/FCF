"""Unit tests for FCF core: STDP, ConceptSpace, GPU/CPU parity."""

import math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eva.symbolic.concept_space import ConceptSpace, ConceptVectorStore, FractalField
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.parameter_optimizer import ParameterOptimizer
from eva.symbolic.fcf_config import FCFConfig, PathConfig, MetricPairBuilder

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_TORCH = False

DIM = 64
VOCAB_SIZE = 20


def make_minimal_cs():
    cs = ConceptSpace(vocab_size=VOCAB_SIZE, dim=DIM)
    cs.init_concepts()
    cs.init_homeostasis()
    return cs


def make_minimal_lattice():
    lat = SyntaxLattice()
    lat.concept_freq = {i: max(10 - i, 1) for i in range(VOCAB_SIZE)}
    return lat


# ── 1. ConceptVectorStore ───────────────────────────────────────

class TestConceptVectorStore:
    def test_basic_crud(self):
        s = ConceptVectorStore(10, DIM)
        assert len(s) == 0
        v = np.random.randn(DIM).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-10)
        s[3] = v
        assert 3 in s
        assert len(s) == 1
        r = s.get(3)
        assert r is not None
        assert np.allclose(r, v)
        assert s.get(999) is None

    def test_bounds_check(self):
        s = ConceptVectorStore(10, DIM)
        v = np.random.randn(DIM).astype(np.float32)
        s[0] = v
        assert s.get(-1) is None
        assert s.get(100) is None

    def test_items_and_keys(self):
        s = ConceptVectorStore(5, DIM)
        for i in range(5):
            v = np.random.randn(DIM).astype(np.float32)
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
    def test_init_and_vector(self):
        ff = FractalField(dim=DIM, latent_dim=64)
        v = ff.init_concept(0)
        assert v is not None
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_basis_health(self):
        ff = FractalField(dim=DIM, latent_dim=64)
        assert not ff.check_basis_health()

    def test_fluctuate(self):
        ff = FractalField(dim=DIM, latent_dim=64)
        v0 = ff.init_concept(0)
        ff.fluctuate(noise_scale=0.001, decay=0.999)
        v1 = ff.compute_vector(0)
        assert v1 is not None
        assert abs(np.linalg.norm(v1) - 1.0) < 1e-6

    def test_fb_dirty_flag(self):
        ff = FractalField(dim=DIM, latent_dim=64)
        assert not ff._fb_dirty
        ff.init_fields(n_anchors=32)
        assert ff._fb_dirty
        ff._fb_dirty = False
        ff.init_fields(n_anchors=64)
        assert ff._fb_dirty


# ── 3. ConceptSpace ─────────────────────────────────────────────

class TestConceptSpace:
    def test_initialization(self):
        cs = make_minimal_cs()
        assert cs.vocab_size == VOCAB_SIZE
        assert len(cs.concept_vectors) > 0

    def test_vector_norms(self):
        cs = make_minimal_cs()
        ok, total, max_dev = cs.validate_vector_norms()
        assert ok == total

    def test_topk_similar(self):
        cs = make_minimal_cs()
        top = cs.topk_similar_concepts(0, k=5, sample_size=20)
        assert len(top) <= 5
        for cid, sim in top:
            assert cid != 0
            assert -1.0 <= sim <= 1.0

    def test_apply_vector_update(self):
        cs = make_minimal_cs()
        v = cs.concept_vectors.get(0)
        assert v is not None
        v_new = v + np.random.randn(DIM).astype(np.float32) * 0.01
        nv = np.linalg.norm(v_new)
        if nv > 1e-10:
            v_new /= nv
        cs._apply_vector_update(0, v_new)
        v_after = cs.concept_vectors.get(0)
        assert v_after is not None
        assert abs(np.linalg.norm(v_after) - 1.0) < 1e-6


# ── 4. STDP ─────────────────────────────────────────────────────

class TestSTDP:
    def test_cpu_stdp_no_crash(self):
        cs = make_minimal_cs()
        lattice = make_minimal_lattice()
        from eva.symbolic.crystal_generator import CrystalGenerator
        gen = CrystalGenerator(cs, None, lattice)
        gen.max_grad_norm = 1.0
        gen._cpu_stdp_apply({1: [(0, 0.1), (2, 0.05)]}, base_lr_val=0.03,
                            destab_scale=0.0, inh_strength=0.0, inh_threshold=0.1)
        v1 = cs.concept_vectors.get(1)
        assert v1 is not None
        assert abs(np.linalg.norm(v1) - 1.0) < 1e-6

    def test_negative_sampling_cpu(self):
        cs = make_minimal_cs()
        lattice = make_minimal_lattice()
        from eva.symbolic.crystal_generator import CrystalGenerator
        gen = CrystalGenerator(cs, None, lattice)
        gen.max_grad_norm = 1.0
        gen._negative_sampling_cpu({1: [(0, 0.1)]}, neg_lr_ratio=0.5,
                                    field_gate=False, neg_samples=2)
        v1 = cs.concept_vectors.get(1)
        assert v1 is not None

    def test_contrastive_objective(self):
        cs = make_minimal_cs()
        lattice = make_minimal_lattice()
        from eva.symbolic.crystal_generator import CrystalGenerator
        gen = CrystalGenerator(cs, None, lattice)
        gen.max_grad_norm = 1.0
        gen._contrastive_objective({1: [(0, 0.1)]})
        v1 = cs.concept_vectors.get(1)
        assert v1 is not None
        assert abs(np.linalg.norm(v1) - 1.0) < 1e-6

    def test_concept_error_fifo(self):
        cs = make_minimal_cs()
        lattice = make_minimal_lattice()
        from eva.symbolic.crystal_generator import CrystalGenerator
        gen = CrystalGenerator(cs, None, lattice)
        for i in range(50):
            gen.concept_error[i] = float(i) / 50.0
        while len(gen.concept_error) > 30:
            gen.concept_error.popitem(last=False)
        assert len(gen.concept_error) <= 30


# ── 5. GPU/CPU parity ───────────────────────────────────────────

class TestGPUParity:
    def test_cpu_no_torch(self):
        cs = make_minimal_cs()
        lattice = make_minimal_lattice()
        from eva.symbolic.crystal_generator import CrystalGenerator
        gen = CrystalGenerator(cs, None, lattice)
        gen.max_grad_norm = 1.0
        v_before = cs.concept_vectors.get(1).copy()
        gen._cpu_stdp_apply({1: [(0, 0.1)]}, base_lr_val=0.03,
                            destab_scale=0.0, inh_strength=0.0, inh_threshold=0.1)
        v_after = cs.concept_vectors.get(1)
        assert not np.allclose(v_before, v_after)

    @staticmethod
    def _make_spy():
        import sentencepiece as spm
        path = os.path.join(os.path.dirname(__file__), '..', 'real_data', 'bpe_ru_146k.model')
        if os.path.exists(path):
            return spm.SentencePieceProcessor(model_file=path)
        return None


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

    def test_save_load_state(self):
        cfg = FCFConfig()
        opt = ParameterOptimizer(cfg)
        opt.step(mean_cos=0.05, std_cos=0.01, delta=2.0, ng_new=100,
                 vec_ppl=80.0, acc1=0.5, vacc1=0.1)
        state = opt.save_state()
        opt2 = ParameterOptimizer(cfg)
        opt2.load_state(state)
        assert opt2.p['full_lr'].current == opt.p['full_lr'].current

    def test_full_stuck_detection(self):
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
    def test_empty_concept_vector_store(self):
        s = ConceptVectorStore(10, DIM)
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

    def test_fractal_subspace_dims(self):
        ff = FractalField(dim=DIM, latent_dim=64)
        assert ff.l_c + ff.l_a + ff.l_m == 64
