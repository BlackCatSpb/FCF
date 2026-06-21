"""FCFConfig — единый источник истины для всех параметров FCF.

Всё, что можно менять без переписывания кода — здесь.
Алгоритмические константы (формулы, шкалы, структуры) остаются в коде.
"""

from __future__ import annotations
import os, json, math, random
from dataclasses import dataclass, field
from typing import Optional


def _auto_base_dir() -> str:
    """Определить корень проекта (FCF/ — родитель eva/)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────
#  Адаптивное правило для ParameterOptimizer
# ──────────────────────────────────────────────

@dataclass
class AdaptRule:
    """Правило: при условии → изменить параметр.

    trigger — имя триггера из набора:
      'mean_cos > X', 'mean_cos < X',
      'std_cos < X*TARGET', 'std_cos > X*TARGET',
      'delta < X', 'delta > X',
      'ng_new < X', 'ng_new > X',
      'vacc1_stuck >= N',
      'vacc1 > X',
      'vec_ppl_plateau', 'acc1_plateau'
    """
    trigger: str
    param: str
    action: str          # 'scale', 'shift', 'set', 'toward_default'
    value: float
    rate: float = 0.0    # для toward_default


# ──────────────────────────────────────────────
#  Параметр с коридором
# ──────────────────────────────────────────────

@dataclass
class ParamDef:
    name: str
    min_val: float
    max_val: float
    default: float
    step_scale: float = 0.1
    rules: list = field(default_factory=list)  # list[AdaptRule]


# ──────────────────────────────────────────────
#  Метрическая пара
# ──────────────────────────────────────────────

@dataclass
class MetricPair:
    a: str
    b: str
    label: str = ''       # 'antonym', 'synonym', 'morph', 'random'
    expected_sim: float = 0.0  # ожидаемая близость (опционально)


# ──────────────────────────────────────────────
#  Пути к файлам
# ──────────────────────────────────────────────

@dataclass
class PathConfig:
    """All file/directory paths for the FCF project."""
    base_dir: str = field(default_factory=_auto_base_dir)

    @property
    def data_dir(self) -> str:
        return os.path.join(self.base_dir, 'real_data')

    @property
    def corpus_path(self) -> str:
        return os.path.join(self.data_dir, 'full_corpus_ru_clean.txt')

    @property
    def bpe_model_path(self) -> str:
        return os.path.join(self.data_dir, 'bpe_ru_146k.model')

    @property
    def morph_vocab_path(self) -> str:
        return os.path.join(self.data_dir, 'morph_vocab.json')

    @property
    def cs_path(self) -> str:
        return os.path.join(self.data_dir, 'concept_space.json')

    @property
    def lattice_path(self) -> str:
        return os.path.join(self.data_dir, 'syntax_lattice.json')

    @property
    def log_path(self) -> str:
        return os.path.join(self.data_dir, 'train_log.txt')

    @property
    def val_corpus_path(self) -> str:
        return os.path.join(self.data_dir, 'val_corpus.txt')

    @property
    def vis_dir(self) -> str:
        return os.path.join(self.data_dir, 'vis')

    @property
    def status_path(self) -> str:
        return os.path.join(self.data_dir, '_train_status.json')

    @property
    def ckpt_state_path(self) -> str:
        return os.path.join(self.data_dir, 'checkpoint_state.json')


# ──────────────────────────────────────────────
#  Построитель метрических пар
# ──────────────────────────────────────────────

class MetricPairBuilder:
    """Static builder for MetricPair lists from morph_vocab and lattice."""

    @staticmethod
    def build_antonym_pairs(morph_vocab, sp, n=5) -> list:
        """Построить антонимичные пары через приставки не-/без-/бес-."""
        pairs = []
        wc = morph_vocab.word_cache
        words = list(wc.keys())
        random.seed(42)
        random.shuffle(words)
        found = 0
        seen = set()
        for w in words:
            if found >= n:
                break
            if w.startswith('не') and len(w) > 4:
                base = w[2:]
                if base in wc and base not in seen:
                    pairs.append(MetricPair(w, base, 'antonym'))
                    seen.add(w); seen.add(base)
                    found += 1
            elif w.startswith('без') or w.startswith('бес'):
                base = w[4:] if w.startswith('бес') else w[3:]
                if base in wc and base not in seen:
                    pairs.append(MetricPair(w, base, 'antonym'))
                    seen.add(w); seen.add(base)
                    found += 1
        return pairs

    @staticmethod
    def build_morph_pairs(morph_vocab, n=5) -> list:
        """Пары форм одной леммы — проверка LCP в октантных путях."""
        pairs = []
        from collections import defaultdict
        by_lemma = defaultdict(set)
        wi = getattr(morph_vocab, '_word_info', {})
        if not wi:
            return pairs
        for word in morph_vocab.word_cache:
            if word in wi:
                lemma = wi[word][0]
                by_lemma[lemma].add(word)
        found = 0
        for lemma, forms in by_lemma.items():
            if found >= n:
                break
            forms = list(forms)
            if len(forms) >= 2:
                a, b = forms[0], forms[-1]
                if a in morph_vocab.word_cache and b in morph_vocab.word_cache:
                    pairs.append(MetricPair(a, b, 'morph'))
                    found += 1
        return pairs

    @staticmethod
    def build_high_pmi_pairs(lattice, sp, n=10) -> list:
        """Top-PMI биграммы из корпуса — сильные коллокации."""
        pairs = []
        if not lattice.ngrams or not lattice.ngrams.get(2):
            return pairs
        scored = []
        for (a,), counter in lattice.ngrams[2].items():
            total = sum(counter.values())
            if total < 5:
                continue
            for b, cnt in counter.most_common(5):
                pmi = math.log(cnt / total / max(lattice.concept_freq.get(b, 1) / max(sum(lattice.concept_freq.values()), 1), 1e-10))
                if pmi > 3:
                    scored.append((pmi, a, b))
        scored.sort(key=lambda x: -x[0])
        for pmi, a, b in scored[:n]:
            ta = sp.IdToPiece(a) if a < sp.vocab_size() else str(a)
            tb = sp.IdToPiece(b) if b < sp.vocab_size() else str(b)
            pairs.append(MetricPair(ta, tb, 'collocation', expected_sim=min(pmi/5, 1.0)))
        return pairs

    @staticmethod
    def build_defaults() -> tuple:
        """Return (live_pairs, eval_pairs) with built-in antonym pairs."""
        builtin = [
            MetricPair('да', 'нет', 'antonym'),
            MetricPair('хороший', 'плохой', 'antonym'),
            MetricPair('большой', 'маленький', 'antonym'),
            MetricPair('высокий', 'низкий', 'antonym'),
            MetricPair('правда', 'ложь', 'antonym'),
            MetricPair('жизнь', 'смерть', 'antonym'),
            MetricPair('война', 'мир', 'antonym'),
            MetricPair('любовь', 'ненависть', 'antonym'),
            MetricPair('всегда', 'никогда', 'antonym'),
            MetricPair('начало', 'конец', 'antonym'),
            MetricPair('новый', 'старый', 'antonym'),
            MetricPair('белый', 'чёрный', 'antonym'),
            MetricPair('день', 'ночь', 'antonym'),
            MetricPair('добро', 'зло', 'antonym'),
        ]
        live = [
            MetricPair('соба', 'ка', 'bpe'),
            MetricPair('человек', 'война', 'bpe'),
            MetricPair('князь', 'Андрей', 'bpe'),
            MetricPair('любовь', 'смерть', 'bpe'),
        ]
        return live, builtin


# ──────────────────────────────────────────────
#  Главный конфиг
# ──────────────────────────────────────────────

@dataclass
class FCFConfig:
    # ── Пути ─────────────────────────────────
    paths: PathConfig = field(default_factory=PathConfig)

    @property
    def data_dir(self) -> str:
        return self.paths.data_dir

    @property
    def corpus_path(self) -> str:
        return self.paths.corpus_path

    @property
    def bpe_model_path(self) -> str:
        return self.paths.bpe_model_path

    @property
    def morph_vocab_path(self) -> str:
        return self.paths.morph_vocab_path

    @property
    def cs_path(self) -> str:
        return self.paths.cs_path

    @property
    def lattice_path(self) -> str:
        return self.paths.lattice_path

    @property
    def log_path(self) -> str:
        return self.paths.log_path

    @property
    def val_corpus_path(self) -> str:
        return self.paths.val_corpus_path

    @property
    def vis_dir(self) -> str:
        return self.paths.vis_dir

    @property
    def status_path(self) -> str:
        return self.paths.status_path

    @property
    def ckpt_state_path(self) -> str:
        return self.paths.ckpt_state_path

    # ── Архитектура ─────────────────────────
    dim: int = 384
    latent_dim: int = 512
    n_anchors: int = 2048
    max_n: int = 4
    octree_levels: int = 16

    @property
    def l_c(self) -> int: return self.latent_dim // 2

    @property
    def l_a(self) -> int: return self.latent_dim // 4

    @property
    def l_m(self) -> int: return self.latent_dim - self.l_c - self.l_a

    def get_field_dims(self):
        return {'l_c': self.l_c, 'l_a': self.l_a, 'l_m': self.l_m}

    # ── Seed ─────────────────────────────────
    global_seed: int = 42

    # ── Параметры адаптации ──────────────────
    params: list = field(default_factory=lambda: [
        ParamDef('full_lr',        0.003,  0.50,   0.03,   0.10, rules=[
            AdaptRule('cos_trend > 0.001 and mean_cos > 0.005', 'full_lr', 'scale', 0.95),
            AdaptRule('cos_trend < -0.001 and mean_cos < -0.005', 'full_lr', 'scale', 1.05),
            AdaptRule('vec_ppl_plateau', 'full_lr', 'scale', 1.08),
            AdaptRule('cos_flat >= 3', 'full_lr', 'scale', 1.15),
        ]),
        ParamDef('repel_strength', 0.01,   0.20,   0.08,   0.05, rules=[
            AdaptRule('mean_cos > 0.01', 'repel_strength', 'scale', 1.10),
            AdaptRule('mean_cos < -0.005', 'repel_strength', 'scale', 0.90),
        ]),
        ParamDef('gradient_noise_scale', 0.0,    0.01,   0.001,  0.05, rules=[
            AdaptRule('std_cos < 0.80*TARGET', 'gradient_noise_scale', 'scale', 1.15),
            AdaptRule('std_cos > 1.30*TARGET', 'gradient_noise_scale', 'scale', 0.90),
        ]),
        ParamDef('fluctuation_amp',   0.0002, 0.01,   0.003,  0.05, rules=[
            AdaptRule('std_cos < 0.80*TARGET', 'fluctuation_amp', 'scale', 1.15),
            AdaptRule('std_cos > 1.30*TARGET', 'fluctuation_amp', 'scale', 0.90),
        ]),
        ParamDef('inh_threshold',  0.05,   0.30,   0.10,   0.05, rules=[
            AdaptRule('est_frac > 0.15', 'inh_threshold', 'shift', 0.02),
            AdaptRule('est_frac < 0.01', 'inh_threshold', 'shift', -0.02),
        ]),
        ParamDef('inh_strength',   0.01,   0.15,   0.05,   0.05),
        ParamDef('inh_sample',     100,    600,    200,    100),
        ParamDef('context_window', 1,      6,      2,      0.5, rules=[
            AdaptRule('vec_ppl_plateau', 'context_window', 'shift', 0.5),
            AdaptRule('cos_flat >= 5', 'context_window', 'shift', 0.5),
        ]),
        ParamDef('theta_tau',      5,      30,     15,     2.0, rules=[
            AdaptRule('acc1_plateau', 'theta_tau', 'shift', 2),
        ]),
        ParamDef('neg_samples',    0,      8,      2,      0.5, rules=[
            AdaptRule('vacc1_stuck >= 3', 'neg_samples', 'shift', 1),
            AdaptRule('vacc1 > 0.01', 'neg_samples', 'shift', -1),
            AdaptRule('cos_flat >= 5', 'neg_samples', 'shift', 1),
        ]),
        ParamDef('pmi_strength',   0.0,    1.0,    1.0,    0.02, rules=[
            AdaptRule('cos_trend > 0.001 and mean_cos > 0.01', 'pmi_strength', 'shift', -0.02),
            AdaptRule('cos_trend < -0.001 and mean_cos < 0.005', 'pmi_strength', 'shift', 0.02),
            AdaptRule('cos_flat >= 3', 'pmi_strength', 'shift', 0.05),
            AdaptRule('full_stuck', 'pmi_strength', 'shift', -0.05),
        ]),
        ParamDef('pmi_gate_min',   0.05,   0.5,    0.20,   0.02, rules=[
            AdaptRule('delta < 2.0', 'pmi_gate_min', 'shift', -0.01),
            AdaptRule('delta > 20.0', 'pmi_gate_min', 'shift', 0.01),
        ]),
        ParamDef('decay_rate',     0.998,  0.9999, 0.9998, 0.00005, rules=[
            AdaptRule('ng_new < 100', 'decay_rate', 'shift', -0.0001),
            AdaptRule('ng_new > 10000', 'decay_rate', 'shift', 0.00005),
        ]),
        ParamDef('destab_decay_lines', 5000, 60000, 30000, 2000, rules=[
            AdaptRule('cos_flat >= 3', 'destab_decay_lines', 'shift', 2000),
            AdaptRule('full_stuck', 'destab_decay_lines', 'shift', -2000),
        ]),
        ParamDef('field_gate_threshold', 0.0, 1.0, 1.0, 0.05, rules=[
            AdaptRule('cos_flat >= 3', 'field_gate_threshold', 'shift', -0.05),
            AdaptRule('mean_cos > 0.01', 'field_gate_threshold', 'shift', 0.05),
        ]),
    ])

    # ── Метрические пары (заполняются из MorphVocab/корпуса) ──
    live_pairs: list = field(default_factory=list)  # list[MetricPair]
    eval_pairs: list = field(default_factory=list)  # полный список для финальной диагностики

    # ── Seeds для тестовой генерации ──────────
    test_seeds: list = field(default_factory=lambda: [
        'князь', 'человек', 'война', 'любовь', 'дом', 'жизнь'
    ])

    # ── Расписания ────────────────────────────
    lr_warmup_lines: int = 1000
    lr_cosine_T0: int = 5000
    lr_cosine_mult: float = 1.5
    checkpoint_every: int = 5000
    eval_every_fast: int = 1000
    eval_every_slow: int = 2000
    eval_fast_lines: int = 64      # TN-9: lines for fast eval (PPL only)
    eval_full_lines: int = 300     # TN-9: lines for full eval (all metrics)
    eval_every_full: int = 5000   # TN-12: full eval interval (fast eval at eval_every_fast)
    batch_size_start: int = 8      # TN-5: initial batch size (warmup)
    batch_size_end: int = 32       # TN-5: final batch size
    fluctuate_every: int = 2000
    decay_every_fast: int = 2000
    decay_every_slow: int = 3000
    decay_every_pairs: int = 32000
    decay_warmup_lines: int = 5000   # TN-15: ramp decay_rate from 0.998 to target

    # ── Гиперы FAST-режима ────────────────────
    fast_lr: float = 0.15
    fast_neg_samples: int = 3

    # ── Defaults для CrystalGenerator ─────────
    beam_width: int = 5
    max_words: int = 30
    min_words: int = 3
    concept_temp: float = 0.5

    # ── Defaults для build_octree_fields ──────
    octree_min_lcp: int = 2
    octree_gamma: float = 0.5

    # ── Val split ─────────────────────────────
    val_pct: float = 0.05

    # ── Destabilisation (PPMI noise) ──────────
    destab_scale_start: float = 0.6
    destab_scale_end: float = 0.02
    # NOTE: destab_decay_lines is a ParamDef in params list (opt.p['destab_decay_lines'])

    # ── Drift guard ───────────────────────────
    code_bound: float = 10.0
    vec_dev_warn: float = 0.01

    # ── Параметры max_shift для STDP ──────────
    # Default 0.5 matches hardcoded max_shift in concept_space._apply_vector_update
    stdp_max_shift: float = 0.5
    max_grad_norm: float = 1.0  # clip gradient norm to prevent explosive updates
    neg_lr_ratio: float = 0.5

    # ── Engine ────────────────────────────────
    use_torch: bool = True
    field_gate: bool = True
    momentum_mu: float = 0.9

    # ── Seeds для генерации на чекпоинтах ─────
    gen_max_words: int = 25
    eval_max_lines: int = 300

    # ── Manage ────────────────────────────────
    cleanup_keep: int = 5       # сколько чекпоинтов хранить
    periodic_save_every: int = 5000

    # ──────────────────────────────────────────
    #  Генерация пар из MorphVocab/корпуса
    # ──────────────────────────────────────────

    def build_metric_pairs(self, morph_vocab=None, lattice=None, sp=None):
        """Заполнить live_pairs и eval_pairs автоматически."""
        builder = MetricPairBuilder()
        live, eval_p = builder.build_defaults()

        if morph_vocab is not None:
            eval_p.extend(builder.build_antonym_pairs(morph_vocab, sp, 5))
            morph = builder.build_morph_pairs(morph_vocab, 3)
            live.extend(morph)

        self.live_pairs = live
        self.eval_pairs = eval_p
        return eval_p

    # ──────────────────────────────────────────
    #  Сериализация
    # ──────────────────────────────────────────

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.data_dir, 'fcf_config.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def to_dict(self):
        d = {}
        for k, v in self.__class__.__dataclass_fields__.items():
            if k.startswith('_'):
                continue
            val = getattr(self, k)
            if isinstance(val, (str, int, float, bool)):
                d[k] = val
            elif isinstance(val, list):
                items = []
                for item in val:
                    if isinstance(item, ParamDef):
                        d2 = {}
                        for k2, v2 in item.__dict__.items():
                            if isinstance(v2, list) and v2 and all(isinstance(r, AdaptRule) for r in v2):
                                d2[k2] = [r.__dict__ for r in v2]
                            else:
                                d2[k2] = v2
                        items.append(d2)
                    elif isinstance(item, MetricPair):
                        items.append(item.__dict__)
                    else:
                        items.append(item)
                d[k] = items
        return d

    @staticmethod
    def load(path=None):
        if path is None:
            path = os.path.join(_auto_base_dir(), 'real_data', 'fcf_config.json')
        if not os.path.exists(path):
            return FCFConfig()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = FCFConfig()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def __post_init__(self):
        # AM-8: Configuration Schema Validation
        assert self.dim > 0 and self.dim % 8 == 0, f"dim={self.dim} must be >0 and divisible by 8"
        assert 0.0 <= self.destab_scale_end <= self.destab_scale_start <= 1.0
        assert self.code_bound > 0
        name_set = set()
        for p in self.params:
            assert p.name not in name_set, f"Duplicate param name: {p.name}"
            name_set.add(p.name)
        # Конвертировать dict-правила обратно в объекты при загрузке из JSON
        for i, p in enumerate(self.params):
            if isinstance(p, dict):
                pd = p.copy()
                rules = pd.pop('rules', [])
                p_def = ParamDef(**pd)
                p_def.rules = [AdaptRule(**r) if isinstance(r, dict) else r for r in rules]
                self.params[i] = p_def
        for i, p in enumerate(self.live_pairs):
            if isinstance(p, dict):
                self.live_pairs[i] = MetricPair(**p)
        for i, p in enumerate(self.eval_pairs):
            if isinstance(p, dict):
                self.eval_pairs[i] = MetricPair(**p)
