"""
FCFConfig — единая конфигурация системы.

Загружается из config.json, все параметры доступны как атрибуты.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SRGConfig:
    w_sim: float = 0.4
    w_ent: float = 0.3
    w_eth: float = 0.3
    ethics_threshold: float = 0.3
    snapshot_confidence_threshold: float = 0.8
    curiosity_confidence_threshold: float = 0.6


@dataclass
class GrowthConfig:
    width_threshold: float = 0.5
    depth_threshold: float = 0.3
    gradient_threshold: float = 1.0
    patience: int = 20


@dataclass
class CuriosityConfig:
    threshold: int = 10
    max_clarification_tokens: int = 64


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    lora_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_steps: Optional[int] = None


@dataclass
class SleepConfig:
    idle_timeout_seconds: int = 300
    sleep_interval_seconds: int = 7200
    min_cluster_size: int = 5
    ttl_idle_days: int = 7
    distill_usage_threshold: int = 100


@dataclass
class KCAConfig:
    max_iterations: int = 5
    rho: float = 0.85
    eta0: float = 0.01
    osc_threshold: float = -0.5
    gate_threshold: float = 0.05
    lambda_gap: float = 0.3
    lambda_contra: float = 0.2


@dataclass
class LayersConfig:
    max_layers: int = 100
    min_layer_interval_seconds: int = 300
    max_recursion_depth: int = 5


@dataclass
class FCFConfig:
    d_model: int = 2560
    num_heads: int = 32
    ff_mult: int = 4
    max_snapshots: int = 10000
    vocab_size: int = 50257
    max_seq_len: int = 2048

    srg: SRGConfig = field(default_factory=SRGConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    kca: KCAConfig = field(default_factory=KCAConfig)
    layers: LayersConfig = field(default_factory=LayersConfig)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @classmethod
    def from_json(cls, path: str) -> "FCFConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            d_model=data.get("d_model", 2560),
            num_heads=data.get("num_heads", 32),
            ff_mult=data.get("ff_mult", 4),
            max_snapshots=data.get("max_snapshots", 10000),
            vocab_size=data.get("vocab_size", 50257),
            max_seq_len=data.get("max_seq_len", 2048),
            srg=SRGConfig(**data.get("srg", {})),
            growth=GrowthConfig(**data.get("growth", {})),
            curiosity=CuriosityConfig(**data.get("curiosity", {})),
            training=TrainingConfig(**data.get("training", {})),
            sleep=SleepConfig(**data.get("sleep", {})),
            kca=KCAConfig(**data.get("kca", {})),
            layers=LayersConfig(**data.get("layers", {})),
        )

    def to_json(self, path: str) -> None:
        data = {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_mult": self.ff_mult,
            "max_snapshots": self.max_snapshots,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "srg": self.srg.__dict__,
            "growth": self.growth.__dict__,
            "curiosity": self.curiosity.__dict__,
            "training": self.training.__dict__,
            "sleep": self.sleep.__dict__,
            "kca": self.kca.__dict__,
            "layers": self.layers.__dict__,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_config(path: str = None) -> FCFConfig:
    if path and os.path.exists(path):
        return FCFConfig.from_json(path)
    default_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    if os.path.exists(default_path):
        return FCFConfig.from_json(default_path)
    return FCFConfig()
