"""
EnvironmentAutoTuner — автонастройка среды исполнения.

Аналог UES (Universal Execution Subsystem) из EVA-Ai, адаптированный под PyTorch.

Автоматически определяет:
- Оптимальное устройство (CPU/CUDA)
- Число потоков CPU
- Размер батча по доступной памяти
- Точность вычислений (float32/mixed)
- Расписание фонового обучения
"""

import os
import sys
import torch
import threading
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class HardwareProfile:
    device: str = "cpu"
    device_name: str = "CPU"
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    cpu_cores_physical: int = 4
    cpu_cores_logical: int = 8
    recommended_threads: int = 4
    recommended_batch_size: int = 1
    mixed_precision: bool = False
    cuda_available: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class RuntimeStats:
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_free_gb: float = 0.0
    vram_used_gb: float = 0.0
    vram_free_gb: float = 0.0
    active_queries: int = 0
    training_active: bool = False
    timestamp: float = field(default_factory=time.time)


class EnvironmentAutoTuner:

    def __init__(self):
        self.profile: Optional[HardwareProfile] = None
        self.stats: RuntimeStats = RuntimeStats()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def discover(self) -> HardwareProfile:
        cuda_available = torch.cuda.is_available()
        device_name = (
            torch.cuda.get_device_name(0) if cuda_available else "CPU"
        )
        vram_gb = 0.0
        if cuda_available:
            vram_gb = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )

        ram_gb = 0.0
        cpu_phys = 4
        cpu_log = 8

        if HAS_PSUTIL:
            ram_gb = psutil.virtual_memory().total / 1e9
            cpu_phys = psutil.cpu_count(logical=False) or 4
            cpu_log = psutil.cpu_count(logical=True) or 8
        else:
            cpu_phys = os.cpu_count() or 4
            cpu_log = cpu_phys

        recommended_threads = min(cpu_phys - 2, 16)  # Оставляем 2 ядра системе
        if recommended_threads < 2:
            recommended_threads = 2

        recommended_batch_size = 1
        if cuda_available:
            if vram_gb >= 16:
                recommended_batch_size = 8
            elif vram_gb >= 8:
                recommended_batch_size = 4
            elif vram_gb >= 4:
                recommended_batch_size = 2
            else:
                recommended_batch_size = 1

        mixed_precision = cuda_available and vram_gb >= 4

        self.profile = HardwareProfile(
            device="cuda" if cuda_available else "cpu",
            device_name=device_name,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            cpu_cores_physical=cpu_phys,
            cpu_cores_logical=cpu_log,
            recommended_threads=recommended_threads,
            recommended_batch_size=recommended_batch_size,
            mixed_precision=mixed_precision,
            cuda_available=cuda_available,
        )

        logger.info(
            f"[AutoTune] Профиль: {self.profile.device_name} "
            f"({self.profile.device}), "
            f"VRAM={self.profile.vram_gb:.1f}GB, "
            f"RAM={self.profile.ram_gb:.1f}GB, "
            f"cores={self.profile.cpu_cores_physical}/{self.profile.cpu_cores_logical}, "
            f"batch={self.profile.recommended_batch_size}, "
            f"threads={self.profile.recommended_threads}"
        )

        if cuda_available and vram_gb < 4.0:
            logger.info(
                f"[AutoTune] GPU {device_name}: VRAM={vram_gb:.1f}GB < 4GB. "
                f"Обучение только на CPU. GPU будет использован для инференса."
            )

        return self.profile

    def apply(self) -> HardwareProfile:
        if self.profile is None:
            self.discover()

        if self.profile is None:
            return HardwareProfile()

        torch.set_num_threads(self.profile.recommended_threads)

        if self.profile.cuda_available:
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

        logger.info(
            f"[AutoTune] Применено: threads={self.profile.recommended_threads}, "
            f"device={self.profile.device}"
        )

        return self.profile

    def get_optimal_device(self) -> str:
        if self.profile is None:
            self.discover()
        if self.profile:
            return self.profile.device
        return "cpu"

    def get_training_device(self) -> str:
        """Возвращает устройство для обучения. GPU только если VRAM >= 4GB."""
        if self.profile is None:
            self.discover()
        if self.profile and self.profile.vram_gb >= 4.0:
            return "cuda"
        return "cpu"

    def get_inference_device(self) -> str:
        """Возвращает устройство для инференса. GPU можно даже с 2GB."""
        if self.profile is None:
            self.discover()
        if self.profile and self.profile.cuda_available:
            return "cuda"
        return "cpu"

    def can_use_gpu_for_training(self) -> bool:
        if self.profile is None:
            self.discover()
        if self.profile:
            return self.profile.vram_gb >= 4.0
        return False

    def get_optimal_batch_size(self, model_params: int) -> int:
        if self.profile is None:
            self.discover()
        if self.profile is None:
            return 1

        if self.profile.device == "cpu":
            ram_free = self._get_ram_free_gb()
            per_sample_mb = (model_params * 4 * 4) / 1e6
            max_by_ram = max(1, int(ram_free * 0.3 * 1024 / per_sample_mb))
            return min(max_by_ram, 4)
        else:
            vram_free = self._get_vram_free_gb()
            per_sample_mb = (model_params * 4 * 3) / 1e6
            max_by_vram = max(1, int(vram_free * 0.5 * 1024 / per_sample_mb))
            return min(max_by_vram, self.profile.recommended_batch_size)

    def can_train(self) -> bool:
        stats = self.get_runtime_stats()

        if stats.training_active:
            return False

        if stats.active_queries > 0:
            return False

        if stats.cpu_percent > 80:
            return False

        if self.profile and self.profile.device == "cuda":
            if stats.vram_free_gb < 0.5:
                return False

        return True

    def get_runtime_stats(self) -> RuntimeStats:
        stats = RuntimeStats()

        try:
            if HAS_PSUTIL:
                stats.cpu_percent = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                stats.ram_used_gb = mem.used / 1e9
                stats.ram_free_gb = mem.available / 1e9
        except Exception:
            pass

        try:
            if torch.cuda.is_available():
                stats.vram_used_gb = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                stats.vram_free_gb = total - stats.vram_used_gb
        except Exception:
            pass

        return stats

    def start_monitoring(self, interval: float = 30.0):
        if self._running:
            return

        self._running = True

        def _monitor():
            while self._running:
                try:
                    self.stats = self.get_runtime_stats()
                    self._balance_threads()
                    time.sleep(interval)
                except Exception:
                    time.sleep(interval)

        self._monitor_thread = threading.Thread(target=_monitor, daemon=True)
        self._monitor_thread.start()
        logger.info(f"[AutoTune] Мониторинг запущен (интервал={interval}с)")

    def _balance_threads(self):
        """Периодически меняет affinity потоков для равномерной нагрузки на ядра."""
        try:
            if self.profile is None:
                return
            n_phys = self.profile.cpu_cores_physical
            n_threads = torch.get_num_threads()
            if n_phys <= 2 or n_threads <= 2:
                return
            import random
            new_threads = n_threads + random.choice([-1, 0, 1])
            new_threads = max(2, min(n_phys, new_threads))
            if new_threads != n_threads:
                torch.set_num_threads(new_threads)
        except Exception:
            pass

    def stop_monitoring(self):
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("[AutoTune] Мониторинг остановлен")

    def get_training_config(self) -> Dict[str, Any]:
        if self.profile is None:
            self.discover()

        training_device = self.get_training_device()

        config = {
            "device": training_device,
            "threads": 4,
            "batch_size": 1,
            "mixed_precision": False,
            "gradient_accumulation_steps": 4,
        }

        if self.profile:
            config["threads"] = self.profile.recommended_threads
            config["batch_size"] = self.profile.recommended_batch_size
            config["mixed_precision"] = (
                self.profile.mixed_precision and training_device == "cuda"
            )

            if config["batch_size"] == 1:
                config["gradient_accumulation_steps"] = 4
            elif config["batch_size"] == 2:
                config["gradient_accumulation_steps"] = 2
            else:
                config["gradient_accumulation_steps"] = 1

        return config

    def _get_ram_free_gb(self) -> float:
        try:
            if HAS_PSUTIL:
                return psutil.virtual_memory().available / 1e9
        except Exception:
            pass
        return 4.0

    def _get_vram_free_gb(self) -> float:
        try:
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                used = torch.cuda.memory_allocated() / 1e9
                return total - used
        except Exception:
            pass
        return 0.0

    def summary(self) -> str:
        if self.profile is None:
            self.discover()
        if self.profile is None:
            return "EnvironmentAutoTuner: не инициализирован"

        lines = [
            f"EnvironmentAutoTuner:",
            f"  Device: {self.profile.device_name} ({self.profile.device})",
            f"  VRAM: {self.profile.vram_gb:.1f} GB",
            f"  RAM: {self.profile.ram_gb:.1f} GB",
            f"  CPU cores: {self.profile.cpu_cores_physical} phys / {self.profile.cpu_cores_logical} log",
            f"  Threads: {self.profile.recommended_threads} (оставлено 2 ядра системе)",
            f"  Batch: {self.profile.recommended_batch_size}",
            f"  Mixed precision: {self.profile.mixed_precision}",
        ]

        if self.profile.cuda_available:
            if self.profile.vram_gb >= 4.0:
                lines.append(f"  GPU training: ДА (VRAM={self.profile.vram_gb:.1f}GB >= 4GB)")
            else:
                lines.append(f"  GPU training: НЕТ (VRAM={self.profile.vram_gb:.1f}GB < 4GB нужно)")
            lines.append(f"  GPU inference: ДА")
        else:
            lines.append(f"  GPU: отсутствует")

        stats = self.get_runtime_stats()
        lines.append(
            f"  Runtime CPU: {stats.cpu_percent:.0f}%, "
            f"RAM free: {stats.ram_free_gb:.1f}GB"
        )

        return "\n".join(lines)
