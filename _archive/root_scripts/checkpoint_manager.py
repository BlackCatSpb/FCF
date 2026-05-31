"""
EVA Checkpoint Manager — production-grade checkpoint management.

Features:
- Keep last N checkpoints
- Keep best by validation SRG
- Auto-prune old (< best SRG - threshold, age > max_days)
- Atomic saves (write to temp, then rename)
"""
import os, json, time, glob, shutil
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class CheckpointEntry:
    path: str
    step: int
    srg: float = 0.0
    timestamp: float = 0.0
    filesize_mb: float = 0.0

    def to_dict(self):
        return {
            'path': self.path,
            'step': self.step,
            'srg': self.srg,
            'timestamp': self.timestamp,
            'filesize_mb': self.filesize_mb,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, keep_last: int = 5,
                 keep_best: int = 3, max_age_days: int = 14,
                 prune_threshold: float = 0.2):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.max_age_days = max_age_days
        self.prune_threshold = prune_threshold
        self.entries: List[CheckpointEntry] = []
        self._load_index()

    def _index_path(self):
        return os.path.join(self.checkpoint_dir, '.checkpoint_index.json')

    def _load_index(self):
        path = self._index_path()
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self.entries = [CheckpointEntry.from_dict(e) for e in data.get('entries', [])]
            except Exception:
                self.entries = []

    def _save_index(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = self._index_path()
        tmp_path = path + '.tmp'
        try:
            with open(tmp_path, 'w') as f:
                json.dump({'entries': [e.to_dict() for e in self.entries]}, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            pass

    def scan(self):
        """Scan checkpoint directory for .pt files."""
        pattern = os.path.join(self.checkpoint_dir, '*.pt')
        files = glob.glob(pattern)
        for fp in files:
            step = self._extract_step(fp)
            if step is None:
                continue
            if not any(e.path == fp for e in self.entries):
                sz = os.path.getsize(fp) / 1024 / 1024
                self.entries.append(CheckpointEntry(
                    path=fp, step=step,
                    timestamp=os.path.getmtime(fp),
                    filesize_mb=round(sz, 1),
                ))
        self.entries.sort(key=lambda e: e.step, reverse=True)
        self._save_index()

    def _extract_step(self, path: str) -> Optional[int]:
        """Extract step number from filename like train_v3_step_13000.pt"""
        import re
        m = re.search(r'step_(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else None

    def register(self, path: str, step: int, srg: float = 0.0):
        """Register a new checkpoint."""
        sz = os.path.getsize(path) / 1024 / 1024 if os.path.exists(path) else 0
        entry = CheckpointEntry(
            path=path, step=step, srg=srg,
            timestamp=time.time(), filesize_mb=round(sz, 1),
        )
        self.entries.append(entry)
        self._save_index()

    def prune(self) -> List[str]:
        """
        Remove old checkpoints according to policy.
        Returns list of deleted paths.
        """
        if len(self.entries) <= self.keep_last + self.keep_best:
            return []

        now = time.time()
        deleted = []

        # Find best SRG
        best_srg = max((e.srg for e in self.entries if e.srg > 0), default=0)

        # Identify entries to keep:
        # 1. Last N by step
        sorted_by_step = sorted(self.entries, key=lambda e: e.step, reverse=True)
        keep_steps = set(e.step for e in sorted_by_step[:self.keep_last])

        # 2. Best by SRG
        sorted_by_srg = sorted([e for e in self.entries if e.srg > 0],
                               key=lambda e: e.srg, reverse=True)
        keep_srg = set(e.step for e in sorted_by_srg[:self.keep_best])

        keep = keep_steps | keep_srg

        for entry in list(self.entries):
            # Keep if it's in the protected set
            if entry.step in keep:
                continue

            # Prune if old AND below threshold
            age_days = (now - entry.timestamp) / 86400
            if age_days > self.max_age_days and (best_srg - entry.srg) > self.prune_threshold:
                try:
                    if os.path.exists(entry.path):
                        os.remove(entry.path)
                        deleted.append(entry.path)
                except Exception:
                    pass
                self.entries.remove(entry)

        # Always keep at least keep_last + keep_best entries
        while len(self.entries) > self.keep_last + self.keep_best + 10:
            oldest = min(self.entries, key=lambda e: e.timestamp)
            try:
                if os.path.exists(oldest.path):
                    os.remove(oldest.path)
                    deleted.append(oldest.path)
            except Exception:
                pass
            self.entries.remove(oldest)

        self._save_index()
        return deleted

    def latest(self) -> Optional[CheckpointEntry]:
        """Get the latest checkpoint by step."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.step)

    def best(self) -> Optional[CheckpointEntry]:
        """Get the best checkpoint by SRG."""
        valid = [e for e in self.entries if e.srg > 0]
        if not valid:
            return None
        return max(valid, key=lambda e: e.srg)

    def summary(self) -> str:
        lines = [f"CheckpointManager: {len(self.entries)} entries"]
        for e in sorted(self.entries, key=lambda e: e.step, reverse=True)[:5]:
            srg_str = f" SRG={e.srg:.3f}" if e.srg > 0 else ""
            lines.append(f"  step {e.step:>6d}: {e.filesize_mb:.1f}MB{srg_str}")
        return "\n".join(lines)
