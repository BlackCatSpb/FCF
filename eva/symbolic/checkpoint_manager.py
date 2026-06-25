"""CheckpointManager — atomic write-ahead checkpointing with crash recovery."""

import os
import json
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Raised when checkpoint save fails.
    Carries tag and original exception."""


class AtomicCheckpointManager:
    """Threaded async checkpoint manager with write-ahead tmp + atomic rename.

    - Every save() writes to .tmp files first, then atomic rename.
    - Stale .tmp files are recovered on construction.
    - Errors are logged and accumulated; wait() re-raises on request.

    Args:
        data_dir: directory for checkpoint files
        cleanup_keep: number of checkpoints to keep (oldest removed)
        max_workers: thread pool size (default 1)
    """

    WAL_FILE = '_checkpoint_wal.json'

    def __init__(self, data_dir='.', cleanup_keep=5, max_workers=1):
        self.data_dir = Path(data_dir)
        self.cleanup_keep = max(cleanup_keep, 1)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list = []
        self._saved_tags: list = []
        self._errors: list = []
        self._lock = threading.Lock()
        self._recover_tmp_files()

    def _recover_tmp_files(self):
        """Recover stale .tmp files from a previous crash."""
        for tmp in self.data_dir.glob('*.tmp'):
            final = tmp.with_suffix('')
            if final.exists():
                tmp.unlink(missing_ok=True)
                logger.warning("Recovered: removed stale %s", tmp.name)
            else:
                try:
                    tmp.rename(final)
                    logger.info("Recovered: renamed %s \u2192 %s", tmp.name, final.name)
                except OSError:
                    tmp.unlink(missing_ok=True)

    def save(self, tag, cs, lattice, opt=None, extras=None, ckpt_state=None):
        """Submit async save. Returns immediately.

        Args:
            tag: checkpoint tag (e.g. 'e1_l5000')
            cs: ConceptSpace instance (must have .save method)
            lattice: SyntaxLattice instance (must have .save method)
            opt: optional ParameterOptimizer (must have save_state)
            extras: dict of {suffix: callable(path)} for extra files
            ckpt_state: optional dict saved as checkpoint_state.json after save
        """
        future = self._executor.submit(
            self._sync_save, tag, cs, lattice, opt, extras or {}, ckpt_state)
        with self._lock:
            self._futures.append(future)
            self._saved_tags.append(tag)

    def wait(self, raise_on_error=True):
        """Wait for all pending saves to complete.

        Args:
            raise_on_error: if True, raises CheckpointError on first failure.
                            if False, errors are logged and accumulated.
        """
        errors = []
        with self._lock:
            futures = self._futures[:]
            self._futures.clear()
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                errors.append(e)
                logger.error("Checkpoint failed: %s", e)
        if errors and raise_on_error:
            raise CheckpointError(f"{len(errors)} checkpoint(s) failed") from errors[0]

    def shutdown(self):
        """Shut down the thread pool after flushing pending saves."""
        self.wait(raise_on_error=False)
        self._executor.shutdown(wait=True)

    def _sync_save(self, tag, cs, lattice, opt, extras, ckpt_state):
        """Synchronous save (runs in thread pool)."""
        try:
            self._do_save(tag, cs, lattice, opt, extras, ckpt_state)
        except Exception as e:
            logger.error("Checkpoint %s: %s", tag, e)
            raise

    def _do_save(self, tag, cs, lattice, opt, extras, ckpt_state):
        cs_path = self.data_dir / f'concept_space_{tag}.json'
        lat_path = self.data_dir / f'syntax_lattice_{tag}.json'
        tmp_cs = cs_path.with_suffix('.json.tmp')
        tmp_lat = lat_path.with_suffix('.json.tmp')

        cs.save(str(tmp_cs))
        lattice.save(str(tmp_lat))
        os.replace(str(tmp_cs), str(cs_path))
        os.replace(str(tmp_lat), str(lat_path))

        if opt is not None:
            opt_path = self.data_dir / f'concept_space_{tag}.opt.json'
            tmp_opt = opt_path.with_suffix('.opt.json.tmp')
            state = opt.save_state()
            with open(tmp_opt, 'w', encoding='utf-8') as f:
                json.dump(state, f)
            os.replace(str(tmp_opt), str(opt_path))

        if ckpt_state is not None:
            ckpt_state_path = ckpt_state.get('_path')
            if ckpt_state_path:
                ckpt_data = {k: v for k, v in ckpt_state.items() if k != '_path'}
                state_path = Path(ckpt_state_path)
                tmp_state = state_path.with_suffix('.json.tmp')
                with open(tmp_state, 'w', encoding='utf-8') as f:
                    json.dump(ckpt_data, f)
                os.replace(str(tmp_state), str(state_path))

        for suffix, data_callable in extras.items():
            path = self.data_dir / f'{tag}_{suffix}'
            data_callable(str(path))

        with self._lock:
            self._cleanup_old()

    def _cleanup_old(self):
        """Remove old checkpoints, keeping the `cleanup_keep` most recent."""
        if len(self._saved_tags) <= self.cleanup_keep:
            return
        for tag in self._saved_tags[:-self.cleanup_keep]:
            self._remove_tag(tag)
        self._saved_tags = self._saved_tags[-self.cleanup_keep:]

    def _remove_tag(self, tag):
        """Remove all files associated with a tag."""
        for prefix in ['concept_space_', 'syntax_lattice_']:
            for ext in ['.json', '.npz', '.codes.npz', '.opt.json']:
                path = self.data_dir / f'{prefix}{tag}{ext}'
                if path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        pass


CheckpointManager = AtomicCheckpointManager
