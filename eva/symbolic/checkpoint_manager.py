"""CheckpointManager — threaded async checkpoint save/cleanup."""

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor


class CheckpointManager:
    """Threaded async checkpoint manager.

    Saves concept space + lattice + optimizer state to temporary files
    and atomically renames them in a background thread. The main training
    loop is not blocked by disk I/O.

    Args:
        data_dir: directory for temporary files during save
        cleanup_keep: number of checkpoints to keep (oldest removed)
        max_workers: thread pool size (default 1)
    """

    def __init__(self, data_dir='.', cleanup_keep=5, max_workers=1):
        self.data_dir = data_dir
        self.cleanup_keep = max(cleanup_keep, 1)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []
        self._saved_tags = []

    def save(self, tag, cs, lattice, opt=None, extras=None):
        """Submit async save. Returns immediately.

        Args:
            tag: checkpoint tag (e.g. 'e1_l5000')
            cs: ConceptSpace instance (must have .save method)
            lattice: SyntaxLattice instance (must have .save method)
            opt: optional ParameterOptimizer (must have save_state)
            extras: dict of {path: data_callable} for extra files (e.g. 3D vis)
        """
        future = self._executor.submit(
            self._sync_save, tag, cs, lattice, opt, extras or {})
        self._futures.append(future)
        self._saved_tags.append(tag)

    def wait(self):
        """Wait for all pending saves to complete."""
        for f in self._futures:
            f.result()
        self._futures.clear()

    def cleanup(self, keep=None):
        """Remove old checkpoints, keeping the `keep` most recent."""
        keep = keep if keep is not None else self.cleanup_keep
        self.wait()
        if len(self._saved_tags) <= keep:
            return
        for tag in self._saved_tags[:-keep]:
            self._remove_tag(tag)
        self._saved_tags = self._saved_tags[-keep:]

    def shutdown(self):
        """Shut down the thread pool."""
        self.wait()
        self._executor.shutdown(wait=True)

    def _sync_save(self, tag, cs, lattice, opt, extras):
        """Synchronous save (runs in thread pool)."""
        cs_path = os.path.join(self.data_dir, f'cs_{tag}.npz')
        lat_path = os.path.join(self.data_dir, f'lat_{tag}.npz')
        tmp_cs = cs_path + '.tmp'
        tmp_lat = lat_path + '.tmp'
        try:
            cs.save(tmp_cs)
            lattice.save(tmp_lat)
            os.replace(tmp_cs, cs_path)
            os.replace(tmp_lat, lat_path)
        except Exception as e:
            print(f"[CheckpointManager] save({tag}) failed: {e}")
            for p in [tmp_cs, tmp_lat]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
            return
        if opt is not None:
            opt_path = os.path.join(self.data_dir, f'opt_{tag}.json')
            tmp_opt = opt_path + '.tmp'
            try:
                opt.save_state(tmp_opt)
                os.replace(tmp_opt, opt_path)
            except Exception as e:
                print(f"[CheckpointManager] opt save({tag}) failed: {e}")
                try:
                    if os.path.exists(tmp_opt):
                        os.remove(tmp_opt)
                except OSError:
                    pass
        for suffix, data_callable in extras.items():
            path = os.path.join(self.data_dir, f'{tag}_{suffix}')
            try:
                data_callable(path)
            except Exception as e:
                print(f"[CheckpointManager] extras({suffix}) failed: {e}")

    def _remove_tag(self, tag):
        """Remove all files associated with a tag."""
        for prefix in ['cs_', 'lat_', 'opt_']:
            path = os.path.join(self.data_dir, f'{prefix}{tag}.npz')
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            json_path = os.path.join(self.data_dir, f'{prefix}{tag}.json')
            if os.path.exists(json_path):
                try:
                    os.remove(json_path)
                except OSError:
                    pass
