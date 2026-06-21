"""CheckpointManager — fully async checkpoint save/cleanup."""

import os
import json
from concurrent.futures import ThreadPoolExecutor


class CheckpointManager:
    """Threaded async checkpoint manager.

    Saves concept space + lattice + optimizer state in a background thread.
    The main training loop is not blocked by disk I/O.

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

    def save(self, tag, cs, lattice, opt=None, extras=None, ckpt_state=None):
        """Submit async save. Returns immediately.

        Args:
            tag: checkpoint tag (e.g. 'e1_l5000')
            cs: ConceptSpace instance (must have .save method)
            lattice: SyntaxLattice instance (must have .save method)
            opt: optional ParameterOptimizer (must have save_state)
            extras: dict of {path: data_callable} for extra files
            ckpt_state: optional dict saved as checkpoint_state.json after save
        """
        future = self._executor.submit(
            self._sync_save, tag, cs, lattice, opt, extras or {}, ckpt_state)
        self._futures.append(future)
        self._saved_tags.append(tag)

    def wait(self):
        """Wait for all pending saves to complete."""
        for f in self._futures:
            f.result()
        self._futures.clear()

    def shutdown(self):
        """Shut down the thread pool."""
        self.wait()
        self._executor.shutdown(wait=True)

    def _sync_save(self, tag, cs, lattice, opt, extras, ckpt_state):
        """Synchronous save (runs in thread pool)."""
        cs_path = os.path.join(self.data_dir, f'concept_space_{tag}.json')
        lat_path = os.path.join(self.data_dir, f'syntax_lattice_{tag}.json')
        tmp_cs = cs_path + '.tmp'
        tmp_lat = lat_path + '.tmp'
        try:
            cs.save(tmp_cs)
            lattice.save(tmp_lat)
            os.replace(tmp_cs, cs_path)
            os.replace(tmp_lat, lat_path)
        except Exception as e:
            print(f"[CheckpointManager] save({tag}) failed: {e}", file=__import__('sys').stderr)
            for p in [tmp_cs, tmp_lat]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
            return
        if opt is not None:
            opt_path = os.path.join(self.data_dir, f'concept_space_{tag}.opt.json')
            tmp_opt = opt_path + '.tmp'
            try:
                state = opt.save_state()
                with open(tmp_opt, 'w', encoding='utf-8') as f:
                    json.dump(state, f)
                os.replace(tmp_opt, opt_path)
            except Exception as e:
                print(f"[CheckpointManager] opt save({tag}) failed: {e}", file=__import__('sys').stderr)
                try:
                    if os.path.exists(tmp_opt):
                        os.remove(tmp_opt)
                except OSError:
                    pass
        # Save checkpoint_state AFTER successful rename (atomic consistency)
        if ckpt_state is not None:
            ckpt_state_path = ckpt_state.get('_path')
            if ckpt_state_path:
                ckpt_data = {k: v for k, v in ckpt_state.items() if k != '_path'}
                try:
                    tmp_state = ckpt_state_path + '.tmp'
                    with open(tmp_state, 'w', encoding='utf-8') as f:
                        json.dump(ckpt_data, f)
                    os.replace(tmp_state, ckpt_state_path)
                except Exception as e:
                    print(f"[CheckpointManager] state save failed: {e}", file=__import__('sys').stderr)
        # Cleanup old checkpoints (also async)
        self._cleanup_old()
        for suffix, data_callable in extras.items():
            path = os.path.join(self.data_dir, f'{tag}_{suffix}')
            try:
                data_callable(path)
            except Exception as e:
                print(f"[CheckpointManager] extras({suffix}) failed: {e}", file=__import__('sys').stderr)

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
                path = os.path.join(self.data_dir, f'{prefix}{tag}{ext}')
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
