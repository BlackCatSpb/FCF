"""Watcher: auto-generates PCA viz for each new checkpoint."""

import argparse
import glob
import os
import time
import subprocess
import sys


def get_checkpoints(data_dir):
    """Return set of (prefix, mtime) for completed checkpoints."""
    ckpts = set()
    for f in glob.glob(os.path.join(data_dir, "concept_space_*.codes.npz")):
        base = f.replace(".codes.npz", "")
        json_f = base + ".json"
        if os.path.exists(json_f):
            ckpts.add((base, os.path.getmtime(f)))
    return ckpts


def html_path(prefix):
    return prefix + "_viz.html"


def png_path(prefix):
    return prefix + "_viz.png"


def main():
    ap = argparse.ArgumentParser(
        description="Watch FCF checkpoints and auto-visualize")
    ap.add_argument("--data-dir", default=r"real_data",
                    help="Directory with concept_space_*.codes.npz")
    ap.add_argument("--sp-model", default=r"real_data\bpe_ru_146k.model",
                    help="SentencePiece model path")
    ap.add_argument("--interval", type=int, default=60,
                    help="Poll interval in seconds (default 60)")
    ap.add_argument("--n-tokens", type=int, default=5000,
                    help="Tokens to plot (default 5000)")
    ap.add_argument("--labels", type=int, default=30,
                    help="Label annotations (default 30)")
    args = ap.parse_args()

    # Normalise path
    data_dir = os.path.abspath(args.data_dir)
    sp_model = os.path.abspath(args.sp_model) if not os.path.isabs(args.sp_model) else args.sp_model

    print(f"[Watcher] Watching {data_dir} every {args.interval}s ...")
    print(f"[Watcher] SP model: {sp_model}")

    done = set()
    # Skip already-visualised
    for prefix_mtime in get_checkpoints(data_dir):
        prefix = prefix_mtime[0]
        if os.path.exists(html_path(prefix)) and os.path.exists(png_path(prefix)):
            done.add(prefix)

    if done:
        print(f"[Watcher] Already have {len(done)} checkpoints")

    while True:
        now = get_checkpoints(data_dir)
        for prefix, mtime in now:
            if prefix in done:
                continue
            if os.path.exists(html_path(prefix)) and os.path.exists(png_path(prefix)):
                done.add(prefix)
                continue

            print(f"[Watcher] New checkpoint: {os.path.basename(prefix)}")
            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "visualize.py"),
                prefix,
                "--n-tokens", str(args.n_tokens),
                "--labels", str(args.labels),
                "--sp-model", sp_model,
                "--html",
                "--output", png_path(prefix),
            ]
            try:
                subprocess.run(cmd, check=True, timeout=300)
                done.add(prefix)
                print(f"[Watcher] Done: {os.path.basename(prefix)}_viz.html")
            except subprocess.TimeoutExpired:
                print(f"[Watcher] Timeout on {os.path.basename(prefix)}")
            except subprocess.CalledProcessError as e:
                print(f"[Watcher] Error on {os.path.basename(prefix)}: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
