"""
Minimal live monitor for Phase 2 training process.
"""
import os, sys, time, argparse, subprocess, re
from datetime import datetime

LOG_FILE = 'checkpoints/v4/phase2.log'
PID_FALLBACK = 8196

def get_gpu():
    try:
        out = subprocess.check_output(
            'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total '
            '--format=csv,noheader,nounits',
            shell=True, timeout=3).decode().strip().split(', ')
        return int(out[0]), int(out[1]), int(out[2])
    except Exception:
        return None, None, None

def get_proc():
    r = subprocess.check_output(
        ['powershell', f'Get-Process -Id {PID_FALLBACK} -ErrorAction SilentlyContinue | Format-List WorkingSet64'],
        timeout=3).decode()
    m = re.search(r'WorkingSet64\s*:\s*(\d+)', r)
    return int(m.group(1)) / 1024 / 1024 if m else 0

def read_log(path):
    if not os.path.exists(path):
        return {}
    mtime = os.path.getmtime(path)
    if time.time() - mtime > 60:
        return {'stale': True}
    with open(path) as f:
        lines = f.readlines()
    for line in reversed(lines):
        if line.startswith('[PHASE2'):
            p = {}
            for k in ['ce', 'nxt', 'bc', 'align', 'ac', 'dv', 'hf', 'acc', 'b_acc']:
                m = re.search(rf'{k}=([\d.]+)', line)
                if m: p[k] = float(m.group(1))
            m = re.search(r'hk=(\d+)', line); p['hk'] = int(m.group(1)) if m else 0
            m = re.search(r'hr=([\d.]+)', line); p['hr'] = float(m.group(1)) if m else 0
            m = re.search(r'att=(\d+)', line); p['att'] = int(m.group(1)) if m else 0
            m = re.search(r'haf_att=(\d+)', line); p['haf_att'] = int(m.group(1)) if m else 0
            m = re.search(r'\[PHASE2 (\d+)/(\d+)\]', line); p['step'] = int(m.group(1)); p['n_steps'] = int(m.group(2))
            return p
    return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=5)
    args = parser.parse_args()
    print(f'\nEVA Phase 2  (refresh {args.interval}s)')
    t0 = time.time()
    try:
        while True:
            gpu_util, gpu_mem, gpu_total = get_gpu()
            mem = get_proc()
            d = read_log(LOG_FILE)

            parts = []
            if gpu_util is not None:
                parts.append(f'GPU:{gpu_util}% VRAM:{gpu_mem}/{gpu_total}MB')
            if mem:
                parts.append(f'RAM:{mem:.0f}MB')
            if d and 'stale' not in d:
                s = d['step']; ns = d['n_steps']
                pct = s / ns * 100
                eta_s = (ns - s) / max(s / (time.time() - t0), 0.01)
                parts.append(f'step:{s}/{ns}({pct:.1f}%) ETA:{eta_s/3600:.1f}h')
                parts.append(f'CE:{d.get("ce",0):.3f} NXT:{d.get("nxt",0):.3f}')
                parts.append(f'BND:{d.get("bc",0):.3f} ACC:{d.get("acc",0):.3f}')
                parts.append(f'HAF:{d.get("hf",0):.4f} K:{d.get("hk",0)}')
                parts.append(f'att:{d.get("att",0)} H:{d.get("haf_att",0)}')
            else:
                est = int((time.time() - t0) / 3600 * 3600 * 2.1)
                parts.append(f'est:{est:,}/{200000}({est/2000:.1f}%)')

            sys.stdout.write('\r' + ' | '.join(parts) + '   ')
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
