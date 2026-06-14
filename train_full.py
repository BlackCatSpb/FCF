"""Full training (146K BPE). Process line-by-line. Checkpoints overwrite."""

import os
# Prevent numpy/BLAS thread contention on single-CPU workloads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import time, json, os, shutil, argparse, glob
import io
import numpy as np

def _quiet(fn, *args, **kwargs):
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn(*args, **kwargs)
    finally:
        sys.stdout = old

def _load_morph(path, sp_path):
    """Load or build MorphVocab (builds from corpus if no cached file)."""
    from eva.symbolic.morph_vocab import MorphVocab
    if os.path.exists(path):
        return MorphVocab.load(path, sp_model_path=sp_path)
    print("  MorphVocab not cached — building from corpus...")
    mv = MorphVocab.build(sp_model_path=sp_path)
    mv.save(path)
    return mv
import sentencepiece as spm
from sklearn.decomposition import PCA
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator
from eva.symbolic.parameter_optimizer import ParameterOptimizer
from eva.symbolic.fcf_config import FCFConfig, MetricPair

# ── Config ──────────────────────────────────────────────────────
CFG = FCFConfig()
os.makedirs(CFG.data_dir, exist_ok=True)
os.makedirs(CFG.vis_dir, exist_ok=True)

# Redirect stdout to UTF-8 log file (terminal cp1251 can't print ▁)
LOG_FILE = CFG.log_path
_log_fh = open(LOG_FILE, 'w', encoding='utf-8')

class TeeOut:
    def write(self, s):
        _log_fh.write(s)
        _log_fh.flush()
        try:
            sys.__stdout__.write(s)
        except UnicodeEncodeError:
            sys.__stdout__.write(s.encode('ascii', errors='replace').decode('ascii'))
        sys.__stdout__.flush()
    def flush(self):
        _log_fh.flush()
        sys.__stdout__.flush()

sys.stdout = TeeOut()

sp = spm.SentencePieceProcessor(model_file=CFG.bpe_model_path)
V = sp.vocab_size()

# ── Helpers ─────────────────────────────────────────────────────

def save_checkpoint_state(line_idx, epoch=1):
    with open(CFG.ckpt_state_path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump({'line': line_idx, 'epoch': epoch, 'timestamp': time.time()}, f)
    os.replace(CFG.ckpt_state_path + '.tmp', CFG.ckpt_state_path)

def load_checkpoint_state():
    if os.path.exists(CFG.ckpt_state_path):
        with open(CFG.ckpt_state_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('line'), data.get('epoch', 1)
    return None, None

def cleanup_old_checkpoints(keep=None):
    if keep is None:
        keep = CFG.cleanup_keep
    import re
    base_dir = CFG.data_dir
    files = []
    for p in glob.glob(os.path.join(base_dir, 'concept_space_*k.json')):
        m = re.search(r'_(\d+)k\.json$', os.path.basename(p))
        if m:
            files.append((p, os.path.getmtime(p), int(m.group(1))))
    files.sort(key=lambda x: -x[1])
    files = files[:keep]
    keep_ks = set(f[2] for f in files)
    for p in glob.glob(os.path.join(base_dir, 'concept_space_*k.json')):
        m = re.search(r'_(\d+)k\.json$', os.path.basename(p))
        if m and int(m.group(1)) not in keep_ks:
            k_label = m.group(0).replace('.json', '')
            for ext in ['.json', '.codes.npz', '.opt.json']:
                fp = os.path.join(base_dir, f'concept_space_{k_label}{ext}')
                if os.path.exists(fp): os.remove(fp)
            for ext in ['.json', '.lattice.npz', '.meta.json']:
                fp = os.path.join(base_dir, f'syntax_lattice_{k_label}{ext}')
                if os.path.exists(fp): os.remove(fp)
print(f"vocab_size = {V}")

# ── Parse args ──────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', type=int, default=1, help='number of epochs (default: 1). Auto-resume wraps corpus for epoch 2+')
parser.add_argument('--resume', '-r', nargs='?', const='', default='',
                    help='resume from checkpoint. Default: auto from checkpoint_state.json. With Nk: load numbered (e.g. 6k)')
parser.add_argument('--fast', '-f', action='store_true', help='fast mode: higher lr + negative sampling, always fresh')
parser.add_argument('--fresh', action='store_true', help='force fresh start even if checkpoint exists')
args = parser.parse_args()
RESUME = args.resume
FAST = args.fast
FRESH = args.fresh or FAST

if FRESH:
    RESUME = None

if FAST:
    print("FAST mode: base_lr=0.15, neg_samples=3, pmi_gate=off, decay_every=250, eval_every=1000")

if RESUME is not None:
    if RESUME == '':
        rl, r_epoch = load_checkpoint_state()
        if rl is None:
            print("No checkpoint found — starting fresh.")
            RESUME = None
        else:
            resume_line = rl
            resume_epoch = r_epoch
            ckpt_k = resume_line // 1000
            resume_tag = f"{ckpt_k}k"
            print(f"\nAuto-resuming from line {resume_line} (checkpoint_state.json) — loading {resume_tag}")
            cs_path = CFG.cs_path.replace('.json', f'_{resume_tag}.json')
            lat_path = CFG.lattice_path.replace('.json', f'_{resume_tag}.json')
            lat_ok = os.path.exists(lat_path) or os.path.exists(lat_path.replace('.json', '.lattice.npz'))
            if not os.path.exists(cs_path) or not lat_ok:
                print(f"Checkpoint {resume_tag} not found: {cs_path} / {lat_path}")
                cs_path = CFG.cs_path
                lat_path = CFG.lattice_path
                if not os.path.exists(cs_path) or not os.path.exists(lat_path.replace('.json', '.lattice.npz')):
                    print(f"Base checkpoint also not found. Giving up.")
                    sys.exit(1)
                resume_tag = 'base'
                print(f"  (fallback to base paths)")
    else:
        resume_tag = RESUME
        r = RESUME.lower().rstrip('k')
        try: resume_line = int(r) * 1000
        except ValueError:
            print(f"Invalid resume value: {RESUME}")
            sys.exit(1)
        resume_epoch = 1
        print(f"Resuming from '{RESUME}' (line {resume_line})")
        cs_path = CFG.cs_path.replace('.json', f'_{RESUME}.json')
        lat_path = CFG.lattice_path.replace('.json', f'_{RESUME}.json')
        if not os.path.exists(cs_path) or not (os.path.exists(lat_path) or os.path.exists(lat_path.replace('.json', '.lattice.npz'))):
            print(f"Checkpoint not found: {cs_path} / {lat_path}")
            sys.exit(1)

if RESUME is not None:
    cs = _quiet(ConceptSpace.load, cs_path)
    lattice = SyntaxLattice()
    load_ng = not FAST
    _quiet(lattice.load, lat_path, load_ngrams=load_ng)
    ng_info = f" ({[len(v) for v in lattice.ngrams.values()]} prefixes)" if load_ng else " (ngrams skipped)"
    print(f"  Loaded {len(cs.concept_vectors)} vectors, {len(lattice.concept_freq)} concepts{ng_info}")

    mv = _load_morph(CFG.morph_vocab_path, CFG.bpe_model_path)
    path_overrides = mv.get_path_overrides()
    if not hasattr(cs, 'H') or cs.H is None:
        print("  Rebuilding H matrix + octree fields from loaded lattice...")
        _quiet(cs.build_octree_fields, lattice, n_anchors=CFG.n_anchors, min_lcp=CFG.octree_min_lcp,
                                gamma=CFG.octree_gamma, path_overrides=path_overrides)

else:
    print(f"\nInitializing {V} concepts @ {CFG.dim}D...")
    cs = ConceptSpace(vocab_size=V, dim=CFG.dim)
    _quiet(cs.init_concepts)
    cs.init_homeostasis()

    mv = _load_morph(CFG.morph_vocab_path, CFG.bpe_model_path)
    path_overrides = mv.get_path_overrides()

    print("Building SyntaxLattice...")
    lattice = SyntaxLattice()
    _quiet(lattice.build, CFG.corpus_path, sp, max_n=CFG.max_n)

    print("Octree H + fields...")
    _quiet(cs.build_octree_fields, lattice, n_anchors=CFG.n_anchors, min_lcp=CFG.octree_min_lcp,
                            gamma=CFG.octree_gamma, path_overrides=path_overrides)

    # ── Diagnostics ────────────────────────────────────────────────
    # (functions defined unconditionally below; baseline run once)

def mean_cosine_sim(cs, sample=2000):
    all_cids = list(cs.concept_vectors.keys())
    rng_state = np.random.RandomState(42)
    cids = rng_state.choice(all_cids, size=min(sample, len(all_cids)), replace=False).tolist()
    vecs = np.array([cs.concept_vectors[c] for c in cids], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    vecs /= norms
    n = len(vecs)
    n_pairs = min(5000, n * (n - 1) // 2)
    rng = np.random.RandomState(42)
    sims = np.empty(n_pairs, dtype=np.float32)
    for k in range(n_pairs):
        i = rng.randint(n); j = rng.randint(n)
        while j == i: j = rng.randint(n)
        sims[k] = float(vecs[i] @ vecs[j])
    return float(np.mean(sims)), float(np.std(sims))

def check_consistency(cs, sample=500):
    all_cids = list(cs.concept_vectors.keys())
    rng_state = np.random.RandomState(42)
    cids = rng_state.choice(all_cids, size=min(sample, len(all_cids)), replace=False).tolist()
    ok = 0
    for cid in cids:
        v_stored = cs.concept_vectors[cid]
        v_code = cs.fractal.compute_vector(cid)
        if v_code is not None and abs(float(np.dot(v_stored, v_code)) - 1.0) < 1e-6:
            ok += 1
    return ok, len(cids)

# ── Baseline ────────────────────────────────────────────────────

# ── Metric pairs (from config) ──────────────────────────────────
CFG.build_metric_pairs(morph_vocab=mv, lattice=lattice, sp=sp)

if RESUME is None:
    mean_sim, std_sim = mean_cosine_sim(cs)
    ok, total = check_consistency(cs)
    print(f"  cos={mean_sim:.4f}±{std_sim:.4f} con={ok}/{total}")

# ── Parameter Optimizer (auto-tuning with config-driven rules) ──

opt = ParameterOptimizer(CFG)
if FAST:
    opt.p['full_lr'].set(CFG.fast_lr)
    opt.p['neg_samples'].set(CFG.fast_neg_samples)

# Restore optimizer state from checkpoint (metric buffers + adapted params)
if RESUME is not None:
    opt_path = CFG.cs_path.replace('.json', f'_{resume_tag}.opt.json')
    if not os.path.exists(opt_path):
        opt_path = CFG.cs_path.replace('.json', '.opt.json')
    if os.path.exists(opt_path):
        with open(opt_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        opt.load_state(saved)
        print(f"Loaded optimizer state")

pmi_gate = not FAST and args.epochs == 1

def get_lr(line_idx):
    if line_idx < CFG.lr_warmup_lines:
        return opt.p['full_lr'].current * (line_idx + 1) / CFG.lr_warmup_lines
    return opt.p['full_lr'].current


# ── Train/val split ──────────────────────────────────────────────

with open(CFG.corpus_path, 'r', encoding='utf-8') as f:
    all_lines = [l.strip() for l in f if l.strip()]

n_val = max(1, int(len(all_lines) * CFG.val_pct))
train_lines = all_lines[:-n_val]
val_lines = all_lines[-n_val:]

# ── Curriculum: sort train lines by BPE length (short → easy first) ──
train_lens = [len(sp.encode(l)) for l in train_lines]
train_pairs = sorted(zip(train_lens, train_lines), key=lambda x: x[0])
train_lens = [p[0] for p in train_pairs]
train_lines = [p[1] for p in train_pairs]

# Per-epoch max length filter
EPOCH_MAX_LEN = {1: 32, 2: 128, 3: 10**9}

print(f"  {len(train_lines)} train, {len(val_lines)} val")
print(f"  Shortest: {train_lens[0]} BPE tokens, Longest: {train_lens[-1]} BPE tokens")
print(f"  Median: {train_lens[len(train_lens)//2]} BPE tokens")
if RESUME is None:
    with open(CFG.val_corpus_path, 'w', encoding='utf-8') as f:
        for l in val_lines:
            f.write(l + '\n')

# ── STDP Training (line by line) ────────────────────────────────

LIVE_REFRESH = 1.0  # seconds between live status updates
COS_REFRESH = 5.0   # seconds between cos/pair recomputation

print("STDP training...")
gen = CrystalGenerator(cs, sp, lattice, morph_vocab=mv)
gen.train_lr = opt.p['full_lr'].current
t_start = time.time()

total_lines = len(train_lines)
CHECKPOINT_EVERY = CFG.checkpoint_every
EVAL_EVERY_F = CFG.eval_every_fast if FAST else CFG.eval_every_slow
FLUCTUATE_EVERY = CFG.fluctuate_every
DECAY_EVERY = CFG.decay_every_fast if FAST else CFG.decay_every_slow

n_trained = 0
ngram_last_total = 0
ppl_history = []
vppl_history = []
last_stat_time = 0.0
last_cos_time = 0.0
last_cos_sim = (0.0, 0.0)
last_fluct_lines = 0
last_decay_lines = 0

def live_status(text):
    """Write one-line status to terminal only — \r updates in place."""
    try:
        sys.__stdout__.write('\r' + text)
    except UnicodeEncodeError:
        safe = text.encode('cp1251', errors='replace').decode('cp1251')
        sys.__stdout__.write('\r' + safe)
    sys.__stdout__.flush()

def save_3d_vis(cs, sp, checkpoint_name):
    vis_dir = CFG.vis_dir

    cids = sorted(cs.concept_vectors.keys())
    if len(cids) == 0:
        return

    # Build matrix 32K×384
    X = np.array([cs.concept_vectors[c] for c in cids], dtype=np.float32)
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    # Randomized SVD for speed
    pca = PCA(n_components=3, random_state=0)
    proj = pca.fit_transform(Xc)  # N×3

    # Rescale to fill [-1, 1] cube
    scale = np.max(np.abs(proj))
    if scale > 0:
        proj = proj / scale

    # Build JSON
    freq = cs.fractal.codes
    tokens = []
    for i, cid in enumerate(cids):
        tok = sp.IdToPiece(cid) if cid < sp.vocab_size() else f'[ID{cid}]'
        f = float(np.linalg.norm(freq.get(cid, np.zeros(384))))
        tokens.append({
            't': tok, 'x': float(proj[i, 0]), 'y': float(proj[i, 1]), 'z': float(proj[i, 2]),
            'f': round(f, 1), 'id': int(cid)
        })

    json_path = os.path.join(vis_dir, f'points_{checkpoint_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False)

    # Latest copy for viewer.html
    latest_path = os.path.join(vis_dir, 'points_latest.json')
    if json_path != latest_path:
        shutil.copy2(json_path, latest_path)

    # Write HTML if not exists
    html_path = os.path.join(vis_dir, 'viewer.html')
    if not os.path.exists(html_path):
        _write_viewer_html(html_path)

    return json_path

def _write_viewer_html(path):
    html = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>EVA 3D</title>
<style>
body{margin:0;overflow:hidden;background:#0a0a12;font-family:monospace;color:#ccc}
#info{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
  color:#888;font-size:13px;background:rgba(0,0,0,.7);padding:4px 14px;border-radius:6px;
  pointer-events:none;z-index:10}
#popup{position:absolute;display:none;pointer-events:none;z-index:20;
  background:rgba(10,10,18,.92);border:1px solid #3a3a50;border-radius:8px;
  padding:8px 14px;color:#ccc;font:14px monospace;white-space:nowrap}
#loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  color:#666;font:24px monospace;z-index:30;text-align:center}
#error{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  color:#f66;font:16px monospace;z-index:30;text-align:center;display:none;
  background:rgba(0,0,0,.8);padding:20px 30px;border-radius:10px;max-width:500px}
</style></head>
<body>
<div id=loading>loading points...</div>
<div id=error></div>
<div id=popup></div>
<div id=info>scroll/drag to rotate · right-drag to pan · hover for token</div>

<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
<script>
async function main(){
  const errDiv=document.getElementById('error');
  try{
    const resp=await fetch('points_latest.json');
    if(!resp.ok) throw new Error('HTTP '+resp.status+': '+resp.statusText);
    const points=await resp.json();
    document.getElementById('loading').remove();

    const scene=new THREE.Scene();
    const cam=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,0.1,10);
    cam.position.set(2,1.5,2.5);
    const renderer=new THREE.WebGLRenderer({antialias:true});
    renderer.setSize(innerWidth,innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    document.body.prepend(renderer.domElement);

    const controls=new THREE.OrbitControls(cam,renderer.domElement);
    controls.enableDamping=true;controls.dampingFactor=0.08;
    controls.minDistance=0.3;controls.maxDistance=8;

    const grid=new THREE.GridHelper(2.4,12,0x333355,0x222244);
    scene.add(grid);
    scene.add(new THREE.AmbientLight(0x404060));

    const N=points.length;
    const pos=new Float32Array(N*3);
    const col=new Float32Array(N*3);
    const C0=[0.27,0.67,1.0],C1=[0.09,0.27,0.6],C2=[0.91,0.07,0.05];

    for(let i=0;i<N;i++){
      const p=points[i];
      pos[i*3]=p.x;pos[i*3+1]=p.y;pos[i*3+2]=p.z;
      const f=Math.min(p.f/500,1);
      if(f<0.5){const s=f*2;col[i*3]=C0[0]-s*0.18;col[i*3+1]=C0[1]-s*0.4;col[i*3+2]=C0[2];}
      else{const s=(f-0.5)*2;col[i*3]=C1[0]+s*0.82;col[i*3+1]=C1[1]-s*0.2;col[i*3+2]=C1[2]-s*0.55;}
    }

    const geo=new THREE.BufferGeometry();
    geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
    geo.setAttribute('color',new THREE.BufferAttribute(col,3));

    const canvas=document.createElement('canvas');canvas.width=64;canvas.height=64;
    const ctx=canvas.getContext('2d');
    const grad=ctx.createRadialGradient(32,32,0,32,32,32);
    grad.addColorStop(0,'rgba(255,255,255,1)');grad.addColorStop(0.3,'rgba(255,255,255,0.9)');grad.addColorStop(1,'rgba(255,255,255,0)');
    ctx.fillStyle=grad;ctx.fillRect(0,0,64,64);
    const tex=new THREE.CanvasTexture(canvas);

    const mat=new THREE.PointsMaterial({size:0.03,map:tex,transparent:true,vertexColors:true,sizeAttenuation:true,blending:THREE.AdditiveBlending,depthWrite:false,opacity:0.85});
    const mesh=new THREE.Points(geo,mat);scene.add(mesh);

    const raycaster=new THREE.Raycaster();const pointer=new THREE.Vector2();const popup=document.getElementById('popup');
    renderer.domElement.addEventListener('pointermove',e=>{const r=renderer.domElement.getBoundingClientRect();pointer.x=((e.clientX-r.left)/r.width)*2-1;pointer.y=-((e.clientY-r.top)/r.height)*2+1;});

    function animate(){
      requestAnimationFrame(animate);controls.update();
      raycaster.setFromCamera(pointer,cam);const hits=raycaster.intersectObject(mesh);
      if(hits.length>0){const idx=hits[0].index;const p=points[idx];popup.style.display='block';const r=renderer.domElement.getBoundingClientRect();popup.style.left=((pointer.x*0.5+0.5)*r.width+16)+'px';popup.style.top=((-pointer.y*0.5+0.5)*r.height-30)+'px';popup.innerHTML='<b>'+p.t+'</b> id='+p.id+' freq='+p.f;}
      else{popup.style.display='none';}
      renderer.render(scene,cam);
    }
    animate();
    window.addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
  }catch(e){
    errDiv.style.display='block';
    errDiv.innerHTML='<b>Error</b><br>'+e.message+'<br><br>Open via HTTP server:<br><tt>python serve_vis.py</tt>';
  }
}
main();
</script></body></html>"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)

# Determine starting line and epoch
start_line = resume_line if RESUME is not None else 0
total_epochs = args.epochs
current_epoch = resume_epoch if RESUME is not None else 1

if RESUME is not None:
    if current_epoch == 1 and start_line >= len(train_lines) - 1:
        current_epoch = 2
        start_line = 0
        print(f"Epoch 1 complete — starting epoch 2 (line 0 / {len(train_lines)})")
    if current_epoch > 1:
        print(f"Resuming epoch {current_epoch} at line {start_line}")

_ckpt_epoch = current_epoch
try:
    for epoch in range(current_epoch, total_epochs + 1):
        _ckpt_epoch = epoch
        if epoch > current_epoch:
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch} / {total_epochs}")
            print(f"{'='*60}")
            # Reset start_line for new epoch
            start_line = 0
            # Re-read corpus with fresh decay
            destab_pct = 0.0  # fresh destab for new epoch

        max_len = EPOCH_MAX_LEN.get(epoch, 10**9)
        # Pre-filter lines by curriculum max length
        epoch_mask = [l <= max_len for l in train_lens]
        epoch_train = [l for l, ok in zip(train_lines, epoch_mask) if ok]
        epoch_lines = len(epoch_train)
        if epoch_lines != total_lines:
            print(f"  Curriculum epoch {epoch}: {epoch_lines}/{total_lines} lines "
                  f"(max {max_len} BPE tokens)")

        for idx, line in enumerate(epoch_train[start_line:], start=start_line):
            if not line:
                continue

            # LR warmup
            gen.train_lr = get_lr(idx)

            destab_pct = min(idx / max(int(round(opt.p['destab_decay_lines'].current)), 1), 1.0)
            destab_scale = CFG.destab_scale_start + (CFG.destab_scale_end - CFG.destab_scale_start) * destab_pct
            gen.train_from_text(line, pmi_gate=pmi_gate, pmi_gate_min=opt.p['pmi_gate_min'].current,
                neg_samples=int(round(opt.p['neg_samples'].current)),
                context_window=int(round(opt.p['context_window'].current)),
                inh_strength=opt.p['inh_strength'].current,
                inh_threshold=opt.p['inh_threshold'].current,
                neg_lr_ratio=CFG.neg_lr_ratio, field_gate=CFG.field_gate and epoch == 1, use_torch=CFG.use_torch,
                destab_scale=destab_scale)
            n_trained += 1
            now = time.time()
            elapsed = now - t_start

            # ---- Periodic tasks (line-based) ----
            if idx > 0 and idx - last_fluct_lines >= FLUCTUATE_EVERY:
                cs.fluctuate_fractal(noise_scale=opt.p['noise_scale'].current,
                                     decay=opt.p['decay_rate'].current,
                                     repel_strength=opt.p['repel_strength'].current)
                last_fluct_lines = idx
                if hasattr(gen, '_invalidate_torch'):
                    gen._invalidate_torch()

            if idx > 0 and idx - last_decay_lines >= DECAY_EVERY:
                lattice.decay_all()
                lattice.decay_connections()
                cs.decay_usage(decay=0.98)
                last_decay_lines = idx

            # ---- Live status (every ~1 second on terminal) ----
            if now - last_stat_time >= LIVE_REFRESH:
                rate = idx / max(elapsed, 0.1)
                pct = 100 * idx / epoch_lines
                if rate >= 0.1:
                    eta = (epoch_lines - idx) / rate
                    eta_h, eta_m = int(eta // 3600), int((eta % 3600) // 60)
                    eta_s = f"ETA {eta_h}h{eta_m:02d}m"
                else:
                    eta_s = "ETA ---"

                if now - last_cos_time >= COS_REFRESH:
                    last_cos_sim = mean_cosine_sim(cs)
                    last_cos_time = now

                mean_sim, std_sim = last_cos_sim
                live_status(f"[{pct:4.1f}%] {idx:6d}L | {rate:4.0f} L/s | {eta_s} | "
                            f"{elapsed/60:.0f}min cos={mean_sim:.4f}±{std_sim:.4f}")
                try:
                    with open(CFG.status_path + '.tmp', 'w', encoding='utf-8') as _sf:
                        json.dump({'line': idx, 'total': epoch_lines, 'pct': pct,
                                   'rate': rate, 'eta_s': eta_s, 'elapsed_min': elapsed/60,
                                   'cos_mean': mean_sim, 'cos_std': std_sim}, _sf)
                    os.replace(CFG.status_path + '.tmp', CFG.status_path)
                except Exception:
                    pass
                last_stat_time = now

            # ---- Line-based checkpoint + full report ----
            if idx > 0 and idx % CHECKPOINT_EVERY == 0:
                rate = idx / max(elapsed, 0.1)
                pct = 100 * idx / epoch_lines
                if rate >= 0.1:
                    eta = (epoch_lines - idx) / rate
                    eta_h, eta_m = int(eta // 3600), int((eta % 3600) // 60)
                    eta_s = f"ETA {eta_h}h{eta_m:02d}m"
                else:
                    eta_s = "ETA ---"
                print(f"\n[{pct:5.1f}%] {idx:7d}L | {rate:4.0f} L/s | {eta_s}")
                mean_sim, std_sim = mean_cosine_sim(cs)
                ok, total_c = check_consistency(cs)
                ng_total = sum(len(v) for v in lattice.ngrams.values())
                ng_new = ng_total - ngram_last_total
                ngram_last_total = ng_total

                ckpt_name = f"{idx // 1000}k"
                cs_num = CFG.cs_path.replace('.json', f'_{ckpt_name}.json')
                lat_num = CFG.lattice_path.replace('.json', f'_{ckpt_name}.json')
                print()
                _quiet(cs.save, cs_num)
                _quiet(lattice.save, lat_num)
                opt_state_path = CFG.cs_path.replace('.json', f'_{ckpt_name}.opt.json')
                with open(opt_state_path + '.tmp', 'w', encoding='utf-8') as f:
                    json.dump(opt.save_state(), f, ensure_ascii=False)
                os.replace(opt_state_path + '.tmp', opt_state_path)
                save_checkpoint_state(idx, epoch=_ckpt_epoch)
                cleanup_old_checkpoints(keep=CFG.cleanup_keep)
                if idx % CFG.periodic_save_every == 0:
                    _quiet(cs.save, CFG.cs_path)
                    _quiet(lattice.save, CFG.lattice_path)

                n_upd = cs._update_count
                avg_delta = (cs._total_shift / max(n_upd, 1)) * 1e3
                cs._total_shift = 0.0
                cs._update_count = 0

                n_code_out, max_code_abs = cs.check_code_range(bound=CFG.code_bound)
                vec_ok, vec_total, vec_max_dev = cs.validate_vector_norms()
                if n_code_out > 0 or vec_max_dev > CFG.vec_dev_warn:
                    print(f"  CODE_DRIFT n_out={n_code_out} max|code|={max_code_abs:.1f} vec_dev={vec_max_dev:.6f}")

                opt.step(mean_cos=mean_sim, std_cos=std_sim, delta=avg_delta, ng_new=ng_new)
                gen.train_lr = opt.p['full_lr'].current

                seed = np.random.choice(CFG.test_seeds)
                result = gen.generate(seed_word=seed, max_words=CFG.gen_max_words)
                txt = result['text'].replace('\n', ' ').strip()
                print(f"  cos={mean_sim:.4f}±{std_sim:.4f} con={ok}/{total_c} | gen({seed}): {txt}")

                if idx > 0 and idx % EVAL_EVERY_F == 0:
                    eval_result = _quiet(gen.evaluate, CFG.val_corpus_path, max_lines=CFG.eval_max_lines)
                    ppl = eval_result['perplexity']
                    vppl = eval_result['vec_perplexity']
                    acc1 = eval_result['accuracy_top1']
                    vacc1 = eval_result['vec_accuracy_top1']
                    ppl_history.append((idx, ppl))
                    vppl_history.append((idx, vppl))
                    ppl_trend = ''
                    if len(ppl_history) >= 2:
                        d = ppl - ppl_history[-2][1]
                        ppl_trend = f" {'+' if d > 0 else ''}{d:.0f}"
                    print(f"  PPL={ppl:.0f}{ppl_trend} acc@1={acc1:.3f} vPPL={vppl:.0f} vacc@1={vacc1:.3f}")
                    opt.step(vec_ppl=vppl, acc1=acc1, vacc1=vacc1)
                    gen.train_lr = opt.p['full_lr'].current
                print()

except KeyboardInterrupt:
    print("\n\n[EVA] Training interrupted — saving checkpoint...")
    _quiet(cs.save, CFG.cs_path)
    _quiet(lattice.save, CFG.lattice_path)
    opt_state_path = CFG.cs_path.replace('.json', '.opt.json')
    with open(opt_state_path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(opt.save_state(), f, ensure_ascii=False)
    os.replace(opt_state_path + '.tmp', opt_state_path)
    save_checkpoint_state(idx, epoch=_ckpt_epoch)
    cleanup_old_checkpoints(keep=CFG.cleanup_keep)
    print("[EVA] Checkpoint saved. Exiting.")
    sys.exit(0)

# Final save
_quiet(cs.save, CFG.cs_path)
_quiet(lattice.save, CFG.lattice_path)
opt_state_path = CFG.cs_path.replace('.json', '.opt.json')
with open(opt_state_path + '.tmp', 'w', encoding='utf-8') as f:
    json.dump(opt.save_state(), f, ensure_ascii=False)
os.replace(opt_state_path + '.tmp', opt_state_path)
save_checkpoint_state(idx, epoch=_ckpt_epoch)
cleanup_old_checkpoints(keep=CFG.cleanup_keep)

# ── Final diagnostics ───────────────────────────────────────────

print("--- Final ---")
mean_sim, std_sim = mean_cosine_sim(cs)
ok, total = check_consistency(cs)
print(f"  cos={mean_sim:.4f}±{std_sim:.4f} con={ok}/{total}")

print("--- Generation ---")
for seed in CFG.test_seeds:
    result = gen.generate(seed_word=seed, max_words=CFG.gen_max_words)
    txt = result['text']
    print(f"  [{seed}] {txt}  ({result['score']:.2f})")

t_total = time.time() - t_start
print(f"  {n_trained} lines in {t_total:.0f}s ({n_trained/t_total:.0f} L/s)")
print("Saving...")
_quiet(cs.save, CFG.cs_path)
_quiet(lattice.save, CFG.lattice_path)
opt_state_path = CFG.cs_path.replace('.json', '.opt.json')
with open(opt_state_path + '.tmp', 'w', encoding='utf-8') as f:
    json.dump(opt.save_state(), f, ensure_ascii=False)
os.replace(opt_state_path + '.tmp', opt_state_path)
save_checkpoint_state(idx, epoch=_ckpt_epoch)
cleanup_old_checkpoints(keep=CFG.cleanup_keep)
print("Done.")


