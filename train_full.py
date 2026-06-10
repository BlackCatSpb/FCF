"""Full training (32K BPE). Process line-by-line. Checkpoints overwrite."""

import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import time, json, os, shutil, argparse
import numpy as np
import sentencepiece as spm
from sklearn.decomposition import PCA
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.crystal_generator import CrystalGenerator

# Redirect stdout to UTF-8 log file (terminal cp1251 can't print ▁)
LOG_FILE = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\train_log.txt'
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

CORPUS = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru.txt'
BPE_MODEL = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\bpe_ru_32k.model'
CS_PATH = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\concept_space.json'
LATTICE_PATH = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\syntax_lattice.json'

sp = spm.SentencePieceProcessor(model_file=BPE_MODEL)
V = sp.vocab_size()
print(f"vocab_size = {V}")

# ── Parse args ──────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--resume', '-r', help='resume from checkpoint (e.g. 2k)')
parser.add_argument('--fast', '-f', action='store_true', help='fast mode: higher lr + negative sampling')
args = parser.parse_args()
RESUME = args.resume
FAST = args.fast
if FAST:
    print("FAST mode: base_lr=0.15, neg_samples=3, pmi_gate=off, decay_every=250, eval_every=1000")

if RESUME:
    # Parse resume line number
    r = RESUME.lower().rstrip('k')
    try:
        resume_line = int(r) * 1000
    except ValueError:
        print(f"Invalid resume value: {RESUME}")
        sys.exit(1)
    print(f"\nResuming from checkpoint '{RESUME}' (line {resume_line})")

    cs_path = CS_PATH.replace('.json', f'_{RESUME}.json')
    lat_path = LATTICE_PATH.replace('.json', f'_{RESUME}.json')
    if not os.path.exists(cs_path) or not os.path.exists(lat_path):
        print(f"Checkpoint not found: {cs_path} or {lat_path}")
        sys.exit(1)

    cs = ConceptSpace.load(cs_path)
    lattice = SyntaxLattice()
    lattice.load(lat_path)
    print(f"  Loaded ConceptSpace ({len(cs.concept_vectors)} vectors)")
    print(f"  Loaded SyntaxLattice ({len(lattice.concept_freq)} concepts)")

else:
    print("\nInitializing ConceptSpace (32K fractal vectors @ 384D)...")
    cs = ConceptSpace(vocab_size=V, dim=384)
    cs.init_concepts()
    cs.init_homeostasis()

    print("\nBuilding SyntaxLattice from full corpus...")
    lattice = SyntaxLattice()
    t0 = time.time()
    lattice.build(CORPUS, sp, max_n=4)
    t1 = time.time()
    print(f"  done in {t1-t0:.1f}s")
    print(f"  n-gram prefixes: {[len(v) for v in lattice.ngrams.values()]}")
    print(f"  unique concepts: {len(lattice.concept_freq)}")

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

def pair_sim(cs, sp, a, b):
    id_a = sp.PieceToId(a); id_b = sp.PieceToId(b)
    if id_a < 0 or id_b < 0: return None
    va = cs.concept_vector(id_a); vb = cs.concept_vector(id_b)
    if va is None or vb is None: return None
    return float(va @ vb)

# ── Baseline ────────────────────────────────────────────────────

if not RESUME:
    print("\n--- Baseline diagnostics ---")
    mean_sim, std_sim = mean_cosine_sim(cs)
    print(f"  mean cosine sim: {mean_sim:.4f} ± {std_sim:.4f}")
    ok, total = check_consistency(cs)
    print(f"  code-vector consistency: {ok}/{total}")
    pairs_to_track = [('соба', 'ка'), ('ко', 'шка'), ('человек', 'а'),
                      ('человек', 'война'), ('князь', 'Андрей')]
    for a, b in pairs_to_track:
        s = pair_sim(cs, sp, '▁' + a if not a.startswith('▁') else a,
                     '▁' + b if not b.startswith('▁') else b)
        if s is not None:
            print(f"  sim({a:12s}, {b:12s}) = {s:.4f}")

else:
    pairs_to_track = [('соба', 'ка'), ('человек', 'война'),
                      ('князь', 'Андрей'), ('любовь', 'смерть')]

# ── Adaptive parameter control ────────────────────────────────

TARGET_STD = 1.0 / np.sqrt(384)  # 0.051 — random uniform on sphere
prev_mean_cos = 0.0

repel_strength = 0.08
base_lr = 0.15 if FAST else 0.03
noise_scale = 0.001
decay_rate = 0.9998
neg_samples = 3 if FAST else 0
pmi_gate = not FAST
# LR warmup
LR_WARMUP_LINES = 1000  # linear ramp from 0 to full_lr
full_lr = base_lr
if FAST:
    print(f"  lr={base_lr}, neg_samples={neg_samples}, pmi_gate={pmi_gate}")

def get_lr(line_idx):
    """LR warmup: linear ramp over first LR_WARMUP_LINES."""
    if line_idx < LR_WARMUP_LINES:
        return full_lr * (line_idx + 1) / LR_WARMUP_LINES
    return full_lr

def adapt_params(mean_cos, std_cos):
    global repel_strength, base_lr, noise_scale, prev_mean_cos, full_lr
    changes = []

    if mean_cos > 0.01:
        repel_strength = min(repel_strength * 1.10, 0.20)
        changes.append(f"repel={repel_strength:.3f}")
    elif mean_cos < -0.005:
        repel_strength = max(repel_strength * 0.90, 0.01)
        changes.append(f"repel={repel_strength:.3f}")

    if std_cos < TARGET_STD * 0.80:
        noise_scale = min(noise_scale * 1.15, 0.01)
        changes.append(f"noise={noise_scale:.4f}")
    elif std_cos > TARGET_STD * 1.30:
        noise_scale = max(noise_scale * 0.90, 0.0002)
        changes.append(f"noise={noise_scale:.4f}")

    cos_trend = mean_cos - prev_mean_cos
    if cos_trend > 0.001 and mean_cos > 0.005:
        full_lr = max(full_lr * 0.95, 0.003)
        changes.append(f"lr={full_lr:.4f}")
    elif cos_trend < -0.001 and mean_cos < -0.005:
        full_lr = min(full_lr * 1.05, 0.02)
        changes.append(f"lr={full_lr:.4f}")

    gen.train_lr = full_lr
    prev_mean_cos = mean_cos
    return changes


# ── Train/val split ──────────────────────────────────────────────

VAL_CORPUS = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\val_corpus.txt'
VAL_PCT = 0.05

with open(CORPUS, 'r', encoding='utf-8') as f:
    all_lines = [l.strip() for l in f if l.strip()]

n_val = max(1, int(len(all_lines) * VAL_PCT))
train_lines = all_lines[:-n_val]
val_lines = all_lines[-n_val:]

print(f"\n  Train lines: {len(train_lines)}, Val lines: {len(val_lines)}")
if not RESUME:
    with open(VAL_CORPUS, 'w', encoding='utf-8') as f:
        for l in val_lines:
            f.write(l + '\n')

# ── STDP Training (line by line) ────────────────────────────────

LIVE_REFRESH = 1.0  # seconds between live status updates
COS_REFRESH = 5.0   # seconds between cos/pair recomputation

print("\n--- STDP training (fractal self-organisation) ---")
gen = CrystalGenerator(cs, sp, lattice)
gen.train_lr = full_lr
t_start = time.time()

total_lines = len(train_lines)
CHECKPOINT_EVERY = 500      # save checkpoint (lines)
EVAL_EVERY = 1000 if FAST else 2000  # full eval (lines)
FLUCTUATE_EVERY = 2000       # fluctuation + centroid repel (lines)
DECAY_EVERY = 2000 if FAST else 3000  # lattice decay sweep (lines)

n_trained = 0
ngram_last_total = 0
ppl_history = []
vppl_history = []
last_stat_time = 0.0
last_cos_time = 0.0
last_cos_sim = (0.0, 0.0)
last_pair_strs = ''
last_fluct_lines = 0
last_decay_lines = 0

_live_pairs = [('соба', 'ка'), ('человек', 'война'),
               ('князь', 'Андрей'), ('любовь', 'смерть')]

def live_status(text):
    """Write one-line status to terminal only — \r updates in place."""
    try:
        sys.__stdout__.write('\r' + text)
    except UnicodeEncodeError:
        safe = text.encode('cp1251', errors='replace').decode('cp1251')
        sys.__stdout__.write('\r' + safe)
    sys.__stdout__.flush()

def get_pair_strs():
    parts = []
    for a, b in _live_pairs:
        s = pair_sim(cs, sp, '▁' + a, '▁' + b)
        if s is not None:
            parts.append(f"{a[:3]}/{b[:3]}={s:.2f}")
    return ' '.join(parts)

def save_3d_vis(cs, sp, checkpoint_name):
    """SVD 384D → 3D, save JSON + write HTML viewer."""
    vis_dir = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\vis'
    os.makedirs(vis_dir, exist_ok=True)

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

# Determine starting line
start_line = resume_line if RESUME else 0

try:
    for idx, line in enumerate(train_lines[start_line:], start=start_line):
        if not line:
            continue

        # LR warmup
        gen.train_lr = get_lr(idx)

        gen.train_from_text(line, pmi_gate=pmi_gate, neg_samples=neg_samples)
        n_trained += 1
        now = time.time()
        elapsed = now - t_start

        # ── Periodic tasks (line-based) ──
        if idx > 0 and idx - last_fluct_lines >= FLUCTUATE_EVERY:
            cs.fluctuate_fractal(noise_scale=noise_scale, decay=decay_rate,
                                 repel_strength=repel_strength)
            last_fluct_lines = idx

        if idx > 0 and idx - last_decay_lines >= DECAY_EVERY:
            lattice.decay_all()
            lattice.decay_connections()
            cs.decay_usage(decay=0.98)
            last_decay_lines = idx

        # ── Live status (every ~1 second on terminal) ──
        if now - last_stat_time >= LIVE_REFRESH:
            rate = idx / max(elapsed, 0.1)
            pct = 100 * idx / total_lines
            if rate >= 0.1:
                eta = (total_lines - idx) / rate
                eta_h, eta_m = int(eta // 3600), int((eta % 3600) // 60)
                eta_s = f"ETA {eta_h}h{eta_m:02d}m"
            else:
                eta_s = "ETA ---"

            # Refresh cos + pairs every COS_REFRESH seconds
            if now - last_cos_time >= COS_REFRESH:
                last_cos_sim = mean_cosine_sim(cs)
                last_pair_strs = get_pair_strs()
                last_cos_time = now

            mean_sim, std_sim = last_cos_sim
            tail = f" cos={mean_sim:.4f}±{std_sim:.4f} | {last_pair_strs} | repel={repel_strength:.2f} lr={full_lr:.3f}"
            live_status(f"[{pct:4.1f}%] {idx:6d}L | {rate:4.0f} L/s | {eta_s} | "
                        f"{elapsed/60:.0f}min{tail}")
            last_stat_time = now

        # ── Line-based checkpoint + full report ──
        if idx > 0 and idx % CHECKPOINT_EVERY == 0:
            rate = idx / max(elapsed, 0.1)
            pct = 100 * idx / total_lines
            if rate >= 0.1:
                eta = (total_lines - idx) / rate
                eta_h, eta_m = int(eta // 3600), int((eta % 3600) // 60)
                eta_s = f"ETA {eta_h}h{eta_m:02d}m"
            else:
                eta_s = "ETA ---"
            mean_sim, std_sim = mean_cosine_sim(cs)
            ok, total_c = check_consistency(cs)
            pair_strs = get_pair_strs()

            # Adaptive params
            param_changes = adapt_params(mean_sim, std_sim)
            param_str = f"repel={repel_strength:.2f} lr={full_lr:.4f} noise={noise_scale:.4f}"
            if param_changes:
                param_str = ' '.join(param_changes)

            # Ngram growth
            ng_total = sum(len(v) for v in lattice.ngrams.values())
            ng_new = ng_total - ngram_last_total
            ngram_last_total = ng_total

            # Save checkpoint
            ckpt_name = f"{idx // 1000}k"
            cs_num = CS_PATH.replace('.json', f'_{ckpt_name}.json')
            lat_num = LATTICE_PATH.replace('.json', f'_{ckpt_name}.json')
            print()  # clear live status line before save messages
            cs.save(cs_num)
            lattice.save(lat_num)

            print(f"\n[{pct:5.1f}%] {idx:7d}L | {rate:4.0f} L/s | {eta_s} | {param_str}")
            # Shift stats
            n_upd = cs._update_count
            avg_delta = (cs._total_shift / max(n_upd, 1)) * 1e3  # milliradians
            cs._total_shift = 0.0
            cs._update_count = 0
            shift_str = f"\u03b4={avg_delta:.2f}m" if n_upd > 0 else "\u03b4=?"

            # Drift diagnostics
            n_code_out, max_code_abs = cs.check_code_range(bound=10.0)
            vec_ok, vec_total, vec_max_dev = cs.validate_vector_norms()
            drift_warn = ""
            if n_code_out > 0 or vec_max_dev > 0.01:
                drift_warn = f" CODE_DRIFT(n_out={n_code_out} max|code|={max_code_abs:.1f} vec_dev={vec_max_dev:.6f})"

            print(f"  cos={mean_sim:.4f}\u00b1{std_sim:.4f} | con={ok}/{total_c} | "
                  f"ng={ng_total} (+{ng_new}) | {shift_str}{drift_warn} | {pair_strs}")

            # Test generation
            seed = np.random.choice(['князь', 'человек', 'война', 'любовь', 'дом', 'жизнь'])
            result = gen.generate(seed_word=seed, max_words=8)
            txt = result['text'].replace('\n', ' ').strip()
            print(f"  gen({seed}): {txt[:70]}")

            # Eval
            if idx > 0 and idx % EVAL_EVERY == 0:
                eval_result = gen.evaluate(VAL_CORPUS, max_lines=100)
                ppl = eval_result['perplexity']
                vppl = eval_result['vec_perplexity']
                acc1 = eval_result['accuracy_top1']
                vacc1 = eval_result['vec_accuracy_top1']
                ppl_history.append((idx, ppl))
                vppl_history.append((idx, vppl))
                ppl_trend = ''
                if len(ppl_history) >= 2:
                    d = ppl - ppl_history[-2][1]
                    ppl_trend = f" {'+' if d > 0 else ''}{d:.0f} vs prev"
                print(f"  PPL={ppl:.0f}{ppl_trend} | acc@1={acc1:.3f} | vPPL={vppl:.0f} | vacc@1={vacc1:.3f}")
            print()

except KeyboardInterrupt:
    print("\n\n[EVA] Training interrupted — saving checkpoint...")
    cs.save(CS_PATH)
    lattice.save(LATTICE_PATH)
    print("[EVA] Checkpoint saved. Exiting.")
    sys.exit(0)

# Final save
cs.save(CS_PATH)
lattice.save(LATTICE_PATH)

# ── Final diagnostics ───────────────────────────────────────────

print("\n--- Final diagnostics ---")
mean_sim, std_sim = mean_cosine_sim(cs)
print(f"  mean cosine sim: {mean_sim:.4f} ± {std_sim:.4f}")
ok, total = check_consistency(cs)
print(f"  code-vector consistency: {ok}/{total}")
for a, b in pairs_to_track:
    s = pair_sim(cs, sp, '▁' + a if not a.startswith('▁') else a,
                 '▁' + b if not b.startswith('▁') else b)
    if s is not None:
        print(f"  sim({a:12s}, {b:12s}) = {s:.4f}")

print("\n--- Generation tests ---")
for seed in ['князь', 'человек', 'война', 'любовь']:
    result = gen.generate(seed_word=seed, max_words=10)
    txt = result['text']
    print(f"  [{seed}] {txt[:60]}  (score={result['score']:.2f})")

t_total = time.time() - t_start
print(f"\nTotal: {n_trained} lines in {t_total:.0f}s ({n_trained/t_total:.0f} l/s)")
print("Saving...")
cs.save(CS_PATH)
lattice.save(LATTICE_PATH)
print("Done.")


