"""t-SNE visualization of concept vectors.
Usage:
  python viz_tsne.py [checkpoint_tag] [--n 3000] [--perplexity 30]

Output: real_data/vis/points_{tag}_tsne.json + viewer.html
Open with: python serve_vis.py
"""
import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import os, json, time, glob, re, shutil, tempfile
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import numpy as np
from sklearn.manifold import TSNE
import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.fcf_config import FCFConfig

CFG = FCFConfig()
BASE = CFG.data_dir
sp = spm.SentencePieceProcessor(model_file=CFG.bpe_model_path)


def clean_sp(s):
    return s.replace('\u2581', ' ').strip()


def resolve_checkpoint(tag):
    if tag != 'latest':
        return tag
    state_path = CFG.ckpt_state_path
    if os.path.exists(state_path):
        with open(state_path) as f:
            s = json.load(f)
        return f"{s['line'] // 1000}k"
    files = glob.glob(os.path.join(BASE, 'concept_space_*k.json'))
    if not files:
        raise FileNotFoundError("No checkpoints found")
    return re.search(r'_(\d+k)\.json$', max(files, key=os.path.getmtime)).group(1)


def main():
    tag = resolve_checkpoint(sys.argv[1] if len(sys.argv) > 1 else 'latest')
    n_samples = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 3000
    perplexity = int(sys.argv[sys.argv.index('--perplexity') + 1]) if '--perplexity' in sys.argv else 30

    cs_path = CFG.cs_path.replace('.json', f'_{tag}.json')
    lat_path = CFG.lattice_path.replace('.json', f'_{tag}.json')

    # Copy to temp to avoid file-lock conflicts
    tmpdir = tempfile.mkdtemp(prefix=f'viz_{tag}_')
    for src_f in glob.glob(os.path.join(BASE, f'*{tag}*')):
        shutil.copy2(src_f, tmpdir)

    cs = ConceptSpace.load(os.path.join(tmpdir, f'concept_space_{tag}.json'))
    lattice = SyntaxLattice()
    lattice.load(os.path.join(tmpdir, f'syntax_lattice_{tag}.json'))

    print(f"Loaded {sum(cs.concept_vectors._valid)}/{cs.vocab_size} vectors")

    # Sample valid vectors + high-frequency tokens
    valid_cids = [c for c in range(cs.vocab_size) if cs.concept_vectors._valid[c]]
    freqs = np.array([lattice.concept_freq.get(c, 0) for c in valid_cids])
    # Stratified: mix of high-freq and low-freq tokens
    high = np.where(freqs > 5)[0]
    low = np.where(freqs <= 5)[0]
    n_high = min(n_samples // 2, len(high))
    n_low = min(n_samples - n_high, len(low))
    rng = np.random.RandomState(42)
    chosen = np.concatenate([
        rng.choice(high, size=n_high, replace=False) if n_high > 0 else [],
        rng.choice(low, size=n_low, replace=False) if n_low > 0 else [],
    ])
    rng.shuffle(chosen)
    sampled = [valid_cids[i] for i in chosen]

    vecs = np.array([cs.concept_vectors._data[c] for c in sampled], dtype=np.float64)
    print(f"Running t-SNE on {len(sampled)} vectors ({vecs.shape})...")
    t0 = time.time()
    proj = TSNE(n_components=3, perplexity=perplexity, random_state=42,
                learning_rate='auto', init='random', verbose=1).fit_transform(vecs)
    print(f"  t-SNE done in {time.time() - t0:.1f}s")

    # Scale to [-1, 1] cube
    scale = np.max(np.abs(proj))
    if scale > 0:
        proj = proj / scale

    # Build JSON
    tokens = []
    for i, cid in enumerate(sampled):
        tok = clean_sp(sp.IdToPiece(cid)) if cid < sp.vocab_size() else f'[ID{cid}]'
        f = float(lattice.concept_freq.get(cid, 0))
        tokens.append({
            't': tok, 'x': float(proj[i, 0]), 'y': float(proj[i, 1]), 'z': float(proj[i, 2]),
            'f': f, 'id': int(cid)
        })

    vis_dir = CFG.vis_dir
    os.makedirs(vis_dir, exist_ok=True)
    out = os.path.join(vis_dir, f'points_{tag}_tsne.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False)
    print(f"Saved: {out}")

    latest = os.path.join(vis_dir, 'points_latest.json')
    if out != latest:
        shutil.copy2(out, latest)

    # Write viewer HTML if not exists
    html = os.path.join(vis_dir, 'viewer.html')
    if not os.path.exists(html):
        _write_viewer_html(html)
        print(f"Written: {html}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\nRun: python serve_vis.py\nThen open http://127.0.0.1:8080/viewer.html")


def _write_viewer_html(path):
    html = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>FCF 3D t-SNE</title>
<style>
body{margin:0;overflow:hidden;background:#0a0a12;font-family:monospace;color:#ccc}
#info{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
  color:#888;font-size:13px;background:rgba(0,0,0,.7);padding:4px 14px;border-radius:6px;
  pointer-events:none;z-index:10}
#popup{position:absolute;display:none;pointer-events:none;z-index:20;
  background:rgba(10,10,18,.92);border:1px solid #3a3a50;border-radius:8px;
  padding:8px 14px;color:#ccc;font:14px monospace;white-space:nowrap}
</style>
</head><body>
<div id=info>FCF | scroll zoom | drag rotate | hover for token</div>
<div id=popup></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const POPUP=document.getElementById('popup');
const scene=new THREE.Scene(), cam=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,0.1,10);
cam.position.set(1.8,1.2,1.8);
const ren=new THREE.WebGLRenderer({antialias:true});
ren.setSize(innerWidth,innerHeight);ren.setPixelRatio(devicePixelRatio);
ren.setClearColor(0x0a0a12);document.body.appendChild(ren.domElement);
const controls=new THREE.OrbitControls(cam,ren.domElement);controls.enableDamping=true;
const loader=new THREE.FileLoader();loader.load('points_latest.json',function(d){
  const pts=JSON.parse(d);console.log(pts.length+' points');
  const geo=new THREE.BufferGeometry();
  const pos=new Float32Array(pts.length*3),col=new Float32Array(pts.length*3);
  for(let i=0;i<pts.length;i++){
    const p=pts[i];pos[i*3]=p.x;pos[i*3+1]=p.y;pos[i*3+2]=p.z;
    const h=0.55+0.4*Math.min(p.f/50,1);col[i*3]=h;col[i*3+1]=h*0.7;col[i*3+2]=h*0.3;
  }
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  const mat=new THREE.PointsMaterial({size:0.008,vertexColors:true,sizeAttenuation:true,
    transparent:true,opacity:0.85,depthWrite:false});
  const mesh=new THREE.Points(geo,mat);scene.add(mesh);
  const raycaster=new THREE.Raycaster();
  ren.domElement.addEventListener('mousemove',function(e){
    const m=new THREE.Vector2((e.clientX/innerWidth)*2-1,-(e.clientY/innerHeight)*2+1);
    raycaster.setFromCamera(m,cam);
    const hits=raycaster.intersectObject(mesh);
    if(hits.length>0){
      const i=hits[0].index;const p=pts[i];const freq=p.f>0?p.f+'x':'rare';
      POPUP.style.display='block';POPUP.style.left=e.clientX+14+'px';POPUP.style.top=e.clientY-10+'px';
      POPUP.innerHTML='<b>'+p.t+'</b> ID:'+p.id+' freq:'+freq;
    }else{POPUP.style.display='none';}
  });
});
function anim(){requestAnimationFrame(anim);controls.update();ren.render(scene,cam);}
anim();window.addEventListener('resize',function(){
  cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight);});
</script></body></html>"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    main()
