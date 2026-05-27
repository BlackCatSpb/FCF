"""
EVA — Validation Suite: Anna Karenina test, Multi-level generation, SelfReflection.
Runs evaluation during training every 5000 steps.
"""
import torch, torch.nn.functional as F, re, numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

def evaluate_model(ut, cv, store, data, total, CKPT, rng, device='cuda'):
    """Run full validation suite. Returns metrics dict."""
    VT = cv.vocab_size
    results = {}
    
    # === 1. GENERATIVE ABILITY ===
    seeds = ['привет', 'Пьер', 'князь Андрей', 'Наташа', 'война', 'мир', 'солнце']
    gen_outputs = {}
    total_coherent = 0
    
    for seed in seeds:
        ids = [cv.SENT_OPEN_IDX, cv.WORD_OPEN_IDX] + cv.encode(seed)[1:-1] + [cv.WORD_CLOSE_IDX, cv.SENT_CLOSE_IDX]
        ids_out = list(ids)
        
        with torch.no_grad():
            for _ in range(80):
                _, sc = ut(torch.tensor([ids_out], dtype=torch.long, device=device), return_scores=True)
                logits = sc[0, -1] / 0.6
                if len(ids_out) >= 3 and ids_out[-1] == ids_out[-2] == ids_out[-3]:
                    logits[ids_out[-1]] -= 10.0
                for t in set(ids_out[-8:]):
                    if t < VT: logits[t] -= 3.0
                _, idx = torch.topk(logits, 20)
                p = torch.softmax(logits[idx], dim=-1)
                nt = idx[torch.multinomial(p, 1)].item()
                ids_out.append(nt)
                if nt == cv.SENT_CLOSE_IDX and len(ids_out) > 20:
                    break
        
        text = cv.decode(ids_out)
        clean = text.replace('</W><W>', ' ').replace('</W></S><S><W>', ' ')
        clean = clean.replace('<S>','').replace('</S>','').replace('<W>','').replace('</W>','')
        clean = re.sub(r'\s+', ' ', clean).strip()
        gen_outputs[seed] = clean
        
        # Count actual Russian words (Cyrillic, >1 char)
        import re as re_mod
        words = re_mod.findall(r'[а-яёА-ЯЁ]+', clean)
        if len(words) >= 3: total_coherent += 1
    
    results['generation'] = gen_outputs
    results['coherent_ratio'] = total_coherent / len(seeds)
    
    # === 2. UNFAMILIAR TEXT TEST ===
    anna_path = r"C:\Users\black\OneDrive\Desktop\Анна Каренина.txt"
    if not os.path.exists(anna_path):
        anna_path = None
        # Try to find any other untrained text
        for path in [r"C:\Users\black\OneDrive\Desktop\*.txt"]:
            pass
    
    if anna_path and os.path.exists(anna_path):
        with open(anna_path, 'r', encoding='windows-1251') as f:
            raw = f.read()
        raw = re.sub(r'\r\n|\r', '\n', raw)
        sents = re.split(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z])', raw)[:100]  # first 100 sentences
        
        anna_acc = []
        for sent in sents[:20]:  # evaluate 20 sentences
            sent = sent.strip()
            if len(sent) < 10: continue
            ids = cv.encode_with_boundaries(sent)
            if len(ids) < 5: continue
            
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                _, scores = ut(inp, return_scores=True)
                pred = scores[0].argmax(dim=-1).tolist()
            correct = sum(1 for p, t in zip(pred, ids) if p == t)
            anna_acc.append(correct / len(ids))
        
        results['anna_karenina_acc'] = np.mean(anna_acc) if anna_acc else 0
        results['anna_karenina_samples'] = len(anna_acc)
    else:
        results['anna_karenina_acc'] = None
    
    # === 3. MULTI-LEVEL GENERATION ===
    # Retrieve similar trajectories and boost generation
    if store.total_stored >= 5:
        multi_gen = {}
        for seed in seeds[:3]:
            ids = cv.encode(seed)[1:-1]
            if len(ids) < 2: continue
            
            htraj = None
            if store.hierarchical:
                for h in store.hierarchical:
                    if seed in h.text:
                        htraj = h
                        break
            
            ids_out = list(ids)
            with torch.no_grad():
                for _ in range(60):
                    _, sc = ut(torch.tensor([ids_out], dtype=torch.long, device=device), return_scores=True)
                    logits = sc[0, -1] / 0.6
                    
                    # Boost from retrieved similar trajectories
                    if store.total_stored >= 5:
                        similar = store.find_similar(
                            ut.embed(torch.tensor([ids_out], device=device))[0].cpu().numpy(),
                            top_k=3
                        )
                        for sim in similar:
                            for sid in sim['ids'][len(ids_out):len(ids_out)+5]:
                                if 0 < sid < VT:
                                    logits[sid] += 2.0
                    
                    for t in set(ids_out[-8:]):
                        if t < VT: logits[t] -= 3.0
                    _, idx = torch.topk(logits, 20)
                    p = torch.softmax(logits[idx], dim=-1)
                    nt = idx[torch.multinomial(p, 1)].item()
                    ids_out.append(nt)
                    if nt == cv.SENT_CLOSE_IDX and len(ids_out) > 20:
                        break
            
            text = cv.decode(ids_out)
            clean = text.replace('</W><W>', ' ').replace('</W></S><S><W>', ' ')
            clean = clean.replace('<S>','').replace('</S>','').replace('<W>','').replace('</W>','')
            clean = re.sub(r'\s+', ' ', clean).strip()
            multi_gen[seed] = clean
        
        results['multi_level_gen'] = multi_gen
    
    # === 4. SELF-REFLECTION ===
    from eva.symbolic.self_reflection import SelfReflection
    reflector = SelfReflection()
    
    reflection_metrics = {}
    for seed, text in gen_outputs.items():
        ids = cv.encode(text[1:-1] if text else seed)
        if len(ids) < 4: continue
        
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            traj = ut.embed(inp)[0].cpu().numpy()
        
        diag = reflector.diagnose(traj, ids)
        reflection_metrics[seed] = {
            'length': diag.length,
            'total_distance': round(diag.total_distance, 2),
            'mean_curvature': round(diag.mean_curvature, 4),
            'max_curvature': round(diag.max_curvature, 4),
            'contradictions': diag.n_contradictions,
            'efficiency': round(diag.efficiency, 4),
            'confidence': round(diag.confidence, 4),
        }
    
    results['reflection'] = reflection_metrics
    if reflection_metrics:
        results['avg_confidence'] = np.mean([m['confidence'] for m in reflection_metrics.values()])
        results['avg_curvature'] = np.mean([m['mean_curvature'] for m in reflection_metrics.values()])
    
    return results
