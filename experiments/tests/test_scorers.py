"""
Test: ConceptScorer + ContradictionScorer on real data.
"""
import torch, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.trajectory_store import TrajectoryStore
from eva.symbolic.concept_miner import ConceptScorer
from eva.symbolic.contradiction_filter import ContradictionScorer, ContradictionLabels

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cv = CharacterVocab(); V = cv.vocab_size; D = 128
print(f'Device: {DEVICE}')

ut = UnifiedMultidimensionalTransformer(vocab_size=V, coord_dim=D, max_levels=8,
    total_heads=32, num_layers=6, d_ff=128).to(DEVICE)
ckpt = torch.load('checkpoints/symbolic/full_latest.pt', map_location=DEVICE, weights_only=False)
sd = ckpt['ut']
for k in list(sd.keys()):
    if k in ut.state_dict() and sd[k].shape != ut.state_dict()[k].shape:
        del sd[k]
ut.load_state_dict(sd, strict=False)
ut.eval()
print(f'Loaded step {ckpt.get("step", "?")}')

store = TrajectoryStore(max_trajectories=50000)
store_path = 'checkpoints/symbolic/trajectory_store_full.pkl'
if os.path.exists(store_path):
    store.load(store_path)
print(f'Store: {store.total_stored} trajectories')

concept_scorer = ConceptScorer(trajectory_store=store if store.total_stored > 0 else None)
contra_scorer = ContradictionScorer(trajectory_store=store if store.total_stored > 0 else None)

test_texts = [
    "Привет мир",
    "Солнце встаёт на востоке",
    "Война и мир — великий роман",
]

print('\n' + '=' * 70)
print('TEST: ConceptScorer + ContradictionScorer (word-level)')
print('=' * 70)

with torch.no_grad():
    for text in test_texts:
        print(f'\n--- {text} ---')
        ids = cv.encode_with_boundaries(text)
        if not ids:
            continue

        inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        h, scores = ut(inp, return_scores=True)
        traj = h[0].cpu().numpy()

        # Boundaries
        w_open = cv.WORD_OPEN_IDX; w_close = cv.WORD_CLOSE_IDX
        boundaries = []; in_word = False; start = 0
        for i, tid in enumerate(ids):
            if tid == w_open: in_word = True; start = i + 1
            elif tid == w_close and in_word: boundaries.append((start, i)); in_word = False

        if not boundaries:
            print('  (no words)')
            continue

        # Word centroids
        wc = np.array([traj[s:e].mean(axis=0) for s, e in boundaries])
        sc = traj.mean(axis=0)

        # --- ConceptScorer ---
        cl = concept_scorer.score_trajectory(wc, np.array([wc[i+1]-wc[i] for i in range(len(wc)-1)]) if len(wc)>1 else np.zeros((0,D)), sc)
        print(f'  Concepts: {cl.n_concepts}/{len(wc)} score~{cl.avg_score:.3f}')
        for i in range(len(wc)):
            print(f'    [{i}] s={cl.scores[i]:.3f} {cl.types[i]} {"*" if cl.scores[i]>0.6 else ""}')

        # --- ContradictionScorer ---
        ctl = contra_scorer.score_trajectory(traj, boundaries, sc, wc, ids)
        contra_positions = np.where(ctl.probs > 0.5)[0]
        print(f'  Contradictions: {ctl.n_contradictions} positions')
        for pos in contra_positions:
            tname = ctl.TYPE_NAMES[ctl.types[pos].argmax()]
            char_at_pos = cv.idx_to_char(ids[pos]) if pos < len(ids) else '?'
            print(f'    Pos {pos} ("{char_at_pos}"): {tname} (p={ctl.probs[pos]:.3f})')

        # Show full contradiction probs for reference
        non_zero = np.where(ctl.probs > 0.01)[0]
        if len(non_zero) <= 15:
            for p in non_zero:
                ci = ctl.types[p].argmax()
                print(f'      [{p}] tot={ctl.probs[p]:.3f} type={ctl.TYPE_NAMES[ci]}')

print('\nDone.')
