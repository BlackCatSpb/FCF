"""Quick check PC1/PC2 across checkpoints."""
import glob, os, numpy as np
from sklearn.decomposition import PCA

data_dir = r'C:\Users\black\OneDrive\Desktop\FCF\real_data'
files = sorted(glob.glob(os.path.join(data_dir, 'concept_space_*k.codes.npz')))
for fp in files:
    base = os.path.basename(fp).replace('.codes.npz', '').replace('concept_space_', '')
    f = np.load(fp)
    n = min(5000, len(f['cids']))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(f['cids']), n, replace=False)
    vecs = f['codes'][idx] @ f['basis']
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    vecs /= norms
    pca = PCA(2, random_state=0).fit(vecs)
    print(f'  {base}: PC1={pca.explained_variance_ratio_[0]:.4f} PC2={pca.explained_variance_ratio_[1]:.4f}')
