"""Read-only vector-space diagnostics for FCF concept space.

Functions:
  - check_antonym_collapse(cs, sp, antonym_pairs): log cosine of known
    antonym pairs (should be low; high means STDP collapsed opposites).
  - detect_concept_clusters(cs, n_clusters=100): K-means + gap detection
    (find dense token regions whose centroid is far from any existing token).
  - prune_near_duplicates(cs, threshold=0.95): merge near-identical vectors.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import math


ANTONYM_PAIRS = [
    ("да", "нет"), ("хороший", "плохой"), ("большой", "маленький"),
    ("высокий", "низкий"), ("правда", "ложь"), ("жизнь", "смерть"),
    ("война", "мир"), ("любовь", "ненависть"), ("всегда", "никогда"),
    ("начало", "конец"), ("новый", "старый"), ("белый", "чёрный"),
    ("день", "ночь"), ("добро", "зло"),
]


def check_antonym_collapse(cs, sp, antonym_pairs=None):
    """Compute cosine for each antonym pair. Returns dict: pair_key -> cos.

    A cos > 0.6 means the antonym vectors have been pulled together by
    STDP — they share similar contexts but should be opposite.
    """
    if antonym_pairs is None:
        antonym_pairs = ANTONYM_PAIRS
    results = {}
    for a, b in antonym_pairs:
        id_a = sp.PieceToId('▁' + a)
        if id_a < 0:
            id_a = sp.PieceToId(a)
        id_b = sp.PieceToId('▁' + b)
        if id_b < 0:
            id_b = sp.PieceToId(b)
        if id_a < 0 or id_b < 0:
            continue
        va = cs.concept_vector(id_a)
        vb = cs.concept_vector(id_b)
        if va is None or vb is None:
            continue
        cos = float(np.dot(va, vb))
        results[f"{a}/{b}"] = cos
    return results


def antonym_collapse_rate(cs, sp, antonym_pairs=None):
    """Return fraction of antonym pairs with cos > 0.5."""
    results = check_antonym_collapse(cs, sp, antonym_pairs)
    if not results:
        return 0.0
    return sum(1 for v in results.values() if abs(v) > 0.5) / len(results)


def detect_concept_clusters(cs, n_clusters=100, min_cosine=0.3):
    """Cluster concept vectors and detect semantic gaps.

    A gap is a cluster whose centroid has cosine distance > min_cosine
    from every existing token — it represents a missing or under-learned
    concept region.

    Returns:
        clusters: list of (centroid_id, cluster_size, max_cos_to_tokens)
        where centroid_id is None (conceptual — not linked to any token).
    """
    cids = sorted(cs.concept_vectors.keys())
    if len(cids) < n_clusters:
        n_clusters = max(10, len(cids) // 2)

    vecs = np.array([cs.concept_vectors[c] for c in cids], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    vecs /= norms

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    labels = kmeans.fit_predict(vecs)

    clusters = []
    for i in range(n_clusters):
        mask = labels == i
        size = int(mask.sum())
        if size == 0:
            continue
        centroid = kmeans.cluster_centers_[i].astype(np.float32)
        cn = np.linalg.norm(centroid)
        if cn > 1e-10:
            centroid /= cn

        # Max cosine between centroid and any existing token in this cluster
        cluster_vecs = vecs[mask]
        sims = cluster_vecs @ centroid
        max_cos = float(sims.max())
        clusters.append({
            'cluster_id': i,
            'size': size,
            'max_cos_to_tokens': max_cos,
            'is_gap': max_cos < min_cosine,
        })

    return clusters


def prune_near_duplicates(cs, threshold=0.95):
    """Log near-duplicate concept pairs (cos > threshold).

    Returns list of (cid_a, cid_b, cos) for pairs that are effectively
    identical.  Does NOT modify the concept space.
    """
    cids = sorted(cs.concept_vectors.keys())
    n = len(cids)
    if n < 2:
        return []

    vecs = np.array([cs.concept_vectors[c] for c in cids], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    vecs /= norms

    sim_matrix = vecs @ vecs.T
    triu_i, triu_j = np.triu_indices(n, k=1)
    sims = sim_matrix[triu_i, triu_j]
    mask = sims > threshold
    if not np.any(mask):
        return []

    pairs = []
    for idx in np.where(mask)[0]:
        i, j = triu_i[idx], triu_j[idx]
        pairs.append((cids[i], cids[j], float(sims[idx])))
    # Sort by cos descending
    pairs.sort(key=lambda x: -x[2])
    return pairs
