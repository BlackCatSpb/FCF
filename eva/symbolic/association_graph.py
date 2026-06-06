"""
AssociationGraph — иерархическая ассоциативная сеть.
Активация распространяется иерархически: токен → концепт → мета → концепты → токены.
"""
import math
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix


def _aggregate_row(csr_mat, row_idx, V):
    """Get transition distribution as dictionary from sparse matrix row."""
    if row_idx >= csr_mat.shape[0]:
        return {}
    row = csr_mat[row_idx].tocoo()
    return {int(col): data for col, data in zip(row.col, row.data) if col < V}


class AssociationGraph:
    """
    Трёхуровневая иерархия:
      Level 0: токены (WORD_STARTER)
      Level 1: концепты (SVD + k-means кластеры токенов)
      Level 2: мета-концепты (кластеры концептов по transition patterns)
    
    Связи между уровнями:
      P(concept_j | concept_i) — вероятности перехода между концептами
      similar(concept_i, concept_j) — концепты в одной мета
      members(c) → tokens — токены, входящие в концепт
    """

    def __init__(self, n_clusters=48, n_metas=12, config=None):
        if config is None:
            from eva.symbolic.auto_config import AutoConfig
            config = AutoConfig()
        self.config = config
        self.n_clusters = n_clusters if n_clusters is not None else config.n_clusters
        self.n_metas = n_metas if n_metas is not None else config.n_metas
        
        self.tid_to_cid = {}        # token_id -> concept_id (level 1)
        self.cid_to_tids = {}       # concept_id -> list[token_id]
        self.cid_to_mid = {}        # concept_id -> meta_id (level 2)
        self.mid_to_cids = {}       # meta_id -> list[concept_id]
        
        self.cid_label = {}         # concept_id -> readable name
        self.mid_label = {}         # meta_id -> readable name
        
        # Concept profiles: discriminative weight of each member token
        self.cid_profiles = {}      # concept_id -> {token_id: weight}
        self.cid_top_tokens = {}    # concept_id -> [(token_id, weight)] sorted desc
        
        # Concept Vector Space: SVD embeddings for DIRECT composition
        self.starter_list = []      # list[token_id] in embedding order
        self.starter_embeddings = None  # ndarray (n_starters, ndim) — SVD embedding
        self.starter_token_to_idx = {}  # token_id -> index in starter_list
        self.centroid_vectors = {}      # concept_id -> centroid SVD vector
        
        # Transition: P(c_j | c_i)
        self.transition = {}        # (ci, cj) -> log_prob
        self.transition_ci = {}     # ci -> [(cj, log_prob)]  (outgoing)
        self.transition_cj = {}     # cj -> [(ci, log_prob)]  (incoming)
        self.pmi = {}               # (ci, cj) -> PMI (log ratio over background)
        self.pmi_ci = {}            # ci -> [(cj, pmi)] sorted by PMI desc
        
        self.L1_OFFSET = self.config.bpe_limit
        self.L2_OFFSET = self.config.bpe_limit + self.n_clusters

    def _cluster_l1_hdbscan(self, emb_norm, starters):
        """L1 concept clustering via HDBSCAN (density-based, auto cluster count)."""
        import hdbscan
        min_size = max(3, int(len(starters) * self.config.hdbscan_min_cluster_ratio))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_epsilon=0.3,
            gen_min_span_tree=True,
        )
        labels = clusterer.fit_predict(emb_norm)
        n_clusters = len(set(l for l in labels if l >= 0))
        n_noise = sum(1 for l in labels if l < 0)
        print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise tokens")
        return labels, clusterer

    def _cluster_l1_kmeans(self, emb_norm, starters):
        """L1 concept clustering via KMeans (fallback)."""
        from sklearn.cluster import KMeans
        k = min(self.n_clusters, len(starters)//2)
        km = KMeans(n_clusters=k, random_state=self.config.random_state, n_init='auto')
        labels = km.fit_predict(emb_norm)
        print(f"  KMeans: {k} clusters")
        return labels, km

    def _cluster_l2_louvain(self):
        """L2 meta-concept clustering via Louvain community detection on transition graph."""
        import networkx as nx
        from community import community_louvain
        G = nx.Graph()
        for (ci, cj), lp in self.transition.items():
            w = math.exp(lp)
            if w > 1e-6:
                G.add_edge(ci, cj, weight=w)
        partition = community_louvain.best_partition(G, weight='weight')
        meta_labels = {}
        for node, community_id in partition.items():
            meta_labels[node] = community_id
        n_metas = len(set(meta_labels.values()))
        print(f"  Louvain: {n_metas} meta-communities")
        return meta_labels, n_metas

    def _cluster_l2_kmeans(self, cid_list):
        """L2 meta-concept clustering via KMeans (fallback)."""
        from sklearn.cluster import KMeans
        nC = len(cid_list)
        cid_to_row = {c: i for i, c in enumerate(cid_list)}
        cmat = np.zeros((nC, nC), dtype=np.float32)
        for (ci, cj), lp in self.transition.items():
            ri, rj = cid_to_row.get(ci), cid_to_row.get(cj)
            if ri is not None and rj is not None:
                cmat[ri, rj] = math.exp(lp)
        km2 = min(self.n_metas, max(2, nC//2))
        km_meta = KMeans(n_clusters=km2, random_state=42, n_init='auto')
        km_labels = km_meta.fit_predict(cmat)
        meta_labels = {cid: km_labels[i] for i, cid in enumerate(cid_list)}
        n_metas = km2
        print(f"  KMeans: {n_metas} meta-clusters")
        return meta_labels, n_metas

    def build(self, log_prob_csr, token_type, decode_fn=None):
        V = min(self.config.bpe_limit, log_prob_csr.shape[0], len(token_type))
        starters = [t for t in range(V) if token_type[t] == 2]
        print(f"Building AssociationGraph: {len(starters)} word starters")
        
        # --- Level 1: Concept clusters via SVD on OUTGOING transitions ---
        # Outgoing = what token PREDICTS = P(next | token)
        # Это семантически богаче, чем incoming (что предшествует токену)
        from scipy.sparse import vstack
        from sklearn.decomposition import TruncatedSVD
        
        blocks = []
        for tid in starters:
            blocks.append(log_prob_csr[tid] if tid < log_prob_csr.shape[0] else csr_matrix((1, log_prob_csr.shape[1])))
        mat = vstack(blocks, format='csr')
        
        ndim = min(self.config.svd_dim, len(starters)-1, mat.shape[1]-1)
        svd = TruncatedSVD(n_components=ndim, random_state=self.config.random_state)
        emb = svd.fit_transform(mat)
        
        # Normalize embeddings to unit length for angular clustering
        emb_norm = emb.copy()
        norms = np.linalg.norm(emb_norm, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_norm /= norms

        # L1 clustering: HDBSCAN (default) or KMeans (fallback)
        method_l1 = self.config.cluster_method_l1
        if method_l1 == 'hdbscan':
            labels, clusterer = self._cluster_l1_hdbscan(emb_norm, starters)
        else:
            labels, clusterer = self._cluster_l1_kmeans(emb_norm, starters)

        # Handle noise tokens from HDBSCAN: assign to nearest cluster
        if method_l1 == 'hdbscan' and -1 in labels:
            unique_labels = sorted(set(l for l in labels if l >= 0))
            if unique_labels:
                centroids = {}
                for ul in unique_labels:
                    mask = labels == ul
                    centroids[ul] = emb_norm[mask].mean(axis=0)
                for i in range(len(labels)):
                    if labels[i] < 0:
                        best_clust = min(unique_labels,
                            key=lambda ul: np.linalg.norm(emb_norm[i] - centroids[ul]))
                        labels[i] = best_clust

        # Remap labels to contiguous 0..K-1
        unique_labels = sorted(set(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        n_clusters_found = len(unique_labels)
        
        self.cid_to_tids = defaultdict(list)
        self.cid_profiles = {}
        
        # Store SVD embeddings as concept vector space
        self.starter_list = starters
        self.starter_embeddings = emb  # (n_starters, ndim)
        self.starter_token_to_idx = {tid: i for i, tid in enumerate(starters)}
        self.n_clusters = n_clusters_found  # update actual cluster count
        
        for i, tid in enumerate(starters):
            cid = self.L1_OFFSET + int(labels[i])
            self.tid_to_cid[tid] = cid
            self.cid_to_tids[cid].append(tid)
        print(f"  Level 1: {len(self.cid_to_tids)} concept clusters")
        
        # Concept names + profiles: distance from centroid
        for c in range(n_clusters_found):
            cid = self.L1_OFFSET + c
            members = self.cid_to_tids[cid]
            centroid = emb_norm[labels == c].mean(axis=0) if method_l1 == 'hdbscan' else clusterer.cluster_centers_[c]
            self.centroid_vectors[cid] = centroid
            
            best = min(members, key=lambda t: np.linalg.norm(
                emb[starters.index(t)] - centroid))
            self.cid_label[cid] = decode_fn([best]).strip() if decode_fn else f'C{c}'
            
            # Profile: weight = exp(-distance_to_centroid), normalized
            profile = {}
            dists = []
            for t in members:
                idx = starters.index(t)
                dist = float(np.linalg.norm(emb[idx] - centroid))
                w = math.exp(-dist * 2.0)
                profile[t] = w
                dists.append((t, w))
            max_w = max(w for _, w in dists) if dists else 1.0
            profile = {t: w / max_w for t, w in profile.items()}
            self.cid_profiles[cid] = profile
            self.cid_top_tokens[cid] = sorted(profile.items(), key=lambda x: -x[1])
        
        # --- Concept transition matrix ---
        trans_count = defaultdict(lambda: defaultdict(float))
        src_total = defaultdict(float)
        
        for tid in range(V):
            ci = self.tid_to_cid.get(tid)
            if ci is None: continue
            trans = _aggregate_row(log_prob_csr, tid, V)
            for col, prob in trans.items():
                cj = self.tid_to_cid.get(col)
                if cj is None: continue
                trans_count[ci][cj] += math.exp(prob)
                src_total[ci] += math.exp(prob)
        
        self.transition_ci = defaultdict(list)
        self.transition_cj = defaultdict(list)
        
        for ci in trans_count:
            total = max(src_total.get(ci, 1e-10), 1e-10)
            for cj, cnt in trans_count[ci].items():
                prob = cnt / total
                lp = math.log(max(prob, 1e-10))
                self.transition[(ci, cj)] = lp
                self.transition_ci[ci].append((cj, lp))
                self.transition_cj[cj].append((ci, lp))
        
        # Sort by probability
        for ci in self.transition_ci:
            self.transition_ci[ci].sort(key=lambda x: -x[1])
        for cj in self.transition_cj:
            self.transition_cj[cj].sort(key=lambda x: -x[1])
        
        nz = sum(len(v) for v in self.transition_ci.values())
        print(f"  Concept transitions: {nz}")
        
        # --- Level 2: Meta-concepts ---
        # Louvain community detection on transition graph (default)
        # or KMeans on transition matrix (fallback)
        cid_list = sorted(self.cid_to_tids.keys())

        method_l2 = self.config.cluster_method_l2
        if method_l2 == 'louvain':
            meta_labels_raw, n_metas_found = self._cluster_l2_louvain()
        else:
            meta_labels_raw, n_metas_found = self._cluster_l2_kmeans(cid_list)
        
        self.mid_to_cids = defaultdict(list)
        for cid in cid_list:
            raw_mid = meta_labels_raw.get(cid, 0)
            mid = self.L2_OFFSET + int(raw_mid)
            self.cid_to_mid[cid] = mid
            self.mid_to_cids[mid].append(cid)
            self.mid_label[mid] = f'M{raw_mid}'
        
        print(f"  Level 2: {len(self.mid_to_cids)} meta-concepts")
        
        # --- PMI (Pointwise Mutual Information) ---
        # PMI(ci, cj) = log(P(cj|ci) / P(cj))
        # Высокий PMI = ассоциация, а не частотность
        cj_marginal = defaultdict(float)
        for ci, targets in trans_count.items():
            total = max(src_total.get(ci, 1e-10), 1e-10)
            for cj, cnt in targets.items():
                cj_marginal[cj] += cnt / total
        n_concepts = len(cid_list)
        for cj in cj_marginal:
            cj_marginal[cj] /= max(n_concepts, 1)
        
        self.pmi_ci = defaultdict(list)
        for (ci, cj), lp in self.transition.items():
            p_cj = max(cj_marginal.get(cj, 1e-10), 1e-10)
            pmi_val = lp - math.log(p_cj)
            self.pmi[(ci, cj)] = pmi_val
            self.pmi_ci[ci].append((cj, pmi_val))
        
        for ci in self.pmi_ci:
            self.pmi_ci[ci].sort(key=lambda x: -x[1])  # sort by PMI desc
        
        self._decode_fn = decode_fn  # keep for save/load
        self._heads_csr = log_prob_csr  # keep for transition rebuilding
        self._print_summary(decode_fn)
        return self

    def _print_summary(self, decode_fn=None, limit=5):
        for cid in sorted(self.cid_to_tids.keys())[:limit]:
            name = self.cid_label.get(cid, '?')
            members = self.cid_to_tids[cid]
            mid = self.cid_to_mid.get(cid)
            mname = self.mid_label.get(mid, '?') if mid else '?'
            txt = []
            for t in members[:5]:
                if decode_fn:
                    txt.append(decode_fn([t]).strip())
            print(f"    C{cid - self.L1_OFFSET} ({name:10s}) [{len(members):3d}] "
                  f"meta={mname}: {txt}")

    def online_update(self, new_embeddings=None, starter_tids=None):
        """
        Online concept update via BIRCH.
        Called after SVD training to incrementally adjust concepts
        without full recomputation.
        
        Args:
            new_embeddings: (n, ndim) array of updated SVD vectors
            starter_tids: list of token IDs corresponding to new_embeddings rows
        """
        from sklearn.cluster import Birch
        emb = self.starter_embeddings
        if new_embeddings is not None and starter_tids is not None:
            emb = new_embeddings

        emb_norm = emb.copy()
        norms = np.linalg.norm(emb_norm, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_norm /= norms

        birch = Birch(
            threshold=self.config.birch_threshold,
            n_clusters=None,
            compute_labels=True,
        )
        birch_labels = birch.fit_predict(emb_norm)

        # Remap BIRCH labels to contiguous L1_OFFSET+c
        unique = sorted(set(birch_labels))
        label_map = {old: new for new, old in enumerate(unique)}
        remapped = np.array([label_map[l] for l in birch_labels])
        n_new = len(unique)

        # Update concept structures
        old_cids = set(self.cid_to_tids.keys())
        self.cid_to_tids = defaultdict(list)
        self.tid_to_cid = {}
        self.centroid_vectors = {}
        self.cid_profiles = {}
        self.cid_top_tokens = {}
        self.cid_label = {}

        tids = starter_tids if starter_tids is not None else self.starter_list
        for i, tid in enumerate(tids):
            cid = self.L1_OFFSET + int(remapped[i])
            self.tid_to_cid[tid] = cid
            self.cid_to_tids[cid].append(tid)

        for c in range(n_new):
            cid = self.L1_OFFSET + c
            members = self.cid_to_tids[cid]
            mask = remapped == c
            centroid = emb_norm[mask].mean(axis=0)
            self.centroid_vectors[cid] = centroid
            best = min(members, key=lambda t: np.linalg.norm(
                emb[tids.index(t)] - centroid))
            self.cid_label[cid] = 'B' + str(c)

            profile = {}
            dists = []
            for t in members:
                idx = tids.index(t)
                dist = float(np.linalg.norm(emb[idx] - centroid))
                w = math.exp(-dist * 2.0)
                profile[t] = w
                dists.append((t, w))
            max_w = max(w for _, w in dists) if dists else 1.0
            profile = {t: w / max_w for t, w in profile.items()}
            self.cid_profiles[cid] = profile
            self.cid_top_tokens[cid] = sorted(profile.items(), key=lambda x: -x[1])

        # Rebuild transition matrix with new concept IDs
        self._rebuild_transitions()
        print(f"  BIRCH online update: {n_new} new concepts ({len(old_cids)} old)")

    def _rebuild_transitions(self):
        """Rebuild concept transition/PPMI after reclustering."""
        self.transition = {}
        self.transition_ci = defaultdict(list)
        self.transition_cj = defaultdict(list)
        self.pmi = {}
        self.pmi_ci = defaultdict(list)

        # Collect raw transitions from token-level log_prob_csr
        if not hasattr(self, '_decode_fn'):
            return
        V = self.L1_OFFSET
        trans_count = defaultdict(lambda: defaultdict(float))
        src_total = defaultdict(float)
        for tid in range(V):
            ci = self.tid_to_cid.get(tid)
            if ci is None:
                continue
            # Use stored CSR from heads if available
            if hasattr(self, '_heads_csr'):
                trans = _aggregate_row(self._heads_csr, tid, V)
                for col, prob in trans.items():
                    cj = self.tid_to_cid.get(col)
                    if cj is None:
                        continue
                    trans_count[ci][cj] += math.exp(prob)
                    src_total[ci] += math.exp(prob)

        for ci in trans_count:
            total = max(src_total.get(ci, 1e-10), 1e-10)
            for cj, cnt in trans_count[ci].items():
                prob = cnt / total
                lp = math.log(max(prob, 1e-10))
                self.transition[(ci, cj)] = lp
                self.transition_ci[ci].append((cj, lp))
                self.transition_cj[cj].append((ci, lp))

        for ci in self.transition_ci:
            self.transition_ci[ci].sort(key=lambda x: -x[1])
        for cj in self.transition_cj:
            self.transition_cj[cj].sort(key=lambda x: -x[1])

    def save(self, path_prefix):
        """Сохраняет AssociationGraph: pickle (данные) + JSON (разметка).
        path_prefix: без расширения, создаст .pkl и .json
        """
        import pickle, json, os
        decode_fn = getattr(self, '_decode_fn', None)
        pkl_path = path_prefix + '.pkl'
        json_path = path_prefix + '.json'
        
        data = {
            'n_clusters': self.n_clusters,
            'n_metas': self.n_metas,
            'L1_OFFSET': self.L1_OFFSET,
            'L2_OFFSET': self.L2_OFFSET,
            'cluster_method_l1': self.config.cluster_method_l1,
            'cluster_method_l2': self.config.cluster_method_l2,
            'starter_list': self.starter_list,
            'starter_embeddings': self.starter_embeddings,
            'starter_token_to_idx': self.starter_token_to_idx,
            'tid_to_cid': self.tid_to_cid,
            'cid_to_tids': dict(self.cid_to_tids),
            'cid_to_mid': self.cid_to_mid,
            'mid_to_cids': dict(self.mid_to_cids),
            'cid_label': self.cid_label,
            'mid_label': self.mid_label,
            'cid_profiles': self.cid_profiles,
            'cid_top_tokens': self.cid_top_tokens,
            'centroid_vectors': self.centroid_vectors,
            'transition': self.transition,
            'transition_ci': dict(self.transition_ci),
            'transition_cj': dict(self.transition_cj),
            'pmi': self.pmi,
            'pmi_ci': dict(self.pmi_ci),
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f, protocol=5)
        
        # Human-readable JSON markup
        markup = {
            'n_clusters': self.n_clusters,
            'n_metas': self.n_metas,
            'concepts': {},
            'metas': {},
            'transitions': {},
        }
        for cid in sorted(self.cid_to_tids.keys()):
            c = cid - self.L1_OFFSET
            name = self.cid_label.get(cid, f'C{c}')
            mid = self.cid_to_mid.get(cid)
            mname = self.mid_label.get(mid, '?') if mid else '?'
            members = []
            for t in self.cid_to_tids[cid][:10]:
                txt = decode_fn([t]).strip() if decode_fn else str(t)
                members.append({'tid': t, 'text': txt})
            top_trans = []
            for cj, lp in self.transition_ci.get(cid, [])[:5]:
                cj_name = self.cid_label.get(cj, f'C{cj-self.L1_OFFSET}')
                top_trans.append({'to': cj_name, 'prob': f'{math.exp(lp):.4f}'})
            markup['concepts'][name] = {
                'cluster': c,
                'size': len(self.cid_to_tids[cid]),
                'meta': mname,
                'top_members': members,
                'top_transitions': top_trans,
            }
        for mid in sorted(self.mid_to_cids.keys()):
            mname = self.mid_label.get(mid, f'M{mid-self.L2_OFFSET}')
            members = []
            for cid in self.mid_to_cids[mid]:
                cname = self.cid_label.get(cid, f'C{cid-self.L1_OFFSET}')
                members.append(cname)
            markup['metas'][mname] = {
                'size': len(self.mid_to_cids[mid]),
                'concepts': members,
            }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(markup, f, ensure_ascii=False, indent=2)
        
        print(f"AssociationGraph saved: {pkl_path} + {json_path}")
    
    @classmethod
    def load(cls, path_prefix, n_clusters=48, n_metas=12):
        """Загружает AssociationGraph из pickle."""
        import pickle
        pkl_path = path_prefix + '.pkl'
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        ag = cls(n_clusters=data.get('n_clusters', n_clusters),
                 n_metas=data.get('n_metas', n_metas))
        for key, val in data.items():
            if key in ('n_clusters', 'n_metas'):
                continue
            setattr(ag, key, val)
        # Restore defaultdicts
        if not isinstance(ag.cid_to_tids, defaultdict):
            ag.cid_to_tids = defaultdict(list, ag.cid_to_tids)
        if not isinstance(ag.mid_to_cids, defaultdict):
            ag.mid_to_cids = defaultdict(list, ag.mid_to_cids)
        if not isinstance(ag.transition_ci, defaultdict):
            ag.transition_ci = defaultdict(list, ag.transition_ci)
        if not isinstance(ag.transition_cj, defaultdict):
            ag.transition_cj = defaultdict(list, ag.transition_cj)
        if not isinstance(ag.pmi_ci, defaultdict):
            ag.pmi_ci = defaultdict(list, ag.pmi_ci)
        return ag
    
    # ---- Query API ----
    def get_concept(self, tid):
        return self.tid_to_cid.get(tid)

    def get_meta(self, cid):
        return self.cid_to_mid.get(cid)

    def concept_name(self, cid):
        return self.cid_label.get(cid, str(cid))

    def meta_name(self, mid):
        return self.mid_label.get(mid, str(mid))

    def concept_members(self, cid):
        return self.cid_to_tids.get(cid, [])

    def meta_members(self, mid):
        return self.mid_to_cids.get(mid, [])

    def transition_out(self, cid):
        """Concepts that follow cid, sorted by log_prob."""
        return self.transition_ci.get(cid, [])

    def transition_in(self, cid):
        """Concepts that precede cid, sorted by log_prob."""
        return self.transition_cj.get(cid, [])

    # ---- Concept Vector Space API ----
    def token_to_vector(self, tid):
        """Возвращает SVD-вектор type-2 токена (32-dim). None если не type-2."""
        idx = self.starter_token_to_idx.get(tid)
        if idx is None or self.starter_embeddings is None:
            return None
        return self.starter_embeddings[idx]
    
    def word_to_vector(self, word, hv):
        """Конвертирует слово в concept vector через type-2 токен."""
        bpe = hv.encode(word)
        if not bpe:
            return None
        first = bpe[0]
        first_decoded = hv.decode([first]).strip()
        for t in range(4096):
            if t < len(hv.token_type) and hv.token_type[t] == 2:
                if hv.decode([t]).strip() == first_decoded:
                    return self.token_to_vector(t)
        return None
    
    def compose_vectors(self, vectors, weights=None):
        """
        Композиция concept vectors: взвешенная интерполяция.
        vectors: list of (32,) ndarrays
        weights: list of floats (default: equal)
        Returns: composed (32,) ndarray
        """
        if not vectors:
            return None
        if weights is None:
            weights = [1.0 / len(vectors)] * len(vectors)
        w = np.array(weights, dtype=np.float32)
        w = w / w.sum()
        result = np.zeros_like(vectors[0])
        for v, weight in zip(vectors, w):
            result += v * weight
        return result
    
    def nearest_concepts(self, vector, top_k=3):
        """
        Находит ближайшие концепты к vector в SVD-пространстве.
        Returns: [(concept_id, distance)] sorted by distance asc.
        """
        if vector is None:
            return []
        dists = []
        for cid, centroid in self.centroid_vectors.items():
            d = float(np.linalg.norm(vector - centroid))
            dists.append((cid, d))
        dists.sort(key=lambda x: x[1])
        return dists[:top_k]
    
    def generate_from_composition(self, seed_word, modifier_word, hv, top_k=5):
        """
        Генерация из композиции концептов.
        seed_word: основное слово (напр. 'корабль')
        modifier_word: модификатор (напр. 'летучий')
        Returns: tokens — список type-2 токенов из ближайшего к композиции концепта.
        """
        v1 = self.word_to_vector(seed_word, hv)
        v2 = self.word_to_vector(modifier_word, hv)
        if v1 is None or v2 is None:
            return []
        
        # Composition: weighted, seed dominant
        composed = self.compose_vectors([v1, v2], [0.7, 0.3])
        
        nearest = self.nearest_concepts(composed, top_k=3)
        if not nearest:
            return []
        
        target_cid = nearest[0][0]
        dist = nearest[0][1]
        
        # Get discriminative tokens from nearest concept
        essence = self.get_profile(target_cid, top_k=top_k)
        return essence, target_cid, dist, composed
    
    def get_profile(self, cid, top_k=20, min_weight=0.3):
        """
        Возвращает токены концепта, взвешенные по дискриминативности (centroid proximity).
        Только токены с weight >= min_weight, top-k.
        Это "сущность" концепта — то, что его ОПРЕДЕЛЯЕТ.
        """
        profile = self.cid_profiles.get(cid, {})
        if not profile:
            return self.cid_to_tids.get(cid, [])
        sorted_items = sorted(profile.items(), key=lambda x: -x[1])
        result = [t for t, w in sorted_items if w >= min_weight][:top_k]
        return result
    
    # ---- API for ConstrainedDecoder ----
    def get_members(self, cid):
        """Return token IDs in this concept (like ConceptGraph.get_members)."""
        return self.cid_to_tids.get(cid, [])

    def concept_scores(self, cid, V_size=4101):
        """Boost scores for tokens in a concept (like ConceptGraph.concept_scores)."""
        boost = np.zeros(min(V_size, 4096), dtype=np.float32)
        members = self.cid_to_tids.get(cid, [])
        for tid in members:
            if tid < len(boost):
                boost[tid] = 2.0
        if V_size > len(boost):
            boost = np.pad(boost, (0, V_size - len(boost)), 'constant')
        return boost

    def predict_next(self, prev_token_id):
        """(Compatibility with ConceptGraph API) — not the primary API."""
        cid = self.tid_to_cid.get(prev_token_id)
        if cid is None:
            return []
        return [(cj, lp) for cj, lp in self.transition_ci.get(cid, [])]


    # ---- Activation propagation ----
    def activate(self, seed_token_id, max_depth=3, decay=0.5):
        """
        Иерархическое распространение активации через PMI-ассоциации.
        
        Ключевое отличие: использует PMI вместо сырой вероятности перехода.
        PMI(ci, cj) = log(P(cj|ci) / P(cj))
        Это даёт ИСТИННЫЕ ассоциации, а не частотные шумы.
        
        Путь: token (L0) → concept (L1) → PMI-top concepts (L1) 
              → meta (L2) → sibling concepts (L1) → tokens (L0)
        """
        activation = {}
        
        def _add(nid, energy):
            activation[nid] = activation.get(nid, 0) + energy
        
        seed_cid = self.tid_to_cid.get(seed_token_id)
        if seed_cid is None:
            _add(seed_token_id, 1.0)
            return activation
        
        _add(seed_token_id, 1.0)
        _add(seed_cid, 1.0)
        
        e1 = decay
        e2 = decay ** 2 if max_depth >= 2 else 0
        e3 = decay ** 3 if max_depth >= 3 else 0
        
        # Depth 1: PMI-ассоциации (forward: что следует за seed)
        # Берём top-10 по PMI (истинные ассоциации, а не частотные)
        for cj, pmi_val in self.pmi_ci.get(seed_cid, [])[:10]:
            # PMI > 0 = genuine association
            if pmi_val <= 0:
                continue
            # Normalize PMI to [0, 1] range
            w = min(1.0, pmi_val / max(abs(pmi_val) + 1, 1))
            _add(cj, e1 * w)
            for t in self.cid_to_tids.get(cj, []):
                _add(t, e1 * w * 0.3)
        
        # Depth 1: Обратные PMI-ассоциации (что предшествует seed)
        # Ищем: какие концепты имеют высокий PMI к seed_cid?
        reverse_pmi = []
        for ci, targets in self.pmi_ci.items():
            for cj, pmi_val in targets:
                if cj == seed_cid and pmi_val > 0.5:
                    reverse_pmi.append((ci, pmi_val))
                    break
        reverse_pmi.sort(key=lambda x: -x[1])
        for ci, pmi_val in reverse_pmi[:5]:
            w = min(1.0, pmi_val / max(abs(pmi_val) + 1, 1))
            _add(ci, e1 * w * 0.4)
            for t in self.cid_to_tids.get(ci, []):
                _add(t, e1 * w * 0.1)
        
        # Depth 1-2: Meta-concept (обобщение)
        mid = self.cid_to_mid.get(seed_cid)
        if mid:
            _add(mid, e1)
            if e2 > 0:
                for cid2 in self.mid_to_cids.get(mid, []):
                    if cid2 != seed_cid:
                        _add(cid2, e2 * 0.5)
                        for t in self.cid_to_tids.get(cid2, []):
                            _add(t, e3 * 0.3)
        
        # Depth 2: 2-hop PMI (ассоциации через ассоциации)
        if e2 > 0:
            for cj1, pmi1 in self.pmi_ci.get(seed_cid, [])[:5]:
                if pmi1 <= 0:
                    continue
                for cj2, pmi2 in self.pmi_ci.get(cj1, [])[:3]:
                    if pmi2 <= 0:
                        continue
                    w = min(1.0, (abs(pmi1) + abs(pmi2)) / 4)
                    _add(cj2, e2 * w * 0.2)
                    for t in self.cid_to_tids.get(cj2, []):
                        _add(t, e3 * w * 0.05)
        
        return activation

    def activate_text(self, text, decode_fn=None):
        """Activate from a text string."""
        if not isinstance(text, str):
            text = str(text)
        tl = text.strip().lower()
        
        # Find token IDs matching this text
        for tid in range(4096):
            try:
                if decode_fn:
                    t = decode_fn([tid]).strip().lower()
                    if t == tl or (len(tl) > 2 and tl in t):
                        return self.activate(tid)
            except:
                pass
        return {}


class ConstrainedGenerator:
    """ConstrainedDecoder + AssociationGraph = генерация через ассоциации."""
    
    def __init__(self, decoder, assoc_graph, decode_fn=None):
        self.decoder = decoder
        self.ag = assoc_graph
        self.decode_fn = decode_fn
    
    def generate_from_association(self, seed_text, temperature=0.2, novelty=False):
        """
        Даёшь слово → активируются ассоциации → генерация с этими ассоциациями.
        
        Алгоритм:
        1. Слово → активация (токены + концепты + мета)
        2. Выбираем самый активированный концепт (≠ seed-концепт)
        3. Генерируем слово в этом концепте
        """
        act = self.ag.activate_text(seed_text, self.decode_fn)
        if not act:
            print(f"  No activation for '{seed_text}'")
            # Fallback: regular generation
            tokens = self.decoder.generate(temperature=temperature)
            return self.decode_fn(tokens) if self.decode_fn else str(tokens)
        
        # Сортируем концепты (level 1) по активации, исключая seed-концепт
        seed_tid = None
        for tid in range(4096):
            try:
                if self.decode_fn and self.decode_fn([tid]).strip().lower() == seed_text.strip().lower():
                    seed_tid = tid
                    break
            except:
                pass
        
        seed_cid = self.ag.get_concept(seed_tid) if seed_tid else None
        
        concepts = [(n, e) for n, e in act.items() 
                    if self.ag.L1_OFFSET <= n < self.ag.L2_OFFSET
                    and n != seed_cid]
        concepts.sort(key=lambda x: -x[1])
        
        if not concepts:
            print("  No alternative concepts activated, using seed concept")
            target_cid = seed_cid
        else:
            # Выбираем top-3 и берём случайный (для разнообразия)
            top = concepts[:3]
            if len(top) > 1 and novelty:
                import random
                target_cid, _ = random.choice(top)
            else:
                target_cid, _ = top[0]
        
        # Генерируем слово в целевом концепте
        tokens = self.decoder.generate_with_target(
            target_cid, context=[2], temperature=temperature, novelty=novelty)
        text = self.decode_fn(tokens).strip() if self.decode_fn else str(tokens)
        
        # Показываем ассоциативную цепочку
        cid_name = self.ag.concept_name(target_cid) if target_cid else '?'
        act_chain = [seed_text]
        if seed_cid:
            act_chain.append(self.ag.concept_name(seed_cid))
        act_chain.append(f"→{cid_name}")
        # Meta
        if target_cid:
            mid = self.ag.get_meta(target_cid)
            if mid:
                act_chain.append(self.ag.meta_name(mid))
        
        return text, act_chain


class DynamicConceptSpace:
    """
    Динамическое пространство концептов.
    Позволяет СОЗДАВАТЬ, УНИЧТОЖАТЬ, ОБЪЕДИНЯТЬ и УКРУПНЯТЬ концепты
    в многомерном SVD-пространстве.
    
    Каждый динамический концепт имеет:
      - вектор в SVD-пространстве (32-dim)
      - профиль дискриминативных токенов
      - вес связи с исходными концептами
      - метку (человекочитаемое имя)
    """
    
    def __init__(self, assoc_graph, hv, starting_id=4156):
        self.ag = assoc_graph
        self.hv = hv
        self.next_id = starting_id  # IDs for dynamic concepts
        
        # Dynamic concept storage
        self.dyn_vectors = {}       # dyn_cid -> SVD vector
        self.dyn_profiles = {}      # dyn_cid -> {token_id: weight}
        self.dyn_top_tokens = {}    # dyn_cid -> [(token_id, weight)]
        self.dyn_labels = {}        # dyn_cid -> name
        self.dyn_parents = {}       # dyn_cid -> [parent_cid, ...] (for traceability)
        self.dyn_pmi_to_base = {}   # dyn_cid -> {base_cid: pmi}
        
        # All dynamic concept IDs are > 4095 (not type-2 tokens)
        # They extend the AssociationGraph's concept space
        
    def _decode_token(self, tid):
        """Безопасное декодирование токена."""
        try:
            return self.hv.decode([tid]).strip()
        except:
            return '?'
    
    def _concept_vector(self, cid):
        """Возвращает вектор концепта (из базовых или динамических)."""
        if cid in self.dyn_vectors:
            return self.dyn_vectors[cid]
        return self.ag.centroid_vectors.get(cid)
    
    def _concept_tokens(self, cid, top_k=20):
        """Возвращает дискриминативные токены концепта."""
        if cid in self.dyn_top_tokens:
            return [t for t, w in self.dyn_top_tokens[cid][:top_k]]
        return self.ag.get_profile(cid, top_k=top_k)
    
    def _word_to_type2_token(self, word):
        """Находит type-2 токен для слова."""
        bpe = self.hv.encode(word)
        if not bpe:
            return None
        first = bpe[0]
        first_d = self.hv.decode([first]).strip()
        for t in range(4096):
            if t < len(self.hv.token_type) and self.hv.token_type[t] == 2:
                if self.hv.decode([t]).strip() == first_d:
                    return t
        return None
    
    def word_vector(self, word):
        """Вектор для слова через его type-2 токен."""
        tid = self._word_to_type2_token(word)
        if tid is None:
            return None
        return self.ag.token_to_vector(tid)
    
    # ---- CREATE ----
    def create_concept(self, vector, name, parent_cids=None):
        """
        Создать новый концепт из произвольного вектора.
        vector: 32-dim SVD vector
        name: human-readable label
        parent_cids: [cid, ...] — откуда произошёл (для traceability)
        Returns: dyn_cid
        """
        cid = self.next_id
        self.next_id += 1
        
        self.dyn_vectors[cid] = vector.copy()
        self.dyn_labels[cid] = name
        self.dyn_parents[cid] = parent_cids or []
        
        # Build profile: find nearest base concepts and blend their profiles
        nearest = self.ag.nearest_concepts(vector, top_k=5)
        if nearest:
            # Weighted blend of profiles from nearest base concepts
            profile = {}
            total_w = 0
            for base_cid, dist in nearest:
                w = math.exp(-dist / 20.0)  # distance decay
                total_w += w
                base_profile = self.ag.cid_profiles.get(base_cid, {})
                for tid, pw in base_profile.items():
                    profile[tid] = profile.get(tid, 0) + pw * w
            
            if total_w > 0:
                profile = {t: w / total_w for t, w in profile.items()}
            
            # Also add tokens from the nearest type-2 token to vector
            nearest_tid = self._nearest_type2_token(vector)
            if nearest_tid is not None:
                profile[nearest_tid] = profile.get(nearest_tid, 0) + 1.0
                
            self.dyn_profiles[cid] = profile
            self.dyn_top_tokens[cid] = sorted(profile.items(), key=lambda x: -x[1])
            
            # PMI to base concepts
            pmi_to_base = {}
            for base_cid, dist in nearest:
                pmi_val = 1.0 / max(dist, 0.1)  # simple inverse-distance PMI proxy
                pmi_to_base[base_cid] = pmi_val
            self.dyn_pmi_to_base[cid] = pmi_to_base
        else:
            self.dyn_profiles[cid] = {}
            self.dyn_top_tokens[cid] = []
            self.dyn_pmi_to_base[cid] = {}
        
        return cid
    
    def _nearest_type2_token(self, vector):
        """Найти type-2 токен, чей вектор ближе всего к заданному."""
        if self.ag.starter_embeddings is None:
            return None
        dists = []
        for i, tid in enumerate(self.ag.starter_list):
            tok_vec = self.ag.starter_embeddings[i]
            d = np.linalg.norm(vector - tok_vec)
            dists.append((tid, d))
        dists.sort(key=lambda x: x[1])
        return dists[0][0] if dists else None
    
    # ---- MERGE ----
    def merge_concepts(self, cid_a, cid_b, name=None, weight_a=0.6, weight_b=0.4):
        """
        Объединить два концепта в один через интерполяцию векторов.
        weight_a, weight_b: вес каждого родителя.
        Returns: dyn_cid (новый концепт)
        """
        va = self._concept_vector(cid_a)
        vb = self._concept_vector(cid_b)
        if va is None or vb is None:
            return None
        
        w = weight_a + weight_b
        va_norm = va / max(np.linalg.norm(va), 1e-10)
        vb_norm = vb / max(np.linalg.norm(vb), 1e-10)
        merged = (va_norm * weight_a + vb_norm * weight_b) / w
        merged = merged * (np.linalg.norm(va) * weight_a + np.linalg.norm(vb) * weight_b) / w
        
        label_a = self._concept_label(cid_a)
        label_b = self._concept_label(cid_b)
        merged_name = name or '%s+%s' % (label_a, label_b)
        
        cid = self.create_concept(merged, merged_name, 
                                   parent_cids=[cid_a, cid_b])
        return cid
    
    def _concept_label(self, cid):
        """Human-readable label for any concept."""
        if cid in self.dyn_labels:
            return self.dyn_labels[cid]
        return self.ag.cid_label.get(cid, 'C%d' % (cid - 4096 if cid >= 4096 else cid))
    
    # ---- SPLIT ----
    def split_concept(self, cid, name_a=None, name_b=None):
        """
        Разделить большой концепт на два меньших.
        Использует k=2 sub-clustering на членах концепта.
        Returns: (cid_a, cid_b) — два новых динамических концепта
        """
        members = self.ag.cid_to_tids.get(cid, [])
        if len(members) < 4:
            return None, None
        
        # Get vectors for all members
        member_vectors = []
        valid_members = []
        for t in members:
            v = self.ag.token_to_vector(t)
            if v is not None:
                member_vectors.append(v)
                valid_members.append(t)
        
        if len(member_vectors) < 4:
            return None, None
        
        from sklearn.cluster import KMeans
        mat = np.array(member_vectors)
        km = KMeans(n_clusters=2, random_state=42, n_init='auto')
        labels = km.fit_predict(mat)
        
        cid_a_vec = km.cluster_centers_[0]
        cid_b_vec = km.cluster_centers_[1]
        
        base_label = self._concept_label(cid)
        ca = self.create_concept(cid_a_vec, name_a or '%s_A' % base_label, [cid])
        cb = self.create_concept(cid_b_vec, name_b or '%s_B' % base_label, [cid])
        
        # Override profiles: only members of each sub-cluster
        self.dyn_profiles[ca] = {}
        for i, tid in enumerate(valid_members):
            if labels[i] == 0:
                self.dyn_profiles[ca][tid] = 1.0
        for i, tid in enumerate(valid_members):
            if labels[i] == 1:
                self.dyn_profiles[cb][tid] = 1.0
        self.dyn_top_tokens[ca] = sorted(self.dyn_profiles[ca].items(), key=lambda x: -x[1])
        self.dyn_top_tokens[cb] = sorted(self.dyn_profiles[cb].items(), key=lambda x: -x[1])
        
        return ca, cb
    
    # ---- PROJECT ----
    def project(self, cid_from, cid_to, t=0.5, name=None):
        """
        Проецирование: интерполяция от from к to.
        t=0 → from, t=1 → to, t=0.5 → середина (новое качество).
        """
        v_from = self._concept_vector(cid_from)
        v_to = self._concept_vector(cid_to)
        if v_from is None or v_to is None:
            return None
        
        projected = v_from * (1 - t) + v_to * t
        label = name or 'project(%s→%s, t=%.1f)' % (
            self._concept_label(cid_from), self._concept_label(cid_to), t)
        
        return self.create_concept(projected, label,
                                    parent_cids=[cid_from, cid_to])
    
    # ---- VERIFY ----
    def verify_connection(self, cid, min_pmi=0.1):
        """
        Проверить, что концепт имеет связь с существующими.
        Возвращает True если есть хотя бы одна PMI-связь > min_pmi.
        """
        if cid in self.dyn_pmi_to_base:
            for base_cid, pmi in self.dyn_pmi_to_base[cid].items():
                if pmi > min_pmi:
                    return True, base_cid, pmi
        
        # Fallback: check PMI to all base concepts from the seed
        # by looking at what concepts are nearest in space
        vec = self.dyn_vectors.get(cid)
        if vec is not None:
            nearest = self.ag.nearest_concepts(vec, top_k=3)
            for base_cid, dist in nearest:
                pmi_proxy = 1.0 / max(dist, 0.1)
                if pmi_proxy > min_pmi:
                    return True, base_cid, pmi_proxy
        
        return False, None, 0.0
    
    # ---- GENERATION SUPPORT ----
    def get_profile(self, cid, top_k=20, min_weight=0.3):
        """Возвращает дискриминативные токены концепта (единый API с AssociationGraph)."""
        if cid in self.dyn_top_tokens:
            result = [t for t, w in self.dyn_top_tokens[cid] if w >= min_weight][:top_k]
            if result:
                return result
            return [t for t, w in self.dyn_top_tokens[cid][:top_k]]
        return self.ag.get_profile(cid, top_k, min_weight)
    
    def print_dynamic(self, limit=10):
        """Показать все динамические концепты."""
        if not self.dyn_labels:
            print("  (no dynamic concepts)")
            return
        for i, (cid, name) in enumerate(sorted(self.dyn_labels.items())):
            if i >= limit:
                break
            top = [self._decode_token(t) for t in self._concept_tokens(cid, 5)]
            parents = [self._concept_label(p) for p in self.dyn_parents.get(cid, [])]
            parents_str = ' <- %s' % '+'.join(parents) if parents else ''
            print("  D%d %s: %s%s" % (cid, name, ' '.join(top), parents_str))
