"""
ConceptGraph — семантические кластеры токенов через дистрибутивную семантику.
Позволяет: контекст → концепт → новые валидные токены.
"""
import math
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


class ConceptGraph:
    def __init__(self, V=4096):
        self.V = V
        self.clusters = {}          # cid -> list of token_ids
        self.token_to_cluster = {}  # tid -> cid
        self.cluster_name = {}      # cid -> representative text
        
        self.concept_transition = {}  # (cid_i, cid_j) -> log_prob
        self.token_embeddings = None  # (n_tokens, dim)
        
    def build(self, log_prob_csr: csr_matrix, token_type: np.ndarray,
              decode_fn=None, n_clusters=64, embed_dim=32):
        """
        Строит концепт-кластеры через SVD + k-means на входящих переходах.
        
        Входящие переходы (incoming) отражают КОНТЕКСТ, в котором токен встречается,
        т.е. distributional semantics: слова в одинаковых контекстах = один концепт.
        """
        print("Building concept graph with distributional semantics...")
        csr_shape = log_prob_csr.shape[0]
        n_tokens = min(self.V, csr_shape, len(token_type))
        
        # Собираем только WORD_STARTER токены (type == 2)
        starter_ids = [tid for tid in range(n_tokens) if token_type[tid] == 2]
        print(f"  WORD_STARTER tokens: {len(starter_ids)}")
        
        if len(starter_ids) < 10:
            print("  Too few word starter tokens, aborting.")
            return self
        
        # Строим dense-эмбеддинги через SVD на матрице входящих переходов
        # incoming[t_i, t_j] = P(t_j | t_i) — как часто t_i следует за t_j
        # Каждый столбец матрицы — это входящий профиль токена
        
        # Извлекаем подматрицу только для стартеров
        tid_to_idx = {tid: i for i, tid in enumerate(starter_ids)}
        n_starters = len(starter_ids)
        
        # Строим входящие векторы через SVD на транспонированной CSR
        print("  Computing SVD embeddings from transition matrix...")
        
        # Транспонируем: incoming[row, col] = P(col | row) для col=starter
        # Берём log_prob_csr, транспонируем
        csr_t = log_prob_csr.T.tocsr()
        
        # Собираем входящие профили для каждого стартера
        # (вектор размера V, но sparse)
        from scipy.sparse import vstack
        
        # Строим sparse-матрицу: (n_starters, V) — входящие переходы
        incoming_blocks = []
        for tid in starter_ids:
            if tid < csr_t.shape[0]:
                row = csr_t[tid]
                incoming_blocks.append(row)
            else:
                incoming_blocks.append(csr_matrix((1, csr_t.shape[1])))
        
        incoming_mat = vstack(incoming_blocks, format='csr')
        print(f"  Incoming matrix: {incoming_mat.shape}")
        
        # SVD: уменьшаем размерность до embed_dim
        n_components = min(embed_dim, n_starters - 1, incoming_mat.shape[1] - 1)
        n_components = max(2, n_components)
        
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = svd.fit_transform(incoming_mat)
        print(f"  SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")
        
        self.token_embeddings = np.zeros((n_tokens, embeddings.shape[1]), dtype=np.float32)
        for i, tid in enumerate(starter_ids):
            if tid < n_tokens:
                self.token_embeddings[tid] = embeddings[i]
        
        # K-means кластеризация в SVD-пространстве
        k = min(n_clusters, n_starters // 2)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        print(f"  K-means: {k} clusters")
        
        # Собираем кластеры
        cluster_members = defaultdict(list)
        for i, tid in enumerate(starter_ids):
            cid = int(labels[i])
            cluster_members[cid].append(tid)
        
        # Сортируем кластеры по размеру (desc) и переименовываем
        sorted_clusters = sorted(cluster_members.items(), key=lambda x: -len(x[1]))
        
        cid = 0
        for _, members in sorted_clusters:
            if len(members) < 2:
                continue
            self.clusters[cid] = members
            for t in members:
                self.token_to_cluster[t] = cid
            
            # Имя кластера — самый частотный/центральный токен
            if decode_fn:
                # Берём ближайший к центру кластера
                center = kmeans.cluster_centers_[labels[starter_ids.index(members[0])]]
                # Находим ближайший к центру
                best_dist = float('inf')
                best_tid = members[0]
                for tid in members:
                    idx = starter_ids.index(tid)
                    dist = np.linalg.norm(embeddings[idx] - center)
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid
                try:
                    self.cluster_name[cid] = decode_fn([best_tid]).strip()
                except:
                    self.cluster_name[cid] = f"C{cid}"
            else:
                self.cluster_name[cid] = f"C{cid}"
            cid += 1
        
        print(f"  Built {len(self.clusters)} concept clusters")
        
        # --- Строим concept_transition: P(cj | ci) ---
        cid_pairs = defaultdict(float)
        cid_src_total = defaultdict(float)
        
        for tid in range(n_tokens):
            ci = self.token_to_cluster.get(tid)
            if ci is None:
                continue
            if tid >= csr_shape:
                continue
            
            row = log_prob_csr[tid].tocoo()
            for col, prob in zip(row.col, row.data):
                col = int(col)
                if col >= self.V:
                    continue
                cj = self.token_to_cluster.get(col)
                if cj is None:
                    continue
                cid_pairs[(ci, cj)] += math.exp(prob)
                cid_src_total[ci] += math.exp(prob)
        
        for (ci, cj), cnt in cid_pairs.items():
            total = cid_src_total.get(ci, 1e-10)
            prob = cnt / max(total, 1e-10)
            self.concept_transition[(ci, cj)] = math.log(max(prob, 1e-10))
        
        print(f"  Concept transitions: {len(self.concept_transition)}")
        return self
    
    def get_concept(self, token_id):
        return self.token_to_cluster.get(token_id)
    
    def get_members(self, cid):
        return self.clusters.get(cid, [])
    
    def predict_next(self, prev_token_id):
        """Предсказать следующий концепт. Возвращает список (cid, log_prob)."""
        ci = self.token_to_cluster.get(prev_token_id)
        if ci is None:
            return []
        candidates = [(cj, lp) for (ci2, cj), lp in self.concept_transition.items() 
                      if ci2 == ci]
        candidates.sort(key=lambda x: -x[1])
        return candidates
    
    def concept_scores(self, target_cid, V):
        """Буст для токенов из target_cid."""
        boost = np.zeros(V, dtype=np.float32)
        if target_cid in self.clusters:
            for tid in self.clusters[target_cid]:
                if tid < V:
                    boost[tid] = 2.0
        return boost
    
    def sample_alternative(self, target_cid, exclude_ids=set(), rng=None):
        """Выбрать альтернативный токен из концепта."""
        if rng is None:
            rng = np.random
        if target_cid not in self.clusters:
            return None
        candidates = [t for t in self.clusters[target_cid] if t not in exclude_ids]
        if not candidates:
            return None
        return rng.choice(candidates) if len(candidates) > 1 else candidates[0]
    
    def summarize(self, decode_fn=None, limit=10):
        print(f"ConceptGraph: {len(self.clusters)} clusters")
        sizes = [len(m) for m in self.clusters.values()]
        print(f"  Avg size: {np.mean(sizes):.1f}, max: {max(sizes)}, min: {min(sizes)}")
        for cid in sorted(self.clusters.keys())[:limit]:
            members = self.clusters[cid]
            name = self.cluster_name.get(cid, f"C{cid}")
            print(f"  C{cid:3d} ({name:10s}): {len(members):3d} tokens", end="")
            if decode_fn:
                texts = []
                for t in sorted(members)[:5]:
                    texts.append(decode_fn([t]).strip())
                print(f"  [{', '.join(texts)}]", end="")
            print()
