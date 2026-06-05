"""
ConceptTransformer — causal transformer на последовательностях концептов.
Учится предсказывать: P(concept_t | concept_1..t-1).
Гораздо компактнее токен-трансформера (48 концептов против 4101 токенов).
"""
import sys, os, math, pickle, json
import numpy as np
from collections import Counter

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptTransformer(nn.Module):
    """
    Причинный трансформер для последовательностей концептов.
    
    Вход: concept_ids (0..n_concepts-1)
    Выход: logits над n_concepts концептами
    
    n_concepts = 48 (из AssociationGraph)
    d_model = 64
    n_layers = 3
    n_heads = 4
    """
    
    def __init__(self, n_concepts=48, d_model=64, n_layers=3, 
                 n_heads=4, d_ff=256, max_seq=512):
        super().__init__()
        self.n_concepts = n_concepts
        self.d_model = d_model
        
        # Embedding
        self.embed = nn.Embedding(n_concepts + 1, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, n_concepts)
        
        # causal_mask generated on the fly in forward
        self._mask_cache = None
    
    def _get_causal_mask(self, seq_len):
        """Создаёт causal mask для self-attention."""
        device = next(self.parameters()).device
        if self._mask_cache is None or self._mask_cache.shape[0] < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
            self._mask_cache = mask
        return self._mask_cache[:seq_len, :seq_len]
    
    def forward(self, x):
        """
        x: (batch, seq_len) — concept_ids (0 = padding, 1..n_concepts = real)
        Returns: (batch, seq_len, n_concepts) — logits
        """
        batch, seq_len = x.shape
        device = x.device
        
        # Embed + position
        tok_embed = self.embed(x)  # (B, S, d)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        pos_embed = self.pos_embed(positions)
        h = tok_embed + pos_embed
        
        # Causal mask
        mask = self._get_causal_mask(seq_len)
        
        # Transformer
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.norm(h)
        
        # Output
        logits = self.output(h)  # (B, S, n_concepts)
        return logits
    
    def predict_next(self, concept_seq, temperature=1.0):
        """
        Предсказать следующий концепт.
        concept_seq: list[int] or (1, S) tensor
        Returns: (concept_id, logits)
        """
        if isinstance(concept_seq, list):
            concept_seq = torch.tensor([concept_seq], dtype=torch.long)
        if concept_seq.dim() == 1:
            concept_seq = concept_seq.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.forward(concept_seq)  # (1, S, n_c)
            next_logits = logits[0, -1, :]  # (n_c,)
            
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                cid = torch.multinomial(probs, 1).item()
            else:
                cid = next_logits.argmax().item()
            
            return cid, next_logits
    
    def train_step(self, x, y, optimizer):
        """
        x: (batch, seq_len) — input concept sequences
        y: (batch, seq_len) — target concept sequences (shifted by 1)
        Returns: loss
        """
        logits = self.forward(x)  # (B, S, n_c)
        loss = F.cross_entropy(
            logits.reshape(-1, self.n_concepts),
            y.reshape(-1),
            ignore_index=0
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return loss.item()


def build_concept_sequences(data_dir, assoc_graph, hv, max_seqs=100000):
    """
    Конвертирует BPE-encoded данные в последовательности концептов.
    Читает .tokens.npy + .lengths.npy для hierarchical/conceptnet/wikipedia.
    Только WORD_STARTER -> concept_id.
    """
    import glob
    names = ['hierarchical', 'conceptnet', 'wikipedia']
    
    sequences = []
    n_skipped = 0
    total_sents = 0
    
    for name in names:
        tpath = os.path.join(data_dir, f'{name}.tokens.npy')
        lpath = os.path.join(data_dir, f'{name}.lengths.npy')
        if not os.path.exists(tpath) or not os.path.exists(lpath):
            print(f"  Missing: {name}")
            continue
        
        tokens = np.load(tpath).astype(np.int32)
        lengths = np.load(lpath)
        n_sents = len(lengths)
        total_sents += n_sents
        print(f"  {name}: {n_sents} sentences, {len(tokens)} tokens")
        
        ptr = 0
        for si in range(n_sents):
            if len(sequences) >= max_seqs:
                break
            slen = int(lengths[si])
            sent_tokens = tokens[ptr:ptr+slen]
            ptr += slen
            
            # Get concepts for each WORD_STARTER token
            concepts = []
            for tid in sent_tokens:
                if tid < 4096 and hv.token_type[tid] == 2:  # WORD_STARTER
                    cid = assoc_graph.get_concept(tid)
                    if cid is not None:
                        concepts.append(cid - assoc_graph.L1_OFFSET + 1)  # +1 since 0=padding
                    elif concepts:
                        concepts.append(concepts[-1])  # fallback: repeat last
            
            if len(concepts) >= 2:
                sequences.append(np.array(concepts, dtype=np.int16))
            else:
                n_skipped += 1
            
            if si % 50000 == 0:
                print(f"    processed {si}/{n_sents} sents, {len(sequences)} sequences", end='\r')
        
        if len(sequences) >= max_seqs:
            break
    
    print(f"\nBuilt {len(sequences)} concept sequences (skipped {n_skipped})")
    return sequences


def train_concept_transformer(sequences, n_concepts=48, d_model=64,
                               n_layers=3, n_heads=4, epochs=3,
                               batch_size=64, lr=1e-3, device='cpu'):
    """Train the ConceptTransformer on concept sequences."""
    
    model = ConceptTransformer(
        n_concepts=n_concepts + 1,  # +1 for padding
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Pad/cut sequences to fixed length
    max_len = 64
    padded = []
    for seq in sequences:
        if len(seq) > max_len:
            seq = seq[:max_len]
        padded_seq = np.zeros(max_len, dtype=np.int64)
        padded_seq[:len(seq)] = seq
        padded.append(padded_seq)
    
    X = np.array(padded, dtype=np.int64)
    n_samples = len(X)
    print(f"Training data: {n_samples} sequences, max_len={max_len}")
    
    # Target = input shifted left by 1
    Y = np.zeros_like(X)
    Y[:, :-1] = X[:, 1:]
    
    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(n_samples)
        X_shuffled, Y_shuffled = X[perm], Y[perm]
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            bx = torch.tensor(X_shuffled[i:i+batch_size], device=device)
            by = torch.tensor(Y_shuffled[i:i+batch_size], device=device)
            
            loss = model.train_step(bx, by, optimizer)
            total_loss += loss
            n_batches += 1
            
            if n_batches % 50 == 0:
                print(f"  epoch {epoch+1}, batch {n_batches}: loss={loss:.4f}")
        
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")
    
    return model


if __name__ == '__main__':
    # Quick test
    from eva.symbolic.association_graph import AssociationGraph
    from eva.symbolic.heads import HeadsEnsemble
    from eva.symbolic.bpe_tokenizer import HierarchicalVocab
    
    print("Loading...")
    hv = HierarchicalVocab()
    heads = HeadsEnsemble('real_data/v8/heads_meta.pkl', 'real_data/v8')
    
    ag = AssociationGraph(n_clusters=48, n_metas=12)
    ag.build(heads.log_prob_csr, hv.token_type, decode_fn=hv.decode)
    
    print("Building concept sequences...")
    seqs = build_concept_sequences('hierarchical_data_clean', ag, hv, max_seqs=50000)
    
    if len(seqs) < 100:
        print(f"ERROR: only {len(seqs)} sequences, cannot train")
        sys.exit(1)
    
    print("Training ConceptTransformer...")
    model = train_concept_transformer(seqs, n_concepts=ag.n_clusters, 
                                       device='cpu', epochs=3, batch_size=128)
    
    # Save
    torch.save(model.state_dict(), 'real_data/v8/concept_transformer.pt')
    print("Model saved to real_data/v8/concept_transformer.pt")
