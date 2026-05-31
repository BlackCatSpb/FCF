"""
train_concept_basis.py — Phase 0: обучение базиса концептов на ConceptNet.

Сброс модели: остаются только CoordinateEmbedding + CoordinateDecoder + RMSNorm.
Всё остальное (transformer, heads, potentials, topology) — reinit.

Обучение: concept pairs из ConceptNet → contrastive learning в 128-dim space.
- Связанные концепты: координаты близко (cosine ≈ 1)
- Несвязанные концепты: координаты далеко (margin triplet loss)
- ConceptHead: учится выдавать 1.0 для реальных концептов
- Temporal smoothness: траектория внутри концепта гладкая
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, time, math, json
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.char_vocab import CharacterVocab as CharVocab
from eva.symbolic.train_v3 import TrainingConfig, MultiTaskLoss


# ============================================================
# 0. Config
# ============================================================

@dataclass
class ConceptConfig:
    conceptnet_path: str = "real_data/conceptnet_ru.txt"
    checkpoint_dir: str = "checkpoints/concept_basis"
    resume: Optional[str] = "checkpoints/v3/eva_v3_latest.pt"
    
    batch_size: int = 16
    seq_len: int = 32
    lr: float = 3e-4
    num_epochs: int = 5
    warmup_steps: int = 100
    save_every: int = 500
    log_every: int = 50
    
    margin_pos: float = 0.5  # triplet: положительные ближе margin
    margin_neg: float = 1.0  # отрицательные дальше margin
    neg_samples: int = 4      # негативных пар на каждую позитивную
    
    w_concept: float = 0.3
    w_contrastive: float = 0.4
    w_srg: float = 0.2
    w_head_consistency: float = 0.1
    w_residual: float = 0.05
    w_uncertainty: float = 0.05


# ============================================================
# 1. ConceptNet Data Preparation
# ============================================================

class ConceptNetDataset:
    """
    ConceptNet triples: concept1 relation_type concept2.
    Два режима:
    - positive pairs: concept1 + concept2 (связаны)
    - negative pairs: concept1 + random concept2 (не связаны)
    """
    
    def __init__(self, path: str, cv: CharVocab, max_len: int = 32):
        self.cv = cv
        self.max_len = max_len
        self.pairs: List[Tuple[str, str, str]] = []
        self.concepts: Dict[str, int] = {}
        
        import re
        CYRILLIC = re.compile(r'[а-яёА-ЯЁ]')
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                rel = parts[-2]
                c2 = parts[-1]
                c1 = line[:-(len(rel) + len(c2) + 2)].strip()
                if not c1 or not c2:
                    continue
                # Filter: at least one concept must have Cyrillic
                if not CYRILLIC.search(c1) and not CYRILLIC.search(c2):
                    continue
                # Filter: max 40 chars to avoid garbage
                if len(c1) > 40 or len(c2) > 40:
                    continue
                self.pairs.append((c1, rel, c2))
                for c in (c1, c2):
                    if c not in self.concepts:
                        self.concepts[c] = len(self.concepts)
        
        self.concept_list = list(self.concepts.keys())
        print(f'[ConceptNet] {len(self.pairs)} pairs, {len(self.concepts)} unique concepts')
    
    def _encode(self, text: str) -> List[int]:
        """Encode concept as boundary sequence with truncation."""
        ids = self.cv.encode_with_boundaries(text)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len-1] + [self.cv.SENT_CLOSE_IDX]
        return ids
    
    def get_positive_batch(self, batch_size: int, device):
        """Random positive pairs."""
        idx = np.random.randint(0, len(self.pairs), size=batch_size)
        batch = []
        for i in idx:
            c1, rel, c2 = self.pairs[i]
            batch.append((c1, rel, c2))
        return self._batch_to_tensors(batch, device)
    
    def get_contrastive_batch(self, batch_size: int, device):
        """Positive pairs + negative pairs."""
        pos = self.get_positive_batch(batch_size, device)
        # negative: replace c2 with random concept
        neg_c2 = []
        for c1, rel, c2 in pos['raw']:
            while True:
                rand_c = self.concept_list[np.random.randint(0, len(self.concept_list))]
                if rand_c != c2:
                    break
            neg_c2.append(rand_c)
        neg_batch = [(pos['raw'][i][0], pos['raw'][i][1], neg_c2[i]) for i in range(batch_size)]
        neg_tensors = self._batch_to_tensors(neg_batch, device)
        return pos, neg_tensors
    
    def _batch_to_tensors(self, batch, device):
        c1_ids = [self._encode(c1) for c1, _, _ in batch]
        c2_ids = [self._encode(c2) for _, _, c2 in batch]
        max_l = max(len(ids) for ids in c1_ids + c2_ids)
        c1_t = torch.zeros(len(batch), max_l, dtype=torch.long, device=device)
        c2_t = torch.zeros(len(batch), max_l, dtype=torch.long, device=device)
        for i, ids in enumerate(c1_ids):
            c1_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        for i, ids in enumerate(c2_ids):
            c2_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return {'c1': c1_t, 'c2': c2_t, 'raw': batch}


# ============================================================
# 2. Model Reset
# ============================================================

def reset_model_for_concept_learning(model: UnifiedMultidimensionalTransformer):
    """
    Keep only CoordinateEmbedding + CoordinateDecoder + RMSNorm.
    Reinitialize everything else.
    """
    # Save symbol knowledge
    embed_state = {
        'coordinates': model.embed.coordinates.clone(),
        'scale': model.embed.scale.clone(),
    }
    decoder_state = {
        'linear.weight': model.decoder.linear.weight.clone(),
        'linear.bias': model.decoder.linear.bias.clone() if model.decoder.linear.bias is not None else None,
        'temperature': model.decoder.temperature.clone(),
        'nn_weight': model.decoder.nn_weight.clone(),
        'group_classifier.weight': model.decoder.group_classifier.weight.clone(),
        'group_classifier.bias': model.decoder.group_classifier.bias.clone() if model.decoder.group_classifier.bias is not None else None,
    }
    norm_state = {
        'norm_final.weight': model.norm_final.weight.clone(),
    }
    
    # Full reinit
    for name, module in model.named_modules():
        if name in ('', 'embed', 'decoder', 'norm_final', 'rope'):
            continue
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            for p in module.parameters(recurse=False):
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                elif p.dim() == 1:
                    nn.init.zeros_(p)
    
    # Restore symbol knowledge
    with torch.no_grad():
        model.embed.coordinates.copy_(embed_state['coordinates'])
        model.embed.scale.copy_(embed_state['scale'])
        model.decoder.linear.weight.copy_(decoder_state['linear.weight'])
        if decoder_state['linear.bias'] is not None:
            model.decoder.linear.bias.copy_(decoder_state['linear.bias'])
        model.decoder.temperature.copy_(decoder_state['temperature'])
        model.decoder.nn_weight.copy_(decoder_state['nn_weight'])
        model.decoder.group_classifier.weight.copy_(decoder_state['group_classifier.weight'])
        if decoder_state['group_classifier.bias'] is not None:
            model.decoder.group_classifier.bias.copy_(decoder_state['group_classifier.bias'])
        model.norm_final.weight.copy_(norm_state['norm_final.weight'])
    
    # Reset teacher (don't use distillation for concept learning)
    if hasattr(model, '_teacher'):
        model._teacher = None
    if hasattr(model, '_distill_head'):
        model._distill_head = None
    model.set_weight_context(False)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Reset] Model reset. Total params: {total:,}, trainable: {trainable:,}')
    print(f'[Reset] Kept: embed.coordinates, decoder.{list(decoder_state.keys())}, norm_final.weight')
    return model


# ============================================================
# 3. Concept Training
# ============================================================

def concept_loss_fn(model, h1, h2, h1_neg, head_out1, head_out2, config: ConceptConfig):
    """
    Concept learning losses:
    - concept_head: high for real concepts (both pos and neg are real concepts)
    - contrastive: pos pairs close, neg pairs far
    - temporal smoothness: smooth trajectory within each concept
    - head consistency: concept ≈ 1 - contradiction
    - residual: low residual error for real concepts
    - uncertainty: low uncertainty for real concepts
    """
    losses = {}
    
    # Concept head: real concepts should have high concept score
    conc1 = head_out1['concept'].mean()
    conc2 = head_out2['concept'].mean()
    losses['concept'] = (1.0 - conc1).pow(2) + (1.0 - conc2).pow(2)
    
    # Contrastive: pos pairs close (cosine), neg pairs far (margin triplet)
    z1 = h1.mean(dim=1)  # [B, D]
    z2 = h2.mean(dim=1)
    z1n = h1_neg.mean(dim=1)
    
    cos_pos = F.cosine_similarity(z1, z2)  # [B]
    cos_neg = F.cosine_similarity(z1, z1n)  # [B]
    # triplet: pos > neg + margin
    triplet = F.relu(cos_neg - cos_pos + config.margin_pos).mean()
    # also push neg below margin_neg
    neg_push = F.relu(cos_neg - (1.0 - config.margin_neg)).mean()
    losses['contrastive'] = (1.0 - cos_pos).mean() + triplet + neg_push
    
    # Temporal smoothness
    if h1.shape[1] > 1:
        d1 = (h1[:, 1:] - h1[:, :-1]).pow(2).mean()
        d2 = (h2[:, 1:] - h2[:, :-1]).pow(2).mean()
        losses['srg'] = (d1 + d2) * 0.5
    else:
        losses['srg'] = torch.tensor(0.0, device=h1.device)
    
    # Head consistency: concept ≈ 1 - contradiction
    contra1 = head_out1['contradiction'].mean()
    contra2 = head_out2['contradiction'].mean()
    losses['head_consistency'] = ((conc1 - (1.0 - contra1)).pow(2) + 
                                   (conc2 - (1.0 - contra2)).pow(2)) * 0.5
    
    # Residual (low = good delta prediction)
    res1 = head_out1.get('residual_error', torch.zeros(1, device=h1.device)).mean()
    res2 = head_out2.get('residual_error', torch.zeros(1, device=h1.device)).mean()
    losses['residual'] = (res1 + res2) * 0.5
    
    # Uncertainty: should be low for known concepts
    unc1 = head_out1['uncertainty'].mean()
    unc2 = head_out2['uncertainty'].mean()
    losses['uncertainty'] = (unc1 + unc2) * 0.5
    
    return losses


def train_concept_epoch(model, dataset: ConceptNetDataset, config: ConceptConfig,
                         optimizer, scheduler, device, start_step=0):
    model.train()
    loss_keys = ['total', 'concept', 'contrastive', 'srg', 
                 'head_consistency', 'residual', 'uncertainty']
    total_losses = {k: 0.0 for k in loss_keys}
    n_batches = 0
    step = start_step
    t0 = time.time()
    
    n_steps = max(100, len(dataset.concepts) * config.num_epochs // config.batch_size)
    
    for ep in range(config.num_epochs):
        for batch_idx in range(n_steps):
            pos, neg = dataset.get_contrastive_batch(config.batch_size, device)
            
            optimizer.zero_grad()
            
            # Forward concept1 (pos)
            h1, _, _, head1 = model.forward(pos['c1'], return_heads=True, capture_attn=True)
            # Forward concept2 (pos)
            h2, _, _, head2 = model.forward(pos['c2'], return_heads=True)
            # Forward negative concept2
            h1n, _, _, head1n = model.forward(neg['c2'], return_heads=True)
            
            losses = concept_loss_fn(model, h1, h2, h1n, head1, head2, config)
            
            total = (config.w_concept * losses['concept'] +
                     config.w_contrastive * losses['contrastive'] +
                     config.w_srg * losses['srg'] +
                     config.w_head_consistency * losses['head_consistency'] +
                     config.w_residual * losses['residual'] +
                     config.w_uncertainty * losses['uncertainty'])
            
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            for k in total_losses:
                total_losses[k] += losses.get(k, torch.tensor(0.0, device=device)).item()
            total_losses['total'] += total.item()
            n_batches += 1
            
            if step % config.log_every == 0:
                avg = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
                elapsed = time.time() - t0
                cos_pos_avg = F.cosine_similarity(
                    h1.mean(dim=1), h2.mean(dim=1)).mean().item()
                print(f'[Epoch {ep+1}/{config.num_epochs} Step {step}] '
                      f'tot={avg["total"]:.4f} conc={avg["concept"]:.4f} '
                      f'ctr={avg["contrastive"]:.4f} srg={avg["srg"]:.4f} '
                      f'cos_pos={cos_pos_avg:.4f} | {elapsed:.0f}s')
            
            if step % config.save_every == 0 and step > start_step:
                save_concept_checkpoint(model, optimizer, step, config)
            
            step += 1
    
    return step


def save_concept_checkpoint(model, optimizer, step, config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f'concept_basis_step_{step}.pt')
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': config,
    }, path)
    latest = os.path.join(config.checkpoint_dir, 'concept_basis_latest.pt')
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
    }, latest)
    # Lightweight inference checkpoint (model only, no optimizer)
    inf_path = os.path.join(config.checkpoint_dir, 'concept_basis_inference.pt')
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
    }, inf_path)
    print(f'[Save] Concept checkpoint saved to {path}')


# ============================================================
# 4. Inference — проверка концептов
# ============================================================

def get_concept_coord(model, concept_text: str, cv: CharVocab, device):
    ids = cv.encode_with_boundaries(concept_text, max_len=32)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        h, _, _, heads = model.forward(inp, return_heads=True)
        coord = h.mean(dim=1)[0]  # [D]
        concept_score = heads['concept'][0].mean().item()
    return coord, concept_score


def find_similar_concepts(model, query: str, concepts: List[str], cv, device, top_k=10):
    q_coord, q_score = get_concept_coord(model, query, cv, device)
    sims = []
    for c in concepts:
        if c == query:
            continue
        c_coord, _ = get_concept_coord(model, c, cv, device)
        sim = F.cosine_similarity(q_coord.unsqueeze(0), c_coord.unsqueeze(0)).item()
        sims.append((sim, c))
    sims.sort(reverse=True)
    return sims[:top_k], q_score


# ============================================================
# 5. Main
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='checkpoints/v3/eva_v3_latest.pt',
                       help='Source checkpoint (will be reset if no --continue)')
    parser.add_argument('--continue', action='store_true', dest='continue_training',
                       help='Continue from concept_basis checkpoint (skip reset)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--conceptnet-path', type=str, default='real_data/conceptnet_ru.txt')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/concept_basis')
    parser.add_argument('--test', action='store_true', help='Test mode: train and eval')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')
    
    cv = CharVocab()
    
    # ---- Load & Reset Model ----
    print('[Model] Loading UnifiedMultidimensionalTransformer...')
    model = UnifiedMultidimensionalTransformer().to(device)
    
    start_step = 0
    if args.continue_training:
        # Continue from concept_basis checkpoint — skip reset
        print(f'[Model] Continuing from: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state'], strict=False)
        start_step = ckpt.get('step', 0)
        print(f'[Model] Resumed from step {start_step}')
    elif args.resume and os.path.exists(args.resume):
        print(f'[Model] Loading checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        keep_prefixes = ('embed.', 'decoder.', 'norm_final.')
        filtered = {k: v for k, v in ckpt['model_state'].items()
                    if k.startswith(keep_prefixes)}
        model.load_state_dict(filtered, strict=False)
        print(f'[Model] Loaded {len(filtered)} keys (embed, decoder, norm) from step {ckpt.get("step", "?")}')
        reset_model_for_concept_learning(model)
    else:
        reset_model_for_concept_learning(model)
    
    # ---- Dataset ----
    dataset = ConceptNetDataset(args.conceptnet_path, cv, max_len=32)
    
    # ---- Config ----
    config = ConceptConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
    
    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    n_steps = max(100, len(dataset.concepts) * config.num_epochs // config.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    
    # ---- Train ----
    print(f'[Train] Starting concept basis training ({config.num_epochs} epochs)')
    step = train_concept_epoch(model, dataset, config, optimizer, scheduler, device, start_step=start_step)
    save_concept_checkpoint(model, optimizer, step, config)
    
    # ---- Test: find similar concepts ----
    if args.test:
        print('\n[Test] Finding similar concepts...')
        test_concepts = ['кошка', 'собака', 'дом', 'вода', 'человек', 'машина']
        all_concepts = list(dataset.concepts.keys())[:1000]
        for q in test_concepts:
            if q not in dataset.concepts:
                continue
            similar, score = find_similar_concepts(model, q, all_concepts, cv, device)
            print(f'\n  {q} (concept_score={score:.3f}):')
            for sim, c in similar[:5]:
                print(f'    {c}: {sim:.4f}')
    
    print(f'\n[Done] Concept basis trained. Final step: {step}')
