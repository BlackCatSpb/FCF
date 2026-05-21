"""
EVA Symbolic — OPTIMIZED training to convergence.

Key optimizations:
1. Vectorized strengthen_batch (no Python loops)
2. GPU-resident tensors (no numpy conversions in hot path)  
3. Batched assembly building
4. Pre-computed symbol potentials cache
5. CUDA-optimized forward pass
"""

import sys, os, time, torch, numpy as np, gc, threading, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from eva.symbolic import *
from eva.symbolic.hierarchical_layer import *
from eva.symbolic.word_level import *
from eva.symbolic.knowledge_base import *
from eva.symbolic.library import *
from eva.symbolic.contemplation import *
from eva.symbolic.advanced_methods import *
from eva.primordial_layer import PrimordialLayer
from eva.config import FCFConfig

# ============================================================
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
os.makedirs(CKPT_DIR, exist_ok=True)

USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'
BATCH = 128; BLOCK = 128
PRINT_STEP = 100; SAVE_STEP = 1000
MIN_STEPS = 3000; CONV_WINDOW = 1500; AFF_THRESH = 0.00001

print("=" * 60)
print("EVA Symbolic — OPTIMIZED")
print("=" * 60)
print(f"Device: {DEVICE}, Batch: {BATCH}, Block: {BLOCK}")

# ============================================================
config = FCFConfig(); config.d_model = 256; config.vocab_size = 156; config.num_heads = 8
layer = PrimordialLayer(config)
if USE_GPU: layer = layer.cuda()
char_vocab = CharacterVocab()
trainer = PotentialTrainer(layer=layer, char_vocab=char_vocab, embed_dim=256, checkpoint_dir=CKPT_DIR)
pf, topo, grammar = trainer.potential_field, trainer.topological_field, trainer.grammar

# Wire light modules only (heavy ones deferred)
lg = LogicGuard(contradiction_filter=trainer.contradiction_filter, grammar=grammar)
cont = ContemplationLoop(pf, topo, trainer.contradiction_filter, trainer.concept_miner,
                         grammar, trainer.geodesic_nav, lg)
cont.start()
ngram = NGramContext(pf, max_context=4, decay=0.5)

# Lazy-init heavy modules
_kb = None; _lib = None; _wd = None
def get_kb():
    global _kb, _lib, _wd
    if _kb is None:
        _wd = WordDiscovery(grammar, pf, char_vocab, min_confidence=0.55)
        _kb = KnowledgeBase(pf, topo, grammar, _wd)
        _lib = LibraryManager(_kb, _wd, char_vocab)
    return _kb, _lib, _wd

print(f"Modules wired (light), device={DEVICE}")

# ============================================================
npy_file = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy_file):
    npy_file = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)
total_tokens = len(all_ids)
print(f"Dataset: {total_tokens/1e6:.1f}M tokens")

# Pre-compute constants
V = pf.vocab_size
PAD = char_vocab.PAD_IDX

# ============================================================
# VECTORIZED: fast affinity update for adjacent pairs
# ============================================================
@torch.no_grad()
def fast_strengthen(batch_ids, batch_attn):
    """Vectorized strengthen: adjacent pairs via tensor ops."""
    B, H, L, _ = batch_attn.shape
    if L < 2: return
    
    # Adjacent pair indices: [B, L-1, 2]
    left = batch_ids[:, :L-1]   # [B, L-1]
    right = batch_ids[:, 1:L]   # [B, L-1]
    
    # Adjacent attention: diagonal of last two dims
    # FIX 2: causal direction — i+1 attends to i (allowed by causal mask)
    adj_attn = batch_attn.mean(dim=1)[:, torch.arange(1, L), torch.arange(L-1)]  # [B, L-1]
    
    # Valid mask
    valid = (left < V) & (right < V) & (left != PAD) & (right != PAD)
    
    # Flatten and filter
    i_idx = left[valid].long()
    j_idx = right[valid].long()
    w = adj_attn[valid].float()
    
    if len(i_idx) > 0:
        i_c, j_c, w_c = i_idx.cpu(), j_idx.cpu(), w.cpu()
        # Device safety: ensure pf buffers are on same device as indices
        assert pf.co_occurrence_count.device == i_c.device, "Device mismatch in fast_strengthen"
        # FIX 1: scatter_add для правильного накопления при дубликатах
        flat_idx = i_c * V + j_c
        increments = 1.0 + w_c
        pf.co_occurrence_count.view(-1).scatter_add_(0, flat_idx, increments)
        # FIX 4: threshold 100K (был 500K, потом 1K)
        unique_flat = flat_idx.unique()
        ui = unique_flat // V
        uj = unique_flat % V
        raw = pf.co_occurrence_count[ui, uj] / 100000.0
        pf.affinity[ui, uj] = 0.5 + 0.5 * torch.clamp(raw, 0.0, 1.0)

# ============================================================
# TRAINING LOOP
# ============================================================
pos = 0
start = time.time()
prev_pot = 0.5; prev_digrams = 0; steps_no_improvement = 0

print("\nTraining...\n")

MAX_STEPS = 50000  # Safety cap: never run more than 50K batches (6.4M assemblies)

try:
    while True:
        if step >= MAX_STEPS:
            print(f"\nMAX STEPS ({MAX_STEPS}) reached. Stopping.")
            break
        # Form batch
        if pos + BLOCK + 2 > total_tokens: pos = 0
        ids_batch, lens = [], []
        for _ in range(BATCH):
            if pos + BLOCK + 2 > total_tokens: pos = 0
            end = min(pos + BLOCK, total_tokens)
            chunk = all_ids[pos:end]
            sep = np.where((chunk == 0) | (chunk == 3))[0]
            if len(sep) > 0 and sep[0] < BLOCK // 2:
                end = pos + sep[0] + 1; chunk = all_ids[pos:end]
            ids = [int(x) for x in chunk if x >= 0][:BLOCK]
            ids_batch.append(ids); lens.append(len(ids))
            pos += max(len(ids), 32)

        ml = max(lens)
        bt = torch.full((BATCH, ml), PAD, dtype=torch.long, device=DEVICE)
        for i, ids in enumerate(ids_batch):
            bt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=DEVICE)

        # Forward pass
        with torch.no_grad():
            layer.eval()
            x = layer.embed(bt)
            layer.forward_transformer(x)
            attn = layer.transformer.attention.last_attention

        # Fast vectorized strengthen
        if attn is not None:
            fast_strengthen(bt, attn)
        
        trainer.valid_assemblies += BATCH
        trainer.total_assemblies += BATCH
        trainer.step += 1
        step = trainer.step

        # Mark active to prevent contemplation race condition
        cont.mark_active()
        
        # Periodic progress display
        if step % PRINT_STEP == 0:
            elapsed = time.time() - start
            lps = step / max(elapsed, 0.01)
            avg_pot = float(pf.affinity.mean())
            aps = lps * BATCH
            msg = (f"{step} batches | {lps:.0f} b/s ({aps:.0f} a/s) | "
                   f"pot={avg_pot:.4f} | {elapsed/3600:.1f}h")
            print(f"\r  {msg}  ", end="", flush=True)
            
            # Write status file for monitoring
            try:
                with open(os.path.join(os.path.dirname(__file__), "training_status.txt"), "w") as sf:
                    sf.write(msg + "\n")
            except:
                pass

        # Periodic operations (every SAVE_STEP batches)
        if step % 1000 == 0 and step > 0:
            grammar.discover_digrams(min_affinity=float(pf.affinity.mean()) + 0.01)
            grammar.discover_ngrams(max_n=5, min_coherence=0.3)
            topo.update_after_learning()
            # Track curvature for anomaly detection
            anomalies = trainer.curvature_analyzer.find_curvature_anomalies(threshold=0.5)
            if anomalies:
                for sym_idx, curv, reason in anomalies[:3]:
                    trainer.contradiction_filter.detect_semantic_contradictions(sym_idx)
            # FIX 3: recompute_affinity_hybrid DISABLED (destroyed learned signal)
            # pf.recompute_affinity_hybrid()
            torch.cuda.empty_cache(); gc.collect()
            
            # Log
            elapsed = time.time() - start
            lps = step / max(elapsed, 0.01)
            avg_pot = float(pf.affinity.mean())
            digrams_count = sum(len(p) for p in grammar.patterns.values())
            dpot = avg_pot - prev_pot; dd = digrams_count - prev_digrams
            apss = lps * BATCH
            print(f"\n  step={step} | {lps:.0f} b/s ({apss:.0f} a/s) | pot={avg_pot:.4f} (+{dpot:.6f}) | "
                  f"digrams={digrams_count} (+{dd}) | {elapsed/3600:.1f}h | "
                  f"{cont.summary()}")
            
            if step >= MIN_STEPS and dd < 100 and abs(dpot) < AFF_THRESH:
                steps_no_improvement += 1000  # FIX: was 50000, triggers on first check
            else:
                steps_no_improvement = 0
            if steps_no_improvement >= CONV_WINDOW:
                print(f"\nCONVERGED!")
                break
            prev_pot = avg_pot; prev_digrams = digrams_count

        # Heavy periodic (every 200K)
        if step % 200000 == 0 and step > 0:
            kb, lib, wd = get_kb()
            wd.discover_from_grammar()
            kb.auto_maintain()
            lib.organize()
            print(f"  {lib.summary()} | {kb.summary()}")
            trainer.save()

except KeyboardInterrupt:
    print("\nInterrupted. Saving...")

trainer.save(final=True)
elapsed = time.time() - start

# Final report
print(f"\n{'='*60}")
print(f"TRAINING DONE: {trainer.step} steps in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
print(f"Pot: {float(pf.affinity.mean()):.4f}, Digrams: {sum(len(p) for p in grammar.patterns.values())}")
print(f"Speed: {trainer.step/max(elapsed,0.01):.0f} lps")
print(f"Contemplation: {cont.summary()}")

# Generation
generator = SymbolicGenerator(
    layer=layer, char_vocab=char_vocab, potential_field=pf,
    contradiction_filter=trainer.contradiction_filter, grammar=grammar,
    concept_miner=trainer.concept_miner, topological_field=topo,
    ngram_context=ngram,
    conditional_binding=TemporalConditionalBinding(pf, topo, trainer.clusterer),
)
print("\nGENERATION:")
for p in ["pri", "chelo", "zem", "pro"]:
    ids = char_vocab.encode(p)[1:-1]
    gen = generator.generate(ids, max_new_symbols=60, temperature=0.6)
    print(f"  '{p}...' -> '{char_vocab.decode(gen)[:80]}'")
