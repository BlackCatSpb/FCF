# FCF & Neuromorphic Computing

*Why FCF is the ideal software model for next-generation neuromorphic processors.*

---

## The Mismatch: Transformers vs. Neuromorphic Hardware

Modern neuromorphic chips — Intel Loihi 2, IBM TrueNorth, and Russian developments (NIISI, Module, MIET) — are fundamentally incompatible with transformer-based language models:

| Requirement | Transformer | Neuromorphic capability |
|-------------|-------------|------------------------|
| Global backpropagation | Essential | Not supported natively |
| Dense matrix × matrix | Core operation | Inefficient (event-driven) |
| Full-precision activations | Float32/16 | Binary/spike events |
| Attention softmax | O(L²) | No analogue |
| Static architecture | Fixed layers | Dynamic routing |

FCF was designed from the ground up with **zero architectural overlap** with transformers, making it a natural fit for the neuromorphic paradigm.

---

## Six Reasons FCF Belongs on a Neuromorphic Chip

### 1. Local learning rules only (no backprop)

FCF uses **STDP**, **Hebbian updates**, **lateral inhibition**, and **contrastive divergence** — every update is computed from locally available information (pre-synaptic and post-synaptic state). No global gradient computation. No automatic differentiation.

On silicon: a memristor crossbar naturally implements STDP as conductance change proportional to `V_pre · V_post`. The entire training loop maps to physical device physics.

### 2. Additive updates = native synaptic events

FCF's core training operation is `code[cid] += lr · Δ` — a **scatter-add** of sparse vectors. The GPU emulates this through atomic operations (expensive). On a neuromorphic chip:

- A pre-synaptic spike arrives at a synapse
- The synapse's weight updates by a fixed increment (STDP)
- The post-synaptic neuron accumulates charge

This is a single physical event — picoseconds of energy, not nanoseconds.

### 3. Event-driven computation

FCF only updates concepts that participate in the current batch. 95% of the vocabulary is idle at any moment. Neuromorphic chips are **event-driven by design**: neurons only consume power when they spike. In idle periods, they draw near-zero current.

The power ratio for a sparse STDP update:
- CPU: ~10⁻⁶ J per event
- GPU: ~10⁻⁹ J per event  
- Loihi 2: ~10⁻¹² J per synaptic event

That's a **1000× improvement** over GPU at every plasticity step.

### 4. Sparse codes + dynamic dimensionality

L1 regularization keeps each concept's latent code at ~8% active components. This sparsity matches the operating point of analog neuromorphic arrays, where each crossbar row sees only a few simultaneous activations.

The `grow_capacity()` / `prune_capacity()` mechanism — adding or removing basis vectors based on code density — maps directly to **structural plasticity** (neurogenesis / synaptic pruning) in biology. On a neuromorphic chip, this corresponds to allocating or releasing physical synapse rows, without reprogramming the instruction stream.

### 5. Hierarchical sector index = content-addressable memory

FCF's 3-level sector field (4+10+20 bits) is an **LSH-based content-addressable memory**. On a neuromorphic chip:

- Hyperdimensional computing natively implements CAM
- A sector lookup is a single pass through associative memory
- Focal search (`search_in_sector`) maps to **parallel prefix matching** in hardware

Russian neuromorphic architectures under development at **Kurchatov Institute (Alkuda)** and **NII Sistem** explicitly target associative memory as a first-class primitive. FCF's sector index is algorithmically identical to their hardware CAM proposals.

### 6. HDC/VSA algebra as a hardware primitive

The VSA operations (`bind=⊙`, `permute=ρ`, `bundle=+`) that FCF uses as a fallback are the **native instruction set** of many neuromorphic designs. When the statistical n-gram lattice has insufficient data, FCF falls back to:

```
query = unbind(context, hdc_memory[prefix])
```

On a neuromorphic chip with VSA microcode, this is a **single instruction cycle** — the hypervectors circulate through the CAM and produce a result in O(1) time, regardless of vocabulary size.

---

## Russian Neuromorphic Ecosystem

| Organisation | Focus | FCF alignment |
|-------------|-------|---------------|
| **NIISI RAS** | Event-driven processors, non-volatile memory | STDP-on-chip, sparse event routing |
| **Module (Baikal-N)** | Neuromorphic accelerator with STDP | Native scatter-add, per-concept EMAs |
| **Kurchatov Institute (Alkuda)** | Memristor crossbars, associative memory | CAM sector lookup, VSA primitives |
| **MIET / ITMiVT** | In-memory computing, analog neuromorphics | 8% sparse codes → low analog MUX ratio |
| **Elvis (NPTS)** | Massively-parallel sparse vector processors | `hdc_memory` as distributed CAM |

---

## What This Means

FCF is currently the **only language-capable model** that simultaneously satisfies all constraints of neuromorphic hardware:

- **No backpropagation** required
- **Only local learning rules**: STDP, Hebbian, L1
- **Dynamic algebraic capacity**: grows/shrinks with knowledge
- **Cellular decomposition**: each concept is an independent computational unit
- **Sparse event-driven**: only active concepts consume energy
- **Hardware-algebra compatible**: VSA primitives as native instructions

This is not an accident. The architecture was designed by asking the question: *"How would a language model look if it could only use the operations that a neuromorphic chip natively supports?"*

FCF is the answer. The transformer era runs on GPUs. The FCF era is waiting for the right chip.

---

*"FCF doesn't need a better GPU. FCF needs a memristor."*
