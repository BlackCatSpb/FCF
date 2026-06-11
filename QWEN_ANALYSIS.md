# QWEN3-4B Architecture Analysis → FCF Insights

## 1. QWEN Architecture (from config.json + OpenVINO)

```
vocab_size:          146,260 (BPE)
hidden_size:         2,560
num_layers:          36 (all full_attention)
num_heads:           32 (GQA: 8 KV heads, 4:1 ratio)
head_dim:            128
intermediate_size:   9,728 (SwiGLU FFN, ≈3.8× hidden)
max_position:        40,960
rope_theta:          1,000,000
rms_norm_eps:        1e-6
tie_word_embeddings: true

Model size: 2.15GB (OpenVINO, u4+f16 mixed quant)
```

## 2. Tokenizer

- Qwen2Tokenizer (BPE)
- 146,260 tokens, 35 special (IDs 151643-151677)
- includes `<fractal_*>` tokens at IDs 151665-151677 (13 tokens)
- chat template: `<|im_start|>role\ncontent<|im_end|>`
- max_length: 131,072

## 3. OpenVINO layer ops (11,759 total)

| Op | Count | Purpose |
|----|-------|---------|
| Const | 3,146 | weights |
| MatMul | 254 | ~7 per layer (QKV + output + FFN gate + FFN up + FFN down + RoPE embed × 2) |
| ScaledDotProductAttention | 36 | 1 per layer |
| Swish | 36 | 1 per layer (SwiGLU gate) |
| RMSNorm ops (Sqrt+Divide+ReduceMean+Power) | 145×4 | 4 per layer, 1 extra |
| RoPE ops (Broadcast+Slice+Transpose) | 145×3 | cos/sin computation |

## 4. FCF Insights

### 4.1 Vocabulary
QWEN: 146K tokens, fine-grained BPE
FCF:  32K tokens, coarse SentencePiece
→ FCF's sparsity is a design choice: each token carries more semantic weight.
→ Anchor matrix H[1024×1024] covers the top-1024 most meaningful tokens.

### 4.2 Position encoding (RoPE → z_a rotation)
QWEN uses rotary position encoding: each attention head rotates queries/keys by pos×θ.
FCF can add position to z_a subspace:

```python
# RoPE-inspired position in z_a (128 dim, split into 64 pairs)
pos_angle = position_ids * inv_theta  # θ = 10000^(2i/128)
z_a_rotated = z_a * cos(pos_angle) + rotate_half(z_a) * sin(pos_angle)
```

No learned embeddings, no position table — pure rotation in z_a space.

### 4.3 GQA (Grouped Query Attention) → Anchor sharing
QWEN: 32 Q heads, 8 KV heads (4:1 ratio)
FCF:  N_a = 1024 anchors. Each anchor's field acts as a KV head.
      Multiple concepts share the same anchor field via H matrix.
      Ratio: V / N_a = 32K / 1K = 32:1 — even sparser.

### 4.4 SwiGLU → Meta-gate
QWEN's SwiGLU: output = (silu(xW_gate) ⊙ xW_up) W_down
FCF meta-gate:  lr_mod = sigmoid(z_m · w_lr)
                 gate = sigmoid(z_m · w_gate + b)
Both use element-wise gating, but FCF's is per-concept, not per-layer.

### 4.5 RMSNorm → Unit sphere
QWEN normalizes: x / sqrt(mean(x²) + ε) × γ
FCF normalizes:  v / ||v||
Both remove magnitude, keep direction. FCF's approach has no learned params.

### 4.6 Layer structure
QWEN: 36 sequential layers, each re-encoding full hidden state:
  h = RMSNorm(h + attention(RMSNorm(h)))
  h = RMSNorm(h + FFN(RMSNorm(h)))

FCF: single hierarchical field, subspaces in parallel:
  z_c (identity) — lr 0.01×  (≈ embedding, stable)
  z_a (attention) — lr 1.0×  (≈ attention, fast)
  z_m (meta) — lr 0.1×       (≈ gate params, medium)

FCF compresses 36 sequential layers into 3 parallel subspaces with
different plasticity rates. The "depth" is hierarchical (BMSSP levels),
not sequential.

### 4.7 Weight tying
QWEN: embed_tokens.weight = lm_head.weight
FCF:  FractalField codes form both encoder (cid→code) and
      decoder (code→vector). Already naturally tied.

## 5. Key Takeaways for Implementation

1. **z_a should use RoPE-like position encoding** — no separate position embeddings needed
2. **GQA insight**: anchor field sharing — not every concept needs its own field matrix
3. **RMSNorm-less**: unit sphere normalization is sufficient, remove learned γ
4. **Parallel subspaces > sequential layers**: 3 subspaces with different lr replace 36 layers
5. **Meta-gate as SwiGLU-lite**: sigmoid gate, no need for full activation
6. **Vocabulary size mismatch**: our BPE 32K vs QWEN 146K → need to ensure our tokenizer captures enough granularity, OR accept the sparsity tradeoff
