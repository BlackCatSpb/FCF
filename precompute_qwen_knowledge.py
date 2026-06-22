"""
Precompute Qwen Knowledge for FCF Distillation.

Usage (Cloud GPU — recommended):
  python precompute_qwen_knowledge.py \\
    --corpus real_data/full_corpus_ru_clean.txt \\
    --fcf-bpe real_data/bpe_ru_146k.model \\
    --qwen-model RefalMachine/RuadaptQwen3-4B-Hybrid \\
    --backend transformers \\
    --output real_data/qwen_knowledge.npz

Usage (Local, OpenVINO):
  python precompute_qwen_knowledge.py \\
    --corpus real_data/full_corpus_ru_clean.txt \\
    --fcf-bpe real_data/bpe_ru_146k.model \\
    --qwen-model models/ruadapt_qwen3_4b_openvino_ModelB \\
    --backend openvino

Output: qwen_knowledge.npz with keys:
  - rows: uint32[N] — FCF CID A
  - cols: uint32[N] — FCF CID B
  - vals: float32[N] — mean cosine similarity from Qwen
  - counts: uint32[N] — number of observations (for confidence weighting)
"""

import os, sys, time, gc, json, math
import numpy as np

# ═══════════════════════════════════════════════════
# Character-level token alignment
# ═══════════════════════════════════════════════════

def encode_fcf_spans(text, sp):
    """Tokenize with FCF SentencePiece BPE. Returns (cids, spans)."""
    pieces = sp.EncodeAsPieces(text)
    cids = [sp.PieceToId(p) for p in pieces]
    spans = []
    pos = 0
    for piece in pieces:
        decoded = piece.replace('\u2581', ' ')
        has_space = decoded.startswith(' ')
        if has_space:
            decoded = decoded[1:]
            while pos < len(text) and text[pos].isspace():
                pos += 1
        idx = text.find(decoded, pos)
        if idx == -1:
            idx = pos
        spans.append((idx, idx + len(decoded)))
        pos = idx + len(decoded)
    return cids, spans

def align_and_similarities(text, cids, spans, qw_hidden, qw_spans, context_window):
    """
    Align FCF tokens to Qwen hidden states and compute pairwise similarities.
    Returns list of ((cid_a, cid_b), cos_sim, count).
    """
    n_fcf = len(cids)
    n_qw = len(qw_hidden)

    # Align: each FCF token → aggregated Qwen vector
    fcf_vecs = []
    for fi in range(n_fcf):
        fs, fe = spans[fi]
        vec = None
        n_ov = 0
        for qi in range(n_qw):
            qs, qe = qw_spans[qi]
            if qs < fe and qe > fs:
                h = qw_hidden[qi]
                if vec is None:
                    vec = h.copy()
                else:
                    vec += h
                n_ov += 1
        if vec is not None and n_ov > 0:
            vec /= n_ov
        fcf_vecs.append(vec)

    # Compute pairwise similarities within context_window
    pairs = {}
    for i in range(n_fcf):
        vi = fcf_vecs[i]
        if vi is None:
            continue
        ni = np.linalg.norm(vi)
        if ni < 1e-10:
            continue
        vi_n = vi / ni
        start = max(0, i - context_window)
        end = min(n_fcf, i + context_window + 1)
        for j in range(start, end):
            if j <= i:
                continue
            vj = fcf_vecs[j]
            if vj is None:
                continue
            nj = np.linalg.norm(vj)
            if nj < 1e-10:
                continue
            cos = float(vi_n @ (vj / nj))
            if abs(cos) < 0.15:
                continue
            a, b = (cids[i], cids[j]) if cids[i] <= cids[j] else (cids[j], cids[i])
            key = (a << 32) | b
            if key not in pairs:
                pairs[key] = [cos, 1]
            else:
                pairs[key][0] += cos
                pairs[key][1] += 1

    results = []
    for key, (sum_cos, cnt) in pairs.items():
        a = key >> 32
        b = key & 0xFFFFFFFF
        results.append(((a, b), sum_cos / cnt, cnt))
    return results


# ═══════════════════════════════════════════════════
# Streaming accumulator
# ═══════════════════════════════════════════════════

class StreamingAccum:
    """Accumulates pairs in-memory, flushes to disk periodically."""
    def __init__(self, output_base, max_entries=3_000_000):
        self.data = {}  # key64 → [sum_cos, count]
        self.max_entries = max_entries
        self._flush_n = 0
        self._output_base = output_base
        self._partials = []

    def add(self, results):
        for (a, b), cos, cnt in results:
            key = (a << 32) | b
            if key in self.data:
                self.data[key][0] += cos * cnt
                self.data[key][1] += cnt
            else:
                self.data[key] = [cos * cnt, cnt]

    def should_flush(self):
        return len(self.data) >= self.max_entries

    def flush(self):
        if not self.data:
            return
        n = len(self.data)
        rows = np.empty(n, dtype=np.uint32)
        cols = np.empty(n, dtype=np.uint32)
        vals = np.empty(n, dtype=np.float32)
        cnts = np.empty(n, dtype=np.uint32)
        for i, (key, (sum_cos, count)) in enumerate(self.data.items()):
            rows[i] = key >> 32
            cols[i] = key & 0xFFFFFFFF
            vals[i] = np.float32(sum_cos / count)
            cnts[i] = min(count, 4294967295)
        path = f"{self._output_base}.part{self._flush_n:04d}.npz"
        np.savez_compressed(path, rows=rows, cols=cols, vals=vals, counts=cnts)
        self._partials.append(path)
        self._flush_n += 1
        self.data.clear()


def merge_partials(partials, output_path):
    """Merge all partial files into one final npz."""
    if not partials:
        return
    all_rows, all_cols, all_vals, all_counts = [], [], [], []
    for p in partials:
        d = np.load(p)
        all_rows.append(d['rows'])
        all_cols.append(d['cols'])
        all_vals.append(d['vals'])
        all_counts.append(d['counts'])
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.concatenate(all_vals)
    counts = np.concatenate(all_counts)
    np.savez_compressed(output_path, rows=rows, cols=cols, vals=vals, counts=counts)
    for p in partials:
        try:
            os.remove(p)
        except OSError:
            pass
    print(f"  Saved {len(rows)} pairs to {output_path}")


# ═══════════════════════════════════════════════════
# Qwen backends
# ═══════════════════════════════════════════════════

def _load_qwen_tokenizer(model_path):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def _load_hf_backend(model_path, device=None, num_layers=None, layer_offset=0):
    """HF transformers backend — recommended for cloud GPU."""
    import torch
    from transformers import AutoModelForCausalLM
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else device,
        output_hidden_states=True, low_cpu_mem_usage=True,
    )
    # Slice layers for early exit
    if num_layers is not None:
        total = len(model.model.layers)
        model.model.layers = model.model.layers[layer_offset:layer_offset + num_layers]
        print(f"  Truncated to layers {layer_offset}-{layer_offset + num_layers - 1}/{total}")
    model.eval()
    dev = next(model.parameters()).device
    def infer(qw_ids):
        with torch.no_grad():
            inputs = torch.tensor([qw_ids], device=dev)
            outputs = model(inputs, output_hidden_states=True)
            hs = outputs.hidden_states[-1][0].cpu().numpy()
        return hs
    return infer, model

def _load_ov_backend(model_path):
    """OpenVINO backend — for local CPU/GPU."""
    import openvino as ov
    core = ov.Core()
    xml_path = os.path.join(model_path, "openvino_model.xml")
    model = core.read_model(xml_path)
    model.add_outputs("hidden_states")
    # Try GPU first
    dev = "CPU"
    for d in core.available_devices:
        if 'GPU' in d:
            dev = d
            break
    compiled = core.compile_model(model, dev)
    outs = {p.any_name: p for p in compiled.outputs}
    hs_key = 'hidden_states' if 'hidden_states' in outs else list(outs.keys())[0]
    def infer(qw_ids):
        seq_len = len(qw_ids)
        results = compiled({
            "input_ids": np.array([qw_ids], dtype=np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, seq_len),
            "beam_idx": np.zeros(1, dtype=np.int32),
        })
        return results[outs[hs_key]][0]
    return infer, compiled


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def precompute(corpus_path, fcf_bpe_path, qwen_model_path, output_path,
               backend="transformers", max_lines=0, checkpoint_every=1000,
               context_window=8, device=None, num_layers=None, layer_offset=0):
    import sentencepiece as spm

    # Load FCF BPE
    print("Loading FCF BPE...")
    sp = spm.SentencePieceProcessor()
    sp.Load(fcf_bpe_path)
    V = sp.vocab_size()
    print(f"  vocab_size = {V}")

    # Load Qwen
    print(f"Loading Qwen ({backend})...")
    tokenizer = _load_qwen_tokenizer(qwen_model_path)
    if backend == "transformers":
        infer_fn, _ = _load_hf_backend(qwen_model_path, device=device,
                                        num_layers=num_layers, layer_offset=layer_offset)
    else:
        infer_fn, _ = _load_ov_backend(qwen_model_path)

    # Read corpus
    print("Reading corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_lines > 0:
        lines = lines[:max_lines]
    print(f"  {len(lines)} texts")

    # Process
    accum = StreamingAccum(output_path)
    t_start = time.time()
    n_pairs_total = 0
    n_texts = 0

    for idx, text in enumerate(lines):
        if not text:
            continue

        # 1. Tokenize both
        cids, fcf_spans = encode_fcf_spans(text, sp)
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        qw_ids = enc['input_ids']
        qw_spans = enc['offset_mapping']

        if not qw_ids or not cids:
            continue

        # 2. Qwen inference
        qw_hidden = infer_fn(qw_ids)
        if qw_hidden is None or len(qw_hidden) != len(qw_ids):
            continue

        # 3. Align + compute similarities
        results = align_and_similarities(text, cids, fcf_spans, qw_hidden, qw_spans, context_window)

        # 4. Accumulate
        if results:
            accum.add(results)
            n_pairs_total += len(results)
        n_texts += 1

        # 5. Periodic flush + status
        if accum.should_flush():
            accum.flush()

        if (idx + 1) % checkpoint_every == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(lines) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{100*(idx+1)/len(lines):4.1f}%] {idx+1}/{len(lines)} texts, "
                  f"{n_pairs_total} pairs, {rate:.0f} L/s, "
                  f"ETA {eta/3600:.1f}h, accum={len(accum.data)} pending")

        if (idx + 1) % 100 == 0:
            gc.collect()

    # Final flush
    if accum.data:
        accum.flush()

    # Merge
    print(f"Merging {len(accum._partials)} partial files...")
    merge_partials(accum._partials, output_path)

    total = time.time() - t_start
    print(f"Done: {n_texts} texts, {n_pairs_total} pairs in {total/3600:.1f}h")
    print(f"  Rate: {n_texts/total:.0f} L/s, {n_pairs_total/total:.0f} pairs/s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Precompute Qwen knowledge for FCF")
    p.add_argument("--corpus", required=True)
    p.add_argument("--fcf-bpe", required=True)
    p.add_argument("--qwen-model", required=True)
    p.add_argument("--output", default="qwen_knowledge.npz")
    p.add_argument("--backend", choices=["transformers", "openvino"], default="transformers")
    p.add_argument("--max-lines", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--context-window", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=None,
                   help="Number of transformer layers to use (None=all)")
    p.add_argument("--layer-offset", type=int, default=0,
                   help="Start layer index for slicing")
    args = p.parse_args()
    precompute(**vars(args))
