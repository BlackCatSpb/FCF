"""
EVA Production API — FastAPI сервер для когнитивной генерации.

Статус: PRODUCTION
Архитектура: Coordinate-navigation cognitive AI (не LLM)
MetaWeighter: 3 источника [know, conc, contr]

Endpoints:
  POST /v1/generate         — генерация текста
  WS   /v1/generate/stream  — WebSocket streaming
  POST /v1/train_step       — один шаг continuous learning
  GET  /v1/metrics          — метрики модели
  GET  /v1/health           — health check
"""
import sys, os, json, time, asyncio, logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.char_vocab import CharacterVocab as CharVocab
from eva.symbolic.potential_fields import SemanticRelevanceGate, GradientFlowSolver
from eva.symbolic.trajectory_store import TrajectoryStore
from eva.symbolic.train_v3 import TrainingConfig, MultiTaskLoss
from eva.symbolic.continuous_runtime import continuous_learning_loop, RuntimeConfig
from eva.symbolic.thought_loop import ThoughtLoopConfig

logging.basicConfig(level=logging.INFO, format='[EVA API] %(message)s')
log = logging.getLogger('eva_api')


# ---- Globals (set during startup) ----
model = None
cv = None
device = None
trajectory_store = None
srg_module = None
flow_solver = None
_config_cache = {
    'total_generations': 0,
    'total_train_steps': 0,
    'start_time': time.time(),
    'avg_srg': 0.0,
    'srg_samples': 0,
}


class GenerationRequest(BaseModel):
    prompt: str = ""
    max_tokens: int = 128
    temperature: float = 0.8
    use_thought_loop: bool = False
    use_flow_solver: bool = False


class GenerationResponse(BaseModel):
    text: str
    tokens_generated: int
    srg_score: float
    generation_time_ms: float


class TrainStepRequest(BaseModel):
    n_cycles: int = 1
    max_generations: int = 3
    train_steps_per_cycle: int = 3


class TrainStepResponse(BaseModel):
    cycles_completed: int
    avg_composite: float
    avg_train_loss: float
    trajectories: int


class MetricsResponse(BaseModel):
    model: str
    uptime_hours: float
    total_generations: int
    total_train_steps: int
    trajectories_stored: int
    avg_srg: float
    device: str
    params: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    memory_allocated_mb: float
    memory_reserved_mb: float


def _guard_model():
    if model is None or cv is None:
        raise HTTPException(status_code=503, detail="Model not loaded (startup incomplete)")


def _update_avg_srg(srg_val: float):
    c = _config_cache
    c['avg_srg'] = (c['avg_srg'] * c['srg_samples'] + srg_val) / (c['srg_samples'] + 1)
    c['srg_samples'] += 1


def _eval_srg(text: str) -> float:
    ids = cv.encode_with_boundaries(text)
    if len(ids) < 2:
        return 0.0
    inp = torch.tensor([ids], device=device)
    h_full, _, _, _ = model.forward(inp, return_heads=True)
    with torch.no_grad():
        c_query = h_full[0].mean(dim=0)
        c_response = h_full[0, -1]
        dummy = torch.zeros(model.vocab_size, device=device)
        return float(srg_module.evaluate(
            c_query.unsqueeze(0), c_response.unsqueeze(0), dummy.unsqueeze(0)
        ))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, cv, device, trajectory_store, srg_module, flow_solver
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Loading model on {device}...")

        model = UnifiedMultidimensionalTransformer().to(device)
        # Load latest checkpoint if available
        import glob
        ckpts = sorted(glob.glob('checkpoints/v3/train_v3_step_*.pt'))
        if ckpts:
            latest = ckpts[-1]
            from eva.symbolic.train_v3 import safe_load_state_dict
            ckpt = torch.load(latest, map_location=device, weights_only=True)
            safe_load_state_dict(model, ckpt['model_state'])
            log.info(f"Loaded checkpoint: {latest} (step {ckpt.get('step', '?')})")
        else:
            log.info("No checkpoint found — starting with untrained model")
        model.eval()
        cv = CharVocab()
        trajectory_store = TrajectoryStore(max_trajectories=100000)
        srg_module = SemanticRelevanceGate()
        flow_solver = GradientFlowSolver(eta=0.05, max_steps=20)

        params = sum(p.numel() for p in model.parameters())
        log.info(f"Model loaded: {params:,} params, device={device}")
    except Exception as e:
        log.error(f"Startup failed: {e}")
        model = None
    yield
    log.info("Shutting down...")


app = FastAPI(
    title="EVA Cognitive API",
    version="1.0.0",
    description="Coordinate-navigation cognitive AI — not an LLM",
    lifespan=lifespan,
)


@app.post("/v1/generate", response_model=GenerationResponse)
async def generate(req: GenerationRequest):
    _guard_model()
    t0 = time.time()

    try:
        prompt_ids = cv.encode_with_boundaries(req.prompt) if req.prompt.strip() else [cv.SENT_OPEN_IDX]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Encoding failed: {e}")

    try:
        with torch.no_grad():
            if req.use_thought_loop:
                from eva.symbolic.thought_loop import generate_with_thought
                tc = ThoughtLoopConfig()
                tc.max_iterations = 3
                text, _ = generate_with_thought(
                    model, prompt_ids, cv,
                    config=tc,
                    max_new=req.max_tokens,
                    temperature=req.temperature,
                )
            elif req.use_flow_solver:
                text, _ = model.generate_text(
                    prompt_ids, cv,
                    max_new=req.max_tokens,
                    temperature=req.temperature,
                    flow_solver=flow_solver,
                    kca_cycle=None,
                    srg_module=srg_module,
                )
            else:
                text = model.enhanced_generate(
                    prompt_ids, cv,
                    max_new=req.max_tokens,
                    temperature=req.temperature,
                )
    except Exception as e:
        log.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    elapsed_ms = (time.time() - t0) * 1000

    try:
        srg_val = _eval_srg(text)
        _update_avg_srg(srg_val)
    except Exception as e:
        log.warning(f"SRG eval failed: {e}")
        srg_val = 0.0

    try:
        ids = cv.encode_with_boundaries(text)
        if len(ids) > 0:
            inp = torch.tensor([ids], device=device)
            h_full, _, _, _ = model.forward(inp, return_heads=True)
            if len(h_full[0]) > 0:
                trajectory_store.store(text, ids, h_full[0].cpu().numpy().astype(np.float32))
        _config_cache['total_generations'] += 1
    except Exception as e:
        log.warning(f"Trajectory store failed: {e}")

    return GenerationResponse(
        text=text,
        tokens_generated=len(ids),
        srg_score=srg_val,
        generation_time_ms=round(elapsed_ms, 1),
    )


@app.websocket("/v1/generate/stream")
async def generate_stream(websocket: WebSocket):
    await websocket.accept()
    if model is None or cv is None:
        await websocket.send_json({"error": "Model not loaded"})
        await websocket.close()
        return

    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        max_new = data.get("max_tokens", 128)
        temperature = data.get("temperature", 0.8)

        prompt_ids = cv.encode_with_boundaries(prompt) if prompt.strip() else [cv.SENT_OPEN_IDX]
        ids = list(prompt_ids)

        for pos in range(max_new):
            inp = torch.tensor([ids], device=device)
            h, _, _, heads_out = model.forward(inp, return_heads=True)

            end, nxt, conn = model.boundary_predictor(h[:, -1:])
            z_curr = h[0, -1]
            z_pred = z_curr + nxt[0, 0]

            context = h.mean(dim=1)
            meta_w = model.meta_weighter(context)[0]
            sym_coords = model.embed.coordinates

            bias_tpf = torch.zeros(model.vocab_size, device=device)
            bias_wvf = torch.zeros(model.vocab_size, device=device)
            if len(ids) > 1:
                last_sym = ids[-1]
                if last_sym < model.tensor_potential.num_symbols:
                    bias_tpf = model.tensor_potential.recursive_bias(z_pred, inp[0])
                bias_wvf = model.word_valence.get_valence_bias(z_pred, inp[0]).to(device)

            logits_know = model.decoder(z_pred.unsqueeze(0).unsqueeze(0))[0, 0] + bias_tpf + bias_wvf

            concept_score = heads_out['concept'][0, -1].item()
            contra_score = heads_out['contradiction'][0, -1].item()
            dists = -torch.cdist(z_pred.unsqueeze(0), sym_coords, p=2).squeeze(0)
            logits_conc = dists * (1.0 + concept_score)
            logits_contr = dists * (1.0 - contra_score * 0.5)

            w = meta_w
            final = (w[0] * logits_know + w[1] * logits_conc + w[2] * logits_contr) / temperature

            # ---- Mask special tokens (PAD=0, UNK=1, BOS=2, EOS=3) ----
            final[:4] = -float('inf')

            # ---- Repetition penalty (logits-level) ----
            freq = set(ids)
            for t in freq:
                final[t] -= 1.0

            # ---- Sample from top-20 ----
            sl, si = final.sort(descending=True)
            v, idx = sl[:20], si[:20]
            p = F.softmax(v, dim=-1)
            nt = idx[torch.multinomial(p, 1)].item()

            ids.append(nt)
            token_char = cv.idx_to_char(nt)
            await websocket.send_json({
                "token": nt,
                "char": token_char,
                "pos": pos,
                "concept": round(concept_score, 4),
                "contra": round(contra_score, 4),
            })

            if nt == cv.SENT_CLOSE_IDX:
                break

        text = cv.decode(ids)
        await websocket.send_json({"done": True, "text": text})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@app.post("/v1/train_step", response_model=TrainStepResponse)
async def train_step(req: TrainStepRequest):
    _guard_model()
    t0 = time.time()

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        config = TrainingConfig(
            batch_size=4, seq_len=64,
            drop_ce=True, use_weight_context=False,
        )
        loss_fn = MultiTaskLoss(config)
        runtime_cfg = RuntimeConfig(
            max_generations=req.max_generations,
            train_steps_per_cycle=req.train_steps_per_cycle,
        )
        thought_cfg = ThoughtLoopConfig()
        thought_cfg.max_iterations = 2

        model.train()
        results = continuous_learning_loop(
            model, cv, optimizer, loss_fn,
            n_cycles=req.n_cycles,
            runtime_cfg=runtime_cfg,
            thought_cfg=thought_cfg,
            train_cfg=config,
            log_fn=lambda msg: log.info(f"Train: {msg}"),
        )
        model.eval()

        avg_comp = float(np.mean([r['best_composite'] for r in results])) if results else 0
        avg_loss = float(np.mean([r['train_loss'] for r in results])) if results else 0
        _config_cache['total_train_steps'] += req.n_cycles * req.train_steps_per_cycle

        return TrainStepResponse(
            cycles_completed=req.n_cycles,
            avg_composite=round(avg_comp, 4),
            avg_train_loss=round(avg_loss, 4),
            trajectories=trajectory_store.total_stored,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Train step failed: {e}")
        model.eval()
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.get("/v1/metrics", response_model=MetricsResponse)
async def get_metrics():
    try:
        params = sum(p.numel() for p in model.parameters()) if model else 0
        uptime = (time.time() - _config_cache['start_time']) / 3600
        traj_count = trajectory_store.total_stored if trajectory_store else 0
        return MetricsResponse(
            model="EVA Symbolic v3",
            uptime_hours=round(uptime, 2),
            total_generations=_config_cache['total_generations'],
            total_train_steps=_config_cache['total_train_steps'],
            trajectories_stored=traj_count,
            avg_srg=round(_config_cache['avg_srg'], 4),
            device=str(device),
            params=params,
        )
    except Exception as e:
        log.error(f"Metrics failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    try:
        if not torch.cuda.is_available():
            mem_alloc, mem_reserved = 0, 0
        else:
            mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024
            mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024

        return HealthResponse(
            status="ok" if model is not None else "loading",
            model_loaded=model is not None,
            device=str(device),
            memory_allocated_mb=round(mem_alloc, 1),
            memory_reserved_mb=round(mem_reserved, 1),
        )
    except Exception as e:
        log.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error", model_loaded=False,
            device="unknown", memory_allocated_mb=0, memory_reserved_mb=0,
        )


@app.get("/")
async def root():
    return {
        "name": "EVA Cognitive API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "production",
        "endpoints": [
            "POST /v1/generate",
            "WS  /v1/generate/stream",
            "POST /v1/train_step",
            "GET  /v1/metrics",
            "GET  /v1/health",
        ],
    }


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('EVA_PORT', 8000))
    host = os.environ.get('EVA_HOST', '0.0.0.0')
    log.info(f"Starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
