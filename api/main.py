"""FastAPI server for FCF concept model."""
import sys, os, time, threading
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager

from api.schemas import GenerateRequest, GenerateResponse, HealthResponse
from model.modeling_fcf import FCFModel
from model.configuration_fcf import FCFConfig


model: FCFModel = None
_trained_lines = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading FCF model...")
    config = FCFConfig()
    model = FCFModel(config)
    model._load()
    print(f"  Loaded: {len(model.space.concept_vectors)} concepts @ {model.space.dim}D")
    yield
    print("Shutting down.")


_rate_limit = defaultdict(list)
_rate_limit_lock = threading.Lock()
RATE_LIMIT_PER_MIN = 10

def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        _rate_limit[client_ip] = [t for t in _rate_limit[client_ip] if now - t < 60]
        if len(_rate_limit[client_ip]) >= RATE_LIMIT_PER_MIN:
            return False
        _rate_limit[client_ip].append(now)
    return True


app = FastAPI(title="FCF Concept Model", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    if not _check_rate_limit(request.client.host):
        raise HTTPException(429, "Rate limit exceeded")
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return HealthResponse(
        status="ok",
        concepts=len(model.space.concept_vectors),
        dimensions=model.space.dim,
        transitions=_trained_lines,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request):
    if not _check_rate_limit(request.client.host):
        raise HTTPException(429, "Rate limit exceeded")
    if model is None:
        raise HTTPException(503, "Model not loaded")

    if not req.prompt and not req.seed_word:
        raise HTTPException(400, "Provide prompt or seed_word")

    result = model.generate(
        prompt=req.prompt or None,
        seed_word=req.seed_word,
        max_words=req.max_words,
        temperature=req.temperature,
    )
    global _trained_lines
    _trained_lines += len(result.text.split())
    return GenerateResponse(
        text=result.text,
        concept_path=result.concept_path,
        hormones=result.hormones,
        confidence=result.confidence,
        intent_anchor=result.intent_anchor,
        semantic_delta=result.semantic_delta,
    )


@app.post("/chat", response_model=GenerateResponse)
async def chat(req: GenerateRequest, request: Request):
    """Alias for /generate — chat-compatible endpoint."""
    if not _check_rate_limit(request.client.host):
        raise HTTPException(429, "Rate limit exceeded")
    return await generate(req, request)
