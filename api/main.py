"""FastAPI server for FCF concept model."""
import sys, os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from api.schemas import GenerateRequest, GenerateResponse, HealthResponse
from model.modeling_fcf import FCFModel
from model.configuration_fcf import FCFConfig


model: FCFModel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading FCF model...")
    config = FCFConfig()
    model = FCFModel(config)
    model._load()
    print(f"  Loaded: {len(model.space.cid_list)} concepts @ {model.space.dim}D")
    yield
    print("Shutting down.")


app = FastAPI(title="FCF Concept Model", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return HealthResponse(
        status="ok",
        concepts=len(model.space.cid_list),
        dimensions=model.space.dim,
        transitions=model.space.concept_transitions.nnz if model.space.concept_transitions else 0,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
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
    return GenerateResponse(
        text=result.text,
        concept_path=result.concept_path,
        hormones=result.hormones,
        confidence=result.confidence,
        intent_anchor=result.intent_anchor,
        semantic_delta=result.semantic_delta,
    )


@app.post("/chat", response_model=GenerateResponse)
async def chat(req: GenerateRequest):
    """Alias for /generate — chat-compatible endpoint."""
    return await generate(req)
