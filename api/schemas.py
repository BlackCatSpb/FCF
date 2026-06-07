"""API schemas for FCF model serving."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List


class GenerateRequest(BaseModel):
    prompt: str = Field(default="", description="Query text")
    seed_word: Optional[str] = Field(default=None, description="Single seed word")
    max_words: int = Field(default=20, ge=1, le=200, description="Max words to generate")
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    beam_width: int = Field(default=3, ge=1, le=10)


class GenerateResponse(BaseModel):
    text: str
    concept_path: List[int]
    hormones: Optional[Dict[str, float]] = None
    confidence: float
    intent_anchor: Optional[str] = None
    semantic_delta: float = 0.0


class HealthResponse(BaseModel):
    status: str
    concepts: int
    dimensions: int
    transitions: int
