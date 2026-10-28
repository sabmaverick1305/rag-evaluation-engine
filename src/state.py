from typing import Any

from pydantic import BaseModel, Field


class RAGState(BaseModel):
    query: str
    rewritten_query: str | None = None
    retrieved_candidates: list[dict[str, Any]] = Field(default_factory=list)
    reranked_chunks: list[dict[str, Any]] = Field(default_factory=list)
    kg_augmented_context: list[dict[str, Any]] = Field(default_factory=list)
    final_context_window: str = ""
    generated_answer: str | None = None
