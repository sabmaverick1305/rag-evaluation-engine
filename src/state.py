from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RAGState(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    retrieved_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    reranked_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    kg_augmented_context: List[Dict[str, Any]] = Field(default_factory=list)
    final_context_window: str = ""
    generated_answer: Optional[str] = None