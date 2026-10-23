import os
from pathlib import Path
from dotenv import load_dotenv

# Load local configuration before importing LangSmith or initializing models.
# Explicit shell/deployment settings retain precedence over .env values.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from langsmith import traceable
from langsmith import utils as langsmith_utils
from src.state import RAGState
from src.retrieval import ProductionGraphRAGRetriever
from src.generator import ProductionGenerator
from src.guardrails import InputGuardrail, OutputGuardrail

app = FastAPI(
    title="Production Graph RAG API",
    description="Zero-cost offline Hybrid Search + KG + Cross-Encoder Reranker + Local LLM with LangSmith Observability",
    version="1.0.0"
)

# Initialize database retriever and local generator on startup
DB_URL = os.getenv("DATABASE_URL", "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag")
retriever = ProductionGraphRAGRetriever(db_url=DB_URL)
generator = ProductionGenerator(model_id="Qwen/Qwen2.5-1.5B-Instruct")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 1

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_candidates: List[Dict[str, Any]]

@traceable(name="Hybrid-Graph-Retriever", run_type="retriever")
def traced_retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Traces the multi-stage dense, sparse, KG traversal and cross-encoder reranking."""
    state = RAGState(query=query)
    result_state = retriever.retrieve(state, top_k=top_k)
    return result_state.retrieved_candidates

@traceable(name="Grounded-LLM-Generator", run_type="llm")
def traced_generate(query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Traces the prompt construction and local LLM generation step with confidence fallback."""
    return generator.generate_answer(query, candidates)

@traceable(name="GraphRAG-Pipeline", run_type="chain")
def run_rag_pipeline(query: str, top_k: int) -> Dict[str, Any]:
    # 1. Input Guardrail: Sanitize & Redact PII
    safe_query = InputGuardrail.sanitize_and_redact(query)
    
    # 2. Retrieval Step
    candidates = traced_retrieve(safe_query, top_k)
    
    # 3. Generation Step
    gen_result = traced_generate(safe_query, candidates)
    
    # 4. Output Guardrail: Verify & Filter Response
    final_answer = OutputGuardrail.verify_and_filter(gen_result["answer"], candidates)
    
    return {
        "query": query,
        "answer": final_answer,
        "retrieved_candidates": candidates
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        result = run_rag_pipeline(request.query, request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    enabled = bool(langsmith_utils.tracing_is_enabled())
    has_key = bool(langsmith_utils.get_env_var("API_KEY"))
    return {
        "status": "healthy",
        "database": "connected",
        "tracing": "configured" if enabled and has_key else "disabled" if not enabled else "missing_api_key",
        "tracing_project": langsmith_utils.get_tracer_project() if enabled else None,
    }
