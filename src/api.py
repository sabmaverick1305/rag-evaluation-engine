import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from typing import Any

from fastapi import FastAPI, HTTPException
from langsmith import traceable
from langsmith import utils as langsmith_utils
from pydantic import BaseModel

from src.guardrails import InputGuardrail, OutputGuardrail
from src.orchestrator import ProductionGraphRAGOrchestrator
from src.schemas import GroundedRAGResponse
from src.state import RAGState

app = FastAPI(
    title="Production Graph RAG API",
    description="Hybrid Search + Neo4j KG + Cross-Encoder Reranker + AWS Bedrock Agent Core & Bedrock LLM",
    version="1.0.0"
)

# Initialize database retriever and generator on startup
DB_URL = os.getenv("DATABASE_URL", "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag")
orchestrator = ProductionGraphRAGOrchestrator(db_url=DB_URL)
retriever = orchestrator.retriever
generator = orchestrator.generator

# Optional Bedrock Agent Core client
_bedrock_agent_runner = None

def get_bedrock_agent_runner():
    global _bedrock_agent_runner
    if _bedrock_agent_runner is None:
        from src.aws.bedrock_agent_runtime import BedrockAgentRunner
        _bedrock_agent_runner = BedrockAgentRunner()
    return _bedrock_agent_runner

class QueryRequest(BaseModel):
    query: str
    top_k: int = 1

class BedrockAgentQueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    enable_trace: bool = True

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_candidates: list[dict[str, Any]]

@traceable(name="Hybrid-Graph-Retriever", run_type="retriever")
def traced_retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
    """Traces the multi-stage dense, sparse, KG traversal and cross-encoder reranking."""
    state = RAGState(query=query)
    result_state = retriever.retrieve(state, top_k=top_k)
    return result_state.retrieved_candidates

@traceable(name="Grounded-LLM-Generator", run_type="llm")
def traced_generate(query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Traces the prompt construction and LLM generation step with confidence fallback."""
    return generator.generate_grounded_answer(query, candidates).model_dump()

@traceable(name="GraphRAG-Pipeline", run_type="chain")
def run_rag_pipeline(query: str, top_k: int) -> dict[str, Any]:
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

@app.post("/query", response_model=GroundedRAGResponse)
async def query_endpoint(request: QueryRequest):
    try:
        # Execute end-to-end orchestrated pipeline
        result = orchestrator.run(request.query, request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/research", response_model=GroundedRAGResponse)
async def agent_research_endpoint(request: QueryRequest):
    """
    Executes the autonomous multi-step Deep Research Agent.
    Routes to Local LangGraph ReAct workflow or AWS Bedrock Agent Core based on AGENT_MODE.
    """
    try:
        from src.main_agent import DeepResearchAgent
        agent = DeepResearchAgent()
        result = agent.run_research(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bedrock-agent/query", response_model=GroundedRAGResponse)
async def bedrock_agent_query_endpoint(request: BedrockAgentQueryRequest):
    """Directly invokes the AWS Bedrock Agent Core deployed in the cloud."""
    try:
        runner = get_bedrock_agent_runner()
        result = runner.invoke_agent(
            query=request.query,
            session_id=request.session_id,
            enable_trace=request.enable_trace,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    enabled = bool(langsmith_utils.tracing_is_enabled())
    has_key = bool(langsmith_utils.get_env_var("API_KEY"))
    llm_provider = os.getenv("LLM_PROVIDER", "local")
    bedrock_agent_id = os.getenv("BEDROCK_AGENT_ID")
    return {
        "status": "healthy",
        "database": "connected",
        "tracing": "configured" if enabled and has_key else "disabled" if not enabled else "missing_api_key",
        "tracing_project": langsmith_utils.get_tracer_project() if enabled else None,
        "llm_provider": llm_provider,
        "agent_mode": os.getenv("AGENT_MODE", "local"),
        "bedrock_agent_id": bedrock_agent_id,
        "bedrock_model_id": os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0") if llm_provider == "bedrock" else None,
    }
