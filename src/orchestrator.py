import os
from typing import Dict, Any
from langsmith import traceable
from src.state import RAGState
from src.retrieval import ProductionGraphRAGRetriever
from src.generator import ProductionGenerator
from src.guardrails import InputGuardrail, OutputGuardrail
from src.schemas import GroundedRAGResponse

class ProductionGraphRAGOrchestrator:
    def __init__(self, db_url: str = None):
        # 1. Component Definition: Registering models, databases, and retrieval engines
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag")
        print("Initializing orchestrator components...")
        self.retriever = ProductionGraphRAGRetriever(db_url=self.db_url)
        self.generator = ProductionGenerator(model_id="Qwen/Qwen2.5-1.5B-Instruct")

    @traceable(name="Agentic-GraphRAG-Pipeline", run_type="chain")
    def run(self, raw_query: str, top_k: int = 1) -> GroundedRAGResponse:
        """
        Chaining Step-by-Step Execution:
        1. Sanitize query via Input Guardrails.
        2. Retrieve relevant context via Hybrid Search (pgvector + BM25 + Neo4j KG + Cross-Encoder).
        3. Generate grounded response with structured citations.
        4. Verify and filter response via Output Guardrails.
        """
        # Step 1: Input Guardrail & Sanitization
        safe_query = InputGuardrail.sanitize_and_redact(raw_query)
        
        # Step 2: Context Construction / Retrieval
        state = RAGState(query=safe_query)
        result_state = self.retriever.retrieve(state, top_k=top_k)
        candidates = result_state.retrieved_candidates
        
        # Step 3: Grounded Generation (Structured Output with Citations)
        structured_response = self.generator.generate_grounded_answer(safe_query, candidates)
        
        # Step 4: Output Guardrail Verification
        structured_response.answer = OutputGuardrail.verify_and_filter(structured_response.answer, candidates)
        
        return structured_response
