"""LangGraph Multi-Step Agent Workflow for GraphRAG Deep Research.

Runs as an autonomous multi-step ReAct agent locally, mapping 1-to-1 to
AWS Bedrock Agent Core tools, schemas, and verification guardrails.
"""

import logging
import os
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.guardrails import InputGuardrail, OutputGuardrail
from src.schemas import GroundedRAGResponse

logger = logging.getLogger(__name__)


class AgentWorkflowState(TypedDict, total=False):
    raw_query: str
    safe_query: str
    research_query: str
    iteration: int
    max_iterations: int
    selected_tool: str
    retrieved_chunks: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    reflection_notes: str
    is_sufficient: bool
    final_response: GroundedRAGResponse


class LocalGraphRAGWorkflow:
    """
    StateGraph-driven autonomous agent running locally.
    Communicates via shared state, executing ReAct loops with circuit-breaking resilience.
    """

    def __init__(
        self,
        retriever: Any | None = None,
        generator: Any | None = None,
        neo4j_client: Any | None = None,
        max_iterations: int = 2,
    ):
        self.retriever = retriever
        self.generator = generator
        self.neo4j_client = neo4j_client
        self.max_iterations = max_iterations
        self._app = self._build_graph()

    def _get_retriever(self) -> Any:
        if self.retriever is None:
            from src.retrieval import ProductionGraphRAGRetriever

            db_url = os.getenv(
                "DATABASE_URL",
                "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag",
            )
            self.retriever = ProductionGraphRAGRetriever(db_url=db_url)
        return self.retriever

    def _get_generator(self) -> Any:
        if self.generator is None:
            llm_provider = os.getenv("LLM_PROVIDER", "local").lower()
            if llm_provider == "bedrock":
                from src.aws.bedrock_generator import BedrockGenerator

                self.generator = BedrockGenerator()
            else:
                from src.generator import ProductionGenerator

                self.generator = ProductionGenerator(model_id="Qwen/Qwen2.5-1.5B-Instruct")
        return self.generator

    def _get_neo4j_client(self) -> Any:
        if self.neo4j_client is None:
            from src.neo4j_client import Neo4jGraphClient

            self.neo4j_client = Neo4jGraphClient()
        return self.neo4j_client

    # -------------------------------------------------------------
    # Tool Execution Handlers (1-to-1 parity with Bedrock Action Group)
    # -------------------------------------------------------------
    def tool_retrieve_and_rerank(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Tool 1: Hybrid search (pgvector + BM25) + Neo4j traversal + Cross-Encoder reranking."""
        try:
            from src.state import RAGState

            retriever = self._get_retriever()
            state = RAGState(query=query)
            result_state = retriever.retrieve(state, top_k=top_k)
            return result_state.retrieved_candidates
        except Exception as e:
            logger.warning(f"Error in retrieve_and_rerank: {e}. Attempting hybrid fallback...")
            return self.tool_hybrid_search(query, top_k=top_k)

    def tool_hybrid_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Tool 2: Dense vector search + BM25 sparse keyword search."""
        try:
            retriever = self._get_retriever()
            return retriever._run_hybrid_search(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"Error in hybrid_search: {e}")
            return []

    def tool_neo4j_traverse(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Tool 3: Neo4j ontology multi-hop traversal."""
        try:
            client = self._get_neo4j_client()
            return client.ontology_multihop_traverse(query, limit=limit)
        except Exception as e:
            logger.warning(f"Error in neo4j_traverse: {e}")
            return []

    # -------------------------------------------------------------
    # LangGraph Nodes
    # -------------------------------------------------------------
    def node_input_guardrail(self, state: AgentWorkflowState) -> dict[str, Any]:
        """Step 1: Sanitize user input and redact PII."""
        raw_query = state.get("raw_query", "")
        safe_query = InputGuardrail.sanitize_and_redact(raw_query)
        logger.info(f"[LangGraph Node: Guardrail] Sanitized query: '{safe_query}'")
        return {
            "safe_query": safe_query,
            "research_query": safe_query,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "retrieved_chunks": [],
            "tool_calls": [],
            "is_sufficient": False,
        }

    def node_planner(self, state: AgentWorkflowState) -> dict[str, Any]:
        """Step 2: Reason about research plan and select tool to invoke."""
        iteration = state.get("iteration", 0)
        chunks = state.get("retrieved_chunks", [])
        is_broadened = state.get("research_query") != state.get("safe_query")

        if iteration == 0 or is_broadened:
            selected_tool = "retrieve_and_rerank"
        elif not chunks:
            selected_tool = "hybrid_search"
        else:
            selected_tool = "neo4j_traverse"

        logger.info(
            f"[LangGraph Node: Planner] Iteration {iteration} -> Selected tool: '{selected_tool}'"
        )
        return {"selected_tool": selected_tool}

    def node_execute_tools(self, state: AgentWorkflowState) -> dict[str, Any]:
        """Step 3: Execute selected tool and accumulate deduplicated context."""
        tool = state.get("selected_tool", "retrieve_and_rerank")
        query = state.get("research_query", state.get("safe_query", ""))
        iteration = state.get("iteration", 0)

        logger.info(f"[LangGraph Node: Tool] Executing '{tool}' for query: '{query}'")

        if tool == "retrieve_and_rerank":
            new_chunks = self.tool_retrieve_and_rerank(query, top_k=3)
        elif tool == "hybrid_search":
            new_chunks = self.tool_hybrid_search(query, top_k=5)
        elif tool == "neo4j_traverse":
            new_chunks = self.tool_neo4j_traverse(query, limit=3)
        else:
            new_chunks = self.tool_retrieve_and_rerank(query, top_k=3)

        if not isinstance(new_chunks, list):
            new_chunks = []

        # Merge and deduplicate chunks by id
        existing_chunks = {c["id"]: c for c in state.get("retrieved_chunks", [])}
        for chunk in new_chunks:
            if isinstance(chunk, dict):
                chunk_id = chunk.get("id") or chunk.get("chunk_id")
                if chunk_id:
                    existing_chunks[chunk_id] = chunk

        updated_chunks = list(existing_chunks.values())
        tool_calls = list(state.get("tool_calls", []))
        tool_calls.append({"tool": tool, "query": query, "results_count": len(new_chunks)})

        return {
            "retrieved_chunks": updated_chunks,
            "tool_calls": tool_calls,
            "iteration": iteration + 1,
        }

    def node_reflection(self, state: AgentWorkflowState) -> dict[str, Any]:
        """Step 4: Reflection & error-correction check on evidence coverage."""
        chunks = state.get("retrieved_chunks", [])
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", self.max_iterations)
        safe_query = state.get("safe_query", "")

        if not chunks and iteration < max_iter:
            logger.info(
                "[LangGraph Node: Reflection] Insufficient evidence. Broadening search query..."
            )
            return {
                "is_sufficient": False,
                "research_query": f"Overview of {safe_query}",
                "reflection_notes": "No chunks found on primary pass. Broadening query.",
            }

        logger.info(
            f"[LangGraph Node: Reflection] Sufficient evidence gathered ({len(chunks)} chunks). Proceeding to synthesis."
        )
        return {
            "is_sufficient": True,
            "reflection_notes": f"Gathered {len(chunks)} grounded chunks.",
        }

    def node_synthesizer(self, state: AgentWorkflowState) -> dict[str, Any]:
        """Step 5: Generate grounded response and enforce output guardrail."""
        safe_query = state.get("safe_query", "")
        chunks = state.get("retrieved_chunks", [])
        generator = self._get_generator()

        logger.info("[LangGraph Node: Synthesizer] Generating grounded answer...")
        response = generator.generate_grounded_answer(safe_query, chunks)

        # Enforce OutputGuardrail verification
        response.answer = OutputGuardrail.verify_and_filter(response.answer, chunks)

        return {"final_response": response}

    def _should_continue(self, state: AgentWorkflowState) -> str:
        """Conditional routing: loop back to planner if evidence is insufficient, else synthesize."""
        if state.get("is_sufficient") or state.get("iteration", 0) >= state.get(
            "max_iterations", self.max_iterations
        ):
            return "synthesizer"
        return "planner"

    def _build_graph(self):
        """Assembles and compiles the StateGraph workflow."""
        graph = StateGraph(AgentWorkflowState)

        # Register nodes
        graph.add_node("input_guardrail", self.node_input_guardrail)
        graph.add_node("planner", self.node_planner)
        graph.add_node("execute_tools", self.node_execute_tools)
        graph.add_node("reflection", self.node_reflection)
        graph.add_node("synthesizer", self.node_synthesizer)

        # Register edges
        graph.set_entry_point("input_guardrail")
        graph.add_edge("input_guardrail", "planner")
        graph.add_edge("planner", "execute_tools")
        graph.add_edge("execute_tools", "reflection")

        # Conditional edge: reflection -> synthesizer OR back to planner
        graph.add_conditional_edges(
            "reflection",
            self._should_continue,
            {
                "synthesizer": "synthesizer",
                "planner": "planner",
            },
        )
        graph.add_edge("synthesizer", END)

        return graph.compile()

    def run(self, query: str) -> GroundedRAGResponse:
        """Executes the multi-step research workflow end-to-end."""
        initial_state: AgentWorkflowState = {
            "raw_query": query,
        }
        final_state = self._app.invoke(initial_state)
        response = final_state.get("final_response")
        if response is None:
            return GroundedRAGResponse(
                answer="Workflow completed without generating a response.",
                citations=[],
                confidence_score=0.0,
            )
        return response
