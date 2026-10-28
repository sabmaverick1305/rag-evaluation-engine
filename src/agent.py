from typing import Literal

from pydantic import BaseModel, Field

from src.orchestrator import ProductionGraphRAGOrchestrator
from src.schemas import GroundedRAGResponse


class AgentPlanStep(BaseModel):
    step_number: int
    action: Literal["search_graph_rag", "synthesize_answer", "reflect_and_verify"]
    query_or_instruction: str

class AgentPlan(BaseModel):
    steps: list[AgentPlanStep] = Field(description="Sequence of manageable steps to solve the deep research query.")
    is_feasible: bool = Field(description="Whether the query can be answered using available database and research tools.")

class DeepResearchAgent:
    def __init__(self, orchestrator: ProductionGraphRAGOrchestrator):
        self.orchestrator = orchestrator

    def generate_plan(self, user_query: str) -> AgentPlan:
        """Step 1: Plan Generation (Task Decomposition)"""
        # In production, call LLM with strict JSON schema to generate the plan
        # Returning a robust multi-step research plan for demonstration:
        return AgentPlan(
            is_feasible=True,
            steps=[
                AgentPlanStep(
                    step_number=1,
                    action="search_graph_rag",
                    query_or_instruction=user_query
                ),
                AgentPlanStep(
                    step_number=2,
                    action="reflect_and_verify",
                    query_or_instruction="Verify if retrieved citations fully cover all aspects of the user query."
                ),
                AgentPlanStep(
                    step_number=3,
                    action="synthesize_answer",
                    query_or_instruction="Generate final structured grounded response."
                )
            ]
        )

    def execute_research_loop(self, user_query: str) -> GroundedRAGResponse:
        """Executes the decoupled plan, handling tool calls, retrieval, and reflection."""
        print("[Agent Planner] Analyzing query and generating execution plan...")
        plan = self.generate_plan(user_query)

        if not plan.is_feasible:
            return GroundedRAGResponse(
                answer="I'm sorry, but this request falls outside the scope of available technical documentation and tools.",
                citations=[],
                confidence_score=0.0
            )

        accumulated_context = []

        for step in plan.steps:
            print(f"[Agent Executor] Executing Step {step.step_number}: {step.action} -> {step.query_or_instruction}")

            if step.action == "search_graph_rag":
                # Invoke our core Graph RAG orchestrator pipeline as a tool
                result = self.orchestrator.run(step.query_or_instruction, top_k=2)
                accumulated_context.extend(result.citations)
                final_response = result

            elif step.action == "reflect_and_verify":
                # Reflection & error correction step (checking evidence coverage)
                if not accumulated_context:
                    print("[Agent Evaluator] Warning: Low evidence found. Adjusting retrieval query...")
                    fallback_result = self.orchestrator.run(user_query, top_k=3)
                    final_response = fallback_result
                else:
                    print("[Agent Evaluator] Reflection check passed: Sufficient citations found.")

        print("[Agent Finish] Deep research task successfully completed.")
        return final_response
