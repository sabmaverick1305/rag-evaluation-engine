import os
from pathlib import Path

# Load database, Neo4j, and tracing settings before importing the pipeline.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from typing import Any

from src.schemas import GroundedRAGResponse


class DeepResearchAgent:
    """
    Unified Deep Research Agent.
    Operates in three primary modes:
    - "agentcore": Amazon Bedrock AgentCore runtime hosting autonomous LangGraph agent on AWS.
    - "local": Autonomous multi-step ReAct agent powered by LangGraph.
    - "bedrock_agent": Production AWS cloud agent orchestrated by AWS Bedrock Classic Agent.
    """

    def __init__(
        self,
        mode: str | None = None,
        orchestrator: Any | None = None,
        bedrock_agent_runner: Any | None = None,
        agentcore_runner: Any | None = None,
        local_workflow: Any | None = None,
        agentcore_runtime_arn: str | None = None,
    ):
        self.mode = (
            mode
            or os.getenv("AGENT_MODE")
            or ("agentcore" if os.getenv("BEDROCK_AGENTCORE_RUNTIME_ARN") else None)
            or ("bedrock_agent" if os.getenv("BEDROCK_AGENT_ID") else "local")
        ).lower()

        self.orchestrator = orchestrator
        self.bedrock_agent_runner = bedrock_agent_runner
        self.agentcore_runner = agentcore_runner
        self.local_workflow = local_workflow
        self.agentcore_runtime_arn = agentcore_runtime_arn or os.getenv("BEDROCK_AGENTCORE_RUNTIME_ARN")

        # Lazy initialization based on mode
        if self.mode == "agentcore" and self.agentcore_runner is None:
            try:
                from src.aws.bedrock_agentcore_runtime import BedrockAgentCoreRuntime
                self.agentcore_runner = BedrockAgentCoreRuntime(runtime_arn=self.agentcore_runtime_arn)
            except Exception as e:
                print(f"Warning: Could not initialize BedrockAgentCoreRuntime: {e}")

        elif self.mode == "bedrock_agent" and self.bedrock_agent_runner is None:
            try:
                from src.aws.bedrock_agent_runtime import BedrockAgentRunner
                self.bedrock_agent_runner = BedrockAgentRunner()
            except Exception as e:
                print(f"Warning: Could not initialize BedrockAgentRunner: {e}")

        elif self.mode in ("local", "langgraph") and self.local_workflow is None:
            try:
                from src.agent_workflow import LocalGraphRAGWorkflow
                self.local_workflow = LocalGraphRAGWorkflow()
            except Exception as e:
                print(f"Warning: Could not initialize LocalGraphRAGWorkflow: {e}")

    def run_research(self, user_query: str) -> GroundedRAGResponse:
        print(f"\n[Agent Pipeline] Step 1: Receiving query -> '{user_query}'")

        # 1. Amazon Bedrock AgentCore Runtime mode
        if self.mode == "agentcore" and self.agentcore_runner is not None:
            print("[Agent Pipeline] Delegating research orchestration to Amazon Bedrock AgentCore Runtime...")
            return self.agentcore_runner.invoke_runtime(user_query)

        # 2. AWS Bedrock Classic Agent mode
        if self.mode == "bedrock_agent" and self.bedrock_agent_runner is not None:
            print("[Agent Pipeline] Delegating research orchestration to AWS Bedrock Agent...")
            return self.bedrock_agent_runner.invoke_agent(user_query)

        # 3. Local LangGraph autonomous multi-step workflow mode
        if self.local_workflow is not None:
            print("[Agent Pipeline] Executing autonomous LangGraph multi-step ReAct workflow...")
            return self.local_workflow.run(user_query)

        # 4. Direct Orchestrator fallback
        if self.orchestrator is not None:
            print("[Agent Pipeline] Executing single-pass hybrid orchestrator...")
            result = self.orchestrator.run(user_query, top_k=3)
            if not result.citations:
                print("[Warning] No direct citations found. Attempting broad fallback query...")
                result = self.orchestrator.run(f"Overview of {user_query}", top_k=5)
            return result

        raise RuntimeError("No operational runner (AgentCore, Bedrock Agent, LangGraph, or Orchestrator) configured.")

if __name__ == "__main__":
    db_url = os.getenv("DATABASE_URL", "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag")

    agent = DeepResearchAgent(mode="local")
    query = "Explain how multi-head attention and transformer blocks function in model architectures."
    final_output = agent.run_research(query)

    print("\n================ FINAL AGENT OUTPUT ================")
    print(final_output.model_dump_json(indent=2))
