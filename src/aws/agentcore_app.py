"""Amazon Bedrock AgentCore Application Runtime for GraphRAG Deep Research.

Deploys the autonomous LangGraph agent directly to the Amazon Bedrock AgentCore
Runtime using the official bedrock-agentcore SDK (BedrockAgentCoreApp).
"""

import logging
import os
from typing import Any, Dict, Optional
import uuid

from bedrock_agentcore import BedrockAgentCoreApp, PingStatus
from src.schemas import GroundedRAGResponse

logger = logging.getLogger(__name__)

# Initialize Bedrock AgentCore Starlette Application
app = BedrockAgentCoreApp()

# Workflow singleton
_workflow: Optional[Any] = None


def get_agent_workflow() -> Any:
    """Lazy loader for the LocalGraphRAGWorkflow instance."""
    global _workflow
    if _workflow is None:
        from src.agent_workflow import LocalGraphRAGWorkflow
        from src.aws.bedrock_generator import BedrockGenerator

        model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
        generator = BedrockGenerator(model_id=model_id)
        _workflow = LocalGraphRAGWorkflow(generator=generator)
    return _workflow


def execute_agent_workflow(prompt: str) -> GroundedRAGResponse:
    """Executes the agent workflow and returns a validated GroundedRAGResponse."""
    workflow = get_agent_workflow()
    return workflow.run(prompt)


@app.ping
def ping_handler() -> PingStatus:
    """AgentCore health and readiness check."""
    return PingStatus.HEALTHY


@app.entrypoint
def agentcore_entrypoint(payload: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """Main invocation entrypoint for Amazon Bedrock AgentCore Runtime.

    Accepts invocation payload from bedrock-agentcore runtime client,
    executes autonomous multi-step reasoning, and returns GroundedRAGResponse.
    """
    if not isinstance(payload, dict):
        return {
            "result": GroundedRAGResponse(
                answer="Invalid payload format: expected JSON object.",
                citations=[],
                confidence_score=0.0,
            ).model_dump(),
            "session_id": str(uuid.uuid4()),
        }

    # Extract prompt from various payload formats
    prompt = payload.get("prompt") or payload.get("query") or payload.get("input") or ""
    session_id = payload.get("session_id") or str(uuid.uuid4())

    prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not prompt:
        return {
            "result": GroundedRAGResponse(
                answer="No prompt provided in invocation request.",
                citations=[],
                confidence_score=0.0,
            ).model_dump(),
            "session_id": session_id,
        }

    try:
        response: GroundedRAGResponse = execute_agent_workflow(prompt)
        return {
            "result": response.model_dump(),
            "session_id": session_id,
        }
    except Exception as exc:
        logger.error(f"Error executing AgentCore workflow: {exc}", exc_info=True)
        return {
            "result": GroundedRAGResponse(
                answer=f"Error executing research workflow: {str(exc)}",
                citations=[],
                confidence_score=0.0,
            ).model_dump(),
            "session_id": session_id,
        }


if __name__ == "__main__":
    # Start AgentCore development server
    port = int(os.getenv("PORT", "8080"))
    app.run(port=port)
