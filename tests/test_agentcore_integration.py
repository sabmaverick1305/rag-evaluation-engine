"""Hermetic unit tests for Amazon Bedrock AgentCore integration.

Tests cover:
1. BedrockAgentCoreApp entrypoint invocation and ping handlers.
2. BedrockAgentCoreRuntime boto3 client wrapper and response parsing.
3. DeepResearchAgent routing to AgentCore mode.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from bedrock_agentcore import PingStatus
from src.schemas import GroundedRAGResponse, Citation
from src.aws.agentcore_app import agentcore_entrypoint, ping_handler
from src.aws.bedrock_agentcore_runtime import BedrockAgentCoreRuntime
from src.main_agent import DeepResearchAgent


def test_agentcore_app_ping():
    """Verify BedrockAgentCoreApp ping handler returns PingStatus.HEALTHY."""
    status = ping_handler()
    assert status == PingStatus.HEALTHY


@patch("src.aws.agentcore_app.execute_agent_workflow")
def test_agentcore_app_entrypoint_success(mock_execute_workflow):
    """Verify BedrockAgentCoreApp entrypoint successfully handles prompt payload."""
    mock_response = GroundedRAGResponse(
        answer="Multi-head attention allows the model to jointly attend to information.",
        citations=[
            Citation(chunk_id="chunk_01", source_doc="attention.pdf", relevant_snippet="Attention is all you need.")
        ],
        confidence_score=0.95,
    )
    mock_execute_workflow.return_value = mock_response

    payload = {
        "prompt": "Explain multi-head attention",
        "session_id": "session_test_123",
    }
    result = agentcore_entrypoint(payload, context=None)

    assert "result" in result
    assert result["result"]["answer"] == mock_response.answer
    assert result["result"]["confidence_score"] == 0.95
    assert len(result["result"]["citations"]) == 1
    assert result["session_id"] == "session_test_123"
    mock_execute_workflow.assert_called_once()


def test_agentcore_app_entrypoint_empty_payload():
    """Verify BedrockAgentCoreApp entrypoint safely handles empty or invalid prompt."""
    payload = {"prompt": "   "}
    result = agentcore_entrypoint(payload, context=None)

    assert "result" in result
    assert result["result"]["confidence_score"] == 0.0
    assert "no prompt" in result["result"]["answer"].lower() or "empty" in result["result"]["answer"].lower()


def test_agentcore_runtime_client_invoke_success():
    """Verify BedrockAgentCoreRuntime client calls boto3 invoke_agent_runtime and parses response."""
    mock_boto_client = MagicMock()
    mock_payload = {
        "result": {
            "answer": "GraphRAG leverages knowledge graph ontologies and vector search.",
            "citations": [
                {"chunk_id": "c1", "source_doc": "graph.pdf", "relevant_snippet": "GraphRAG integrates ontologies."}
            ],
            "confidence_score": 0.92,
        },
        "session_id": "sess_456",
    }
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(mock_payload).encode("utf-8")

    mock_boto_client.invoke_agent_runtime.return_value = {
        "statusCode": 200,
        "runtimeSessionId": "sess_456",
        "response": mock_body,
    }

    client = BedrockAgentCoreRuntime(
        runtime_arn="arn:aws:bedrock-agentcore:us-east-1:159142082968:runtime/rag-agent",
        client=mock_boto_client,
    )

    response = client.invoke_runtime(prompt="What is GraphRAG?")
    assert isinstance(response, GroundedRAGResponse)
    assert response.confidence_score == 0.92
    assert "GraphRAG leverages" in response.answer
    assert len(response.citations) == 1

    mock_boto_client.invoke_agent_runtime.assert_called_once()
    call_kwargs = mock_boto_client.invoke_agent_runtime.call_args.kwargs
    assert call_kwargs["agentRuntimeArn"] == "arn:aws:bedrock-agentcore:us-east-1:159142082968:runtime/rag-agent"
    assert call_kwargs["contentType"] == "application/json"


def test_agentcore_runtime_client_missing_arn_raises():
    """Verify invoking without runtime ARN raises ValueError."""
    client = BedrockAgentCoreRuntime(runtime_arn="", client=MagicMock())
    with pytest.raises(ValueError, match="runtime_arn"):
        client.invoke_runtime(prompt="test")


@patch.object(BedrockAgentCoreRuntime, "invoke_runtime")
def test_deep_research_agent_routes_to_agentcore(mock_invoke):
    """Verify DeepResearchAgent with mode='agentcore' routes invocation to BedrockAgentCoreRuntime."""
    expected = GroundedRAGResponse(
        answer="AgentCore runs autonomous agents on AWS managed runtime.",
        citations=[],
        confidence_score=0.9,
    )
    mock_invoke.return_value = expected

    agent = DeepResearchAgent(
        mode="agentcore",
        agentcore_runtime_arn="arn:aws:bedrock-agentcore:us-east-1:159142082968:runtime/rag-agent",
    )
    result = agent.run_research("How does AgentCore work?")

    assert result.answer == expected.answer
    assert result.confidence_score == 0.9
    mock_invoke.assert_called_once_with("How does AgentCore work?")
