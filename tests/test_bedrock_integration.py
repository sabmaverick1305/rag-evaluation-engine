import json
from unittest.mock import MagicMock, Mock, patch

from src.aws.bedrock_agent_runtime import BedrockAgentRunner
from src.aws.bedrock_generator import BedrockGenerator
from src.aws.lambda_action_group import lambda_handler
from src.main_agent import DeepResearchAgent
from src.schemas import GroundedRAGResponse


def test_bedrock_generator_converse_valid_json():
    mock_client = Mock()
    payload = {
        "answer": "Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
        "citations": [
            {
                "chunk_id": "chunk_101",
                "source_doc": "ai_engineering_ch2",
                "relevant_snippet": "Multi-head attention combines multiple attention heads."
            }
        ],
        "confidence_score": 0.95,
    }
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": json.dumps(payload)}]
            }
        }
    }

    generator = BedrockGenerator(boto3_client=mock_client)
    chunks = [{"id": "chunk_101", "text": "Multi-head attention combines multiple attention heads.", "source_doc": "ai_engineering_ch2"}]
    result = generator.generate_grounded_answer("How does attention work?", chunks)

    assert isinstance(result, GroundedRAGResponse)
    assert "Multi-head attention" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "chunk_101"
    assert result.confidence_score == 0.95


def test_bedrock_generator_markdown_codeblock_cleaning():
    mock_client = Mock()
    raw_markdown = """```json
{
  "answer": "Cleaned response",
  "citations": [],
  "confidence_score": 0.8
}
```"""
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": raw_markdown}]
            }
        }
    }

    generator = BedrockGenerator(boto3_client=mock_client)
    chunks = [{"id": "chunk_1", "text": "Some text", "score": 1.0}]
    result = generator.generate_grounded_answer("query", chunks)
    assert result.answer == "Cleaned response"
    assert result.confidence_score == 0.8


def test_bedrock_generator_skips_when_no_chunks():
    mock_client = Mock()
    generator = BedrockGenerator(boto3_client=mock_client)
    result = generator.generate_grounded_answer("query", [])
    assert "no sufficiently relevant context" in result.answer
    assert result.confidence_score == 0.0
    mock_client.converse.assert_not_called()


def test_bedrock_agent_runner_invoke_streaming():
    mock_client = Mock()
    # Mock event stream returned by invoke_agent
    mock_client.invoke_agent.return_value = {
        "completion": [
            {
                "chunk": {
                    "bytes": json.dumps({
                        "answer": "Bedrock Agent answered via GraphRAG tool.",
                        "citations": [
                            {"chunk_id": "c1", "source_doc": "doc.pdf", "relevant_snippet": "evidence snippet"}
                        ],
                        "confidence_score": 0.92
                    }).encode("utf-8")
                }
            }
        ]
    }

    runner = BedrockAgentRunner(
        agent_id="test-agent-id",
        agent_alias_id="TSTALIASID",
        boto3_client=mock_client
    )

    result = runner.invoke_agent("Explain transformer architecture")
    assert isinstance(result, GroundedRAGResponse)
    assert result.answer == "Bedrock Agent answered via GraphRAG tool."
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"
    assert result.confidence_score == 0.92


def test_bedrock_agent_runner_with_native_citations():
    mock_client = Mock()
    mock_client.invoke_agent.return_value = {
        "completion": [
            {
                "chunk": {
                    "bytes": b"Synthesized technical response based on retrieved docs.",
                    "attribution": {
                        "citations": [
                            {
                                "retrievedReferences": [
                                    {
                                        "location": {
                                            "type": "S3",
                                            "s3Location": {"uri": "s3://my-bucket/doc1.pdf"}
                                        },
                                        "content": {"text": "Retrieved passage content"},
                                        "metadata": {"chunk_id": "chunk_77"}
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        ]
    }

    runner = BedrockAgentRunner(
        agent_id="test-agent-id",
        agent_alias_id="TSTALIASID",
        boto3_client=mock_client
    )

    result = runner.invoke_agent("Explain RAG retrieval")
    assert result.answer == "Synthesized technical response based on retrieved docs."
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "chunk_77"
    assert result.citations[0].source_doc == "s3://my-bucket/doc1.pdf"


def test_lambda_action_group_handler_retrieve_and_rerank():
    event = {
        "actionGroup": "GraphRAGActionGroup",
        "apiPath": "/retrieve_and_rerank",
        "httpMethod": "POST",
        "parameters": [
            {"name": "query", "type": "string", "value": "attention mechanism"},
            {"name": "top_k", "type": "integer", "value": "2"}
        ]
    }

    mock_state = MagicMock()
    mock_state.retrieved_candidates = [
        {"id": "chunk_1", "text": "attention content", "score": 0.88, "source_doc": "test"}
    ]

    with patch("src.aws.lambda_action_group.get_retriever") as mock_get_retriever:
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_state
        mock_get_retriever.return_value = mock_retriever

        response = lambda_handler(event)

        assert response["messageVersion"] == "1.0"
        res = response["response"]
        assert res["actionGroup"] == "GraphRAGActionGroup"
        assert res["httpStatusCode"] == 200

        body = json.loads(res["responseBody"]["application/json"]["body"])
        assert body["status"] == "success"
        assert len(body["data"]["retrieved_candidates"]) == 1
        assert body["data"]["retrieved_candidates"][0]["id"] == "chunk_1"


def test_lambda_action_group_handler_function_schema():
    event = {
        "actionGroup": "GraphRAGActionGroup",
        "function": "sanitize_input",
        "parameters": [
            {"name": "query", "type": "string", "value": "Contact john@example.com for info"}
        ]
    }

    response = lambda_handler(event)
    res = response["response"]
    assert res["httpStatusCode"] == 200
    body = json.loads(res["responseBody"]["application/json"]["body"])
    assert body["status"] == "success"
    assert "[REDACTED_EMAIL]" in body["data"]["safe_query"]


def test_deep_research_agent_routes_to_bedrock_agent(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "bedrock_agent")
    mock_runner = Mock()
    mock_runner.invoke_agent.return_value = GroundedRAGResponse(
        answer="Answer from Bedrock Agent Core",
        citations=[],
        confidence_score=0.9
    )

    agent = DeepResearchAgent(bedrock_agent_runner=mock_runner)
    result = agent.run_research("What is GraphRAG?")

    assert result.answer == "Answer from Bedrock Agent Core"
    mock_runner.invoke_agent.assert_called_once_with("What is GraphRAG?")

