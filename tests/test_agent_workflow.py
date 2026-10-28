from unittest.mock import MagicMock, Mock

import pytest

from src.agent_workflow import LocalGraphRAGWorkflow
from src.schemas import Citation, GroundedRAGResponse


@pytest.fixture
def mock_retriever():
    retriever = Mock()
    mock_state = MagicMock()
    mock_state.retrieved_candidates = [
        {
            "id": "chunk_101",
            "text": "Multi-head attention combines multiple attention heads to attend to information from different representation subspaces.",
            "source_doc": "ch2_attention.pdf",
            "score": 0.92,
        }
    ]
    retriever.retrieve.return_value = mock_state
    retriever._run_hybrid_search.return_value = [
        {
            "id": "chunk_101",
            "text": "Multi-head attention combines multiple attention heads.",
            "source_doc": "ch2_attention.pdf",
        }
    ]
    return retriever


@pytest.fixture
def mock_generator():
    generator = Mock()
    generator.generate_grounded_answer.return_value = GroundedRAGResponse(
        answer="Multi-head attention jointly attends to information from different representation subspaces.",
        citations=[
            Citation(
                chunk_id="chunk_101",
                source_doc="ch2_attention.pdf",
                relevant_snippet="Multi-head attention combines multiple attention heads.",
            )
        ],
        confidence_score=0.95,
    )
    return generator


def test_local_workflow_happy_path(mock_retriever, mock_generator):
    workflow = LocalGraphRAGWorkflow(
        retriever=mock_retriever,
        generator=mock_generator,
        max_iterations=2,
    )

    query = "How does multi-head attention work?"
    result = workflow.run(query)

    assert isinstance(result, GroundedRAGResponse)
    assert "Multi-head attention" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "chunk_101"
    assert result.confidence_score == 0.95

    # Verify retriever was invoked with sanitized query
    assert mock_retriever.retrieve.called
    assert mock_generator.generate_grounded_answer.called


def test_local_workflow_redacts_pii(mock_retriever, mock_generator):
    workflow = LocalGraphRAGWorkflow(
        retriever=mock_retriever,
        generator=mock_generator,
    )

    query = "My email is test@company.com. Explain attention mechanisms."
    workflow.run(query)

    # Check that retrieve was called with redacted query
    call_args = mock_retriever.retrieve.call_args[0]
    called_state = call_args[0]
    assert "[REDACTED_EMAIL]" in called_state.query
    assert "test@company.com" not in called_state.query


def test_local_workflow_broadens_query_on_empty_context(mock_generator):
    retriever = Mock()
    # First call returns empty, second call returns content
    empty_state = MagicMock()
    empty_state.retrieved_candidates = []

    content_state = MagicMock()
    content_state.retrieved_candidates = [
        {"id": "chunk_200", "text": "Broad overview evidence.", "source_doc": "overview.pdf"}
    ]

    retriever.retrieve.side_effect = [empty_state, content_state]

    workflow = LocalGraphRAGWorkflow(
        retriever=retriever,
        generator=mock_generator,
        max_iterations=2,
    )

    workflow.run("Obscure specialized topic")

    assert retriever.retrieve.call_count == 2
    # Verify second call used broadened query
    second_call_state = retriever.retrieve.call_args_list[1][0][0]
    assert "Overview of" in second_call_state.query


def test_local_workflow_circuit_breaker_on_retriever_failure(mock_generator):
    retriever = Mock()
    retriever.retrieve.side_effect = RuntimeError("Database connection timed out")
    retriever._run_hybrid_search.return_value = [
        {"id": "fallback_chunk", "text": "Fallback text from cache.", "source_doc": "fallback"}
    ]

    workflow = LocalGraphRAGWorkflow(
        retriever=retriever,
        generator=mock_generator,
    )

    # Should not raise exception; should trigger hybrid fallback
    result = workflow.run("Query during DB issue")
    assert isinstance(result, GroundedRAGResponse)
    assert retriever._run_hybrid_search.called
