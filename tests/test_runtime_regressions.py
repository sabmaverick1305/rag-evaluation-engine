from unittest.mock import MagicMock, Mock

import pytest

from src import retrieval
from src.generator import ProductionGenerator


def test_retriever_loads_postgres_cache_and_closes_connection(monkeypatch):
    # The real cursor is a context manager; model initialization is unnecessary here.
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        ("chunk_1", "Supporting text", "book")
    ]
    monkeypatch.setattr(retrieval.psycopg2, "connect", Mock(return_value=connection))
    retriever = retrieval.ProductionGraphRAGRetriever.__new__(retrieval.ProductionGraphRAGRetriever)
    retriever.db_url = "unused"
    assert retriever._load_all_chunks() == {
        "chunk_1": {"id": "chunk_1", "text": "Supporting text", "source_doc": "book"}
    }
    connection.close.assert_called_once()


@pytest.mark.parametrize("output", [
    '{"answer": "Grounded answer", "citations": [], "confidence_score": 0.8}',
    "Not valid JSON",
    '{"answer": "Answer", "citations": "invalid", "confidence_score": 0.8}',
])
def test_structured_generation_handles_valid_and_malformed_outputs(output):
    generator = ProductionGenerator.__new__(ProductionGenerator)
    generator.llm = Mock()
    generator.llm.invoke.return_value = output
    result = generator.generate_grounded_answer("question", [
        {"id": "chunk_1", "text": "Supporting text", "source_doc": "book", "score": 1.0}
    ])
    assert result.answer
    if output.startswith('{"answer": "Grounded'):
        assert result.answer == "Grounded answer"
        assert result.confidence_score == 0.8
    else:
        assert result.citations[0].chunk_id == "chunk_1"
