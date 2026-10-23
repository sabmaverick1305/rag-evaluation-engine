import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
import dotenv
from fastapi.testclient import TestClient
from langsmith import Client, get_current_run_tree, tracing_context
from langsmith import utils as langsmith_utils

from src import generator, retrieval
from src.state import RAGState


@pytest.fixture
def api_client(monkeypatch):
    # Keep unit tests independent of local credentials and external trace uploads.
    monkeypatch.setattr(dotenv, "load_dotenv", Mock(return_value=False))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")
    langsmith_utils.get_env_var.cache_clear()
    langsmith_utils.get_tracer_project.cache_clear()
    retriever = Mock()
    llm = Mock()
    monkeypatch.setattr(retrieval, "ProductionGraphRAGRetriever", Mock(return_value=retriever))
    monkeypatch.setattr(generator, "ProductionGenerator", Mock(return_value=llm))
    # Load independently so test doubles do not persist in the application's module.
    path = Path(__file__).resolve().parents[1] / "src" / "api.py"
    spec = importlib.util.spec_from_file_location("api_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with TestClient(module.app) as client:
        yield client, retriever, llm
    langsmith_utils.get_env_var.cache_clear()
    langsmith_utils.get_tracer_project.cache_clear()


def test_query_returns_response_matching_declared_schema(api_client):
    client, retriever, llm = api_client
    chunks = [{"id": "chunk_1", "text": "Supporting text"}]
    retriever.retrieve.return_value = RAGState(query="question", retrieved_candidates=chunks)
    llm.generate_answer.return_value = {"answer": "Grounded answer"}
    response = client.post("/query", json={"query": "question"})
    assert response.status_code == 200
    assert response.json() == {
        "query": "question", "answer": "Grounded answer", "retrieved_candidates": chunks,
    }


def test_no_context_returns_generator_fallback(api_client):
    client, retriever, llm = api_client
    retriever.retrieve.return_value = RAGState(query="question")
    llm.generate_answer.return_value = {"answer": "No sufficiently relevant context was found."}
    response = client.post("/query", json={"query": "question"})
    assert response.status_code == 200
    assert response.json()["retrieved_candidates"] == []
    llm.generate_answer.assert_called_once_with("question", [])


def test_unexpected_failure_returns_500(api_client):
    client, retriever, _ = api_client
    retriever.retrieve.side_effect = RuntimeError("Database unavailable")
    response = client.post("/query", json={"query": "question"})
    assert response.status_code == 500


def test_health_does_not_claim_disabled_tracing_is_active(api_client):
    client, _, _ = api_client
    data = client.get("/health").json()
    assert data["tracing"] == "disabled"
    assert data["tracing_project"] is None


def test_health_reports_configured_project_without_credentials(api_client, monkeypatch):
    client, _, _ = api_client
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-secret")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    langsmith_utils.get_env_var.cache_clear()
    langsmith_utils.get_tracer_project.cache_clear()
    response = client.get("/health")
    assert response.json()["tracing"] == "configured"
    assert response.json()["tracing_project"] == "test-project"
    assert "test-secret" not in response.text


def test_pipeline_records_nested_retrieval_and_generation_spans(api_client):
    client, retriever, llm = api_client
    spans = []

    def retrieve(state, top_k):
        run = get_current_run_tree()
        spans.append((run.name, run.parent_run_id, run.trace_id))
        return state

    def generate(query, candidates):
        run = get_current_run_tree()
        spans.append((run.name, run.parent_run_id, run.trace_id))
        return {"answer": "Test answer"}

    retriever.retrieve.side_effect = retrieve
    llm.generate_answer.side_effect = generate
    trace_client = Mock(spec=Client)
    with tracing_context(enabled="local", client=trace_client):
        assert client.post("/query", json={"query": "question"}).status_code == 200
    assert [span[0] for span in spans] == ["Hybrid-Graph-Retriever", "Grounded-LLM-Generator"]
    assert spans[0][1] is not None
    assert spans[0][1] == spans[1][1] == spans[0][2] == spans[1][2]
