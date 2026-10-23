from types import SimpleNamespace
from unittest.mock import Mock

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import ingest_pdf


def test_pdf_nuls_are_removed_before_embedding_and_storage(monkeypatch):
    reader = SimpleNamespace(pages=[
        SimpleNamespace(extract_text=lambda: "AI\x00 engineering and RAG — café\n第二章"),
        SimpleNamespace(extract_text=lambda: None),
    ])
    monkeypatch.setattr(ingest_pdf, "PdfReader", lambda path: reader)
    connection = Mock()
    monkeypatch.setattr(ingest_pdf.psycopg2, "connect", Mock(return_value=connection))

    # Exercise ingestion without downloading an embedding model or using a database.
    pipeline = ingest_pdf.ProductionPDFIngestion.__new__(ingest_pdf.ProductionPDFIngestion)
    pipeline.db_url = "unused"
    pipeline.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    pipeline.encoder = Mock()
    pipeline.encoder.encode.return_value.tolist.return_value = [0.0] * 384

    pipeline.process_and_store("example.pdf", "example")

    cursor = connection.cursor.return_value
    chunk_params = cursor.execute.call_args_list[0].args[1]
    stored_text = chunk_params[2]
    assert stored_text == "[Page 1]\nAI engineering and RAG — café\n第二章"
    pipeline.encoder.encode.assert_called_once_with(stored_text)
    # Graph extraction must also see the cleaned phrase "AI engineering".
    graph_params = cursor.execute.call_args_list[-1].args[1]
    assert graph_params == ("AI engineering", "RAG", "relates_to", "example_chunk_0")
    connection.commit.assert_called_once()
