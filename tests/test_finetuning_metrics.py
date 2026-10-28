from evaluation.evaluate_finetuning import score_case, summarize
from src.schemas import Citation, GroundedRAGResponse


def response(citations):
    return GroundedRAGResponse(answer="A grounded answer", citations=citations, confidence_score=0.8)


def test_recall_accounts_for_all_targets_and_uses_exact_ids():
    case = {"target_chunk_ids": ["doc_chunk_1", "doc_chunk_2"]}
    chunks = [{"id": "doc_chunk_1", "source_doc": "doc", "text": "evidence"}]
    metrics = score_case(case, chunks, response([]), {"schema_valid": True})
    assert metrics["retrieval_recall"] == 0.5
    assert metrics["retrieval_precision"] == 1.0
    assert metrics["answer_exact_match"] is None


def test_citations_to_other_sources_do_not_count_as_grounded():
    case = {"target_chunk_ids": ["doc_chunk_1"]}
    chunks = [{"id": "doc_chunk_1", "source_doc": "doc", "text": "actual evidence"}]
    citations = [Citation(chunk_id="doc_chunk_1", source_doc="wrong source", relevant_snippet="invented text")]
    metrics = score_case(case, chunks, response(citations), {"schema_valid": True})
    assert metrics["context_valid_citations"] == 0
    assert metrics["verbatim_snippets"] == 0
    assert metrics["target_citation_recall"] == 0


def test_no_citations_and_no_reference_answers_are_not_reported_as_perfect():
    case = {"target_chunk_ids": ["doc_chunk_1"]}
    metrics = score_case(case, [], response([]), {"schema_valid": False})
    summary = summarize([{"metrics": metrics, "generation_seconds": 1, "retrieval_seconds": 2, "error": None}])
    assert summary["citation_precision_in_context"] is None
    assert summary["verbatim_snippet_rate"] is None
    assert summary["answer_exact_match"] is None
    assert summary["answer_token_f1"] is None
    assert summary["schema_valid_rate_all_cases"] == 0
    assert summary["mean_end_to_end_seconds"] == 3
