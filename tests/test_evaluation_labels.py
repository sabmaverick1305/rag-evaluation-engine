import pytest

from evaluation.evaluate_retrieval import load_golden_dataset, validate_golden_dataset


def test_unreviewed_generated_question_cannot_enter_quality_gate():
    cases = [{"query_id": "q1", "review_status": "needs_review", "target_chunk_ids": ["chunk_1"]}]
    with pytest.raises(AssertionError, match="requires source review"):
        validate_golden_dataset(cases, [{"id": "chunk_1", "text": "Source text"}])


def test_missing_golden_chunk_reports_dataset_problem():
    cases = [{"query_id": "q1", "target_chunk_ids": ["missing"]}]
    with pytest.raises(AssertionError, match="golden targets missing"):
        validate_golden_dataset(cases, [{"id": "other", "text": "unrelated"}])


def test_existing_chunk_with_wrong_content_rejects_stale_label():
    cases = [{
        "query_id": "q1",
        "target_chunk_ids": ["chunk_1"],
        "target_evidence": {"chunk_1": "evaluation criteria"},
    }]
    with pytest.raises(AssertionError, match="evidence no longer matches"):
        validate_golden_dataset(cases, [{"id": "chunk_1", "text": "Book endorsements"}])


def test_evidence_allows_pdf_line_wrapping():
    cases = [{
        "query_id": "q1",
        "target_chunk_ids": ["chunk_1"],
        "target_evidence": {"chunk_1": "evaluation criteria"},
    }]
    validate_golden_dataset(cases, [{"id": "chunk_1", "text": "Define evaluation\ncriteria first."}])


def test_dataset_path_is_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert load_golden_dataset()
