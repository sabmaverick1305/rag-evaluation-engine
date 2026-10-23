import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import generate_golden_100 as generator


def generate(monkeypatch, tmp_path, rows):
    connection = Mock()
    connection.cursor.return_value.fetchall.return_value = rows
    monkeypatch.setattr(generator.psycopg2, "connect", Mock(return_value=connection))
    path = tmp_path / "draft.json"
    generator.generate_100_golden_dataset("unused", path)
    return json.loads(path.read_text())


def test_candidates_cover_corpus_and_require_review(monkeypatch, tmp_path):
    rows = [(f"doc_chunk_{i}", f"Example passage number {i}") for i in reversed(range(2379))]
    cases = generate(monkeypatch, tmp_path, rows)
    ids = [case["target_chunk_ids"][0] for case in cases]
    assert len(set(ids)) == 100
    assert ids[0] == "doc_chunk_0"
    assert ids[-1] == "doc_chunk_2378"
    assert all(case["review_status"] == "needs_review" for case in cases)
    assert all(case["source_excerpt"] for case in cases)


def test_small_corpus_does_not_repeat_chunks_to_reach_100(monkeypatch, tmp_path):
    cases = generate(monkeypatch, tmp_path, [("chunk_1", "Some source content")])
    assert len(cases) == 1


def test_generator_cannot_overwrite_active_benchmark():
    path = Path(generator.__file__).resolve().parents[1] / "evaluation" / "golden_dataset.json"
    with pytest.raises(ValueError, match="review questions and labels"):
        generator.generate_100_golden_dataset("unused", path)
