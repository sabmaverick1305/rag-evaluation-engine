import os
import json
from pathlib import Path
import pytest
from src.state import RAGState
from src.retrieval import ProductionGraphRAGRetriever

def load_golden_dataset():
    path = Path(__file__).with_name("golden_dataset.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def validate_golden_dataset(test_cases, chunks):
    """Reject stale labels before interpreting a failure as poor retrieval."""
    if isinstance(chunks, dict):
        chunks = chunks.values()
    corpus = {chunk["id"]: chunk["text"] for chunk in chunks}
    for case in test_cases:
        query_id = case["query_id"]
        assert case.get("review_status") in (None, "approved"), (
            f"{query_id}: draft question requires source review before use in the quality gate."
        )
        targets = set(case["target_chunk_ids"])
        assert targets, f"{query_id}: target_chunk_ids must not be empty."
        missing = targets - corpus.keys()
        assert not missing, (
            f"{query_id}: golden targets missing from the ingested corpus: {sorted(missing)}. "
            "Check the database, document name, and chunking settings."
        )
        for chunk_id, evidence in case.get("target_evidence", {}).items():
            assert chunk_id in targets, f"{query_id}: evidence references a non-target chunk: {chunk_id}"
            assert evidence.strip() and " ".join(evidence.split()) in " ".join(corpus[chunk_id].split()), (
                f"{query_id}: evidence no longer matches {chunk_id}. "
                "Review the golden labels against the current source text."
            )

def test_production_retrieval_evaluation():
    db_url = os.getenv("DATABASE_URL", "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag")
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    assert top_k > 0, "RETRIEVAL_TOP_K must be positive."
    test_cases = load_golden_dataset()
    
    if not test_cases:
        pytest.skip("Golden dataset is empty or missing. Add evaluation cases to run tests.")

    # Initialize the production retriever connected to live Postgres
    retriever = ProductionGraphRAGRetriever(db_url=db_url)
    validate_golden_dataset(test_cases, retriever.chunks)

    recall_scores = []
    precision_scores = []
    reciprocal_ranks = []
    total_queries = len(test_cases)
    diagnostics = []

    for case in test_cases:
        state = RAGState(query=case["query"])
        result_state = retriever.retrieve(state, top_k=top_k)

        retrieved_candidates = result_state.retrieved_candidates
        retrieved_ids = [chunk["id"] for chunk in retrieved_candidates]
        target_ids = set(case["target_chunk_ids"])

        # 1. Context Recall: Check if any retrieved candidate is an acceptable match (including siblings)
        matched = any(is_acceptable_match(rid, target_ids, window=1) for rid in retrieved_ids)
        recall = 1.0 if matched else 0.0
        recall_scores.append(recall)

        # 2. Context Precision: What % of retrieved chunks are acceptable matches?
        relevant_retrieved = sum(1 for rid in retrieved_ids if is_acceptable_match(rid, target_ids, window=1))
        precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0
        precision_scores.append(precision)

        # 3. Mean Reciprocal Rank (MRR): Position of the first acceptable match
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if is_acceptable_match(rid, target_ids, window=1):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
        
        missing_status = [] if matched else sorted(target_ids)
        diagnostics.append(
            f"{case['query_id']}: recall={recall:.2f}, rr={rr:.3f}; "
            f"expected={sorted(target_ids)}; retrieved={retrieved_ids}; "
            f"missing={missing_status}"
        )

    mean_recall = sum(recall_scores) / total_queries
    mean_precision = sum(precision_scores) / total_queries
    mrr = sum(reciprocal_ranks) / total_queries

    print("\n========================================")
    print(f"PRODUCTION HYBRID GRAPH RAG EVALUATION (top_k={top_k})")
    print("========================================")
    print(f"Mean Context Recall@{top_k}    : {mean_recall * 100:.1f}%")
    print(f"Mean Context Precision@{top_k} : {mean_precision * 100:.1f}%")
    print(f"Mean Reciprocal Rank@{top_k}   : {mrr:.3f}")
    print("========================================\n")
    details = "\n".join(diagnostics)
    print(details)

    # CI/CD Quality Gate Assertions
    assert mean_recall >= 0.75, f"Context recall {mean_recall:.2f} is below the 75% threshold!\n{details}"
    assert mrr >= 0.50, f"MRR ranking quality {mrr:.3f} is below threshold!\n{details}"

def is_acceptable_match(retrieved_id: str, target_ids: set, window: int = 1) -> bool:
    """Checks if the retrieved chunk matches the target or is an adjacent sibling chunk."""
    if retrieved_id in target_ids:
        return True
    
    # Parse chunk index (e.g., 'ai_engineering_ch1_chunk_1206' -> 1206)
    try:
        retrieved_prefix, retrieved_suffix = retrieved_id.rsplit("_", 1)
        retrieved_idx = int(retrieved_suffix)
    except ValueError:
        return False
    for tid in target_ids:
        try:
            target_prefix, target_suffix = tid.rsplit("_", 1)
            target_idx = int(target_suffix)
        except ValueError:
            continue
        if retrieved_prefix == target_prefix and abs(retrieved_idx - target_idx) <= window:
            return True
    return False
