"""Compare base and adapted generation on identical retrieved golden contexts."""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

from src.schemas import GroundedRAGResponse

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    return " ".join(text.casefold().split())


def score_case(case, candidates, response, diagnostics):
    targets = set(case["target_chunk_ids"])
    ids = [c["id"] for c in candidates]
    context = {c["id"]: c for c in candidates}
    cited = {c.chunk_id for c in response.citations if c.chunk_id in context and c.source_doc == context[c.chunk_id]["source_doc"]}
    valid = sum(c.chunk_id in context and c.source_doc == context[c.chunk_id]["source_doc"] for c in response.citations)
    verbatim = sum(
        c.chunk_id in context and bool(c.relevant_snippet.strip())
        and normalize(c.relevant_snippet) in normalize(context[c.chunk_id]["text"])
        for c in response.citations
    )
    reference = case.get("reference_answer") or case.get("expected_answer")
    exact = f1 = None
    if reference:
        answer_tokens = Counter(re.findall(r"\w+", response.answer.casefold()))
        reference_tokens = Counter(re.findall(r"\w+", reference.casefold()))
        overlap = sum((answer_tokens & reference_tokens).values())
        denominator = sum(answer_tokens.values()) + sum(reference_tokens.values())
        f1 = 2 * overlap / denominator if denominator else 1.0
        exact = float(normalize(response.answer) == normalize(reference))
    return {
        "retrieval_recall": len(targets & set(ids)) / len(targets),
        "retrieval_precision": len(targets & set(ids)) / len(set(ids)) if ids else 0.0,
        "reciprocal_rank": next((1 / rank for rank, cid in enumerate(ids, 1) if cid in targets), 0.0),
        "schema_valid": bool(diagnostics.get("schema_valid")),
        "generation_skipped": bool(diagnostics.get("generation_skipped")),
        "citation_count": len(response.citations),
        "context_valid_citations": valid,
        "verbatim_snippets": verbatim,
        "target_citation_recall": len(targets & cited) / len(targets),
        "nonempty_answer": bool(response.answer.strip()),
        "abstained": "cannot answer" in response.answer.casefold(),
        "self_reported_confidence": response.confidence_score if diagnostics.get("schema_valid") else None,
        "answer_exact_match": exact,
        "answer_token_f1": f1,
    }


def summarize(rows):
    def mean(key):
        values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
        return statistics.mean(values) if values else None
    citations = sum(r["metrics"]["citation_count"] for r in rows)
    times = sorted(r["generation_seconds"] for r in rows)
    attempted = [r for r in rows if not r["metrics"]["generation_skipped"]]
    return {
        "cases": len(rows),
        "retrieval_recall": mean("retrieval_recall"),
        "retrieval_precision": mean("retrieval_precision"),
        "mrr": mean("reciprocal_rank"),
        "schema_valid_rate_all_cases": mean("schema_valid"),
        "schema_valid_rate_attempted": statistics.mean(r["metrics"]["schema_valid"] for r in attempted) if attempted else None,
        "generation_skipped_rate": mean("generation_skipped"),
        "citation_precision_in_context": sum(r["metrics"]["context_valid_citations"] for r in rows) / citations if citations else None,
        "target_citation_recall": mean("target_citation_recall"),
        "verbatim_snippet_rate": sum(r["metrics"]["verbatim_snippets"] for r in rows) / citations if citations else None,
        "total_citations": citations,
        "nonempty_answer_rate": mean("nonempty_answer"),
        "abstention_rate": mean("abstained"),
        "mean_self_reported_confidence": mean("self_reported_confidence"),
        "reference_answer_cases": sum(r["metrics"]["answer_exact_match"] is not None for r in rows),
        "answer_exact_match": mean("answer_exact_match"),
        "answer_token_f1": mean("answer_token_f1"),
        "mean_generation_seconds": statistics.mean(times),
        "p95_generation_seconds": times[max(0, (95 * len(times) + 99) // 100 - 1)],
        "mean_end_to_end_seconds": statistics.mean(r["retrieval_seconds"] + r["generation_seconds"] for r in rows),
        "error_rate": statistics.mean(bool(r.get("error")) for r in rows),
    }


def write_markdown_report(report, path):
    variants = {name: result["summary"] for name, result in report["variants"].items() if result.get("summary")}
    if not variants:
        path.write_text(f"# Fine-tuning evaluation\n\nStatus: **{report['status']}**. No answers completed yet.\n")
        return
    top_k = report["configuration"]["top_k"]
    metrics = [
        (f"Exact target recall@{top_k}", "retrieval_recall", "percent"),
        (f"Exact target precision@{top_k}", "retrieval_precision", "percent"),
        (f"MRR@{top_k}", "mrr", "number"),
        ("Valid structured output", "schema_valid_rate_all_cases", "percent"),
        ("Citation IDs/source documents match context", "citation_precision_in_context", "percent"),
        ("Target citation recall", "target_citation_recall", "percent"),
        ("Verbatim snippet matching", "verbatim_snippet_rate", "percent"),
        ("Execution error rate", "error_rate", "percent"),
        ("Answer exact match (requires references)", "answer_exact_match", "percent"),
        ("Answer token F1 (requires references)", "answer_token_f1", "number"),
        ("Mean generation seconds", "mean_generation_seconds", "number"),
        ("P95 generation seconds", "p95_generation_seconds", "number"),
        ("Mean retrieval + generation seconds", "mean_end_to_end_seconds", "number"),
    ]
    lines = ["# Fine-tuning evaluation", "", f"Status: **{report['status']}**. Greedy decoding; {report['configuration']['max_new_tokens']} output tokens maximum.", "", "| Metric | " + " | ".join(variants) + " |", "|---|" + "---:|" * len(variants)]
    for label, key, kind in metrics:
        values = []
        for summary in variants.values():
            value = summary[key]
            values.append("N/A" if value is None else f"{100 * value:.1f}%" if kind == "percent" else f"{value:.3f}")
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines.extend(["", "Both variants reuse the same retrieved contexts. Retrieval scores cannot improve from generator-only fine-tuning.", "", "Citation ID/source matching does not establish that the answer is factually correct. Literal snippet matching can reject valid paraphrases. Confidence is self-reported. Missing answer references prevent correctness scoring.", "", f"Neo4j status: `{json.dumps(report.get('neo4j', {}))}`.", "", "## Invalid structured-output cases", ""])
    for name, result in report["variants"].items():
        invalid = [r["query_id"] for r in result["results"] if not r["metrics"]["schema_valid"]]
        lines.append(f"- {name}: {', '.join(invalid) if invalid else 'none'} ({len(result['results'])} evaluated queries)")
    lines.extend(["", f"[Full per-query results and raw outputs]({path.with_suffix('.json').name})", ""])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/golden_dataset.json")
    parser.add_argument("--adapter", type=Path, default=ROOT / "qwen-graphrag-adapter-final")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/finetuning.json")
    args = parser.parse_args()
    if args.top_k < 1 or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("top-k, max-new-tokens, and limit must be positive")
    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text())
    if adapter_config.get("base_model_name_or_path") != args.model:
        parser.error("Adapter base model differs from --model")
    cases = json.loads(args.dataset.read_text())
    if args.limit:
        cases = cases[:args.limit]
    if not cases or any(not c.get("target_chunk_ids") for c in cases):
        parser.error("The dataset must contain cases with nonempty target_chunk_ids")
    if len({c["query_id"] for c in cases}) != len(cases):
        parser.error("query_id values must be unique")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import os
    # Avoid a safetensors/GIL deadlock observed with concurrent tensor loading.
    # This only changes weight loading, not generation or the model parameters.
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    import torch
    from langsmith import tracing_context
    from src.generator import ProductionGenerator
    from src.guardrails import InputGuardrail, OutputGuardrail
    from src.retrieval import ProductionGraphRAGRetriever
    from src.state import RAGState
    from evaluation.evaluate_retrieval import validate_golden_dataset

    # Fail with the actual connection error before loading large models.
    from contextlib import closing
    import psycopg2
    db_url = os.getenv("DATABASE_URL", "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag")
    with closing(psycopg2.connect(db_url, connect_timeout=10)) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT chunk_id, content, source_doc FROM document_chunks")
            preflight_chunks = [{"id": row[0], "text": row[1], "source_doc": row[2]} for row in cursor.fetchall()]
    validate_golden_dataset(cases, preflight_chunks)
    del preflight_chunks
    torch.manual_seed(42)
    retriever = ProductionGraphRAGRetriever(db_url=db_url)
    graph_status = {}
    try:
        with retriever.neo4j_client.driver.session() as session:
            record = session.run('MATCH (n) RETURN sum(CASE WHEN "Entity" IN labels(n) THEN 1 ELSE 0 END) AS entities, sum(CASE WHEN "Chunk" IN labels(n) THEN 1 ELSE 0 END) AS chunks').single()
            graph_status = {"entity_nodes": record["entities"], "chunk_nodes": record["chunks"]}
    except Exception as exc:
        graph_status = {"probe_error": type(exc).__name__}
    chunks = list(retriever.chunks.values()) if isinstance(retriever.chunks, dict) else retriever.chunks
    validate_golden_dataset(cases, chunks)
    contexts = []
    for index, case in enumerate(cases, 1):
        query = InputGuardrail.sanitize_and_redact(case["query"])
        start = time.perf_counter()
        state = retriever.retrieve(RAGState(query=query), top_k=args.top_k)
        contexts.append((case, query, state.retrieved_candidates, time.perf_counter() - start))
        print(f"Retrieved {index}/{len(cases)}: {case['query_id']}", flush=True)
    corpus_hash = hashlib.sha256(json.dumps(sorted(chunks, key=lambda c: c["id"]), sort_keys=True).encode()).hexdigest()
    retriever.neo4j_client.close()
    del retriever, chunks
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    generator = ProductionGenerator(model_id=args.model, adapter_path=str(args.adapter), do_sample=False, max_new_tokens=args.max_new_tokens)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "neo4j": graph_status,
        "configuration": {"model": args.model, "adapter": str(args.adapter.resolve()), "top_k": args.top_k, "max_new_tokens": args.max_new_tokens, "do_sample": False, "seed": 42, "matching": "exact chunk IDs", "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(), "corpus_sha256": corpus_hash, "adapter_sha256": hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()},
        "limitations": ["Citation ID/source checks and literal snippet matching do not measure semantic faithfulness or answer correctness.", "Self-reported confidence is not accuracy.", "Reference answer metrics are null when reference_answer/expected_answer labels are absent.", "Retrieval is cached identically across both variants; generator fine-tuning cannot change retrieval metrics.", "This run does not establish a held-out split independent of the training dataset."],
        "variants": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        temporary.replace(args.output)
        write_markdown_report(report, args.output.with_suffix(".md"))
    save()
    for variant in (["finetuned"] if args.adapter_only else ["base", "finetuned"]):
        rows = []
        report["variants"][variant] = {"summary": None, "results": rows}
        with (generator.model.disable_adapter() if variant == "base" else nullcontext()), tracing_context(enabled=False):
            for index, (case, query, candidates, retrieval_seconds) in enumerate(contexts, 1):
                diagnostics = {}
                start = time.perf_counter()
                error = None
                try:
                    response = generator.generate_grounded_answer(query, candidates, diagnostics=diagnostics)
                    response.answer = OutputGuardrail.verify_and_filter(response.answer, candidates)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    diagnostics["schema_valid"] = False
                    response = GroundedRAGResponse(answer="", citations=[], confidence_score=0)
                elapsed = time.perf_counter() - start
                rows.append({"query_id": case["query_id"], "query": case["query"], "target_chunk_ids": case["target_chunk_ids"], "retrieved_candidates": candidates, "response": response.model_dump(), "diagnostics": diagnostics, "metrics": score_case(case, candidates, response, diagnostics), "retrieval_seconds": retrieval_seconds, "generation_seconds": elapsed, "error": error})
                report["variants"][variant]["summary"] = summarize(rows)
                save()
                print(f"{variant} {index}/{len(cases)}: {case['query_id']} {elapsed:.1f}s schema_valid={diagnostics.get('schema_valid', False)}", flush=True)
    if len(report["variants"]) == 2:
        base = report["variants"]["base"]["summary"]
        tuned = report["variants"]["finetuned"]["summary"]
        report["delta_finetuned_minus_base"] = {k: tuned[k] - base[k] for k in base if isinstance(base[k], (int, float)) and isinstance(tuned[k], (int, float))}
    report["status"] = "complete"
    save()
    for name, variant in report["variants"].items():
        print(name, json.dumps(variant["summary"], indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
