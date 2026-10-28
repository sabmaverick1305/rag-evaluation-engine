# RAG evaluation

## Compare the fine-tuned generator with the base model

```bash
uv run python -m evaluation.evaluate_finetuning --top-k 3
```

The default adapter is `qwen-graphrag-adapter-final`. Both variants use greedy
decoding, a 512-token response budget, the same chat template, and identical
cached retrieval contexts. The base run disables the adapter via PEFT; the
fine-tuned run enables the saved adapter. Generation runs locally with LangSmith
uploads disabled. The report is saved after each query to
`evaluation/results/finetuning.json` with aggregate metrics, per-query answers,
raw model outputs, citations, timings, hashes, and fine-tuned-minus-base deltas.
Use `--output` to preserve separate runs, `--adapter-only` for the adapted model
alone, or `--limit 2` for a quick smoke check. Partial runs retain `status=running`.
Sequential weight loading is enabled by default to avoid an observed concurrent
safetensors loader deadlock on the local Mac; generation remains unchanged.

Retrieval recall, precision, and MRR use exact chunk IDs, without sibling credit.
Generator-only fine-tuning does not change retrieval. Generation metrics include
schema validity, citation IDs/source documents matching the retrieved context,
target citation recall, literal snippet matching, abstentions, errors, and latency.
Malformed output is counted as invalid rather than repaired into synthetic citations.
Literal snippet matching does not establish semantic faithfulness; paraphrases
can be correct without matching verbatim. Confidence is self-reported, not accuracy.

The current golden set has no reference answers, so answer exact-match and token
F1 are unavailable (`null`). Add `reference_answer` or `expected_answer` per case
to enable those text-based metrics. A held-out answer set and human or semantic
judgments are needed to assess factual correctness and generalization.

Adapter loading follows [PEFT's inference API](https://huggingface.co/docs/peft/guides/peft_model_config).

## Retrieval-only gate

`golden_dataset.json` is the active relevance benchmark. A target chunk must
answer its question; being the chunk used to generate a topic is insufficient.
Template-generated examples require source review: front-matter phrases such as
"Praise Engineering" do not establish answerable questions.

Generate a separate candidate file with:

```bash
uv run python scripts/generate_golden_100.py
```

This writes `evaluation/golden_dataset.draft.json`, sampling across the corpus.
Keyword templates still require review. For each draft, read `source_excerpt`,
rewrite the question so the source actually answers it, and check other chunks
for relevant answers. Remove unanswerable or duplicate cases. Include a short
`target_evidence` mapping from each target ID to supporting source text, then set
`review_status` to `approved`. Copy reviewed cases to the active benchmark.
Source snippets validate label stability, not the meaning of the question.
Choose labels independently of the retriever's rankings.

Run the retrieval-only gate (default top 3):

```bash
uv run pytest evaluation/evaluate_retrieval.py -v -s
```

For an explicit top-10 comparison:

```bash
RETRIEVAL_TOP_K=10 uv run pytest evaluation/evaluate_retrieval.py -v -s
```

The retrieval-only gate allows adjacent chunks from the same document (window 1)
and measures whether any acceptable target was hit. The fine-tuning comparison
above uses strict IDs and recall over all target IDs instead. Compare reports
with the same cutoff and matching policy. The retrieval-only thresholds remain
75% mean hit rate and 0.50 MRR; these do not measure generated-answer quality.
