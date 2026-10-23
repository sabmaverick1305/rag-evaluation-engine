# Retrieval evaluation

`golden_dataset.json` is the active relevance benchmark. A target chunk must
answer its question; being the chunk used to generate a topic is insufficient.
The current 100 template-generated examples need source review: phrases such as
"Praise Engineering" come from front matter and do not establish answerable questions.

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

Run the current top-1 gate:

```bash
uv run pytest evaluation/evaluate_retrieval.py -v -s
```

For an explicit top-10 comparison:

```bash
RETRIEVAL_TOP_K=10 uv run pytest evaluation/evaluate_retrieval.py -v -s
```

At top 1, with one target per query, recall, precision, and reciprocal rank all
equal the fraction of queries that return that exact target first. The 75%
recall gate therefore requires at least 75 of 100 single-target cases to rank
their target first. Compare results at the same cutoff. The thresholds remain
75% mean recall and 0.50 MRR. A valid larger benchmark can expose real retrieval
failures; retain those failures for analysis rather than changing their labels
to match returned chunks.
