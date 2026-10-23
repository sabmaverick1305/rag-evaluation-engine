import os
import json
import psycopg2

CURATED_QUESTIONS = [
    {
        "query_id": "eval_001",
        "query": "What did OpenAI's 2021 experiment find regarding sampling more outputs for test problems?",
        "keyword_hint": "OpenAI's experiment"
    },
    {
        "query_id": "eval_002",
        "query": "Who first introduced the retrieve-then-generate pattern in 2017?",
        "keyword_hint": "retrieve-then-generate pattern"
    },
    {
        "query_id": "eval_003",
        "query": "Who coined the term retrieval-augmented generation (RAG)?",
        "keyword_hint": "Retrieval-Augmented Generation for Knowledge-Intensive"
    },
    {
        "query_id": "eval_004",
        "query": "How does RAG serve a purpose similar to feature engineering in classical machine learning models?",
        "keyword_hint": "feature engineering for classical ML models"
    },
    {
        "query_id": "eval_005",
        "query": "What are the two main functions of a retriever?",
        "keyword_hint": "indexing and querying"
    },
    {
        "query_id": "eval_006",
        "query": "How is term frequency (TF) defined in term-based retrieval?",
        "keyword_hint": "term frequency (TF)"
    },
    {
        "query_id": "eval_007",
        "query": "How is inverse document frequency (IDF) calculated?",
        "keyword_hint": "IDF value of the term"
    },
    {
        "query_id": "eval_008",
        "query": "What are two common term-based retrieval solutions mentioned?",
        "keyword_hint": "Elasticsearch and BM25"
    },
    {
        "query_id": "eval_009",
        "query": "What is embedding-based retrieval also known as?",
        "keyword_hint": "semantic retrieval"
    },
    {
        "query_id": "eval_010",
        "query": "What are the two steps involved in querying a vector database for semantic retrieval?",
        "keyword_hint": "Embedding model: convert the query"
    },
    {
        "query_id": "eval_011",
        "query": "What approximate nearest neighbor algorithm uses K-means clustering to organize vectors into clusters?",
        "keyword_hint": "IVF (inverted file index)"
    },
    {
        "query_id": "eval_012",
        "query": "Which tree-based approximate nearest neighbor approach builds binary trees using random splitting and was open-sourced by Spotify?",
        "keyword_hint": "Annoy (Approximate Nearest Neighbors"
    },
    {
        "query_id": "eval_013",
        "query": "How do term-based retrieval and embedding-based retrieval compare in querying speed and performance?",
        "keyword_hint": "Comparing retrieval algorithms"
    },
    {
        "query_id": "eval_014",
        "query": "Why is reranking especially useful in a retrieval pipeline?",
        "keyword_hint": "reduce the number of retrieved documents"
    },
    {
        "query_id": "eval_015",
        "query": "What is query rewriting also known as?",
        "keyword_hint": "query reformulation, query normalization"
    },
    {
        "query_id": "eval_016",
        "query": "What is the core idea behind contextual retrieval?",
        "keyword_hint": "contextual retrieval is to augment each chunk"
    },
    {
        "query_id": "eval_017",
        "query": "How does Multimodal RAG work using CLIP embeddings?",
        "keyword_hint": "Generate CLIP embeddings for all your data"
    },
    {
        "query_id": "eval_018",
        "query": "What are the three steps in a RAG system that augments context with tabular data?",
        "keyword_hint": "Text-to-SQL: based on the user query"
    },
    {
        "query_id": "eval_019",
        "query": "What does TTFT stand for, and what does it measure in inference performance?",
        "keyword_hint": "Time to First Token"
    },
    {
        "query_id": "eval_020",
        "query": "What does TPOT stand for, and what does it measure?",
        "keyword_hint": "Time per output token"
    },
    {
        "query_id": "eval_021",
        "query": "How does prompt caching help reduce latency and cost for long documents?",
        "keyword_hint": "prompt cache stores these overlapping segments"
    },
    {
        "query_id": "eval_022",
        "query": "What is the difference between exact caching and semantic caching?",
        "keyword_hint": "two major system caching mechanisms: exact caching and semantic caching"
    },
    {
        "query_id": "eval_023",
        "query": "What is the ultimate goal of dataset engineering?",
        "keyword_hint": "goal of dataset engineering is to create a dataset"
    },
    {
        "query_id": "eval_024",
        "query": "How does disaggregating prefill and decode improve inference server efficiency?",
        "keyword_hint": "disaggregate prefill and decode"
    }
]

def build_curated_dataset():
    db_url = os.getenv("DATABASE_URL", "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    curated_dataset = []
    for item in CURATED_QUESTIONS:
        hint = item["keyword_hint"]
        # Search Postgres chunks for the hint
        cursor.execute(
            "SELECT chunk_id FROM document_chunks WHERE content ILIKE %s LIMIT 1;",
            (f"%{hint}%",)
        )
        row = cursor.fetchone()
        if row:
            chunk_id = row[0]
            curated_dataset.append({
                "query_id": item["query_id"],
                "query": item["query"],
                "target_chunk_ids": [chunk_id]
            })
        else:
            print(f"Warning: Hint '{hint}' not found in database for {item['query_id']}.")

    cursor.close()
    conn.close()

    os.makedirs("evaluation", exist_ok=True)
    output_path = "evaluation/golden_dataset.json"
    with open(output_path, "w") as f:
        json.dump(curated_dataset, f, indent=2)

    print(f"Successfully generated curated golden dataset with {len(curated_dataset)} real-world queries at {output_path}!")

if __name__ == "__main__":
    build_curated_dataset()