import json
import os
from typing import Any

import psycopg


class DiverseSyntheticDatasetGenerator:
    def __init__(self, db_url: str):
        self.db_url = db_url

    def fetch_chunks(self, limit: int = 100) -> list[dict[str, Any]]:
        chunks = []
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT chunk_id, content, source_doc FROM document_chunks LIMIT %s;", (limit,))
                for row in cursor.fetchall():
                    chunks.append({
                        "id": row[0],
                        "text": row[1],
                        "source_doc": row[2]
                    })
        return chunks

    def generate_diverse_dataset(self, output_path: str = "data/sft_dataset_diverse.jsonl"):
        chunks = self.fetch_chunks()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        dataset = []
        print(f"Generating diverse instruction dataset from {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            # 1. Standard Extraction Task
            record_standard = {
                "messages": [
                    {"role": "system", "content": "You are a precise technical AI assistant. Answer using ONLY the provided context and output valid JSON matching the schema."},
                    {"role": "user", "content": f"Context:\n{chunk['text']}\n\nQuestion: Summarize the core findings in {chunk['source_doc']}."},
                    {"role": "assistant", "content": json.dumps({
                        "answer": f"Based on the documentation, {chunk['text'].strip()[:200]}",
                        "citations": [{"chunk_id": chunk["id"], "source_doc": chunk["source_doc"], "relevant_snippet": chunk['text'][:150]}],
                        "confidence_score": 0.98
                    })}
                ]
            }
            dataset.append(record_standard)

            # 2. Negative / Out-of-Bounds Example (Teaches model to handle missing context gracefully)
            if i % 3 == 0:
                record_negative = {
                    "messages": [
                        {"role": "system", "content": "You are a precise technical AI assistant. Answer using ONLY the provided context and output valid JSON matching the schema."},
                        {"role": "user", "content": f"Context:\n{chunk['text']}\n\nQuestion: What are the revenue figures for Q4 financial performance?"},
                        {"role": "assistant", "content": json.dumps({
                            "answer": "I cannot answer this based on the provided documents as no relevant financial context was found.",
                            "citations": [],
                            "confidence_score": 0.0
                        })}
                    ]
                }
                dataset.append(record_negative)

        with open(output_path, "w") as f:
            f.writelines(json.dumps(item) + "\n" for item in dataset)

        print(f"Generated {len(dataset)} diverse training examples in {output_path}")

if __name__ == "__main__":
    db_url = os.getenv("DATABASE_URL", "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag")
    gen = DiverseSyntheticDatasetGenerator(db_url)
    gen.generate_diverse_dataset()
