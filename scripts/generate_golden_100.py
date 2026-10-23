import os
import json
import re
from pathlib import Path
import psycopg2

def generate_100_golden_dataset(db_url: str, output_path=None):
    """Generate review candidates, not validated relevance judgments."""
    evaluation_dir = Path(__file__).resolve().parents[1] / "evaluation"
    output_path = Path(output_path) if output_path is not None else evaluation_dir / "golden_dataset.draft.json"
    if output_path.resolve() == (evaluation_dir / "golden_dataset.json").resolve():
        raise ValueError("Generate to a draft file; review questions and labels before replacing golden_dataset.json.")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, content FROM document_chunks;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        print("No chunks found in database! Please ingest documents first.")
        return

    print(f"Found {len(rows)} chunks. Generating up to 100 draft questions requiring source review...")

    # Template query patterns mapped to topics
    templates = [
        ("What are the key concepts discussed regarding {topic}?", "conceptual"),
        ("How does {topic} impact overall system performance?", "analytical"),
        ("What are the primary challenges and considerations for {topic}?", "diagnostic"),
        ("Can you explain the architecture and workflow of {topic}?", "architectural"),
        ("What metrics and evaluation criteria apply to {topic}?", "evaluation")
    ]

    golden_dataset = []
    query_counter = 1

    # Sample across the corpus rather than taking the first 100 rows (often front matter).
    # Natural ordering keeps chunk_10 after chunk_9 regardless of database row order.
    rows = sorted(rows, key=lambda row: [
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r"(\d+)", row[0])
    ])
    count = min(100, len(rows))
    indices = [i * (len(rows) - 1) // max(1, count - 1) for i in range(count)]
    for index in indices:
        row = rows[index]
        chunk_id, content = row
        
        # Extract simple keyword/topic from content snippet
        words = [w.strip(".,()[]{}\"'-") for w in content.split() if len(w) > 5]
        topic = " ".join(words[:2]) if words else "AI systems"

        template, q_type = templates[(query_counter - 1) % len(templates)]
        query_text = template.format(topic=topic)

        golden_dataset.append({
            "query_id": f"eval_{query_counter:03d}",
            "query": query_text,
            "query_type": q_type,
            "target_chunk_ids": [chunk_id],
            "review_status": "needs_review",
            "generation_method": "keyword_template",
            "source_excerpt": content,
            "review_note": "Verify that the source answers the question; rewrite the question and label all relevant chunks before approval."
        })
        query_counter += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(golden_dataset, f, indent=2)

    print(f"Saved {len(golden_dataset)} draft questions to {output_path}. These are not yet a golden benchmark.")

if __name__ == "__main__":
    DB_URL = os.getenv("DATABASE_URL", "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag")
    generate_100_golden_dataset(DB_URL)
