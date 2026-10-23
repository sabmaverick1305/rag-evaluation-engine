import os
from typing import Dict, List, Any
import psycopg2
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PostgresIngestionPipeline:
    def __init__(self, db_url: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.db_url = db_url
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        # Using a compact, high-efficiency local embedding model
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def ingest_documents(self, raw_documents: List[Dict[str, Any]]):
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        total_inserted = 0
        for doc in raw_documents:
            splits = self.splitter.split_text(doc["text"])
            for idx, text in enumerate(splits):
                chunk_id = f"{doc['id']}_chunk_{idx}"
                embedding = self.encoder.encode(text).tolist()

                cursor.execute(
                    """
                    INSERT INTO document_chunks (chunk_id, parent_doc_id, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE 
                    SET content = EXCLUDED.content, embedding = EXCLUDED.embedding;
                    """,
                    (chunk_id, doc["id"], text, embedding)
                )
                total_inserted += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Successfully ingested, chunked, and embedded {total_inserted} chunks into PostgreSQL!")

if __name__ == "__main__":
    # Example usage targeting your local docker-compose Postgres instance
    DB_URL = os.getenv("DATABASE_URL", "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag")
    
    sample_docs = [
        {
            "id": "doc_networking_01",
            "text": "EADDRNOTAVAIL (99) occurs when a socket tries to bind to an address that is not local or when a connection is made to an invalid address/port combination. Handling procedures include verifying correct port bindings and network configurations."
        },
        {
            "id": "doc_graphrag_02",
            "text": "Graph RAG enhances traditional vector search by building hierarchical community summaries and traversing entity relationships. This bridges context gaps when multi-hop reasoning spans across disparate document sections."
        }
    ]
    
    pipeline = PostgresIngestionPipeline(db_url=DB_URL)
    pipeline.ingest_documents(sample_docs)

        
