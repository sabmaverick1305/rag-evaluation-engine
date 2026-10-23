import os
from typing import List, Dict, Any
import psycopg2
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ProductionPDFIngestion:
    def __init__(self, db_url: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.db_url = db_url
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def parse_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n[Page {page_num + 1}]\n{text}"
        # PDF extraction can emit NULs, which PostgreSQL text columns reject.
        # Clean before splitting so embeddings and stored content use the same text.
        return full_text.replace("\x00", "")

    def extract_graph_triplets(self, chunk_text: str, chunk_id: str) -> List[Dict[str, str]]:
        """
        Production-grade rule/heuristic or lightweight extraction of entities & relations.
        (In production, you can replace this with an LLM structured extraction call).
        """
        triplets = []
        # Simple heuristic example for demonstration: looking for key terms
        keywords = ["AI engineering", "foundation models", "RAG", "fine-tuning", "prompt engineering", "evaluation"]
        found = [kw for kw in keywords if kw.lower() in chunk_text.lower()]
        
        for i in range(len(found) - 1):
            triplets.append({
                "source": found[i],
                "target": found[i+1],
                "relation": "relates_to",
                "chunk_id": chunk_id
            })
        return triplets

    def process_and_store(self, pdf_path: str, doc_name: str):
        print(f"Parsing PDF: {pdf_path}...")
        raw_text = self.parse_pdf(pdf_path)
        
        splits = self.splitter.split_text(raw_text)
        print(f"Split document into {len(splits)} chunks. Generating embeddings & storing in Postgres...")

        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        for idx, text in enumerate(splits):
            chunk_id = f"{doc_name}_chunk_{idx}"
            embedding = self.encoder.encode(text).tolist()

            # 1. Insert into document_chunks
            cursor.execute(
                """
                INSERT INTO document_chunks (chunk_id, source_doc, content, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE 
                SET content = EXCLUDED.content, embedding = EXCLUDED.embedding;
                """,
                (chunk_id, doc_name, text, embedding)
            )

            # 2. Extract and insert Knowledge Graph nodes/edges
            triplets = self.extract_graph_triplets(text, chunk_id)
            for t in triplets:
                # Insert nodes if not exist
                cursor.execute("INSERT INTO kg_nodes (entity_name) VALUES (%s) ON CONFLICT DO NOTHING;", (t["source"],))
                cursor.execute("INSERT INTO kg_nodes (entity_name) VALUES (%s) ON CONFLICT DO NOTHING;", (t["target"],))
                
                # Insert edge
                cursor.execute(
                    """
                    INSERT INTO kg_edges (source_entity, target_entity, relation_type, chunk_id)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (t["source"], t["target"], t["relation"], t["chunk_id"])
                )

        conn.commit()
        cursor.close()
        conn.close()
        print(f"Successfully processed and stored {len(splits)} chunks and extracted KG relations into PostgreSQL!")

if __name__ == "__main__":
    DB_URL = os.getenv("DATABASE_URL", "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag")
    pipeline = ProductionPDFIngestion(db_url=DB_URL)
    
    # Target your local PDF book chapter or sample file
    pdf_file = "data/ai_engineering_by_chip_huyen.pdf"
    if os.path.exists(pdf_file):
        pipeline.process_and_store(pdf_file, doc_name="ai_engineering_ch1")
    else:
        print(f"Please place your PDF file at {pdf_file} to run ingestion.")
