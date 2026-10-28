from contextlib import closing
from typing import Any

import psycopg2
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.neo4j_client import Neo4jGraphClient
from src.state import RAGState


class ProductionGraphRAGRetriever:
    def __init__(
        self,
        db_url: str,
        embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_id: str = "BAAI/bge-reranker-large"
    ):
        self.db_url = db_url
        print(f"Initializing embedding model ({embedding_model_id})...")
        self.embedder = SentenceTransformer(embedding_model_id)

        print(f"Initializing cross-encoder reranker ({cross_encoder_id})...")
        self.reranker = CrossEncoder(cross_encoder_id)

        # Initialize Neo4j graph client for ontology-aware traversal
        self.neo4j_client = Neo4jGraphClient()

        # Load local reference chunks for evaluation alignment
        self.chunks = self._load_all_chunks()

    def _load_all_chunks(self) -> dict[str, dict[str, Any]]:
        """Loads all document chunks from Postgres into memory for validation."""
        chunks_map = {}
        try:
            with closing(psycopg2.connect(self.db_url, connect_timeout=10)) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT chunk_id, content, source_doc FROM document_chunks;")
                    rows = cursor.fetchall()
                    for row in rows:
                        chunks_map[row[0]] = {
                            "id": row[0],
                            "text": row[1],
                            "source_doc": row[2]
                        }
        except Exception as e:
            print(f"Warning: Could not load chunks cache from DB: {e}")
        return chunks_map

    def _run_hybrid_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Executes combined dense vector search and sparse text search in PostgreSQL."""
        query_vector = self.embedder.encode(query).tolist()
        candidates = {}

        with closing(psycopg2.connect(self.db_url, connect_timeout=10)) as conn:
            with conn.cursor() as cursor:
                # 1. Dense Vector Search (pgvector cosine similarity)
                cursor.execute(
                    """
                    SELECT chunk_id, content, source_doc,
                           1 - (embedding <=> %s::vector) AS score
                    FROM document_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_vector, query_vector, top_k)
                )
                for row in cursor.fetchall():
                    candidates[row[0]] = {
                        "id": row[0],
                        "text": row[1],
                        "source_doc": row[2],
                        "score": float(row[3]),
                        "source": "vector_search"
                    }

                # 2. Sparse Text Search (Postgres Full-Text / ILIKE keyword fallback)
                keywords = [f"%{term}%" for term in query.split() if len(term) > 3]
                if keywords:
                    cursor.execute(
                        """
                        SELECT chunk_id, content, source_doc
                        FROM document_chunks
                        WHERE content ILIKE ANY (%s)
                        LIMIT %s;
                        """,
                        (keywords, top_k)
                    )
                    for row in cursor.fetchall():
                        if row[0] not in candidates:
                            candidates[row[0]] = {
                                "id": row[0],
                                "text": row[1],
                                "source_doc": row[2],
                                "score": 0.5,  # Default baseline score for sparse keyword matches
                                "source": "sparse_search"
                            }

        return list(candidates.values())

    def _rerank_candidates(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-scores and sorts candidate chunks using the Cross-Encoder model."""
        if not candidates:
            return []

        pairs = [[query, candidate["text"]] for candidate in candidates]
        scores = self.reranker.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["score"] = float(scores[i])

        # Sort descending by cross-encoder score
        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return reranked

    def retrieve(self, state: RAGState, top_k: int = 1) -> RAGState:
        """
        Executes multi-stage retrieval:
        1. Hybrid Search (pgvector + sparse keywords)
        2. Neo4j Ontology-Driven Multi-Hop Graph Traversal
        3. Deduplication and Candidate Pool Merging
        4. Cross-Encoder Reranking
        5. Returns top_k candidates in state.retrieved_candidates
        """
        query = state.query

        # Step 1: Hybrid Search from Postgres (fetch a wider candidate pool for cross-encoder reranking)
        candidate_pool_size = max(15, top_k * 5)
        hybrid_candidates = self._run_hybrid_search(query, top_k=candidate_pool_size)

        # Step 2: Neo4j Ontology-Driven Graph Traversal
        kg_candidates = self.neo4j_client.ontology_multihop_traverse(query, limit=3)

        # Step 3: Deduplicate and merge candidate pools
        combined_pool = {c["id"]: c for c in hybrid_candidates + kg_candidates}.values()

        # Step 4: Cross-Encoder Reranking
        scored_candidates = self._rerank_candidates(query, list(combined_pool))

        # Step 5: Assign top_k results to state
        state.retrieved_candidates = scored_candidates[:top_k]
        return state
