import os
from typing import List, Dict, Any
import psycopg2
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.state import RAGState

class ProductionGraphRAGRetriever:
    def __init__(self, db_url: str, embedding_model: str = "all-MiniLM-L6-v2", reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.db_url = db_url
        self.encoder = SentenceTransformer(embedding_model)
        # Initialize the cross-encoder for deep semantic re-scoring
        self.reranker = CrossEncoder(reranker_model)
        
        self.chunks = self._load_all_chunks()
        tokenized_corpus = [chunk["text"].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _load_all_chunks(self) -> List[Dict[str, Any]]:
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id, source_doc, content FROM document_chunks;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [{"id": row[0], "source_doc": row[1], "text": row[2]} for row in rows]

    def retrieve(self, state: RAGState, fetch_k: int = 30, top_k: int = 3) -> RAGState:
        query = state.query
        query_vector = self.encoder.encode(query).tolist()

        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        # 1. Broad Dense Semantic Search via pgvector
        cursor.execute(
            """
            SELECT chunk_id, source_doc, content, (embedding <=> %s::vector) as distance
            FROM document_chunks
            ORDER BY distance ASC
            LIMIT %s;
            """,
            (query_vector, fetch_k)
        )
        dense_rows = cursor.fetchall()
        formatted_dense = [
            {"id": row[0], "source_doc": row[1], "text": row[2], "score": 1.0 - float(row[3])}
            for row in dense_rows
        ]

        # 2. Broad Sparse Lexical Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]
        sparse_results = [
            {"id": self.chunks[i]["id"], "source_doc": self.chunks[i]["source_doc"], "text": self.chunks[i]["text"], "score": float(bm25_scores[i])}
            for i in top_sparse_indices
        ]

        # 3. Enhanced Knowledge Graph Relational Traversal (Fuzzy / Multi-word entity matching)
        query_terms = [f"%{term}%" for term in query.lower().split() if len(term) > 3]
        if query_terms:
            cursor.execute(
                """
                SELECT DISTINCT e.chunk_id, dc.content, dc.source_doc
                FROM kg_edges e
                JOIN document_chunks dc ON e.chunk_id = dc.chunk_id
                WHERE (
                    SELECT bool_or(e.source_entity ILIKE t OR e.target_entity ILIKE t)
                    FROM unnest(%s) t
                );
                """,
                (query_terms,)
            )
            kg_rows = cursor.fetchall()
            state.kg_augmented_context = [
                {"id": row[0], "text": row[1], "source_doc": row[2], "relation_source": "knowledge_graph"}
                for row in kg_rows
            ]
        else:
            state.kg_augmented_context = []

        cursor.close()
        conn.close()

        # 4. Reciprocal Rank Fusion (RRF) to merge candidate pools
        fused_candidates = self._reciprocal_rank_fusion(formatted_dense, sparse_results, fetch_k=fetch_k)

        # 5. Cross-Encoder Reranking
        if fused_candidates:
            pairs = [[query, candidate["text"]] for candidate in fused_candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            for idx, candidate in enumerate(fused_candidates):
                candidate["score"] = float(rerank_scores[idx])
            
            # Sort by cross-encoder score descending and truncate to final top_k
            fused_candidates = sorted(fused_candidates, key=lambda x: x["score"], reverse=True)

        state.retrieved_candidates = fused_candidates[:top_k]
        return state

    def _reciprocal_rank_fusion(self, dense: List[Dict], sparse: List[Dict], k: int = 60, fetch_k: int = 30) -> List[Dict]:
        rrf_scores = {}
        
        for rank, item in enumerate(dense):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, {"id": cid, "text": item["text"], "source_doc": item["source_doc"], "score": 0.0})
            rrf_scores[cid]["score"] += 1.0 / (k + rank + 1)

        for rank, item in enumerate(sparse):
            cid = item["id"]
            if cid not in rrf_scores:
                chunk_text = next((c["text"] for c in self.chunks if c["id"] == cid), "")
                source_doc = next((c["source_doc"] for c in self.chunks if c["id"] == cid), "")
                rrf_scores[cid] = {"id": cid, "text": chunk_text, "source_doc": source_doc, "score": 0.0}
            rrf_scores[cid]["score"] += 1.0 / (k + rank + 1)

        sorted_candidates = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return sorted_candidates[:fetch_k]