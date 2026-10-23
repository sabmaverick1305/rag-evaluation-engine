-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Chunks table for Hybrid Search (Dense + Sparse)
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    source_doc VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx 
ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- 2. Knowledge Graph Nodes (Entities)
CREATE TABLE IF NOT EXISTS kg_nodes (
    id SERIAL PRIMARY KEY,
    entity_name VARCHAR(255) UNIQUE NOT NULL,
    entity_type VARCHAR(100)
);

-- 3. Knowledge Graph Edges (Relationships linking entities to chunks)
CREATE TABLE IF NOT EXISTS kg_edges (
    id SERIAL PRIMARY KEY,
    source_entity VARCHAR(255) NOT NULL,
    target_entity VARCHAR(255) NOT NULL,
    relation_type VARCHAR(100) NOT NULL,
    chunk_id VARCHAR(255) REFERENCES document_chunks(chunk_id) ON DELETE CASCADE
);