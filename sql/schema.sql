CREATE TABLE IF NOT EXISTS kg_edges (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    source_entity TEXT NOT NULL,
    source_type TEXT DEFAULT 'Concept',     -- Ontological category (e.g., Architecture, Metric)
    target_entity TEXT NOT NULL,
    target_type TEXT DEFAULT 'Concept',     -- Ontological category
    relation_type VARCHAR(100) NOT NULL,     -- Directed relationship type (e.g., COMPRISES, APPLIES_TO)
    weight FLOAT DEFAULT 1.0,
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(chunk_id) ON DELETE CASCADE
);

-- CREATE TABLE IF NOT EXISTS does not update previously initialized tables.
ALTER TABLE kg_edges
    ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'Concept',
    ADD COLUMN IF NOT EXISTS target_type TEXT DEFAULT 'Concept',
    ADD COLUMN IF NOT EXISTS weight FLOAT DEFAULT 1.0;

CREATE INDEX IF NOT EXISTS idx_kg_source_entity ON kg_edges(source_entity);
CREATE INDEX IF NOT EXISTS idx_kg_target_entity ON kg_edges(target_entity);
CREATE INDEX IF NOT EXISTS idx_kg_relation ON kg_edges(relation_type);
