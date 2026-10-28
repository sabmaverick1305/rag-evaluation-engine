import os
from typing import Any

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

class Neo4jGraphClient:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "localdevonly")
        if GraphDatabase is not None:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            except Exception as e:
                print(f"Warning: Could not initialize Neo4j driver: {e}")
                self.driver = None
        else:
            self.driver = None

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def ontology_multihop_traverse(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Executes an ontology-aware multi-hop Cypher traversal in Neo4j
        matching query terms against entity nodes and fetching linked document chunks.
        """
        if self.driver is None:
            return []

        query_terms = [term.lower() for term in query.split() if len(term) > 3]
        if not query_terms:
            return []

        cypher_query = """
        MATCH (s:Entity)-[r]->(t:Entity)
        WHERE any(term IN $terms WHERE toLower(s.name) CONTAINS term OR toLower(t.name) CONTAINS term)
        MATCH (t)-[:MENTIONED_IN|CONTAINS|LINKED_TO]->(c:Chunk)
        RETURN DISTINCT
            c.chunk_id AS chunk_id,
            c.content AS content,
            c.source_doc AS source_doc,
            s.name AS source_entity,
            labels(s)[0] AS source_type,
            type(r) AS relation,
            t.name AS target_entity,
            labels(t)[0] AS target_type
        LIMIT $limit
        """

        augmented_chunks = []
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, terms=query_terms, limit=limit)
                for record in result:
                    augmented_chunks.append({
                        "id": record["chunk_id"],
                        "text": record["content"],
                        "source_doc": record["source_doc"],
                        "kg_context": f"Neo4j Ontology Edge: ({record['source_entity']}:{record['source_type']}) --[{record['relation']}]--> ({record['target_entity']}:{record['target_type']})",
                        "relation_source": "neo4j_knowledge_graph"
                    })
        except Exception as e:
            print(f"Warning: Neo4j traversal failed or connection unavailable: {e}")

        return augmented_chunks
