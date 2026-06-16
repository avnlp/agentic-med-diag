// Link Entity nodes to the Chunks they were extracted from via :HAS_ENTITY.
// rec must have: entity_id, chunk_id.
UNWIND $records AS rec
MATCH (c:Chunk {id: rec.chunk_id})
MATCH (e:Entity {id: rec.entity_id})
MERGE (c)-[:HAS_ENTITY]->(e)
