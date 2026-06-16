// Link each Chunk to its parent Document via :PART_OF.
// rec must have: id (chunk id), document_id.
UNWIND $records AS rec
MATCH (c:Chunk {id: rec.id})
MATCH (d:Document {id: rec.document_id})
MERGE (c)-[:PART_OF]->(d)
