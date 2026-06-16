// Derive NEXT_CHUNK edges from chunk_index ordering within a document.
// Called once per document_id after all chunks for that document are written.
// $document_id is a parameter (string UUID).
MATCH (c:Chunk) WHERE c.document_id = $document_id
WITH c ORDER BY c.chunk_index
WITH collect(c) AS cs
UNWIND range(0, size(cs)-2) AS i
WITH cs[i] AS a, cs[i+1] AS b
MERGE (a)-[:NEXT_CHUNK]->(b)
