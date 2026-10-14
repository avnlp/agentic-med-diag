// Entity ids mentioned by a set of chunks — BFS seeds for the chunk retriever.
// $ids: list of Chunk UUID strings.
UNWIND $ids AS cid
MATCH (c:Chunk {id: cid})-[:HAS_ENTITY]->(e:Entity)
RETURN DISTINCT e.id AS id
