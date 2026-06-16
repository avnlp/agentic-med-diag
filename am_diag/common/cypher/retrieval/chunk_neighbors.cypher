// Fetch the local context around a Chunk: entities, relations, communities, neighbors.
// $chunk_id: Chunk UUID (string).
MATCH (c:Chunk {id: $chunk_id})
OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:Entity)
OPTIONAL MATCH (e)-[r:RELATES_TO]->(e2:Entity)
OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)
OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
OPTIONAL MATCH (e)-[:IN_COMMUNITY]->(comm:Community)
RETURN
    c.id AS chunk_id,
    c.text AS chunk_text,
    collect(DISTINCT {id: e.id, name: e.name, labels: labels(e)}) AS entities,
    collect(DISTINCT {id: r.id, type: r.type, tail: e2.id}) AS relations,
    collect(DISTINCT {id: comm.id, summary: comm.summary}) AS communities,
    next.id AS next_chunk_id,
    prev.id AS prev_chunk_id
