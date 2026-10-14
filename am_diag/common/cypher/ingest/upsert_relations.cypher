// Upsert generic RELATES_TO relationships between Entity nodes.
// The medical relation type is stored as the `type` property.
// rec must have: id, head_id, tail_id, chunk_ids, sources, and props dict (type, description, score, ...).
// chunk_ids/sources are unioned (not clobbered) so re-ingest accumulates provenance.
UNWIND $records AS rec
MATCH (h:Entity {id: rec.head_id})
MATCH (t:Entity {id: rec.tail_id})
MERGE (h)-[r:RELATES_TO {id: rec.id}]->(t)
SET r += rec.props
SET r.chunk_ids = coalesce(r.chunk_ids, []) +
    [x IN coalesce(rec.chunk_ids, []) WHERE NOT x IN coalesce(r.chunk_ids, [])]
SET r.sources = coalesce(r.sources, []) +
    [x IN coalesce(rec.sources, []) WHERE NOT x IN coalesce(r.sources, [])]
