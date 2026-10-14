// Upsert Entity nodes with a dynamic secondary type label.
// ${Label} is substituted at call time (e.g. `Disease`, `Drug`) — validated + backtick-escaped.
// Called once per entity type group so all nodes in the batch share the same label.
// chunk_ids/surface_forms are unioned (not clobbered) so re-ingest accumulates provenance.
UNWIND $records AS rec
MERGE (n:Entity {id: rec.id})
SET n += rec.props
SET n.chunk_ids = coalesce(n.chunk_ids, []) +
    [x IN coalesce(rec.chunk_ids, []) WHERE NOT x IN coalesce(n.chunk_ids, [])]
SET n.surface_forms = coalesce(n.surface_forms, []) +
    [x IN coalesce(rec.surface_forms, []) WHERE NOT x IN coalesce(n.surface_forms, [])]
SET n:${Label}
