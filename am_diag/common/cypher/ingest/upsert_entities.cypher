// Upsert Entity nodes with a dynamic secondary type label.
// ${Label} is substituted at call time (e.g. `Disease`, `Drug`) — validated + backtick-escaped.
// Called once per entity type group so all nodes in the batch share the same label.
UNWIND $records AS rec
MERGE (n:Entity {id: rec.id})
SET n += rec.props
SET n:${Label}
