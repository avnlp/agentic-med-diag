// Upsert generic RELATES_TO relationships between Entity nodes.
// The medical relation type is stored as the `type` property.
// rec must have: id, head_id, tail_id, and props dict (type, description, score, ...).
UNWIND $records AS rec
MATCH (h:Entity {id: rec.head_id})
MATCH (t:Entity {id: rec.tail_id})
MERGE (h)-[r:RELATES_TO {id: rec.id}]->(t)
SET r += rec.props
