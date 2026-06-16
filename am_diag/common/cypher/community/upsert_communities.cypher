// Upsert Community nodes.
UNWIND $records AS rec
MERGE (n:Community {id: rec.id})
SET n += rec.props
