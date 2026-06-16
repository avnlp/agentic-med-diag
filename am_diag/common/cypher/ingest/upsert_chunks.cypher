// Upsert Chunk nodes from a batch of parameter records.
UNWIND $records AS rec
MERGE (n:Chunk {id: rec.id})
SET n += rec.props
