// Upsert Document nodes from a batch of parameter records.
// Each rec must have: id (string UUID), and all other Document fields as props.
UNWIND $records AS rec
MERGE (n:Document {id: rec.id})
SET n += rec.props
