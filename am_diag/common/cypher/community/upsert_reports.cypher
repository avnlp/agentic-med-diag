// Upsert CommunityReport nodes.
UNWIND $records AS rec
MERGE (n:CommunityReport {id: rec.id})
SET n += rec.props
