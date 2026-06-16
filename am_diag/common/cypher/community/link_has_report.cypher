// Link Community to its CommunityReport via :HAS_REPORT.
// rec must have: community_id, report_id.
UNWIND $records AS rec
MATCH (c:Community {id: rec.community_id})
MATCH (r:CommunityReport {id: rec.report_id})
MERGE (c)-[:HAS_REPORT]->(r)
