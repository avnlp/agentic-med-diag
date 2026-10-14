// All CommunityReports at a given hierarchy level.
// $level: community hierarchy level (int).
MATCH (cr:CommunityReport)-[:HAS_REPORT]-(c:Community)
WHERE c.level = $level
RETURN cr.id AS id, cr.title AS title, cr.summary AS summary
