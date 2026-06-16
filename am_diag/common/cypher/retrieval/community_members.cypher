// Fetch entities belonging to a community, with optional chunk context.
// $community_id: Community UUID (string).
MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community {id: $community_id})
OPTIONAL MATCH (chunk:Chunk)-[:HAS_ENTITY]->(e)
RETURN
    e.id AS entity_id,
    e.name AS entity_name,
    labels(e) AS labels,
    collect(DISTINCT chunk.id) AS chunk_ids
ORDER BY e.name
