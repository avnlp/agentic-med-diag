// Retrieve the induced subgraph of a community: its entities and their internal edges.
// $community_id is the community's UUID (string).
MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community {id: $community_id})
OPTIONAL MATCH (e)-[r:RELATES_TO]->(e2:Entity)-[:IN_COMMUNITY]->(c)
RETURN
    e.id AS entity_id,
    e.name AS entity_name,
    labels(e) AS entity_labels,
    collect(DISTINCT {
        rel_id: r.id,
        rel_type: r.type,
        tail_id: e2.id,
        tail_name: e2.name
    }) AS edges
ORDER BY e.name
