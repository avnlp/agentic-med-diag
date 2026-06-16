// Link Entity nodes to their Community via :IN_COMMUNITY.
// rec must have: entity_id, community_id.
UNWIND $records AS rec
MATCH (e:Entity {id: rec.entity_id})
MATCH (c:Community {id: rec.community_id})
MERGE (e)-[:IN_COMMUNITY]->(c)
