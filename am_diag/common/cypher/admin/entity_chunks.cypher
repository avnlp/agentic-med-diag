// Fetch Chunk nodes that mention an entity via HAS_ENTITY.
// $id: Entity UUID (string).
MATCH (c:Chunk)-[:HAS_ENTITY]->(e:Entity {id: $id})
RETURN c.id AS id, properties(c) AS props
