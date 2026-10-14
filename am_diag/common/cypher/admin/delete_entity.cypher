// Delete an Entity and all incident relationships (cascade).
// $id: Entity UUID (string).
MATCH (e:Entity {id: $id})
DETACH DELETE e
