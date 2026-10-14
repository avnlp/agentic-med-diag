// Fetch one Entity node by id with its labels and properties.
// $id: Entity UUID (string).
MATCH (e:Entity {id: $id})
RETURN e.id AS id, labels(e) AS labels, properties(e) AS props
