// Apply a constrained semantic property patch to an Entity and return it.
// $id: Entity UUID (string). $props: map of semantic fields only.
MATCH (e:Entity {id: $id})
SET e += $props
RETURN e.id AS id, labels(e) AS labels, properties(e) AS props
