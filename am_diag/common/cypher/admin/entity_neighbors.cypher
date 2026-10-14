// Fetch Entity nodes within $depth RELATES_TO hops of a seed entity.
// $id: seed Entity UUID (string).
// $depth: max hop count.
MATCH (e:Entity {id: $id})
MATCH path = (e)-[:RELATES_TO*1..$depth]-(n:Entity)
WHERE n.id <> $id
RETURN DISTINCT n.id AS id, labels(n) AS labels, properties(n) AS props
