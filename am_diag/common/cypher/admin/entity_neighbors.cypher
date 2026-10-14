// Fetch Entity nodes within a bounded number of RELATES_TO hops of a seed entity.
// $id: seed Entity UUID (string).
// ${depth}: max hop count, substituted by the client after clamping.
// $limit: max rows to return.
MATCH (e:Entity {id: $id})
MATCH (e)-[:RELATES_TO*1..${depth}]-(n:Entity)
WHERE n.id <> $id
RETURN DISTINCT n.id AS id, labels(n) AS labels, properties(n) AS props
LIMIT $limit
