// Fetch full Entity node properties for a set of ids.
// $ids: list of string UUIDs.
UNWIND $ids AS eid
MATCH (e:Entity {id: eid})
RETURN e.id AS id, labels(e) AS labels, properties(e) AS props
