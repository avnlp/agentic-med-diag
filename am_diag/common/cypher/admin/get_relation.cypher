// Fetch one RELATES_TO edge by id with its endpoints and properties.
// $id: Relation UUID (string).
MATCH ()-[r:RELATES_TO {id: $id}]->()
RETURN r.id AS id, properties(r) AS props,
       startNode(r).id AS head_id, endNode(r).id AS tail_id
