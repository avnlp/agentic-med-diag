// Fetch full RELATES_TO relationship properties for a set of relation ids.
// $ids: list of string UUIDs.
UNWIND $ids AS rid
MATCH ()-[r:RELATES_TO {id: rid}]->()
RETURN r.id AS id, properties(r) AS props,
       startNode(r).id AS head_id, endNode(r).id AS tail_id
