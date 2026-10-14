// Apply a constrained semantic property patch to a relation and return it.
// $id: Relation UUID (string). $props: map of semantic fields only.
MATCH ()-[r:RELATES_TO {id: $id}]->()
SET r += $props
RETURN r.id AS id, properties(r) AS props,
       startNode(r).id AS head_id, endNode(r).id AS tail_id
