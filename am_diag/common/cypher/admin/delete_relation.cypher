// Delete a single RELATES_TO edge (leaf; endpoints are untouched).
// $id: Relation UUID (string).
MATCH ()-[r:RELATES_TO {id: $id}]->()
DELETE r
