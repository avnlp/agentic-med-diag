// Fetch RELATES_TO edges incident to an entity (in either direction).
// $id: Entity UUID (string).
MATCH (e:Entity {id: $id})-[r:RELATES_TO]-()
RETURN DISTINCT r.id AS id, properties(r) AS props,
       startNode(r).id AS head_id, endNode(r).id AS tail_id
