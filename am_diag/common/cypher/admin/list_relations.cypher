// Page RELATES_TO edges by ascending id, optionally filtered by type.
// $type: optional relation type to filter on (null = all).
// $cursor: optional exclusive lower-bound id for keyset pagination (null = start).
// $limit: max rows to return.
MATCH ()-[r:RELATES_TO]->()
WHERE ($type IS NULL OR r.type = $type)
  AND ($cursor IS NULL OR r.id > $cursor)
RETURN r.id AS id, properties(r) AS props,
       startNode(r).id AS head_id, endNode(r).id AS tail_id
ORDER BY r.id ASC
LIMIT $limit
