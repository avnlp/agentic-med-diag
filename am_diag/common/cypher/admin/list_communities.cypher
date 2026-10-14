// Page Community nodes by ascending id.
// $cursor: optional exclusive lower-bound id (null = start). $limit: max rows.
MATCH (c:Community)
WHERE ($cursor IS NULL OR c.id > $cursor)
RETURN c.id AS id, properties(c) AS props
ORDER BY c.id ASC
LIMIT $limit
