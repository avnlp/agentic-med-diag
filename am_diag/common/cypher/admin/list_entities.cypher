// Page Entity nodes by ascending id, optionally filtered by label.
// $label: optional secondary label to filter on (null = all).
// $cursor: optional exclusive lower-bound id for keyset pagination (null = start).
// $limit: max rows to return.
MATCH (e:Entity)
WHERE ($label IS NULL OR $label IN labels(e))
  AND ($cursor IS NULL OR e.id > $cursor)
RETURN e.id AS id, labels(e) AS labels, properties(e) AS props
ORDER BY e.id ASC
LIMIT $limit
