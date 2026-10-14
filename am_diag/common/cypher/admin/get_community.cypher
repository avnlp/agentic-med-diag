// Fetch one Community node by id with its properties.
// $id: Community UUID (string).
MATCH (c:Community {id: $id})
RETURN c.id AS id, properties(c) AS props
