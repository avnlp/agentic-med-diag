// Fetch one Chunk node by id with its properties.
// $id: Chunk UUID (string).
MATCH (c:Chunk {id: $id})
RETURN c.id AS id, properties(c) AS props
