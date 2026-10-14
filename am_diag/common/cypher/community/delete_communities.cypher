// Remove Community nodes from a previous finalization.
MATCH (c:Community)
DETACH DELETE c
