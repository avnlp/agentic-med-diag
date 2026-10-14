// Member entity ids for a set of communities — BFS seeds for the community retriever.
// $ids: list of Community UUID strings.
UNWIND $ids AS cid
MATCH (e:Entity)-[:IN_COMMUNITY]->(:Community {id: cid})
RETURN DISTINCT e.id AS id
