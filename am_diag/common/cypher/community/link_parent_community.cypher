// Link child Community to parent Community via :PARENT_COMMUNITY.
// rec must have: child_id, parent_id.
UNWIND $records AS rec
MATCH (child:Community {id: rec.child_id})
MATCH (parent:Community {id: rec.parent_id})
MERGE (child)-[:PARENT_COMMUNITY]->(parent)
