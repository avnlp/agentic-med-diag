// GDS Leiden community detection.
// Projects the Entity--RELATES_TO graph, runs Leiden with intermediate
// communities, and writes community IDs back to each Entity node.
//
// Requires the Neo4j Graph Data Science (GDS) library to be installed.
// The graph name uses a UUID suffix to avoid collisions across concurrent runs.

// 1. Drop any existing in-memory graph with this name (idempotent).
CALL gds.graph.drop($graph_name, false) YIELD graphName

// 2. Project the entity graph.
//    Only includes RELATES_TO relationships to avoid structural edges
//    (PART_OF, HAS_ENTITY, IN_COMMUNITY, etc.) skewing community detection.
CALL gds.graph.project(
  $graph_name,
  'Entity',
  'RELATES_TO',
  {
    relationshipProperties: 'score'
  }
) YIELD nodeCount, relationshipCount

// 3. Run Leiden algorithm, writing community IDs to a temporary node property.
CALL gds.leiden.mutate(
  $graph_name,
  {
    mutateProperty: 'gdsLeidenCommunityId',
    includeIntermediateCommunities: true,
    maxLevels: 10,
    randomSeed: $seed
  }
) YIELD modularity, modularities, ranLevels

// 4. Read community assignments at every hierarchical level.
MATCH (n:Entity)
WHERE n.gdsLeidenCommunityId IS NOT NULL
WITH n, n.gdsLeidenCommunityId AS community_ids
UNWIND range(0, size(community_ids) - 1) AS level
WITH
  n,
  level,
  community_ids[level] AS cid,
  CASE
    WHEN level = 0 THEN null
    ELSE community_ids[level - 1]
  END AS parent_cid
RETURN
  n.id AS entity_id,
  level AS community_level,
  toString(cid) AS community_id,
  toString(parent_cid) AS parent_community_id

// 5. Clean up temporary property.
MATCH (n:Entity)
REMOVE n.gdsLeidenCommunityId

// 6. Drop the in-memory graph.
CALL gds.graph.drop($graph_name, false)
