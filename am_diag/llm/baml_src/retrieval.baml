// Retrieval functions: text-to-Cypher generation for the am_diag knowledge graph

class CypherQuery {
  cypher string
    @description("A valid read-only Cypher MATCH/RETURN query answering the question. No mutations.")
  reasoning string
    @description("One sentence: what the query finds and why this structure was chosen")
}

function GenerateCypherQuery(
  question: string,
  schema_description: string,
  examples: string,
) -> CypherQuery {
  client OpenAIGeneric

  prompt #"
    {{ _.role("system") }}
    You are a Neo4j Cypher query expert for a biomedical knowledge graph.
    Generate a read-only Cypher query that answers the clinical question.

    Knowledge graph schema:
    {{ schema_description }}

    {% if examples %}
    Example queries for reference:
    {{ examples }}
    {% endif %}

    RULES:
    1. Only use MATCH / OPTIONAL MATCH / WITH / WHERE / RETURN — no CREATE / MERGE / DELETE / SET
    2. Use ``n.uuid`` to identify nodes. Use ``n.name`` for human-readable names.
    3. Use specific node labels (e.g. ``Drug``, ``Disease``) when the question implies them.
    4. Include a LIMIT clause (maximum 50) to prevent runaway queries.
    5. Return ``n.name``, ``n.uuid``, ``type(r)`` — not raw internal IDs alone.

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question: {{ question }}
  "#
}
