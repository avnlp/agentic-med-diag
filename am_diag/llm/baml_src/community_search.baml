// Community search BAML functions — MAP and REDUCE for global map-reduce search.
// Map uses OpenAIGeneric (5 retries), Reduce uses OpenAIGenericRobust (10 retries).

class MapKeyPoint {
  description string @description("A key point extracted from the community report batch")
  score int @description("Importance score 0-100")
  source_report_ids string[] @description("IDs of community reports supporting this point")
}

class MapResponse {
  points MapKeyPoint[]
}

class ReduceAnswer {
  answer string @description("Synthesized answer from all analyst reports")
  key_themes string[] @description("Major themes identified across communities")
  supporting_reports string[] @description("Community report IDs providing evidence")
}

function MapCommunityBatch(
  context_data: string,
  query: string,
) -> MapResponse {
  client OpenAIGeneric
  prompt #"
    {{ _.role("system") }}
    You are analyzing community report summaries from a medical knowledge graph.
    Extract key points relevant to the query and assign each an importance score (0-100).
    Reference specific report IDs using [Data: Reports (id, ...)] format.

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Query: {{ query }}
    Community report data:
    {{ context_data }}
  "#
}

function ReduceMapResponses(
  map_responses: string,
  query: string,
) -> ReduceAnswer {
  client OpenAIGenericRobust
  prompt #"
    {{ _.role("system") }}
    Synthesize the following analyst reports into a coherent medical answer.
    Reports are sorted by importance (most important first).
    Preserve data references with [Data: Reports (id, ...)] format.
    Do NOT mention "analysts" in the final answer — present as direct knowledge.

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Query: {{ query }}
    Reports:
    {{ map_responses }}
  "#
}
