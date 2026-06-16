// Community report generation BAML functions for the am_diag system.
// Uses OpenAIGeneric (5 retries) — same as extraction.baml.

class Finding {
  summary string
    @description("One-sentence finding headline.")
  explanation string
    @description("Supporting evidence and context.")
}

class CommunityReportOutput {
  title string
    @description("Short descriptive title for this community.")
  summary string
    @description("2-3 sentence summary of the community's medical significance.")
  findings Finding[]
    @description("Key structured findings, ordered by importance.")
  rating float
    @description("Importance rating 0.0-10.0 (higher = more clinically important).")
  rating_explanation string
    @description("One sentence justifying the rating.")
}

function GenerateCommunityReport(
  context_data: string,
  schema_context: string,
  max_report_length: int,
) -> CommunityReportOutput {
  client OpenAIGeneric
  prompt #"
    {{ _.role("system") }}
    You are a biomedical knowledge graph analyst. Generate a structured community report
    summarizing the medical entities and relationships in the provided context.

    The community context contains entities and relationships from a medical knowledge graph.
    Use the schema context to understand entity types and relationship semantics.

    Schema context:
    {{ schema_context }}

    Focus on:
    1. The primary medical concepts and their relationships
    2. Clinical significance and actionable insights
    3. Key findings with supporting evidence from the context
    4. Rating the community's importance for clinical decision-making (0-10)

    Keep the summary concise (target ~{{ max_report_length }} tokens).

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Generate a report for this community:

    {{ context_data }}
  "#
}
