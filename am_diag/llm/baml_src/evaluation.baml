class AnswerEquivalenceJudgement {
  is_correct bool
  explanation string
  matched_reference string?
}

class DiagnosisJudgement {
  is_correct bool
  explanation string
  matched_diagnosis string?
}

class RubricCriterionJudgement {
  criterion_id string?
  criteria_met bool
  explanation string
}

class ReasoningMatch {
  reason_index int
  matched bool
  matched_statements string[]
  explanation string
}

class ReasoningRecallJudgement {
  matches ReasoningMatch[]
  recall_score float
  explanation string
}

function JudgeAnswerEquivalent(
  question: string,
  predicted_answer: string,
  reference_answer: string,
  context: string?,
) -> AnswerEquivalenceJudgement {
  client OpenAIGenericRobust
  prompt #"
    {{ _.role("system") }}
    You are a strict medical QA judge.
    Decide whether the predicted answer is equivalent in meaning to the reference answer.
    Be conservative: only mark correct when the predicted answer clearly matches the reference.

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question:
    {{ question }}

    Predicted answer:
    {{ predicted_answer }}

    Reference answer:
    {{ reference_answer }}

    {% if context != null %}
    Context:
    {{ context }}
    {% endif %}
  "#
}

function JudgeDiagnosisCorrect(
  question: string,
  predicted_diagnosis: string,
  true_diagnosis: string,
  context: string?,
) -> DiagnosisJudgement {
  client OpenAIGenericRobust
  prompt #"
    {{ _.role("system") }}
    You are a strict diagnostic correctness judge.
    Decide whether the predicted diagnosis is medically equivalent to the true diagnosis.
    Accept synonyms and common abbreviations when they refer to the same diagnosis.

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question:
    {{ question }}

    Predicted diagnosis:
    {{ predicted_diagnosis }}

    True diagnosis:
    {{ true_diagnosis }}

    {% if context != null %}
    Context:
    {{ context }}
    {% endif %}
  "#
}

function JudgeRubricCriterion(
  question_or_conversation: string,
  assistant_response: string,
  criterion_text: string,
  criterion_id: string?,
  points: int?,
) -> RubricCriterionJudgement {
  client OpenAIGenericRobust
  prompt #"
    {{ _.role("system") }}
    You are a rubric-based medical response judge.
    Determine whether the assistant response satisfies the rubric criterion.

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Task or conversation:
    {{ question_or_conversation }}

    Assistant response:
    {{ assistant_response }}

    Criterion{% if criterion_id != null %} ({{ criterion_id }}){% endif %}{% if points != null %} [{{ points }} points]{% endif %}:
    {{ criterion_text }}
  "#
}

function JudgeReasoningRecall(
  ground_truth_reasons: string[],
  predicted_trace: string,
) -> ReasoningRecallJudgement {
  client OpenAIGenericRobust
  prompt #"
    {{ _.role("system") }}
    You are a medical reasoning recall judge.
    For each ground-truth reasoning statement, determine whether the predicted trace covers it.

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Ground truth reasons:
    {% for reason in ground_truth_reasons %}
    - {{ reason }}
    {% endfor %}

    Predicted trace:
    {{ predicted_trace }}
  "#
}
