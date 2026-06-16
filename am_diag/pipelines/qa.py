"""Multi-dataset QA runner.

Loads every ``QADataset``, sends each sample through the DeepAgents
clinical-QA harness, and writes per-dataset JSON result files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from am_diag.agents.run import answer_question
from am_diag.agents.settings import AgentSettings
from am_diag.common.data_models import (
    MCQSample,
    OpenEndedSample,
    QASample,
    RARMedSample,
    RubricSample,
)
from am_diag.db.graph.neo4j import create_neo4j_client
from am_diag.db.vector.qdrant import create_qdrant_store
from am_diag.loaders.dataset import (
    CareQADataset,
    CareQAReasoningDataset,
    HealthBenchDataset,
    MedCaseReasoningDataset,
    MedMCQADataset,
    MedQADataset,
    MedXpertQADataset,
    MMLUMedDataset,
    MMLUProHealthDataset,
    NEJMDiagnosticDataset,
    NEJMQADataset,
    PubHealthBenchDataset,
    PubHealthBenchFreeformDataset,
    PubMedQADataset,
    QADataset,
    RARMedDataset,
    SuperGPQAMedDataset,
)
from am_diag.pipelines.config import QaConfig
from am_diag.retrieval.config import RetrievalConfig
from am_diag.retrieval.search import SearchEngine
from am_diag.vector.embedding.sentence_transformers import (
    SentenceTransformersEmbedder,
)


logger = logging.getLogger(__name__)


DATASET_REGISTRY: dict[str, type[QADataset]] = {
    "careqa": CareQADataset,
    "careqa_reasoning": CareQAReasoningDataset,
    "healthbench": HealthBenchDataset,
    "medcase_reasoning": MedCaseReasoningDataset,
    "medmcqa": MedMCQADataset,
    "medqa": MedQADataset,
    "medxpertqa": MedXpertQADataset,
    "mmlu_med": MMLUMedDataset,
    "mmlu_pro_health": MMLUProHealthDataset,
    "nejm_diagnostic": NEJMDiagnosticDataset,
    "nejm_qa": NEJMQADataset,
    "pubhealthbench": PubHealthBenchDataset,
    "pubhealthbench_freeform": PubHealthBenchFreeformDataset,
    "pubmedqa": PubMedQADataset,
    "rar_med": RARMedDataset,
    "supergpqa_med": SuperGPQAMedDataset,
}


def _resolve_datasets(spec: str) -> list[type[QADataset]]:
    if spec == "all":
        return list(DATASET_REGISTRY.values())
    names = [n.strip() for n in spec.split(",")]
    return [DATASET_REGISTRY[n] for n in names]


def _format_sample(sample: QASample) -> str:
    """Format a ``QASample`` into a question prompt for the agent."""
    if isinstance(sample, MCQSample):
        lines = [sample.question_stem or sample.question]
        for key in sorted(sample.options):
            lines.append(f"{key}. {sample.options[key]}")
        return "\n".join(lines)

    if isinstance(sample, RubricSample):
        parts = []
        for turn in sample.conversation:
            parts.append(f"{turn.role}: {turn.content}")
        return "\n".join(parts)

    if isinstance(sample, RARMedSample):
        return sample.question

    if isinstance(sample, OpenEndedSample):
        return sample.question

    return str(sample.question)


def _serialize_result(
    sample: QASample,
    agent_answer: Any,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a serialisable result dict for one QA pair."""
    base = {
        "sample_id": getattr(sample, "sample_id", ""),
        "dataset": getattr(sample, "dataset", ""),
        "question": _format_sample(sample),
    }

    if error:
        base["error"] = error
        return base

    if agent_answer is None:
        base["error"] = "No answer returned"
        return base

    if hasattr(agent_answer, "model_dump"):
        base["answer"] = agent_answer.model_dump()
    elif hasattr(agent_answer, "dict"):
        base["answer"] = agent_answer.dict()
    else:
        base["answer"] = str(agent_answer)

    return base


async def run_single_dataset(
    dataset_cls: type[QADataset],
    search_engine: SearchEngine,
    agent_settings: AgentSettings,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run one dataset through the agent and return result records."""
    name = getattr(dataset_cls, "dataset_name", dataset_cls.__name__)
    logger.info("Starting QA run for dataset: %s", name)

    dataset = dataset_cls()
    samples = dataset.load(limit=limit)

    results: list[dict[str, Any]] = []
    for sample in samples:
        question = _format_sample(sample)
        logger.debug("Question [%s]: %s", name, question[:120])

        try:
            answer = await answer_question(
                question=question,
                search_engine=search_engine,
                settings=agent_settings,
            )
            results.append(_serialize_result(sample, answer))
        except Exception as exc:
            logger.exception("Error processing sample in %s", name)
            results.append(_serialize_result(sample, agent_answer=None, error=str(exc)))

    logger.info(
        "Finished %s: %d / %d samples answered",
        name,
        sum(1 for r in results if "error" not in r),
        len(results),
    )

    return results


async def run_all_datasets(
    search_engine: SearchEngine,
    agent_settings: AgentSettings | None = None,
    config: QaConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run every dataset in ``config`` through the agent.

    Args:
        search_engine: Configured ``SearchEngine`` for retrieval.
        agent_settings: Agent settings.  Defaults to ``AgentSettings()``.
        config: QA configuration.  Defaults to ``QaConfig()``.

    Returns:
        Mapping of dataset name → list of result records.
    """
    cfg = config or QaConfig()
    settings = agent_settings or AgentSettings()
    dataset_classes = _resolve_datasets(cfg.datasets)

    all_results: dict[str, list[dict[str, Any]]] = {}
    for cls in dataset_classes:
        name = getattr(cls, "dataset_name", cls.__name__)
        results = await run_single_dataset(
            dataset_cls=cls,
            search_engine=search_engine,
            agent_settings=settings,
            limit=cfg.limit,
        )
        all_results[name] = results

    return all_results


def _write_results(
    all_results: dict[str, list[dict[str, Any]]],
    output_dir: str,
) -> Path:
    """Write per-dataset JSON files and return the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for dataset_name, records in all_results.items():
        path = out / f"{dataset_name}.json"
        with open(path, "w") as f:
            json.dump(
                {"dataset": dataset_name, "samples": records},
                f,
                indent=2,
                default=str,
            )

    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QA datasets through the DeepAgents clinical-QA harness.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help=(
            "Comma-separated dataset names, or 'all' (default). "
            f"Options: {', '.join(DATASET_REGISTRY)}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per dataset (default: no limit).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Output directory for per-dataset JSON files (default: results/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (``am-diag-qa``).

    Usage::

        uv run am-diag-qa
        uv run am-diag-qa --datasets careqa,medqa --limit 10
    """
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = QaConfig(
        datasets=args.datasets,
        limit=args.limit,
        output_dir=args.output_dir,
    )
    agent_settings = AgentSettings()

    async def _run() -> None:
        async with (
            create_neo4j_client() as graph_store,
            create_qdrant_store() as vector_store,
        ):
            embedder = SentenceTransformersEmbedder()
            search_engine = SearchEngine(
                config=RetrievalConfig(),
                vector_store=vector_store,
                graph_store=graph_store,
                embedder=embedder,
            )
            results = await run_all_datasets(
                search_engine=search_engine,
                agent_settings=agent_settings,
                config=config,
            )
            out = _write_results(results, config.output_dir)
            print(f"\nResults written to {out.resolve()}")

    asyncio.run(_run())


__all__ = [
    "DATASET_REGISTRY",
    "main",
    "run_all_datasets",
    "run_single_dataset",
]
