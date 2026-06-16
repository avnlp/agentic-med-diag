"""Configuration for the extraction pipeline.

Moved from `ingestion/config.py` to break the circular dependency between
`graph_construction/` (normalization, resolution, extraction) and
`ingestion/` (pipeline orchestration).

All fields are env-overridable with the `EXTRACTION_` prefix.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings


class CommunityDetectionConfig(BaseSettings):
    """Configuration for community detection backends."""

    model_config = {
        "env_prefix": "COMMUNITY_",
        "env_file": ".env",
        "extra": "ignore",
    }

    community_backend: Literal["graspologic", "gds"] = "graspologic"
    max_cluster_size: int = 10
    resolution: float = 1.0
    randomness: float = 0.001
    use_modularity: bool = True
    iterations: int = 1
    seed: int = 0xDEADBEEF


class ExtractionPipelineConfig(BaseSettings):
    """Pipeline configuration. All fields env-overridable with EXTRACTION_ prefix."""

    model_config = {"env_prefix": "EXTRACTION_", "env_file": ".env", "extra": "ignore"}

    # Extractor selection
    use_gliner: bool = True
    use_glirel: bool = True
    use_llm: bool = True
    merge_strategy: Literal["union", "intersection", "max_score", "gliner_primary"] = (
        "union"
    )

    # GLiNER
    gliner_model: str = "gliner-community/gliner_medium-v2.5"
    gliner_threshold: float = 0.5
    gliner_batch_size: int = 8

    # GLiREL (no spacy_model — uses regex tokenizer)
    glirel_model: str = "jackboyla/glirel-large-v0"
    glirel_threshold: float = 0.5

    # LLM
    max_text_chars: int = 8000
    llm_max_concurrent: int = 10

    # Normalisation
    min_entity_length: int = 3
    min_entity_score: float = 0.0

    # Aggregation
    # (no toggle — aggregation is always-on)

    # Resolution — clustering
    use_llm_resolution: bool = True
    use_deterministic_dedupe: bool = True
    dedupe_semhash_threshold: float = 0.95
    dedupe_minhash_threshold: float = 0.9
    resolution_cluster_size: int = 128
    resolution_kmeans_max_iter: int = 20

    # Resolution — retrieval (BM25 + embedding fusion)
    resolution_top_k: int = 16
    resolution_bm25_weight: float = 0.5
    resolution_embedding_weight: float = 0.5

    # Resolution — concurrency
    resolution_max_concurrent_clusters: int = 8
    resolution_max_concurrent_llm_calls: int = 10

    # Community detection
    community: CommunityDetectionConfig = CommunityDetectionConfig()

    # Output
    output_dir: str = "/tmp/extraction_output"

    @model_validator(mode="after")
    def _validate_pipeline_settings(self) -> "ExtractionPipelineConfig":
        if self.use_glirel and not self.use_gliner:
            raise ValueError(
                "use_glirel=True requires use_gliner=True. "
                "GLiREL uses GLiNER entity spans as input.",
            )
        if (
            abs(self.resolution_bm25_weight + self.resolution_embedding_weight - 1.0)
            > 1e-9
        ):
            raise ValueError(
                "resolution_bm25_weight + resolution_embedding_weight must sum to 1.0",
            )
        return self
