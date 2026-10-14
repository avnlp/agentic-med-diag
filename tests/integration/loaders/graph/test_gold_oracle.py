"""Cross-check a raw loader against the released OptimusKG gold for the same source.

Because the raw builders adopt OptimusKG's exact canonical CURIE scheme, a raw
subgraph's node/edge identities should align with the released gold's slice for that
source. This test validates that alignment against real data and is skipped unless the
required directories are configured via environment variables:

- ``GRAPH_REFERENCE_DIR`` / ``GRAPH_RAW_DATA_DIR`` - the raw + reference roots.
- ``OPTIMUSKG_GOLD_DIR`` - the released gold Parquet directory.

It does not require Neo4j/Qdrant; it compares ``build_frames`` output only.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from am_diag.loaders.graph.bgee import BgeeGraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.optimuskg import OptimusKGGraphLoader
from am_diag.loaders.graph.reference_data import ReferenceData
from am_diag.loaders.graph.settings import GraphSourcesSettings


_GOLD_DIR = os.environ.get("OPTIMUSKG_GOLD_DIR")


@pytest.mark.skipif(_GOLD_DIR is None, reason="OPTIMUSKG_GOLD_DIR not configured")
def test_bgee_nodes_align_with_gold() -> None:
    """Bgee anatomy/gene CURIEs are a subset of the released gold's nodes."""
    settings = GraphSourcesSettings()
    reference = ReferenceData(settings.reference_dir)
    bgee = BgeeGraphLoader(
        settings.source_dir("bgee"), enricher=NodeEnricher(reference)
    )
    bgee_nodes, _ = bgee.build_frames()

    gold_nodes, _ = OptimusKGGraphLoader(Path(_GOLD_DIR)).build_frames()
    gold_curies = set(gold_nodes["curie"].to_list())

    bgee_curies = set(bgee_nodes["curie"].to_list())
    missing = bgee_curies - gold_curies
    assert not missing, (
        f"{len(missing)} Bgee CURIEs absent from gold (e.g. {list(missing)[:5]})"
    )
