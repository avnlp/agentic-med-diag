"""Single-type OBO ontology backbone loaders (Uberon, Mondo, HPO).

Each builds identity nodes plus an ``is_a`` ``PARENT`` hierarchy from one OBO Graph
JSON file. The three share all logic except the file name, id prefix, node type, and
provenance source, so a small shared base captures the transform and each concrete
loader is just four class variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema.hpo import HPO_SCHEMA
from am_diag.common.schema.mondo import MONDO_SCHEMA
from am_diag.common.schema.uberon import UBERON_SCHEMA
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.formats import obo_isa_edges, obo_terms, read_obograph_json
from am_diag.loaders.graph.vocab import (
    Node,
    Relation,
    Source,
    label_for,
    structural_label,
)


class _OboBackboneLoader(GraphLoader):
    """Shared base for single-node-type OBO ontologies with an is_a hierarchy.

    Subclasses set ``ontology_file``, ``id_prefix``, ``node_type``, and ``source``.
    """

    ontology_file: ClassVar[str]
    id_prefix: ClassVar[str]
    node_type: ClassVar[Node]
    source: ClassVar[Source]

    def __init__(self, source_dir: str | Path) -> None:
        """Args: source_dir: directory holding ``ontology_file`` (the OBO JSON)."""
        self._dir = Path(source_dir)

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        raw = read_obograph_json(self._dir / self.ontology_file)
        nodes = (
            obo_terms(raw)
            .filter(pl.col("id").str.starts_with(self.id_prefix))
            .rename({"id": "curie"})
            .with_columns(pl.lit(label_for(self.node_type)).alias("label"))
        )
        edges = (
            obo_isa_edges(raw)
            .filter(
                pl.col("parent").str.starts_with(self.id_prefix)
                & pl.col("child").str.starts_with(self.id_prefix)
            )
            .select(
                pl.col("parent").alias("from"),
                pl.col("child").alias("to"),
                pl.lit(Relation.PARENT.value).alias("type"),
                pl.lit(structural_label(self.node_type, self.node_type)).alias("label"),
                pl.lit([self.source.value]).alias("sources"),
            )
            .unique(subset=["from", "to"])
        )
        return nodes, edges


class UberonGraphLoader(_OboBackboneLoader):
    """Uberon anatomy backbone (ANA nodes, ANA-ANA hierarchy). License: CC0."""

    dataset_name: ClassVar[str] = "uberon"
    graph_schema: ClassVar = UBERON_SCHEMA
    ontology_file: ClassVar[str] = "uberon.json"
    id_prefix: ClassVar[str] = "UBERON_"
    node_type: ClassVar[Node] = Node.ANATOMY
    source: ClassVar[Source] = Source.UBERON


class MondoGraphLoader(_OboBackboneLoader):
    """Mondo disease backbone (DIS nodes, DIS-DIS hierarchy). License: CC BY 4.0."""

    dataset_name: ClassVar[str] = "mondo"
    graph_schema: ClassVar = MONDO_SCHEMA
    ontology_file: ClassVar[str] = "mondo.json"
    id_prefix: ClassVar[str] = "MONDO_"
    node_type: ClassVar[Node] = Node.DISEASE
    source: ClassVar[Source] = Source.MONDO


class HpoGraphLoader(_OboBackboneLoader):
    """HPO phenotype backbone (PHE nodes, PHE-PHE hierarchy). License: custom open."""

    dataset_name: ClassVar[str] = "hpo"
    graph_schema: ClassVar = HPO_SCHEMA
    ontology_file: ClassVar[str] = "hp.json"
    id_prefix: ClassVar[str] = "HP_"
    node_type: ClassVar[Node] = Node.PHENOTYPE
    source: ClassVar[Source] = Source.HPO
