"""Reactome pathway backbone loader (PWY nodes + PWY-PWY hierarchy).

Reads the two plain Reactome text exports: ``ReactomePathways.txt``
(``stable_id``, ``name``, ``species``) and ``ReactomePathwaysRelation.txt``
(``parent``, ``child``). Restricted to human pathways (``R-HSA-`` stable ids).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema.reactome import REACTOME_SCHEMA
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.vocab import (
    Node,
    Relation,
    Source,
    label_for,
    structural_label,
)


_HUMAN_PREFIX = "R-HSA-"


class ReactomeGraphLoader(GraphLoader):
    """Build the human Reactome pathway subgraph. License: CC0."""

    dataset_name: ClassVar[str] = "reactome"
    graph_schema: ClassVar = REACTOME_SCHEMA

    def __init__(self, source_dir: str | Path) -> None:
        """Args: source_dir: directory with the two Reactome ``.txt`` exports."""
        self._dir = Path(source_dir)

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build human pathway nodes and the PWY-PWY hierarchy from the exports."""
        pathways = pl.read_csv(
            self._dir / "ReactomePathways.txt",
            separator="\t",
            has_header=False,
            new_columns=["stable_id", "name", "species"],
            infer_schema_length=0,
        ).filter(pl.col("stable_id").str.starts_with(_HUMAN_PREFIX))
        relations = pl.read_csv(
            self._dir / "ReactomePathwaysRelation.txt",
            separator="\t",
            has_header=False,
            new_columns=["parent", "child"],
            infer_schema_length=0,
        ).filter(
            pl.col("parent").str.starts_with(_HUMAN_PREFIX)
            & pl.col("child").str.starts_with(_HUMAN_PREFIX)
        )
        nodes = pathways.select(
            pl.col("stable_id").alias("curie"),
            pl.col("name"),
            pl.col("species"),
            pl.lit(label_for(Node.PATHWAY)).alias("label"),
        ).unique("curie")
        edges = relations.select(
            pl.col("parent").alias("from"),
            pl.col("child").alias("to"),
            pl.lit(Relation.PARENT.value).alias("type"),
            pl.lit(structural_label(Node.PATHWAY, Node.PATHWAY)).alias("label"),
            pl.lit([Source.REACTOME.value]).alias("sources"),
        ).unique(subset=["from", "to"])
        return nodes, edges
