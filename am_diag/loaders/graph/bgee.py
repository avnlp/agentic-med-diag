"""Bgee anatomy-gene expression loader (ANA-GEN).

Reads the Homo sapiens advanced-expression TSV. Anatomy ids are already Uberon and
gene ids already Ensembl, so no harmonization is needed; gene nodes are enriched
with Open Targets metadata. This is the largest source (~8.8M edges); ingest it
with ``embedding_scope="entities"`` to avoid relation-vector fan-out.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema.bgee import BGEE_SCHEMA
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.vocab import (
    Node,
    Relation,
    Source,
    label_for,
    structural_label,
)


_COLUMNS = [
    "Gene ID",
    "Gene name",
    "Anatomical entity ID",
    "Anatomical entity name",
    "Expression",
    "Call quality",
    "Expression rank",
]


class BgeeGraphLoader(GraphLoader):
    """Build the Bgee ANA-GEN expression subgraph. License: CC0."""

    dataset_name: ClassVar[str] = "bgee"
    graph_schema: ClassVar = BGEE_SCHEMA

    def __init__(self, source_dir: str | Path, *, enricher: NodeEnricher) -> None:
        """Args: source_dir holds the Bgee TSV; enricher supplies gene metadata."""
        self._dir = Path(source_dir)
        self._enricher = enricher

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build Bgee ANA-GEN anatomy-gene expression nodes and edges from the TSV."""
        raw = (
            pl.read_csv(
                self._dir / "Homo_sapiens_expr_advanced.tsv.gz",
                separator="\t",
                infer_schema_length=0,
                columns=_COLUMNS,
            )
            .filter(pl.col("Anatomical entity ID").str.starts_with("UBERON"))
            .select(
                pl.col("Gene ID").alias("gene_id"),
                pl.col("Gene name").alias("gene_name"),
                pl.col("Anatomical entity ID").alias("anatomy_id"),
                pl.col("Anatomical entity name").alias("anatomy_name"),
                pl.col("Expression").alias("expression"),
                pl.col("Call quality").alias("call_quality"),
                pl.col("Expression rank").alias("expression_rank"),
            )
            .filter(~pl.col("anatomy_id").str.contains("∩"))
        )
        edges = (
            raw.filter(pl.col("expression").is_in(["present", "absent"]))
            .with_columns(
                pl.col("anatomy_id").str.replace("UBERON:", "UBERON_").alias("from"),
                pl.col("gene_id").alias("to"),
                pl.when(pl.col("expression") == "present")
                .then(pl.lit(Relation.EXPRESSION_PRESENT.value))
                .otherwise(pl.lit(Relation.EXPRESSION_ABSENT.value))
                .alias("type"),
                pl.lit(structural_label(Node.ANATOMY, Node.GENE)).alias("label"),
                pl.lit([Source.BGEE.value]).alias("sources"),
                pl.col("expression_rank").cast(pl.Float64),
            )
            .select(
                "from",
                "to",
                "type",
                "label",
                "sources",
                "expression_rank",
                "call_quality",
            )
            .unique(subset=["from", "to"])
        )
        anatomy = (
            raw.select(
                pl.col("anatomy_id").str.replace("UBERON:", "UBERON_").alias("curie"),
                pl.col("anatomy_name").alias("name"),
            )
            .unique("curie")
            .with_columns(pl.lit(label_for(Node.ANATOMY)).alias("label"))
        )
        genes = (
            raw.select(
                pl.col("gene_id").alias("curie"), pl.col("gene_name").alias("name")
            )
            .unique("curie")
            .with_columns(pl.lit(label_for(Node.GENE)).alias("label"))
        )
        genes = self._enricher.enrich_genes(genes)
        nodes = pl.concat([anatomy, genes], how="diagonal")
        return nodes, edges
