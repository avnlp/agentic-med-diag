"""PrimeKG / PPI protein-protein interaction loader (GEN-GEN).

Reads ``protein_protein.csv`` (NCBI entrez endpoints + ``|``-separated source
databases). Both endpoints harmonize to Ensembl; interactions whose endpoints lack
an Ensembl mapping are dropped so the gene nodes merge with the rest of the graph.
License: MIT.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema.ppi import PPI_SCHEMA
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.vocab import (
    Node,
    Relation,
    Source,
    label_for,
    structural_label,
)


class PpiGraphLoader(GraphLoader):
    """Build the PPI gene-gene interaction subgraph."""

    dataset_name: ClassVar[str] = "ppi"
    graph_schema: ClassVar = PPI_SCHEMA

    def __init__(
        self,
        source_dir: str | Path,
        *,
        harmonizer: Harmonizer,
        enricher: NodeEnricher,
    ) -> None:
        """Hold the PPI source dir plus the shared harmonizer and enricher."""
        self._dir = Path(source_dir)
        self._harmonizer = harmonizer
        self._enricher = enricher

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build GEN-GEN interaction edges with Ensembl-harmonized endpoints."""
        raw = pl.read_csv(
            self._dir / "protein_protein.csv", infer_schema_length=0
        ).select(
            pl.col("proteinA_entrezid").alias("ncbi_a"),
            pl.col("proteinB_entrezid").alias("ncbi_b"),
            pl.col("databases").str.split("|").alias("databases"),
        )
        with_a = self._harmonizer.to_ensembl(raw, "ncbi_a").rename(
            {"ensembl_id": "from"}
        )
        mapped = self._harmonizer.to_ensembl(with_a, "ncbi_b").rename(
            {"ensembl_id": "to"}
        )
        edges = mapped.select(
            "from",
            "to",
            pl.lit(Relation.INTERACTS_WITH.value).alias("type"),
            pl.lit(structural_label(Node.GENE, Node.GENE)).alias("label"),
            pl.lit([Source.PRIMEKG.value]).alias("sources"),
            pl.col("databases"),
        ).unique(subset=["from", "to"])
        gene_curies = (
            pl.concat(
                [
                    edges.select(pl.col("from").alias("curie")),
                    edges.select(pl.col("to").alias("curie")),
                ]
            )
            .unique("curie")
            .with_columns(pl.lit(label_for(Node.GENE)).alias("label"))
        )
        nodes = self._enricher.enrich_genes(gene_curies).with_columns(
            pl.coalesce([pl.col("symbol"), pl.col("curie")]).alias("name")
        )
        return nodes, edges
