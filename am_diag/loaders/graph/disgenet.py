"""DisGeNET disease/phenotype-gene association loader (DIS-GEN, PHE-GEN).

Reads the curated gene-disease associations TSV. Gene symbols resolve to Ensembl
via Open Targets, and the UMLS ``diseaseId`` resolves to the Open Targets canonical
disease id; rows split into DIS-GEN or PHE-GEN by ``diseaseType``. License:
CC BY-NC-SA 4.0 (derived-subgraph redistribution is restricted).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema.disgenet import DISGENET_SCHEMA
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.reference_data import ReferenceData
from am_diag.loaders.graph.vocab import (
    Node,
    Relation,
    Source,
    label_for,
    structural_label,
)


class DisGeNETGraphLoader(GraphLoader):
    """Build the DisGeNET disease/phenotype-gene subgraph from the curated TSV."""

    dataset_name: ClassVar[str] = "disgenet"
    graph_schema: ClassVar = DISGENET_SCHEMA

    def __init__(
        self,
        source_dir: str | Path,
        *,
        reference: ReferenceData,
        harmonizer: Harmonizer,
        enricher: NodeEnricher,
    ) -> None:
        """Hold the curated-TSV dir plus shared reference, harmonizer, and enricher."""
        self._dir = Path(source_dir)
        self._ref = reference
        self._harmonizer = harmonizer
        self._enricher = enricher

    def _mapped(self) -> pl.DataFrame:
        raw = pl.read_csv(
            self._dir / "curated_gene_disease_associations.tsv",
            separator="\t",
            infer_schema_length=0,
        ).select(
            pl.col("geneSymbol").alias("gene_symbol"),
            pl.col("diseaseId").alias("umls_id"),
            pl.col("diseaseName").alias("disease_name"),
            pl.col("diseaseType").alias("disease_type"),
            pl.col("score").cast(pl.Float64, strict=False).alias("score"),
            pl.col("EI").cast(pl.Float64, strict=False).alias("evidence_index"),
        )
        target = self._ref.opentargets_target.select(
            pl.col("id").alias("ensembl_id"), pl.col("approved_symbol")
        )
        with_gene = raw.join(
            target, left_on="gene_symbol", right_on="approved_symbol", how="inner"
        )
        return self._harmonizer.disease_umls_to_canonical(with_gene, "umls_id")

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build the DisGeNET disease/phenotype-gene subgraph from the curated TSV."""
        mapped = self._mapped()
        edges = (
            mapped.with_columns(
                pl.when(pl.col("disease_type") == "phenotype")
                .then(pl.lit(structural_label(Node.PHENOTYPE, Node.GENE)))
                .otherwise(pl.lit(structural_label(Node.DISEASE, Node.GENE)))
                .alias("label"),
                pl.lit(Relation.ASSOCIATED_WITH.value).alias("type"),
                pl.lit([Source.DISGENET.value]).alias("sources"),
            )
            .select(
                pl.col("id").alias("from"),
                pl.col("ensembl_id").alias("to"),
                "type",
                "label",
                "sources",
                "score",
                "evidence_index",
            )
            .unique(subset=["from", "to"])
        )
        disease_curies = mapped.select(
            pl.col("id").alias("curie"),
            pl.when(pl.col("disease_type") == "phenotype")
            .then(pl.lit(label_for(Node.PHENOTYPE)))
            .otherwise(pl.lit(label_for(Node.DISEASE)))
            .alias("label"),
            pl.col("disease_name").alias("name"),
        ).unique("curie")
        gene_curies = (
            mapped.select(pl.col("ensembl_id").alias("curie"))
            .unique("curie")
            .with_columns(pl.lit(label_for(Node.GENE)).alias("label"))
        )
        diseases = self._enricher.enrich_diseases(disease_curies)
        genes = self._enricher.enrich_genes(gene_curies).with_columns(
            pl.coalesce([pl.col("symbol"), pl.col("curie")]).alias("name")
        )
        nodes = pl.concat([diseases, genes], how="diagonal")
        return nodes, edges
