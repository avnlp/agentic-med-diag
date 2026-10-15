"""DrugCentral drug-disease / drug-phenotype loader (DRG-DIS, DRG-PHE).

Emits both ``INDICATION`` and ``CONTRAINDICATION`` edges.

Reads the DrugCentral PostgreSQL dump serverlessly (no server): the ``identifier``
table supplies each structure's ChEMBL id and ``omop_relationship`` supplies the
indication/contraindication links keyed by UMLS CUI. Drugs harmonize to ChEMBL and
diseases to the Open Targets canonical id; an ``HP_`` canonical id routes the edge to
DRG-PHE. License: CC BY-SA 4.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema.drugcentral import DRUGCENTRAL_SCHEMA
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.formats import read_pg_dump_table
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.relations import collapse_relations
from am_diag.loaders.graph.vocab import (
    Node,
    Relation,
    Source,
    label_for,
    structural_label,
)


_DUMP = "drugcentral_dump.sql.gz"
_SOURCE = "DRUG_CENTRAL"
_RELATION_MAP = {
    "indication": Relation.INDICATION.value,
    "contraindication": Relation.CONTRAINDICATION.value,
    "off-label use": Relation.OFF_LABEL_USE.value,
}


class DrugCentralGraphLoader(GraphLoader):
    """Build the DrugCentral drug-disease / drug-phenotype subgraph."""

    dataset_name: ClassVar[str] = "drugcentral"
    graph_schema: ClassVar = DRUGCENTRAL_SCHEMA

    def __init__(
        self,
        source_dir: str | Path,
        *,
        harmonizer: Harmonizer,
        enricher: NodeEnricher,
    ) -> None:
        """Hold the dump dir plus the shared harmonizer and enricher."""
        self._dir = Path(source_dir)
        self._harmonizer = harmonizer
        self._enricher = enricher

    def _mapped(self) -> pl.DataFrame:
        dump = self._dir / _DUMP
        omop = read_pg_dump_table(dump, "omop_relationship").select(
            pl.col("struct_id"),
            pl.col("relationship_name"),
            pl.col("umls_cui"),
            pl.col("concept_name"),
        )
        chembl = (
            read_pg_dump_table(dump, "identifier")
            .filter(pl.col("id_type") == "ChEMBL_ID")
            .select(pl.col("struct_id"), pl.col("identifier").alias("chembl_id"))
        )
        joined = omop.join(chembl, on="struct_id", how="inner").filter(
            pl.col("relationship_name").is_in(list(_RELATION_MAP))
            & pl.col("umls_cui").is_not_null()
        )
        return self._harmonizer.disease_umls_to_canonical(joined, "umls_cui")

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build DRG-DIS/DRG-PHE indication edges with harmonized endpoints."""
        mapped = self._mapped().with_columns(
            pl.col("id").str.starts_with("HP_").alias("is_phenotype")
        )
        edges = collapse_relations(
            mapped.select(
                pl.col("chembl_id").alias("from"),
                pl.col("id").alias("to"),
                pl.col("relationship_name").replace_strict(_RELATION_MAP).alias("type"),
                pl.when(pl.col("is_phenotype"))
                .then(pl.lit(structural_label(Node.DRUG, Node.PHENOTYPE)))
                .otherwise(pl.lit(structural_label(Node.DRUG, Node.DISEASE)))
                .alias("label"),
                pl.lit([_SOURCE]).alias("sources"),
            ),
            Source.DRUG_CENTRAL,
        )
        drugs = self._enricher.enrich_drugs(
            mapped.select(pl.col("chembl_id").alias("curie"))
            .unique("curie")
            .with_columns(pl.lit(label_for(Node.DRUG)).alias("label"))
        ).with_columns(
            pl.coalesce([pl.col("drug_name"), pl.col("curie")]).alias("name")
        )
        targets = mapped.select(
            pl.col("id").alias("curie"),
            pl.when(pl.col("is_phenotype"))
            .then(pl.lit(label_for(Node.PHENOTYPE)))
            .otherwise(pl.lit(label_for(Node.DISEASE)))
            .alias("label"),
            pl.col("concept_name").alias("name"),
        ).unique("curie")
        nodes = pl.concat([drugs, targets], how="diagonal")
        return nodes, edges
