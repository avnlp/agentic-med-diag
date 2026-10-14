"""OnSIDES adverse-drug-reaction loader (DRG-PHE).

Reads the OnSIDES release zip (high-confidence drug-effect pairs plus the RxNorm and
MedDRA vocabularies). OnSIDES keeps native RxNorm drug ids and MedDRA effect ids -
OptimusKG does not harmonize these to ChEMBL/HPO - so this loader matches the gold by
keeping them native (no harmonizer/enricher). License: MIT.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema.onsides import ONSIDES_SCHEMA
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.formats import read_zip_csv
from am_diag.loaders.graph.vocab import (
    Node,
    Relation,
    Source,
    label_for,
    structural_label,
)


_ZIP = "onsides.zip"
_HIGH_CONFIDENCE = "csv/high_confidence.csv"
_RXNORM = "csv/vocab_rxnorm_ingredient.csv"
_MEDDRA = "csv/vocab_meddra_adverse_effect.csv"


class OnsidesGraphLoader(GraphLoader):
    """Build the OnSIDES drug-adverse-reaction subgraph from the release zip."""

    dataset_name: ClassVar[str] = "onsides"
    graph_schema: ClassVar = ONSIDES_SCHEMA

    def __init__(self, source_dir: str | Path) -> None:
        """Args: source_dir holds ``onsides.zip``."""
        self._dir = Path(source_dir)

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build DRG-PHE adverse-reaction edges with RxNorm/MedDRA nodes."""
        zip_path = self._dir / _ZIP
        high_confidence = read_zip_csv(
            zip_path, _HIGH_CONFIDENCE, infer_schema_length=0
        ).with_columns(
            ("RXNORM:" + pl.col("ingredient_id")).alias("ingredient_id"),
            ("meddra:" + pl.col("effect_meddra_id")).alias("effect_meddra_id"),
        )
        rxnorm = (
            read_zip_csv(zip_path, _RXNORM, infer_schema_length=0)
            .filter(pl.col("rxnorm_id").str.contains(r"^\d+$"))
            .with_columns(("RXNORM:" + pl.col("rxnorm_id")).alias("rxnorm_id"))
        )
        meddra = read_zip_csv(zip_path, _MEDDRA, infer_schema_length=0).with_columns(
            ("meddra:" + pl.col("meddra_id")).alias("meddra_id")
        )
        joined = high_confidence.join(
            rxnorm, left_on="ingredient_id", right_on="rxnorm_id", how="inner"
        ).join(meddra, left_on="effect_meddra_id", right_on="meddra_id", how="inner")

        edges = joined.select(
            pl.col("ingredient_id").alias("from"),
            pl.col("effect_meddra_id").alias("to"),
            pl.lit(Relation.ADVERSE_DRUG_REACTION.value).alias("type"),
            pl.lit(structural_label(Node.DRUG, Node.PHENOTYPE)).alias("label"),
            pl.lit([Source.ONSIDES.value]).alias("sources"),
        ).unique(subset=["from", "to"])
        drugs = (
            joined.select(
                pl.col("ingredient_id").alias("curie"),
                pl.col("rxnorm_name").alias("name"),
            )
            .unique("curie")
            .with_columns(pl.lit(label_for(Node.DRUG)).alias("label"))
        )
        phenotypes = (
            joined.select(
                pl.col("effect_meddra_id").alias("curie"),
                pl.col("meddra_name").alias("name"),
            )
            .unique("curie")
            .with_columns(pl.lit(label_for(Node.PHENOTYPE)).alias("label"))
        )
        nodes = pl.concat([drugs, phenotypes], how="diagonal")
        return nodes, edges
