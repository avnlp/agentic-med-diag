"""DrugBank drug-target (DRG-GEN) and drug-drug (DRG-DRG) loader.

Drug-target edges come from the per-role polypeptide CSVs joined to the UniProt->NCBI
gene map; drug-drug interactions are streamed from ``full database.xml``. Drug ids
harmonize to ChEMBL and gene ids to Ensembl so the nodes merge with the rest of the
graph. License: CC BY-NC 4.0 (derived-subgraph redistribution is restricted).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema.drugbank import DRUGBANK_SCHEMA
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.formats import read_drugbank_interactions_xml
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.vocab import (
    Node,
    Relation,
    Source,
    label_for,
    structural_label,
)


_ROLE_FILES = {
    "carrier": "drugbank_all_carrier_polypeptide_ids.csv",
    "enzyme": "drugbank_all_enzyme_polypeptide_ids.csv",
    "target": "drugbank_all_target_polypeptide_ids.csv",
    "transporter": "drugbank_all_transporter_polypeptide_ids.csv",
}
_RELATION_BY_ROLE = {
    "target": Relation.TARGET.value,
    "enzyme": Relation.ENZYME.value,
    "transporter": Relation.TRANSPORTER.value,
    "carrier": Relation.CARRIER.value,
}


class DrugBankGraphLoader(GraphLoader):
    """Build the DrugBank drug-target and drug-drug subgraph."""

    dataset_name: ClassVar[str] = "drugbank"
    graph_schema: ClassVar = DRUGBANK_SCHEMA

    def __init__(
        self,
        source_dir: str | Path,
        *,
        harmonizer: Harmonizer,
        enricher: NodeEnricher,
    ) -> None:
        """Args: source_dir holds the DrugBank files; harmonizer/enricher are shared."""
        self._dir = Path(source_dir)
        self._harmonizer = harmonizer
        self._enricher = enricher

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build DrugBank drug-gene and drug-drug edges with enriched nodes."""
        drug_gene = self._drug_gene_edges()
        drug_drug = self._drug_drug_edges()
        edges = pl.concat([drug_gene, drug_drug], how="diagonal")
        nodes = self._nodes(drug_gene, drug_drug)
        return nodes, edges

    def _raw_drug_gene(self) -> pl.DataFrame:
        gene_map = (
            pl.read_csv(
                self._dir / "drugbank_gene_map.tsv",
                separator="\t",
                infer_schema_length=0,
            )
            .select(
                pl.col("UniProt ID(supplied by UniProt)").alias("uniprot_id"),
                pl.col("NCBI Gene ID(supplied by NCBI)").alias("ncbi_gene_id"),
            )
            .drop_nulls()
        )
        frames = []
        for role, filename in _ROLE_FILES.items():
            df = pl.read_csv(self._dir / filename, infer_schema_length=0)
            frames.append(
                df.filter(pl.col("Species") == "Humans")
                .join(
                    gene_map, left_on="UniProt ID", right_on="uniprot_id", how="inner"
                )
                .with_columns(pl.col("Drug IDs").str.split("; ").alias("drug_ids"))
                .explode("drug_ids")
                .select(
                    ("DRUGBANK:" + pl.col("drug_ids")).alias("drug_bank_id"),
                    pl.col("ncbi_gene_id"),
                    pl.lit(role).alias("role"),
                )
            )
        return pl.concat(frames, how="diagonal")

    def _drug_gene_edges(self) -> pl.DataFrame:
        raw = self._raw_drug_gene()
        mapped = self._harmonizer.to_ensembl(
            self._harmonizer.to_chembl(raw, "drug_bank_id"), "ncbi_gene_id"
        )
        return (
            mapped.select(
                pl.col("chembl_id").alias("from"),
                pl.col("ensembl_id").alias("to"),
                pl.col("role")
                .replace_strict(_RELATION_BY_ROLE, default=Relation.OTHER.value)
                .alias("type"),
            )
            .unique(["from", "to", "type"])
            .with_columns(
                pl.lit(structural_label(Node.DRUG, Node.GENE)).alias("label"),
                pl.lit([Source.DRUG_BANK.value]).alias("sources"),
            )
        )

    def _drug_drug_edges(self) -> pl.DataFrame:
        raw = read_drugbank_interactions_xml(self._dir / "full database.xml")
        mapped = (
            self._harmonizer.to_chembl(raw, "head_drug_id")
            .rename({"chembl_id": "head_chembl"})
            .pipe(self._harmonizer.to_chembl, "tail_drug_id")
            .rename({"chembl_id": "tail_chembl"})
        )
        return mapped.select(
            pl.col("head_chembl").alias("from"),
            pl.col("tail_chembl").alias("to"),
            pl.lit(Relation.SYNERGISTIC_INTERACTION.value).alias("type"),
            pl.lit(structural_label(Node.DRUG, Node.DRUG)).alias("label"),
            pl.lit([Source.DRUG_BANK.value]).alias("sources"),
            pl.col("description").alias("interaction_description"),
        ).unique(["from", "to"])

    def _nodes(self, drug_gene: pl.DataFrame, drug_drug: pl.DataFrame) -> pl.DataFrame:
        drug_curies = pl.concat(
            [
                drug_gene.select(pl.col("from").alias("curie")),
                drug_drug.select(pl.col("from").alias("curie")),
                drug_drug.select(pl.col("to").alias("curie")),
            ]
        ).unique("curie")
        gene_curies = drug_gene.select(pl.col("to").alias("curie")).unique("curie")

        drugs = self._enricher.enrich_drugs(
            drug_curies.with_columns(pl.lit(label_for(Node.DRUG)).alias("label"))
        ).with_columns(
            pl.coalesce([pl.col("drug_name"), pl.col("curie")]).alias("name")
        )
        genes = self._enricher.enrich_genes(
            gene_curies.with_columns(pl.lit(label_for(Node.GENE)).alias("label"))
        ).with_columns(pl.coalesce([pl.col("symbol"), pl.col("curie")]).alias("name"))
        return pl.concat([drugs, genes], how="diagonal")
