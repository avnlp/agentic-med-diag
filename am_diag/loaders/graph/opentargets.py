"""Open Targets loader: the multi-relation target-disease subgraph.

Reads the Open Targets parquet exports (already keyed on Ensembl genes, ChEMBL
drugs, and EFO/MONDO/HP/GO/UBERON entities, so no harmonization is needed) and emits:
target-(disease/phenotype/anatomy/GO) ``ASSOCIATED_WITH`` edges, the disease ``PARENT``
hierarchy, disease ``HAS_PHENOTYPE`` links, drug ``INDICATION`` edges, and drug-target
mechanism edges (one relation per action type). License: CC0.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema.opentargets import DRUG_GENE_ACTIONS, OPEN_TARGETS_SCHEMA
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.formats import go_term_namespaces
from am_diag.loaders.graph.reference_data import ReferenceData
from am_diag.loaders.graph.vocab import (
    Node,
    Relation,
    Source,
    label_for,
    structural_label,
)


_SRC = [Source.OPEN_TARGETS.value]
_ABBREV_TO_LABEL = {node.value: label_for(node) for node in Node}
_GO_NAMESPACE_TO_ABBREV = {
    "biological_process": Node.BIOLOGICAL_PROCESS.value,
    "cellular_component": Node.CELLULAR_COMPONENT.value,
    "molecular_function": Node.MOLECULAR_FUNCTION.value,
}


class OpenTargetsGraphLoader(GraphLoader):
    """Build the Open Targets multi-relation subgraph from its parquet exports."""

    dataset_name: ClassVar[str] = "opentargets"
    graph_schema: ClassVar = OPEN_TARGETS_SCHEMA

    def __init__(
        self,
        source_dir: str | Path,
        *,
        reference: ReferenceData,
        enricher: NodeEnricher,
    ) -> None:
        """Hold the OT parquet root plus the shared reference and enricher."""
        self._dir = Path(source_dir)
        self._ref = reference
        self._enricher = enricher

    def _read(self, name: str) -> pl.DataFrame:
        return pl.read_parquet(self._dir / "opentargets" / name / "*.parquet")

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build all Open Targets edge families and their enriched endpoint nodes."""
        edges = pl.concat(
            [
                self._association_edges(),
                self._disease_disease(),
                self._disease_phenotype(),
                self._drug_disease(),
                self._drug_gene(),
            ],
            how="diagonal",
        )
        nodes = self._nodes(edges)
        return nodes, edges

    def _association_edges(self) -> pl.DataFrame:
        assoc = self._read("target_disease_associations").select(
            pl.col("disease_id"),
            pl.col("target_id"),
            pl.col("metadata").struct.field("score").alias("evidence_score"),
            pl.col("metadata").struct.field("evidence_count").alias("evidence_count"),
        )
        non_go = assoc.filter(~pl.col("disease_id").str.starts_with("GO_"))
        simple = non_go.with_columns(
            pl.when(pl.col("disease_id").str.contains("HP"))
            .then(pl.lit(Node.PHENOTYPE.value))
            .when(pl.col("disease_id").str.starts_with("UBERON_"))
            .then(pl.lit(Node.ANATOMY.value))
            .otherwise(pl.lit(Node.DISEASE.value))
            .alias("head_abbrev")
        )
        namespaces = go_term_namespaces(self._ref.ontology("go_plus"))
        go = (
            assoc.filter(pl.col("disease_id").str.starts_with("GO_"))
            .join(namespaces, left_on="disease_id", right_on="id", how="inner")
            .with_columns(
                pl.col("namespace")
                .replace_strict(_GO_NAMESPACE_TO_ABBREV)
                .alias("head_abbrev")
            )
        )
        combined = pl.concat([simple, go.select(simple.columns)], how="vertical")
        return combined.select(
            pl.col("disease_id").alias("from"),
            pl.col("target_id").alias("to"),
            pl.lit(Relation.ASSOCIATED_WITH.value).alias("type"),
            (pl.col("head_abbrev") + "-" + pl.lit(Node.GENE.value)).alias("label"),
            pl.lit(_SRC).alias("sources"),
            pl.col("evidence_score"),
            pl.col("evidence_count"),
        ).unique(subset=["from", "to"])

    def _disease_disease(self) -> pl.DataFrame:
        disease = self._ref.opentargets_disease.select(
            pl.col("id"), pl.col("metadata").struct.field("parents").alias("parents")
        )
        return (
            disease.explode("parents")
            .filter(
                pl.col("parents").is_not_null()
                & ~pl.col("id").str.contains("HP")
                & ~pl.col("id").str.starts_with("GO_")
            )
            .select(
                pl.col("parents").alias("from"),
                pl.col("id").alias("to"),
                pl.lit(Relation.PARENT.value).alias("type"),
                pl.lit(structural_label(Node.DISEASE, Node.DISEASE)).alias("label"),
                pl.lit(_SRC).alias("sources"),
            )
            .unique(subset=["from", "to"])
        )

    def _disease_phenotype(self) -> pl.DataFrame:
        return (
            self._read("disease_phenotype")
            .select(pl.col("disease"), pl.col("phenotype"))
            .filter(pl.col("phenotype").str.contains("HP"))
            .select(
                pl.col("disease").alias("from"),
                pl.col("phenotype").alias("to"),
                pl.lit(Relation.HAS_PHENOTYPE.value).alias("type"),
                pl.lit(structural_label(Node.DISEASE, Node.PHENOTYPE)).alias("label"),
                pl.lit(_SRC).alias("sources"),
            )
            .unique(subset=["from", "to"])
        )

    def _drug_disease(self) -> pl.DataFrame:
        return (
            self._read("drug_indication")
            .select(
                pl.col("id"),
                pl.col("metadata").struct.field("indications").alias("indications"),
            )
            .explode("indications")
            .unnest("indications")
            .filter(
                pl.col("disease").is_not_null()
                & ~pl.col("disease").str.contains("HP")
                & ~pl.col("disease").str.starts_with("GO_")
            )
            .select(
                pl.col("id").alias("from"),
                pl.col("disease").alias("to"),
                pl.lit(Relation.INDICATION.value).alias("type"),
                pl.lit(structural_label(Node.DRUG, Node.DISEASE)).alias("label"),
                pl.lit(_SRC).alias("sources"),
            )
            .unique(subset=["from", "to"])
        )

    def _drug_gene(self) -> pl.DataFrame:
        mechanism = self._read("drug_mechanism_of_action").select(
            pl.col("targets"),
            pl.col("chembl_ids"),
            pl.col("metadata").struct.field("action_type").alias("action_type"),
        )
        normalized = (
            mechanism.explode("chembl_ids")
            .explode("targets")
            .filter(
                pl.col("chembl_ids").is_not_null() & pl.col("targets").is_not_null()
            )
            .with_columns(
                pl.col("action_type")
                .str.to_uppercase()
                .str.replace_all(" ", "_")
                .alias("action")
            )
            .with_columns(
                pl.when(pl.col("action").is_in(DRUG_GENE_ACTIONS))
                .then(pl.col("action"))
                .otherwise(pl.lit("OTHER"))
                .alias("type")
            )
        )
        return normalized.select(
            pl.col("chembl_ids").alias("from"),
            pl.col("targets").alias("to"),
            pl.col("type"),
            pl.lit(structural_label(Node.DRUG, Node.GENE)).alias("label"),
            pl.lit(_SRC).alias("sources"),
        ).unique(subset=["from", "to", "type"])

    def _nodes(self, edges: pl.DataFrame) -> pl.DataFrame:
        heads = edges.select(
            pl.col("from").alias("curie"),
            pl.col("label").str.split("-").list.first().alias("abbrev"),
        )
        tails = edges.select(
            pl.col("to").alias("curie"),
            pl.col("label").str.split("-").list.last().alias("abbrev"),
        )
        endpoints = (
            pl.concat([heads, tails])
            .with_columns(
                pl.col("abbrev").replace_strict(_ABBREV_TO_LABEL).alias("label")
            )
            .drop("abbrev")
            .unique("curie")
        )
        frames = [
            self._enrich_group(endpoints, label) for label in _ABBREV_TO_LABEL.values()
        ]
        return pl.concat([f for f in frames if f.height > 0], how="diagonal")

    def _enrich_group(self, endpoints: pl.DataFrame, label: str) -> pl.DataFrame:
        group = endpoints.filter(pl.col("label") == label)
        if group.height == 0:
            return group
        if label == "Gene":
            return self._enricher.enrich_genes(group).with_columns(
                pl.coalesce([pl.col("symbol"), pl.col("curie")]).alias("name")
            )
        if label == "Drug":
            return self._enricher.enrich_drugs(group).with_columns(
                pl.coalesce([pl.col("drug_name"), pl.col("curie")]).alias("name")
            )
        if label == "Disease":
            return self._enricher.enrich_diseases(group).with_columns(
                pl.coalesce([pl.col("disease_name"), pl.col("curie")]).alias("name")
            )
        ontology = {
            "Phenotype": "hp",
            "AnatomicalStructure": "uberon",
            "GoBiologicalProcess": "go_plus",
            "GoCellularComponent": "go_plus",
            "GoMolecularFunction": "go_plus",
        }.get(label)
        if ontology is None:
            return group
        return self._enricher.enrich_from_ontology(group, ontology).with_columns(
            pl.coalesce([pl.col("term_name"), pl.col("curie")]).alias("name")
        )
