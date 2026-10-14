"""CTD environmental-exposure loader (EXP-DIS/GEN/EXP/BPO/CCO/MFN).

Reads the CTD exposure-events CSV. Exposure stressors stay native MeSH ids; disease
outcomes harmonize to Mondo via Mondo xrefs, gene outcomes to Ensembl via the marker
symbol, and GO outcomes route to BPO/CCO/MFN by namespace. Edges carry only the core
structural properties; OptimusKG's per-edge exposure-statistic aggregations (age,
receptor, smoking-status rollups) are intentionally omitted. License: CTD academic.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema.ctd import CTD_SCHEMA
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.formats import go_term_namespaces, obo_terms
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.reference_data import ReferenceData
from am_diag.loaders.graph.vocab import (
    Node,
    Relation,
    Source,
    label_for,
    structural_label,
)


_EVENTS = "CTD_exposure_events.csv.gz"
_GO_NAMESPACE_TO_NODE = {
    "biological_process": Node.BIOLOGICAL_PROCESS,
    "cellular_component": Node.CELLULAR_COMPONENT,
    "molecular_function": Node.MOLECULAR_FUNCTION,
}
_GO_NAMESPACE_TO_RELATION = {
    "biological_process": Relation.EXPOSURE_BIOLOGICAL_PROCESS.value,
    "cellular_component": Relation.EXPOSURE_CELLULAR_COMPONENT.value,
    "molecular_function": Relation.EXPOSURE_MOLECULAR_FUNCTION.value,
}


class CtdGraphLoader(GraphLoader):
    """Build the CTD environmental-exposure subgraph from the exposure-events CSV."""

    dataset_name: ClassVar[str] = "ctd"
    graph_schema: ClassVar = CTD_SCHEMA

    def __init__(
        self,
        source_dir: str | Path,
        *,
        reference: ReferenceData,
        harmonizer: Harmonizer,
        enricher: NodeEnricher,
    ) -> None:
        """Hold the events dir plus the shared reference, harmonizer, and enricher."""
        self._dir = Path(source_dir)
        self._ref = reference
        self._harmonizer = harmonizer
        self._enricher = enricher

    def _events(self) -> pl.DataFrame:
        raw = pl.read_csv(
            self._dir / _EVENTS,
            comment_prefix="#",
            infer_schema_length=0,
            null_values="",
        )
        return raw.select(
            ("MESH:" + pl.col("exposurestressorid")).alias("exposure_id"),
            pl.col("exposurestressorname").alias("exposure_name"),
            pl.col("exposuremarker").alias("marker_symbol"),
            pl.when(pl.col("exposuremarkerid").is_not_null())
            .then("MESH:" + pl.col("exposuremarkerid"))
            .alias("marker_id"),
            pl.when(pl.col("diseaseid").is_not_null())
            .then("MESH:" + pl.col("diseaseid"))
            .alias("disease_mesh"),
            pl.col("phenotypeid").str.replace("GO:", "GO_").alias("go_id"),
        )

    def _mondo_xrefs(self) -> pl.DataFrame:
        return (
            obo_terms(self._ref.ontology("mondo"))
            .select(pl.col("id"), pl.col("xrefs"))
            .explode("xrefs")
            .drop_nulls("xrefs")
            .select(pl.col("id"), pl.col("xrefs").alias("xref_id"))
            .unique()
        )

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build the six exposure edge families with harmonized outcome endpoints."""
        events = self._events()
        edges = pl.concat(
            [
                self._exposure_disease(events),
                self._exposure_gene(events),
                self._exposure_exposure(events),
                self._exposure_go(events),
            ],
            how="diagonal",
        )
        nodes = self._nodes(events, edges)
        return nodes, edges

    def _exposure_disease(self, events: pl.DataFrame) -> pl.DataFrame:
        return (
            events.filter(pl.col("disease_mesh").is_not_null())
            .join(self._mondo_xrefs(), left_on="disease_mesh", right_on="xref_id")
            .select(
                pl.col("exposure_id").alias("from"),
                pl.col("id").alias("to"),
                pl.lit(Relation.EXPOSURE_DISEASE.value).alias("type"),
                pl.lit(structural_label(Node.EXPOSURE, Node.DISEASE)).alias("label"),
                pl.lit([Source.CTD.value]).alias("sources"),
            )
            .unique(subset=["from", "to"])
        )

    def _exposure_gene(self, events: pl.DataFrame) -> pl.DataFrame:
        target = self._ref.opentargets_target.select(
            pl.col("id").alias("ensembl_id"), pl.col("approved_symbol")
        )
        return (
            events.filter(pl.col("marker_symbol").is_not_null())
            .join(target, left_on="marker_symbol", right_on="approved_symbol")
            .select(
                pl.col("exposure_id").alias("from"),
                pl.col("ensembl_id").alias("to"),
                pl.lit(Relation.EXPOSURE_GENE.value).alias("type"),
                pl.lit(structural_label(Node.EXPOSURE, Node.GENE)).alias("label"),
                pl.lit([Source.CTD.value]).alias("sources"),
            )
            .unique(subset=["from", "to"])
        )

    def _exposure_exposure(self, events: pl.DataFrame) -> pl.DataFrame:
        return (
            events.filter(
                pl.col("marker_id").is_not_null() & pl.col("marker_symbol").is_null()
            )
            .select(
                pl.col("exposure_id").alias("from"),
                pl.col("marker_id").alias("to"),
                pl.lit(Relation.EXPOSURE_EXPOSURE.value).alias("type"),
                pl.lit(structural_label(Node.EXPOSURE, Node.EXPOSURE)).alias("label"),
                pl.lit([Source.CTD.value]).alias("sources"),
            )
            .filter(pl.col("from") != pl.col("to"))
            .unique(subset=["from", "to"])
        )

    def _exposure_go(self, events: pl.DataFrame) -> pl.DataFrame:
        namespaces = go_term_namespaces(self._ref.ontology("go_plus"))
        joined = (
            events.filter(pl.col("go_id").is_not_null())
            .join(namespaces, left_on="go_id", right_on="id")
            .filter(pl.col("namespace").is_in(list(_GO_NAMESPACE_TO_NODE)))
        )
        return joined.select(
            pl.col("exposure_id").alias("from"),
            pl.col("go_id").alias("to"),
            pl.col("namespace").replace_strict(_GO_NAMESPACE_TO_RELATION).alias("type"),
            pl.struct(pl.col("namespace"))
            .map_elements(
                lambda row: structural_label(
                    Node.EXPOSURE, _GO_NAMESPACE_TO_NODE[row["namespace"]]
                ),
                return_dtype=pl.String,
            )
            .alias("label"),
            pl.lit([Source.CTD.value]).alias("sources"),
        ).unique(subset=["from", "to"])

    def _nodes(self, events: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
        exposure_names = events.select(
            pl.col("exposure_id").alias("curie"), pl.col("exposure_name").alias("name")
        ).unique("curie")
        exposure_curies = (
            edges.filter(pl.col("label").str.starts_with("EXP-"))
            .select(pl.col("from").alias("curie"))
            .vstack(
                edges.filter(pl.col("label") == "EXP-EXP").select(
                    pl.col("to").alias("curie")
                )
            )
            .unique("curie")
            .join(exposure_names, on="curie", how="left")
            .with_columns(pl.lit(label_for(Node.EXPOSURE)).alias("label"))
        )
        diseases = self._enricher.enrich_diseases(
            edges.filter(pl.col("label") == "EXP-DIS")
            .select(pl.col("to").alias("curie"))
            .unique("curie")
            .with_columns(pl.lit(label_for(Node.DISEASE)).alias("label"))
        )
        genes = self._enricher.enrich_genes(
            edges.filter(pl.col("label") == "EXP-GEN")
            .select(pl.col("to").alias("curie"))
            .unique("curie")
            .with_columns(pl.lit(label_for(Node.GENE)).alias("label"))
        ).with_columns(pl.coalesce([pl.col("symbol"), pl.col("curie")]).alias("name"))
        go_terms = self._enricher.enrich_from_ontology(
            edges.filter(pl.col("label").str.contains("-BPO|-CCO|-MFN"))
            .select(pl.col("to").alias("curie"))
            .unique("curie"),
            "go_plus",
        ).with_columns(
            pl.coalesce([pl.col("term_name"), pl.col("curie")]).alias("name")
        )
        # GO node labels come from the edges' structural label tail; resolve per curie.
        go_label = (
            edges.filter(pl.col("label").str.contains("-BPO|-CCO|-MFN"))
            .select(
                pl.col("to").alias("curie"),
                pl.col("label").str.split("-").list.last().alias("abbrev"),
            )
            .unique("curie")
            .with_columns(
                pl.col("abbrev")
                .replace_strict(
                    {n.value: label_for(n) for n in _GO_NAMESPACE_TO_NODE.values()}
                )
                .alias("label")
            )
            .drop("abbrev")
        )
        go_nodes = go_terms.join(go_label, on="curie", how="left")
        return pl.concat([exposure_curies, diseases, genes, go_nodes], how="diagonal")
