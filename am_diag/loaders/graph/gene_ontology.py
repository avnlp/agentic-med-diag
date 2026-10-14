"""Gene Ontology backbone loader (BPO/CCO/MFN terms + within-namespace is_a edges).

GO terms carry their sub-ontology in ``meta.basicPropertyValues`` under the
``hasOBONamespace`` predicate; that namespace selects the PascalCase label and the
``is_a`` edges stay within a namespace, so each edge's structural label is derived
from its endpoints' shared namespace.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema.gene_ontology import GENE_ONTOLOGY_SCHEMA
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.formats import obo_isa_edges, read_obograph_json
from am_diag.loaders.graph.vocab import Node, Relation, Source, label_for


_OBO = "http://purl.obolibrary.org/obo/"
_HAS_NAMESPACE = "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"

_NAMESPACE_TO_NODE: dict[str, Node] = {
    "biological_process": Node.BIOLOGICAL_PROCESS,
    "cellular_component": Node.CELLULAR_COMPONENT,
    "molecular_function": Node.MOLECULAR_FUNCTION,
}
_NAMESPACE_TO_LABEL = {ns: label_for(node) for ns, node in _NAMESPACE_TO_NODE.items()}
_NAMESPACE_TO_ABBREV = {ns: node.value for ns, node in _NAMESPACE_TO_NODE.items()}


class GeneOntologyGraphLoader(GraphLoader):
    """Build the GO ontology subgraph from ``go_plus.json``. License: CC BY."""

    dataset_name: ClassVar[str] = "gene_ontology"
    graph_schema: ClassVar = GENE_ONTOLOGY_SCHEMA

    def __init__(self, source_dir: str | Path) -> None:
        """Args: source_dir: directory holding ``go_plus.json``."""
        self._dir = Path(source_dir)

    def _terms(self, raw: pl.DataFrame) -> pl.DataFrame:
        return (
            raw.explode("graphs")
            .select(pl.col("graphs").struct.field("nodes").alias("nodes"))
            .explode("nodes")
            .select(
                pl.col("nodes").struct.field("id").str.replace(_OBO, "").alias("id"),
                pl.col("nodes").struct.field("lbl").alias("name"),
                pl.col("nodes").struct.field("meta").alias("meta"),
            )
            .filter(pl.col("id").str.starts_with("GO_"))
            .with_columns(
                pl.col("meta")
                .struct.field("basicPropertyValues")
                .list.eval(
                    pl.when(pl.element().struct.field("pred") == _HAS_NAMESPACE).then(
                        pl.element().struct.field("val")
                    )
                )
                .list.drop_nulls()
                .list.first()
                .alias("namespace"),
                pl.col("meta")
                .struct.field("definition")
                .struct.field("val")
                .alias("definition"),
                pl.col("meta")
                .struct.field("synonyms")
                .list.eval(pl.element().struct.field("val"))
                .alias("synonyms"),
            )
            .drop("meta")
            .filter(pl.col("namespace").is_in(list(_NAMESPACE_TO_NODE)))
            .unique("id")
        )

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build GO term nodes and within-namespace ``is_a`` edges from the source."""
        raw = read_obograph_json(self._dir / "go_plus.json")
        terms = self._terms(raw)
        nodes = terms.with_columns(
            pl.col("id").alias("curie"),
            pl.col("namespace").replace_strict(_NAMESPACE_TO_LABEL).alias("label"),
        ).drop("id")
        namespaces = terms.select(pl.col("id"), pl.col("namespace"))
        edges = (
            obo_isa_edges(raw)
            .filter(
                pl.col("parent").str.starts_with("GO_")
                & pl.col("child").str.starts_with("GO_")
            )
            .join(namespaces, left_on="child", right_on="id", how="inner")
            .with_columns(
                pl.col("namespace").replace_strict(_NAMESPACE_TO_ABBREV).alias("abbrev")
            )
            .select(
                pl.col("parent").alias("from"),
                pl.col("child").alias("to"),
                pl.lit(Relation.PARENT.value).alias("type"),
                (pl.col("abbrev") + "-" + pl.col("abbrev")).alias("label"),
                pl.lit([Source.GO.value]).alias("sources"),
            )
            .unique(subset=["from", "to"])
        )
        return nodes, edges
