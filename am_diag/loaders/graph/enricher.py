"""Per-entity-type node-metadata enrichment to OptimusKG parity.

Each method takes a node frame whose ``curie`` column holds canonical ids and
left-joins that entity type's metadata columns from the shared ``ReferenceData``.
The joined columns are non-reserved, so the base loader carries them into each
node's queryable ``schema_properties``.
"""

from __future__ import annotations

import polars as pl

from am_diag.loaders.graph.formats import obo_terms
from am_diag.loaders.graph.reference_data import ReferenceData


class NodeEnricher:
    """Enrich node frames with OptimusKG-parity metadata from the reference layer."""

    def __init__(self, reference: ReferenceData) -> None:
        """Store the shared reference layer used by every enrich method."""
        self._ref = reference

    def enrich_genes(self, nodes: pl.DataFrame) -> pl.DataFrame:
        """Join OT ``target`` metadata (symbol, biotype, synonyms) by Ensembl id."""
        meta = self._ref.opentargets_target.select(
            pl.col("id"),
            pl.col("approved_symbol").alias("symbol"),
            pl.col("approved_name").alias("gene_name"),
            pl.col("biotype"),
            pl.col("synonyms"),
            pl.col("db_xrefs").alias("xrefs"),
        )
        return nodes.join(meta, left_on="curie", right_on="id", how="left")

    def enrich_drugs(self, nodes: pl.DataFrame) -> pl.DataFrame:
        """Join OT ``drug_molecule`` identity/chemistry metadata by ChEMBL id."""
        meta = self._ref.opentargets_drug_molecule.select(
            pl.col("id"),
            pl.col("name").alias("drug_name"),
            pl.col("drug_type"),
            pl.col("synonyms"),
            pl.col("trade_names"),
        )
        return nodes.join(meta, left_on="curie", right_on="id", how="left")

    def enrich_diseases(self, nodes: pl.DataFrame) -> pl.DataFrame:
        """Join OT ``disease`` metadata (name, description) by canonical id."""
        meta = self._ref.opentargets_disease.select(
            pl.col("id"),
            pl.col("name").alias("disease_name"),
            pl.col("description"),
        )
        return nodes.join(meta, left_on="curie", right_on="id", how="left")

    def enrich_from_ontology(self, nodes: pl.DataFrame, ontology: str) -> pl.DataFrame:
        """Join OBO term metadata (name, definition, synonyms, xrefs) by Biolink id.

        Used for anatomy (``uberon``), phenotype (``hp``), GO terms (``go``), and the
        disease backbone (``mondo``).
        """
        terms = obo_terms(self._ref.ontology(ontology)).select(
            pl.col("id"),
            pl.col("name").alias("term_name"),
            pl.col("definition"),
            pl.col("synonyms"),
            pl.col("xrefs"),
        )
        return nodes.join(terms, left_on="curie", right_on="id", how="left")
