"""Structured-graph loaders.

Loaders that ingest a pre-built biomedical knowledge graph (OptimusKG and its
per-source subgraphs) directly as `Entity`/`Relation`, bypassing text extraction.
"""

from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.bgee import BgeeGraphLoader
from am_diag.loaders.graph.ctd import CtdGraphLoader
from am_diag.loaders.graph.disgenet import DisGeNETGraphLoader
from am_diag.loaders.graph.drugbank import DrugBankGraphLoader
from am_diag.loaders.graph.drugcentral import DrugCentralGraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.gene_ontology import GeneOntologyGraphLoader
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.identity import curie_to_entity_id, relation_id
from am_diag.loaders.graph.onsides import OnsidesGraphLoader
from am_diag.loaders.graph.ontology_backbones import (
    HpoGraphLoader,
    MondoGraphLoader,
    UberonGraphLoader,
)
from am_diag.loaders.graph.opentargets import OpenTargetsGraphLoader
from am_diag.loaders.graph.optimuskg import OptimusKGGraphLoader
from am_diag.loaders.graph.ppi import PpiGraphLoader
from am_diag.loaders.graph.reactome import ReactomeGraphLoader
from am_diag.loaders.graph.reference_data import ReferenceData
from am_diag.loaders.graph.settings import GraphSourcesSettings


__all__ = [
    "GraphLoader",
    "GraphSourcesSettings",
    "Harmonizer",
    "NodeEnricher",
    "OptimusKGGraphLoader",
    "ReferenceData",
    "UberonGraphLoader",
    "MondoGraphLoader",
    "HpoGraphLoader",
    "GeneOntologyGraphLoader",
    "ReactomeGraphLoader",
    "BgeeGraphLoader",
    "DrugBankGraphLoader",
    "DisGeNETGraphLoader",
    "PpiGraphLoader",
    "OnsidesGraphLoader",
    "DrugCentralGraphLoader",
    "CtdGraphLoader",
    "OpenTargetsGraphLoader",
    "curie_to_entity_id",
    "relation_id",
]
