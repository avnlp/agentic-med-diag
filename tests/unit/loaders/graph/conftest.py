"""Synthetic reference fixtures for structured-graph loader unit tests.

Builds a tiny on-disk reference tree (Open Targets parquet, HGNC TSV, OBO Graph
JSON) under ``tmp_path`` so ``ReferenceData``/``Harmonizer``/``NodeEnricher`` can be
exercised without the multi-gigabyte real inputs. Nested ``metadata`` structs mirror
OptimusKG's bronze layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from am_diag.loaders.graph.reference_data import ReferenceData


def _write_opentargets(root: Path) -> None:
    ot = root / "opentargets"
    target_dir = ot / "target"
    drug_dir = ot / "drug_molecule"
    disease_dir = ot / "disease"
    for directory in (target_dir, drug_dir, disease_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "id": ["ENSG1", "ENSG2"],
            "metadata": [
                {
                    "approved_symbol": "TP53",
                    "approved_name": "tumor protein p53",
                    "biotype": "protein_coding",
                    "synonyms": ["p53"],
                    "db_xrefs": ["HGNC:11998"],
                },
                {
                    "approved_symbol": "EGFR",
                    "approved_name": "epidermal growth factor receptor",
                    "biotype": "protein_coding",
                    "synonyms": ["ERBB1"],
                    "db_xrefs": ["HGNC:3236"],
                },
            ],
        }
    ).write_parquet(target_dir / "part-0.parquet")

    pl.DataFrame(
        {
            "id": ["CHEMBL25", "CHEMBL100"],
            "metadata": [
                {
                    "name": "aspirin",
                    "drug_type": "small molecule",
                    "synonyms": ["ASA"],
                    "trade_names": ["Aspirin"],
                    "cross_references": [{"source": "drugbank", "ids": ["DB00945"]}],
                },
                {
                    "name": "warfarin",
                    "drug_type": "small molecule",
                    "synonyms": ["coumadin"],
                    "trade_names": ["Coumadin"],
                    "cross_references": [{"source": "drugbank", "ids": ["DB01050"]}],
                },
            ],
        }
    ).write_parquet(drug_dir / "part-0.parquet")

    pl.DataFrame(
        {
            "id": ["MONDO_0005148", "EFO_1"],
            "name": ["type 2 diabetes mellitus", "other disease"],
            "description": ["a metabolic disorder", "another"],
            "metadata": [
                {
                    "db_xrefs": ["UMLS:C0011860", "MESH:D003924"],
                    "parents": ["MONDO_0005015"],
                },
                {"db_xrefs": ["MESH:D000001"], "parents": []},
            ],
        }
    ).write_parquet(disease_dir / "part-0.parquet")


def _write_gene_names(root: Path) -> None:
    gene_dir = root / "gene_names"
    gene_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"symbol": ["TP53", "EGFR"], "ncbi_id": ["7157", "1956"]}).write_csv(
        gene_dir / "gene_names.tsv", separator="\t"
    )


def _write_ontology(root: Path) -> None:
    ont_dir = root / "ontology"
    ont_dir.mkdir(parents=True, exist_ok=True)
    obo = "http://purl.obolibrary.org/obo/"
    graph = {
        "graphs": [
            {
                "nodes": [
                    {
                        "id": f"{obo}MONDO_0005148",
                        "lbl": "type 2 diabetes mellitus",
                        "meta": {
                            "definition": {"val": "a metabolic disorder"},
                            "synonyms": [{"val": "T2DM"}],
                            "xrefs": [
                                {"val": "UMLS:C0011860"},
                                {"val": "MESH:D003924"},
                            ],
                        },
                    }
                ],
                "edges": [
                    {
                        "sub": f"{obo}MONDO_0005148",
                        "pred": "is_a",
                        "obj": f"{obo}MONDO_0005015",
                    }
                ],
            }
        ]
    }
    (ont_dir / "mondo.json").write_text(json.dumps(graph), encoding="utf-8")

    has_ns = "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"
    go_graph = {
        "graphs": [
            {
                "nodes": [
                    {
                        "id": f"{obo}GO_0008150",
                        "lbl": "biological process",
                        "meta": {
                            "basicPropertyValues": [
                                {"pred": has_ns, "val": "biological_process"}
                            ],
                            "definition": {"val": "a biological process"},
                            "synonyms": [{"val": "biological process"}],
                            "xrefs": [{"val": "Wikipedia:Biological_process"}],
                        },
                    }
                ],
                "edges": [],
            }
        ]
    }
    (ont_dir / "go_plus.json").write_text(json.dumps(go_graph), encoding="utf-8")

    hp_graph = {
        "graphs": [
            {
                "nodes": [
                    {
                        "id": f"{obo}HP_0001",
                        "lbl": "abnormal phenotype",
                        "meta": {
                            "definition": {"val": "an abnormal phenotype"},
                            "synonyms": [{"val": "abnormality"}],
                            "xrefs": [{"val": "UMLS:C9999"}],
                        },
                    }
                ],
                "edges": [],
            }
        ]
    }
    (ont_dir / "hp.json").write_text(json.dumps(hp_graph), encoding="utf-8")


@pytest.fixture
def reference_dir(tmp_path: Path) -> Path:
    """Build and return a synthetic reference-data root."""
    _write_opentargets(tmp_path)
    _write_gene_names(tmp_path)
    _write_ontology(tmp_path)
    return tmp_path


@pytest.fixture
def reference_data(reference_dir: Path) -> ReferenceData:
    """Return a ``ReferenceData`` over the synthetic reference root."""
    return ReferenceData(reference_dir)
