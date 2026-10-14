"""Tests for the CTD environmental-exposure loader."""

from __future__ import annotations

import gzip
from pathlib import Path

from am_diag.loaders.graph.ctd import CtdGraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.reference_data import ReferenceData


_HEADER = (
    "exposurestressorid,exposurestressorname,exposuremarker,"
    "exposuremarkerid,diseaseid,phenotypeid"
)


def _write_ctd(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        _HEADER,
        "C001,benzene,,,D003924,",
        "C001,benzene,TP53,,,",
        "C001,benzene,,C002,,",
        "C001,benzene,,,,GO:0008150",
    ]
    with gzip.open(
        directory / "CTD_exposure_events.csv.gz", "wt", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(rows) + "\n")


class TestCtdLoader:
    """Each exposure outcome maps to its harmonized edge family."""

    def test_build_frames(self, reference_dir: Path) -> None:
        """Disease/gene/exposure/GO outcomes become the six exposure edge families."""
        src = reference_dir / "ctd"
        _write_ctd(src)
        reference = ReferenceData(reference_dir)
        loader = CtdGraphLoader(
            src,
            reference=reference,
            harmonizer=Harmonizer(reference),
            enricher=NodeEnricher(reference),
        )
        nodes, edges = loader.build_frames()

        by_type = {r["type"]: r for r in edges.iter_rows(named=True)}
        assert by_type["EXPOSURE_DISEASE"]["from"] == "MESH:C001"
        assert by_type["EXPOSURE_DISEASE"]["to"] == "MONDO_0005148"
        assert by_type["EXPOSURE_GENE"]["to"] == "ENSG1"
        assert by_type["EXPOSURE_EXPOSURE"]["to"] == "MESH:C002"
        assert by_type["EXPOSURE_BIOLOGICAL_PROCESS"]["to"] == "GO_0008150"
        assert by_type["EXPOSURE_BIOLOGICAL_PROCESS"]["label"] == "EXP-BPO"

        exposure = nodes.filter(nodes["curie"] == "MESH:C001").row(0, named=True)
        assert exposure["label"] == "Exposure"
        assert exposure["name"] == "benzene"
        go_node = nodes.filter(nodes["curie"] == "GO_0008150").row(0, named=True)
        assert go_node["label"] == "GoBiologicalProcess"

    def test_schema_covers_emitted_types(self, reference_dir: Path) -> None:
        """Emitted labels and relation types fall within the authored schema."""
        src = reference_dir / "ctd"
        _write_ctd(src)
        reference = ReferenceData(reference_dir)
        loader = CtdGraphLoader(
            src,
            reference=reference,
            harmonizer=Harmonizer(reference),
            enricher=NodeEnricher(reference),
        )
        nodes, edges = loader.build_frames()
        assert set(nodes["label"].to_list()) <= loader.graph_schema.entity_type_names()
        assert set(edges["type"].to_list()) <= loader.graph_schema.relation_type_names()
