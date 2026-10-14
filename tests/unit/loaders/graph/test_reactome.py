"""Tests for the Reactome pathway loader: human filter + PWY-PWY hierarchy."""

from __future__ import annotations

from pathlib import Path

from am_diag.loaders.graph.reactome import ReactomeGraphLoader


def _write_reactome(directory: Path) -> None:
    (directory / "ReactomePathways.txt").write_text(
        "R-HSA-1\tGlycolysis\tHomo sapiens\n"
        "R-HSA-2\tMetabolism\tHomo sapiens\n"
        "R-MMU-1\tMouse pathway\tMus musculus\n",
        encoding="utf-8",
    )
    (directory / "ReactomePathwaysRelation.txt").write_text(
        "R-HSA-2\tR-HSA-1\nR-MMU-2\tR-MMU-1\n",
        encoding="utf-8",
    )


class TestReactomeLoader:
    """Only human pathways and their hierarchy edges survive."""

    def test_build_frames_human_only(self, tmp_path: Path) -> None:
        """Non-human pathways and relations are dropped; PARENT edges are emitted."""
        _write_reactome(tmp_path)
        nodes, edges = ReactomeGraphLoader(tmp_path).build_frames()

        assert set(nodes["curie"].to_list()) == {"R-HSA-1", "R-HSA-2"}
        assert nodes["label"].unique().to_list() == ["Pathway"]

        assert edges.height == 1
        edge = edges.row(0, named=True)
        assert edge["from"] == "R-HSA-2"
        assert edge["to"] == "R-HSA-1"
        assert edge["type"] == "PARENT"
        assert edge["label"] == "PWY-PWY"
