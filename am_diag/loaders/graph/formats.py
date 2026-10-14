"""Raw-format readers for structured-graph loaders (polars + stdlib + lxml).

Centralizes every non-trivial input format so per-source loaders stay pure polars
transforms. Ontologies arrive as OBO Graph JSON (polars ``read_json``), DrugBank as
XML (``lxml`` streaming), DrugCentral as a gzipped Postgres dump (parsed serverlessly
from its ``COPY`` data blocks), and zipped CSVs via stdlib ``zipfile``.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import polars as pl
from lxml import etree  # ty: ignore[unresolved-import]


_OBO_PREFIX = "http://purl.obolibrary.org/obo/"
_HAS_OBO_NAMESPACE = "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"


def read_obograph_json(path: str | Path) -> pl.DataFrame:
    """Read an OBO Graph JSON ontology (e.g. ``mondo.json``) into a polars frame.

    Returns the raw frame carrying the top-level ``graphs`` struct column; use
    ``obo_terms`` / ``obo_isa_edges`` to flatten it.
    """
    return pl.read_json(Path(path))


def obo_terms(raw: pl.DataFrame) -> pl.DataFrame:
    """Flatten OBO Graph JSON nodes into ``(id, name, definition, synonyms, xrefs)``.

    The OBO prefix is stripped from ids so they become Biolink CURIEs (``MONDO_1``).
    """
    return (
        raw.explode("graphs")
        .select(pl.col("graphs").struct.field("nodes").alias("nodes"))
        .explode("nodes")
        .select(
            pl.col("nodes").struct.field("id").str.replace(_OBO_PREFIX, "").alias("id"),
            pl.col("nodes").struct.field("lbl").alias("name"),
            pl.col("nodes").struct.field("meta").alias("meta"),
        )
        .with_columns(
            pl.col("meta")
            .struct.field("definition")
            .struct.field("val")
            .alias("definition"),
            pl.col("meta")
            .struct.field("synonyms")
            .list.eval(pl.element().struct.field("val"))
            .alias("synonyms"),
            pl.col("meta")
            .struct.field("xrefs")
            .list.eval(pl.element().struct.field("val"))
            .alias("xrefs"),
        )
        .drop("meta")
        .filter(pl.col("id").is_not_null())
        .unique("id")
    )


def obo_isa_edges(raw: pl.DataFrame) -> pl.DataFrame:
    """Flatten OBO Graph JSON ``is_a`` edges into ``(parent, child)`` pairs.

    Parent is the edge ``obj`` and child is the ``sub`` (OBO convention); ids are
    stripped to Biolink CURIE form. Non-``is_a`` predicates are dropped.
    """
    return (
        raw.explode("graphs")
        .select(pl.col("graphs").struct.field("edges").alias("edges"))
        .explode("edges")
        .select(
            pl.col("edges")
            .struct.field("obj")
            .str.replace(_OBO_PREFIX, "")
            .alias("parent"),
            pl.col("edges")
            .struct.field("sub")
            .str.replace(_OBO_PREFIX, "")
            .alias("child"),
            pl.col("edges").struct.field("pred").alias("pred"),
        )
        .filter(pl.col("pred").str.contains("is_a"))
        .drop("pred")
        .filter(pl.col("parent").is_not_null() & pl.col("child").is_not_null())
    )


def go_term_namespaces(raw: pl.DataFrame) -> pl.DataFrame:
    """Flatten GO OBO Graph JSON nodes into ``(id, namespace)`` pairs.

    The namespace (``biological_process`` / ``cellular_component`` /
    ``molecular_function``) comes from each node's ``hasOBONamespace`` basic property
    value; terms without one are dropped.
    """
    return (
        raw.explode("graphs")
        .select(pl.col("graphs").struct.field("nodes").alias("nodes"))
        .explode("nodes")
        .select(
            pl.col("nodes").struct.field("id").str.replace(_OBO_PREFIX, "").alias("id"),
            pl.col("nodes").struct.field("meta").alias("meta"),
        )
        .filter(pl.col("id").str.starts_with("GO_"))
        .with_columns(
            pl.col("meta")
            .struct.field("basicPropertyValues")
            .list.eval(
                pl.when(pl.element().struct.field("pred") == _HAS_OBO_NAMESPACE).then(
                    pl.element().struct.field("val")
                )
            )
            .list.drop_nulls()
            .list.first()
            .alias("namespace")
        )
        .drop("meta")
        .drop_nulls("namespace")
        .unique("id")
    )


def read_drugbank_interactions_xml(path: str | Path) -> pl.DataFrame:
    """Stream DrugBank ``full database.xml`` into drug-drug interaction rows.

    Mirrors OptimusKG's ``bronze/nodes/drugbank/drug_drug.py``: each ``drug``
    contributes its ``drug-interaction`` children as sorted
    ``(tail_drug_id, head_drug_id, description)`` rows with ``DRUGBANK:`` prefixes.
    Uses ``iterparse`` and clears each ``drug`` subtree so the large file streams.
    """
    rows: list[tuple[str, str, str | None]] = []
    namespace: str | None = None
    context = etree.iterparse(str(path), events=("end",))
    for _, elem in context:
        if namespace is None and elem.nsmap:
            namespace = elem.nsmap.get(None)
        if etree.QName(elem).localname != "drug":
            continue
        prefix = f"{{{namespace}}}" if namespace else ""
        head_el = elem.find(f"{prefix}drugbank-id")
        if head_el is None or head_el.text is None:
            elem.clear()
            continue
        head = f"DRUGBANK:{head_el.text}"
        for interaction in elem.findall(f".//{prefix}drug-interaction"):
            tail_el = interaction.find(f"{prefix}drugbank-id")
            desc_el = interaction.find(f"{prefix}description")
            if tail_el is None or tail_el.text is None:
                continue
            tail = f"DRUGBANK:{tail_el.text}"
            first, second = sorted((head, tail))
            rows.append((first, second, desc_el.text if desc_el is not None else None))
        elem.clear()
    return pl.DataFrame(
        rows,
        schema=["tail_drug_id", "head_drug_id", "description"],
        orient="row",
    ).unique(subset=["tail_drug_id", "head_drug_id"])


def read_zip_csv(path: str | Path, member: str, **csv_args) -> pl.DataFrame:
    """Read one CSV member out of a ``.zip`` archive.

    Args:
        path: Path to the zip archive.
        member: Name of the CSV file inside the archive.
        **csv_args: Forwarded to ``pl.read_csv``.
    """
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        return pl.read_csv(io.BytesIO(handle.read()), **csv_args)


def read_pg_dump_table(
    path: str | Path, table: str, columns: list[str] | None = None
) -> pl.DataFrame:
    r"""Extract one table from a gzipped pg_dump by scanning its ``COPY`` block.

    Locates ``COPY <table> (col, ...) FROM stdin;`` and reads the tab-delimited rows
    until the ``\\.`` terminator, mapping ``\\N`` to null. Serverless — avoids
    Postgres entirely (used for the DrugCentral dump).

    Args:
        path: Path to the gzipped SQL dump.
        table: Table name as it appears after ``COPY`` (e.g. ``public.structures``).
        columns: Column names for the rows. When ``None``, they are parsed from the
            ``COPY`` header's parenthesized column list.

    Returns:
        A frame of the table's rows with all values as strings (``\\N`` -> null).
    """
    rows: list[list[str | None]] = []
    schema = columns
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        in_block = False
        for line in handle:
            if not in_block:
                if line.startswith("COPY ") and table in line and "FROM stdin" in line:
                    in_block = True
                    if schema is None and "(" in line:
                        inner = line[line.index("(") + 1 : line.index(")")]
                        schema = [c.strip().strip('"') for c in inner.split(",")]
                continue
            if line.startswith("\\."):
                break
            fields = line.rstrip("\n").split("\t")
            rows.append([None if field == "\\N" else field for field in fields])
    return pl.DataFrame(rows, schema=schema, orient="row")
