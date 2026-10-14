"""Abstract base for structured-graph loaders.

A ``GraphLoader`` ingests a pre-built biomedical graph (OptimusKG and its
per-source subgraphs) directly as typed ``Entity``/``Relation`` — the shape the
text-extraction front-half *outputs* — so it re-enters the pipeline at the
finalization tail rather than going through chunking/extraction.

This base is the graph-world analog of ``MCQDataset``: a subclass does all of one
source's processing in-class and produces the seam (two polars frames); the base
owns everything generic — ``curie``→id mapping, ``Entity``/``Relation``
construction with deterministic ids, streaming, the whole-graph accessor, and
Parquet export. Subclasses never construct ``Entity``/``Relation`` themselves.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import polars as pl

from am_diag.common.data_models import Entity, KnowledgeGraph, Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.loaders.graph.identity import curie_to_entity_id, relation_id


# Reserved columns map to core Entity/Relation fields; every other column is a
# queryable property. ``description`` and ``score`` are reserved because they are
# core fields: a source property of that name (e.g. a drug's ``description`` or a
# DisGeNET edge ``score``) must populate the core field, not collide with it.
_NODE_RESERVED = frozenset(
    {
        "curie",
        "name",
        "label",
        "description",
        "score",
        "canonical_name",
        "aliases",
        "id",
        "schema_properties",
        "provenance",
    }
)
_EDGE_RESERVED = frozenset(
    {
        "from",
        "to",
        "type",
        "label",
        "description",
        "score",
        "sources",
        "id",
        "head_id",
        "tail_id",
        "head_name",
        "tail_name",
        "structural_label",
        "schema_properties",
        "provenance",
    }
)

# Streaming work prepared off the event loop: the sliced edge frame, a
# CURIE->node-row index, and a CURIE->name map for endpoint hydration.
_PreparedStream = tuple[pl.DataFrame, dict[str, dict[str, Any]], dict[str, str]]


class GraphLoader(ABC):
    """Stream a structured graph into am_diag as ``Entity``/``Relation``.

    Subclasses set ``dataset_name`` and ``graph_schema`` and implement
    ``build_frames`` (raw source -> two am_diag-native polars frames). Node
    reserved columns are ``curie``, ``name``, ``label`` (plus optional
    ``canonical_name``, ``aliases``); edge reserved columns are ``from``, ``to``,
    ``type``, ``label`` (structural), and ``sources``. All other columns become
    queryable properties on the resulting nodes/edges.

    Attributes:
        dataset_name: Short identifier stamped onto every node/edge property bag.
        graph_schema: Entity/relation type definitions for this source.
    """

    dataset_name: ClassVar[str]
    graph_schema: ClassVar[GraphSchema]

    _frames_cache: tuple[pl.DataFrame, pl.DataFrame] | None = None

    @abstractmethod
    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return ``(nodes_df, edges_df)`` in am_diag-native columns.

        Returns:
            A node frame and an edge frame. Reserved columns are interpreted
            structurally; all other columns are carried as queryable properties.
        """

    def _frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build the frames once and cache them on the instance."""
        cached = self._frames_cache
        if cached is None:
            cached = self.build_frames()
            self._frames_cache = cached
        return cached

    def _node_row_to_entity(self, row: dict[str, Any]) -> Entity:
        """Build an ``Entity`` from one node-frame row (deterministic id)."""
        curie = row["curie"]
        props = {
            k: v for k, v in row.items() if k not in _NODE_RESERVED and v is not None
        }
        props["dataset"] = self.dataset_name
        score = row.get("score")
        return Entity(
            id=curie_to_entity_id(curie),
            curie=curie,
            name=row.get("name") or curie,
            label=row["label"],
            description=row.get("description"),
            score=1.0 if score is None else score,
            canonical_name=row.get("canonical_name"),
            aliases=row.get("aliases") or [],
            schema_properties=props,
        )

    @staticmethod
    def _synth_description(head_name: str, rel_type: str, tail_name: str) -> str:
        """Build an embeddable fact string from endpoint names and relation type.

        Structured edges carry no prose, so relation vectors would otherwise be
        empty. This yields a readable fact, e.g. "liver expression present TP53".
        """
        readable = rel_type.replace("_", " ").lower()
        return f"{head_name} {readable} {tail_name}".strip()

    def _edge_row_to_relation(
        self, row: dict[str, Any], names: dict[str, str]
    ) -> Relation:
        """Build a ``Relation`` from one edge-frame row (pre-linked endpoints)."""
        head, tail, rel_type = row["from"], row["to"], row["type"]
        props = {
            k: v for k, v in row.items() if k not in _EDGE_RESERVED and v is not None
        }
        props["dataset"] = self.dataset_name
        # A list ``sources`` (structured builders) maps straight to the field; a
        # scalar ``sources`` (e.g. OptimusKG's gold string) is wrapped so it lands
        # in the first-class field rather than colliding in schema_properties.
        raw_sources = row.get("sources")
        if isinstance(raw_sources, list):
            rel_sources = raw_sources
        elif raw_sources is not None:
            rel_sources = [str(raw_sources)]
        else:
            rel_sources = []
        score = row.get("score")
        head_name = names.get(head, head)
        tail_name = names.get(tail, tail)
        description = row.get("description") or self._synth_description(
            head_name, rel_type, tail_name
        )
        return Relation(
            id=relation_id(head, rel_type, tail),
            head_id=curie_to_entity_id(head),
            tail_id=curie_to_entity_id(tail),
            head_name=head_name,
            tail_name=tail_name,
            type=rel_type,
            description=description,
            score=1.0 if score is None else score,
            structural_label=row.get("label"),
            sources=rel_sources,
            schema_properties=props,
        )

    async def astream(
        self,
        batch_size: int = 5000,
        offset: int = 0,
        limit: int | None = None,
    ) -> AsyncIterator[KnowledgeGraph]:
        """Stream the graph as ``KnowledgeGraph`` batches of edges + endpoints.

        Each batch carries a slice of relations plus the entities at their
        endpoints. The whole graph is never materialized at once; the polars work
        runs in ``asyncio.to_thread`` so the event loop stays free. Endpoint
        entities recur across batches and are deduplicated downstream by the
        deterministic-id ``MERGE`` in the graph store.

        Args:
            batch_size: Maximum relations per yielded batch.
            offset: Number of leading edges to skip.
            limit: Maximum number of edges to stream (``None`` for all).

        Yields:
            ``KnowledgeGraph`` batches with ``entities`` and ``relations`` set.
            Edges whose endpoints are missing from the node frame are skipped, and
            nodes with no edges are emitted in trailing entity-only batches.

        Raises:
            ValueError: If ``batch_size`` is not positive or ``offset``/``limit``
                is negative.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")

        def _prepare() -> _PreparedStream:
            nodes, edges = self._frames()
            node_by_curie = {r["curie"]: r for r in nodes.iter_rows(named=True)}
            known = list(node_by_curie)
            edges = edges.filter(
                pl.col("from").is_in(known) & pl.col("to").is_in(known)
            )
            if offset:
                edges = edges.slice(offset)
            if limit is not None:
                edges = edges.head(limit)
            names = {c: (r.get("name") or c) for c, r in node_by_curie.items()}
            return edges, node_by_curie, names

        edges, node_by_curie, names = await asyncio.to_thread(_prepare)

        for start in range(0, edges.height, batch_size):

            def _batch(start: int = start) -> KnowledgeGraph:
                rows = list(edges.slice(start, batch_size).iter_rows(named=True))
                relations = [self._edge_row_to_relation(r, names) for r in rows]
                touched = {r["from"] for r in rows} | {r["to"] for r in rows}
                entities = [
                    self._node_row_to_entity(node_by_curie[c])
                    for c in touched
                    if c in node_by_curie
                ]
                return KnowledgeGraph(
                    entities=entities,
                    relations=relations,
                    graph_schema=self.graph_schema,
                )

            yield await asyncio.to_thread(_batch)

        if offset or limit is not None:
            return
        linked = set(edges["from"]) | set(edges["to"])
        isolated = [c for c in node_by_curie if c not in linked]
        for start in range(0, len(isolated), batch_size):

            def _isolated(start: int = start) -> KnowledgeGraph:
                chunk = isolated[start : start + batch_size]
                return KnowledgeGraph(
                    entities=[
                        self._node_row_to_entity(node_by_curie[c]) for c in chunk
                    ],
                    relations=[],
                    graph_schema=self.graph_schema,
                )

            yield await asyncio.to_thread(_isolated)

    async def load_graph(self) -> KnowledgeGraph:
        """Return the whole graph as one ``KnowledgeGraph``.

        Convenient for small subgraphs, tests, and notebooks. Prefer ``astream``
        for large sources.

        Returns:
            A ``KnowledgeGraph`` with all entities and relations populated.
        """

        def _all() -> KnowledgeGraph:
            nodes, edges = self._frames()
            names = {
                r["curie"]: (r.get("name") or r["curie"])
                for r in nodes.iter_rows(named=True)
            }
            entities = [
                self._node_row_to_entity(r) for r in nodes.iter_rows(named=True)
            ]
            relations = [
                self._edge_row_to_relation(r, names)
                for r in edges.iter_rows(named=True)
                if r["from"] in names and r["to"] in names
            ]
            return KnowledgeGraph(
                entities=entities,
                relations=relations,
                graph_schema=self.graph_schema,
            )

        return await asyncio.to_thread(_all)

    def export_to_parquet(self, out_dir: str | Path) -> None:
        """Write the am_diag-native frames to ``nodes.parquet`` / ``edges.parquet``.

        Args:
            out_dir: Destination directory; created if it does not exist.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        nodes, edges = self._frames()
        nodes.write_parquet(out / "nodes.parquet")
        edges.write_parquet(out / "edges.parquet")

    def __repr__(self) -> str:
        """Return a concise developer representation."""
        return f"{type(self).__name__}(dataset={self.dataset_name!r})"
