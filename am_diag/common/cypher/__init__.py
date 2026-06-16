"""Externalized Cypher queries for the am_diag Neo4j graph.

All Cypher lives in .cypher files under this package; Python code never
constructs multi-line query strings. The loader module provides cached read
and safe identifier substitution.
"""

from am_diag.common.cypher.loader import load, render, safe_identifier


__all__ = ["load", "render", "safe_identifier"]
