"""Resolution clustering data structures."""

from pydantic import BaseModel, Field


class ItemCluster(BaseModel):
    """One semantic cluster of entity-name or edge-type string indices to resolve."""

    items: list[int]


class ResolvedAliasMap(BaseModel):
    """Canonical alias mapped to the original strings it subsumes, for one item kind."""

    alias_to_members: dict[str, list[str]] = Field(default_factory=dict)
