"""Neo4j connection settings loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Neo4jSettings(BaseSettings):
    """Neo4j connection settings.

    Reads from environment variables with the `NEO4J_` prefix.
    All fields can also be supplied directly as constructor arguments.

    Attributes:
        uri: Bolt URI of the Neo4j instance, e.g. `bolt://localhost:7687`.
        user: Neo4j username.
        password: Neo4j password.
        database: Target database name. Defaults to `"neo4j"`.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        env_file=".env",
        extra="ignore",
    )

    uri: str = ""
    user: str = ""
    password: str = ""
    database: str = "neo4j"
