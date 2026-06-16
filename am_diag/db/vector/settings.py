"""Vector store connection settings loaded from environment variables."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    """Qdrant connection settings.

    Reads from environment variables with the `QDRANT_` prefix.
    All fields can also be supplied directly as constructor arguments.

    Attributes:
        url: HTTP(S) URL of the Qdrant instance, e.g. `http://localhost:6333`
            or a Qdrant Cloud cluster URL.
        api_key: API key for authenticated Qdrant Cloud endpoints. Leave empty
            for local unauthenticated instances.
        quantization: Optional native Qdrant vector quantization mode to apply
            at collection creation.
    """

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=".env",
        extra="ignore",
    )

    url: str = "http://localhost:6333"
    api_key: str = ""
    quantization: Literal["none", "scalar_int8", "binary"] = "none"


class WeaviateSettings(BaseSettings):
    """Weaviate connection settings.

    Reads from environment variables with the `WEAVIATE_` prefix.
    All fields can also be supplied directly as constructor arguments.

    The `url` field controls which connection helper is used:

    * `localhost` / `127.0.0.1` → local instance
    * `*.weaviate.network` / `*.weaviate.cloud` → Weaviate Cloud
    * anything else → custom deployment (requires `api_key` when auth is
      enabled)

    Attributes:
        url: Base URL of the Weaviate instance.
        api_key: API key for authenticated endpoints. Leave empty for local
            unauthenticated instances.
        grpc_port: gRPC port used for batch operations (10–20× faster than
            REST). Set to `0` to disable gRPC and fall back to HTTP-only.
        batch_size: Maximum objects per `insert_many` call inside
            `WeaviateVectorStore.upsert`.
        quantization: Optional native Weaviate vector-index quantizer to apply
            at collection creation.
    """

    model_config = SettingsConfigDict(
        env_prefix="WEAVIATE_",
        env_file=".env",
        extra="ignore",
    )

    url: str = "http://localhost:8080"
    api_key: str = ""
    grpc_port: int = 50051
    batch_size: int = 200
    quantization: Literal["none", "sq", "bq", "rq", "pq"] = "none"
