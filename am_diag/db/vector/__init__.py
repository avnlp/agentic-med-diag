"""Vector store components for ingestion and query pipelines.

Concrete provider implementations are imported directly from their submodules:

* ``from am_diag.db.vector.qdrant import QdrantVectorStore``
* ``from am_diag.db.vector.weaviate import WeaviateVectorStore``

Data models are provided by ``am_diag.common.data_models``:

* ``from am_diag.common.data_models.vector_record import VectorRecord, VectorHit``
* ``from am_diag.common.data_models.enums import Distance``
"""

from am_diag.db.vector.base import VectorStoreBase


__all__ = [
    "VectorStoreBase",
]
