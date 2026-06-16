"""Root base class for all am_diag graph-storable types.

`DataPoint` provides auto-generated UUID (`uuid4`) or deterministic UUID5
from `Dedup`-annotated fields, creation/update timestamps, version counter,
and serialization helpers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_OID, UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field

from am_diag.common.data_models.annotations import _Dedup, _Embeddable
from am_diag.common.data_models.metadata import MetaData


logger = logging.getLogger(__name__)


class DataPoint(BaseModel):
    """Root base class for all am_diag graph-storable types.

    Provides:
    - Auto-generated UUID (`uuid4`) or deterministic UUID5 from
      `Dedup`-annotated fields.
    - Creation/update timestamps (epoch ms).
    - Version counter for optimistic concurrency.
    - `metadata` descriptor for embedding and dedup configuration.
    - Serialization: `to_json()`, `from_json()`, `to_dict()`,
      `from_dict()`.
    - Annotation-driven auto-derivation of `index_fields` and
      `identity_fields`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Identity & Lifecycle ────────────────────────────────────────────
    id: UUID = Field(default_factory=uuid4)
    created_at: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000),
    )
    updated_at: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000),
    )
    version: int = 1

    # ── Kind & Metadata ─────────────────────────────────────────────────
    metadata: MetaData = {"index_fields": [], "identity_fields": []}
    kind: str = Field(default_factory=lambda: DataPoint.__name__)

    def __init__(self, **data: Any) -> None:
        """Initialize a DataPoint, auto-generating UUID if not provided."""
        explicit_id = "id" in data
        super().__init__(**data)
        object.__setattr__(self, "kind", self.__class__.__name__)
        if not explicit_id:
            identity_fields = self.__class__._get_identity_fields()
            if identity_fields:
                identity_id = self.__class__._generate_identity_id(
                    identity_fields,
                    self.model_dump(),
                    self.__class__.__name__,
                )
                if identity_id is not None:
                    object.__setattr__(self, "id", identity_id)

    def model_post_init(self, __context: Any) -> None:
        """Ensure `metadata` uses the class-level default from annotations.

        Pydantic v2 deep-copies mutable defaults (like dicts) during class
        compilation, so modifications to `cls.model_fields["metadata"].default`
        in `__pydantic_init_subclass__` do not propagate to instance creation.
        This hook re-applies the class-level default when the instance has the
        parent's empty default, while preserving explicitly provided metadata.
        """
        cls_default = self.__class__.model_fields.get("metadata")
        if cls_default is not None and cls_default.default is not None:
            current = self.metadata
            # Pydantic v2 deep-copies the parent's default. If the current
            # metadata matches the parent DataPoint's empty default, replace it
            # with the auto-derived class default. Explicitly provided metadata
            # (which would have non-empty lists) is preserved.
            _parent_defaults = (
                {"index_fields": []},
                {"index_fields": [], "identity_fields": []},
            )
            if (
                current in _parent_defaults
                and cls_default.default not in _parent_defaults
            ):
                super().__setattr__("metadata", dict(cls_default.default))

    # ── Identity Field Resolution ───────────────────────────────────────

    @classmethod
    def _get_identity_fields(cls) -> list[str] | None:
        """Get `identity_fields` from the class's `metadata` field default.

        Walks the MRO to detect if a parent class defined `identity_fields`
        that a subclass accidentally dropped when overriding `metadata`.
        """
        metadata_field = cls.model_fields.get("metadata")
        if metadata_field is not None and metadata_field.default is not None:
            identity = metadata_field.default.get("identity_fields")
            if identity is None:
                for parent in cls.__mro__[1:]:
                    parent_meta = getattr(parent, "model_fields", {}).get("metadata")
                    if parent_meta is not None and parent_meta.default is not None:
                        parent_identity = parent_meta.default.get("identity_fields")
                        if parent_identity is not None:
                            logger.warning(
                                "%s overrides metadata but drops identity_fields "
                                "defined in parent %s",
                                cls.__name__,
                                parent.__name__,
                            )
                            break
            return identity
        return []

    @classmethod
    def _generate_identity_id(
        cls,
        identity_fields: list[str],
        data: dict,
        class_name: str,
    ) -> UUID | None:
        """Generate a deterministic UUID5 from identity field values.

        Returns `None` if any identity field is missing from both *data*
        and Pydantic field defaults, which causes fallback to `uuid4()`.
        """
        parts: list[str] = []
        for field_name in identity_fields:
            if field_name in data:
                value = data[field_name]
            else:
                field_info = cls.model_fields.get(field_name)
                if field_info is not None and field_info.default is not None:
                    value = field_info.default
                else:
                    return None
            # A None Dedup field means identity is not fully specified;
            # fall back to uuid4 so each instance is distinct.
            if value is None:
                return None
            if isinstance(value, str):
                value = value.lower().replace(" ", "_").replace("'", "")
            else:
                value = str(value)
            parts.append(value)
        joined = "|".join(parts)
        identity_string = f"{class_name}:{joined}"
        return uuid5(NAMESPACE_OID, identity_string)

    # ── Annotation Auto-Derivation ──────────────────────────────────────

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-derive `metadata.index_fields` and `metadata.identity_fields`.

        If a subclass uses `Annotated[str, Embeddable()]` or
        `Annotated[str, Dedup()]` on its fields, and does **not** explicitly
        set `metadata`, the metadata default is automatically populated from
        those annotations.
        """
        super().__pydantic_init_subclass__(**kwargs)

        # Only auto-derive if the subclass didn't explicitly declare metadata.
        if "metadata" in cls.__annotations__:
            return

        embeddable_fields: list[str] = []
        dedup_fields: list[str] = []
        for field_name, field_info in cls.model_fields.items():
            if field_info.metadata:
                for meta in field_info.metadata:
                    if isinstance(meta, _Embeddable):
                        embeddable_fields.append(field_name)
                    if isinstance(meta, _Dedup):
                        dedup_fields.append(field_name)

        if embeddable_fields or dedup_fields:
            new_metadata: dict[str, Any] = {"index_fields": embeddable_fields}
            if dedup_fields:
                new_metadata["identity_fields"] = dedup_fields
            cls.model_fields["metadata"].default = new_metadata

    # ── Embeddable Data Introspection ───────────────────────────────────

    @classmethod
    def get_embeddable_data(cls, data_point: DataPoint) -> Any | None:
        """Retrieve the value of the first embeddable field from a data point.

        Returns `None` if no index fields are defined or the field is missing.
        """
        if (
            data_point.metadata
            and len(data_point.metadata["index_fields"]) > 0
            and hasattr(data_point, data_point.metadata["index_fields"][0])
        ):
            attribute = getattr(data_point, data_point.metadata["index_fields"][0])
            if isinstance(attribute, str):
                return attribute.strip()
            return attribute
        return None

    @classmethod
    def get_embeddable_properties(cls, data_point: DataPoint) -> list[Any | None]:
        """Return values of all embeddable fields for the given data point."""
        if data_point.metadata and len(data_point.metadata["index_fields"]) > 0:
            return [
                getattr(data_point, field, None)
                for field in data_point.metadata["index_fields"]
            ]
        return []

    @classmethod
    def get_embeddable_property_names(cls, data_point: DataPoint) -> list[str]:
        """Return the field names marked as embeddable."""
        return data_point.metadata["index_fields"] or []

    # ── Versioning ───────────────────────────────────────────────────────

    def update_version(self) -> None:
        """Increment version and refresh `updated_at` timestamp."""
        self.version += 1
        self.updated_at = int(datetime.now(timezone.utc).timestamp() * 1000)

    # ── Serialization ────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> DataPoint:
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)  # type: ignore[return-value]

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Convert to a dictionary."""
        return self.model_dump(**kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataPoint:
        """Instantiate from a dictionary."""
        return cls.model_validate(data)  # type: ignore[return-value]
