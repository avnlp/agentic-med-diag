"""Tests for the ``Chunk`` DataPoint."""

from __future__ import annotations

from uuid import uuid4

from am_diag.common.data_models import Chunk


class TestChunkInstantiation:
    """Basic construction and field defaults."""

    def test_minimal_chunk(self):
        doc_id = uuid4()
        chunk = Chunk(
            text="Metformin is a first-line treatment.",
            document_id=doc_id,
            document_source="textbooks",
            chunk_index=0,
            chunk_size=35,
            cut_type="recursive",
        )
        assert chunk.text == "Metformin is a first-line treatment."
        assert chunk.document_id == doc_id
        assert chunk.document_source == "textbooks"
        assert chunk.chunk_index == 0
        assert chunk.chunk_size == 35
        assert chunk.cut_type == "recursive"
        assert chunk.properties == {}
        assert chunk.title is None

    def test_full_chunk(self):
        doc_id = uuid4()
        chunk = Chunk(
            text="Aspirin inhibits COX enzymes.",
            document_id=doc_id,
            document_source="pubmed",
            chunk_index=3,
            chunk_size=30,
            cut_type="sentence",
            properties={"strategy": "overlap", "overlap_chars": 200},
            title="Pharmacology Chapter 5",
        )
        assert chunk.properties == {"strategy": "overlap", "overlap_chars": 200}
        assert chunk.title == "Pharmacology Chapter 5"


class TestChunkUUID5:
    """Deterministic UUID5 from (document_id, chunk_index)."""

    def test_same_doc_same_index_same_uuid(self):
        doc_id = uuid4()
        c1 = Chunk(
            text="Text A",
            document_id=doc_id,
            document_source="textbooks",
            chunk_index=0,
            chunk_size=6,
            cut_type="recursive",
        )
        c2 = Chunk(
            text="Text A",
            document_id=doc_id,
            document_source="textbooks",
            chunk_index=0,
            chunk_size=6,
            cut_type="recursive",
        )
        assert str(c1.id) == str(c2.id)

    def test_different_index_different_uuid(self):
        doc_id = uuid4()
        c1 = Chunk(
            text="Text A",
            document_id=doc_id,
            document_source="textbooks",
            chunk_index=0,
            chunk_size=6,
            cut_type="recursive",
        )
        c2 = Chunk(
            text="Text B",
            document_id=doc_id,
            document_source="textbooks",
            chunk_index=1,
            chunk_size=6,
            cut_type="recursive",
        )
        assert str(c1.id) != str(c2.id)

    def test_different_doc_same_index_different_uuid(self):
        c1 = Chunk(
            text="Text",
            document_id=uuid4(),
            document_source="textbooks",
            chunk_index=0,
            chunk_size=4,
            cut_type="recursive",
        )
        c2 = Chunk(
            text="Text",
            document_id=uuid4(),
            document_source="textbooks",
            chunk_index=0,
            chunk_size=4,
            cut_type="recursive",
        )
        assert str(c1.id) != str(c2.id)


class TestChunkMetadata:
    """Auto-derived metadata from DataPoint annotations."""

    def test_text_is_embeddable(self):
        chunk = Chunk(
            text="Test text",
            document_id=uuid4(),
            document_source="textbooks",
            chunk_index=0,
            chunk_size=9,
            cut_type="recursive",
        )
        assert "text" in chunk.metadata["index_fields"]

    def test_identity_fields(self):
        chunk = Chunk(
            text="Test text",
            document_id=uuid4(),
            document_source="textbooks",
            chunk_index=0,
            chunk_size=9,
            cut_type="recursive",
        )
        assert "document_id" in chunk.metadata["identity_fields"]
        assert "chunk_index" in chunk.metadata["identity_fields"]

    def test_kind_is_chunk(self):
        chunk = Chunk(
            text="Test",
            document_id=uuid4(),
            document_source="textbooks",
            chunk_index=0,
            chunk_size=4,
            cut_type="recursive",
        )
        assert chunk.kind == "Chunk"


class TestChunkSerialization:
    """Round-trip serialization."""

    def test_json_round_trip(self):
        doc_id = uuid4()
        chunk = Chunk(
            text="Metformin is a first-line treatment for type 2 diabetes.",
            document_id=doc_id,
            document_source="textbooks",
            chunk_index=2,
            chunk_size=57,
            cut_type="recursive",
            properties={"chapter": "Pharmacology"},
            title="Antidiabetic Drugs",
        )
        json_str = chunk.to_json()
        restored = Chunk.from_json(json_str)
        assert restored.text == chunk.text
        assert str(restored.document_id) == str(doc_id)
        assert restored.document_source == "textbooks"
        assert restored.chunk_index == 2
        assert restored.chunk_size == 57
        assert restored.cut_type == "recursive"
        assert restored.properties == {"chapter": "Pharmacology"}
        assert restored.title == "Antidiabetic Drugs"

    def test_dict_round_trip(self):
        chunk = Chunk(
            text="Test text",
            document_id=uuid4(),
            document_source="pubmed",
            chunk_index=0,
            chunk_size=9,
            cut_type="sentence",
        )
        d = chunk.to_dict()
        restored = Chunk.from_dict(d)
        assert str(restored.id) == str(chunk.id)
        assert restored.text == "Test text"
        assert restored.document_source == "pubmed"
