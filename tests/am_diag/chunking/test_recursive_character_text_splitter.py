"""Tests for RecursiveCharacterTextChunker."""

from __future__ import annotations

from am_diag.chunking.recursive_character_text_splitter import (
    RecursiveCharacterTextChunker,
)
from am_diag.common.data_models.document import Document


def _make_doc(
    text: str = "A" * 100,
    external_id: str | None = None,
    source: str = "textbooks",
    title: str | None = None,
) -> Document:
    return Document(
        text=text,
        source=source,
        external_id=external_id,
        title=title,
    )


class TestRecursiveCharacterTextChunker:
    """Core chunking behaviour."""

    async def test_single_document_produces_chunks(self):
        text = "First paragraph.\n\n" + "A" * 100 + "\n\n" + "B" * 100
        doc = _make_doc(text=text)
        chunker = RecursiveCharacterTextChunker(chunk_size=50, chunk_overlap=5)
        chunks = await chunker.chunk([doc])
        assert len(chunks) >= 2

    async def test_chunk_fields_populated(self):
        doc = _make_doc(
            text="Hello world.",
            external_id="unique-doc-1",
            source="pubmed",
            title="Test Title",
        )
        chunker = RecursiveCharacterTextChunker(chunk_size=100, chunk_overlap=10)
        chunks = await chunker.chunk([doc])
        assert len(chunks) == 1
        c = chunks[0]
        assert c.text == "Hello world."
        assert c.document_id == doc.id
        assert c.document_source == "pubmed"
        assert c.chunk_index == 0
        assert c.chunk_size == len("Hello world.")
        assert c.cut_type == "recursive"
        assert c.title == "Test Title"

    async def test_multi_document_flattened(self):
        doc1 = _make_doc(text="A " * 50, external_id="doc-1")
        doc2 = _make_doc(text="B " * 50, external_id="doc-2")
        chunker = RecursiveCharacterTextChunker(chunk_size=20, chunk_overlap=5)
        chunks = await chunker.chunk([doc1, doc2])
        assert len(chunks) > 2
        sources = {c.document_id for c in chunks}
        assert len(sources) == 2

    async def test_chunk_index_sequential_per_document(self):
        doc = _make_doc(text="A\n\nB\n\nC\n\nD\n\nE")
        chunker = RecursiveCharacterTextChunker(chunk_size=5, chunk_overlap=1)
        chunks = await chunker.chunk([doc])
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    async def test_chunk_size_matches_text_length(self):
        doc = _make_doc(text="Short text here.")
        chunker = RecursiveCharacterTextChunker(chunk_size=100, chunk_overlap=10)
        chunks = await chunker.chunk([doc])
        for c in chunks:
            assert c.chunk_size == len(c.text)

    async def test_deterministic_uuid5(self):
        doc = _make_doc(text="Same content.", external_id="stable-doc")
        c1 = (
            await RecursiveCharacterTextChunker(chunk_size=100, chunk_overlap=10).chunk(
                [doc]
            )
        )[0]
        doc2 = _make_doc(text="Same content.", external_id="stable-doc")
        c2 = (
            await RecursiveCharacterTextChunker(chunk_size=100, chunk_overlap=10).chunk(
                [doc2]
            )
        )[0]
        assert c1.id == c2.id

    async def test_different_doc_different_uuid(self):
        # No external_id — uuid4 so different each time
        c1 = (
            await RecursiveCharacterTextChunker(chunk_size=100, chunk_overlap=10).chunk(
                [_make_doc(text="Same content.", external_id=None)]
            )
        )[0]
        c2 = (
            await RecursiveCharacterTextChunker(chunk_size=100, chunk_overlap=10).chunk(
                [_make_doc(text="Same content.", external_id=None)]
            )
        )[0]
        assert c1.id != c2.id

    async def test_empty_document_list(self):
        chunker = RecursiveCharacterTextChunker()
        assert await chunker.chunk([]) == []

    async def test_title_propagated_from_document(self):
        doc = _make_doc(text="Content.", title="Chapter 1")
        chunks = await RecursiveCharacterTextChunker(
            chunk_size=100, chunk_overlap=10
        ).chunk([doc])
        assert chunks[0].title == "Chapter 1"

    async def test_title_none_when_document_has_none(self):
        doc = _make_doc(text="Content.", title=None)
        chunks = await RecursiveCharacterTextChunker(
            chunk_size=100, chunk_overlap=10
        ).chunk([doc])
        assert chunks[0].title is None


class TestConfigurableParameters:
    """Verify that init parameters affect splitting behaviour."""

    async def test_chunk_size_limits_output(self):
        doc = _make_doc(text="A" * 200)
        small = await RecursiveCharacterTextChunker(
            chunk_size=50, chunk_overlap=5
        ).chunk([doc])
        large = await RecursiveCharacterTextChunker(
            chunk_size=200, chunk_overlap=10
        ).chunk([doc])
        assert len(small) > len(large)

    async def test_custom_separators(self):
        doc = _make_doc(text="AA|BB|CC")
        chunks = await RecursiveCharacterTextChunker(
            separators=["|"], chunk_size=100, chunk_overlap=10
        ).chunk([doc])
        texts = [c.text for c in chunks]
        assert any("AA" in t for t in texts)

    async def test_chunk_overlap(self):
        doc = _make_doc(text="A" * 100)
        chunks_no_overlap = await RecursiveCharacterTextChunker(
            chunk_size=30, chunk_overlap=0
        ).chunk([doc])
        chunks_with_overlap = await RecursiveCharacterTextChunker(
            chunk_size=30, chunk_overlap=10
        ).chunk([doc])
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    async def test_add_start_index_in_properties(self):
        doc = _make_doc(text="A\n\nB\n\nC")
        chunks = await RecursiveCharacterTextChunker(
            chunk_size=5, chunk_overlap=1, add_start_index=True
        ).chunk([doc])
        for c in chunks:
            assert "start_index" in c.properties

    async def test_no_start_index_when_disabled(self):
        doc = _make_doc(text="A\n\nB\n\nC")
        chunks = await RecursiveCharacterTextChunker(
            chunk_size=5, chunk_overlap=1, add_start_index=False
        ).chunk([doc])
        for c in chunks:
            assert "start_index" not in c.properties

    async def test_strip_whitespace(self):
        doc = _make_doc(text="  Hello  \n\n  World  ")
        chunks = await RecursiveCharacterTextChunker(
            chunk_size=100, chunk_overlap=10, strip_whitespace=True
        ).chunk([doc])
        for c in chunks:
            assert c.text == c.text.strip()
