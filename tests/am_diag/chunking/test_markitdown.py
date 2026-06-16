"""Tests for MarkItDownLoader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from am_diag.chunking.markitdown import MarkItDownLoader


def _write_html(path: Path, title: str = "Test", body: str = "Hello world") -> Path:
    html = f"<html><head><title>{title}</title></head><body><p>{body}</p></body></html>"
    path.write_text(html)
    return path


class TestMarkItDownLoader:
    """Core loading behaviour."""

    async def test_single_file_produces_document(self, tmp_path: Path):
        html_file = _write_html(tmp_path / "test.html")
        loader = MarkItDownLoader()
        docs = await loader.load([html_file])
        assert len(docs) == 1
        doc = docs[0]
        assert doc.text == "Hello world"
        assert doc.source == "markitdown"
        assert doc.title == "Test"
        assert doc.id is not None

    async def test_properties_populated(self, tmp_path: Path):
        html_file = _write_html(tmp_path / "doc.html")
        loader = MarkItDownLoader()
        docs = await loader.load([html_file])
        props = docs[0].properties
        assert props["file_path"] == str(html_file.resolve())
        assert props["file_name"] == "doc.html"
        assert props["file_extension"] == ".html"
        assert props["file_size_bytes"] == html_file.stat().st_size
        assert props["loader"] == "markitdown"

    async def test_title_from_markitdown(self, tmp_path: Path):
        html_file = _write_html(tmp_path / "titled.html", title="Extracted Title")
        loader = MarkItDownLoader()
        docs = await loader.load([html_file])
        assert docs[0].title == "Extracted Title"

    async def test_title_fallback_to_filename(self, tmp_path: Path):
        plain = tmp_path / "no_title.txt"
        plain.write_text("Just some text.")
        loader = MarkItDownLoader()
        docs = await loader.load([plain])
        assert docs[0].title == "no_title"

    async def test_title_fallback_to_default_title(self, tmp_path: Path):
        plain = tmp_path / "no_title.txt"
        plain.write_text("Just some text.")
        loader = MarkItDownLoader()
        docs = await loader.load([plain], default_title="Fallback")
        assert docs[0].title == "Fallback"

    async def test_multi_file_list(self, tmp_path: Path):
        f1 = _write_html(tmp_path / "a.html", body="File A")
        f2 = _write_html(tmp_path / "b.html", body="File B")
        loader = MarkItDownLoader()
        docs = await loader.load([f1, f2])
        assert len(docs) == 2
        assert docs[0].text == "File A"
        assert docs[1].text == "File B"

    async def test_custom_source(self, tmp_path: Path):
        html_file = _write_html(tmp_path / "test.html")
        loader = MarkItDownLoader()
        docs = await loader.load([html_file], source="my_corpus")
        assert docs[0].source == "my_corpus"

    async def test_empty_file_list(self):
        loader = MarkItDownLoader()
        docs = await loader.load([])
        assert docs == []

    async def test_failed_conversion_skipped(self, tmp_path: Path):
        good = _write_html(tmp_path / "ok.html")
        loader = MarkItDownLoader()
        docs = await loader.load(["/nonexistent/file.pdf", good])
        assert len(docs) == 1
        assert docs[0].text == "Hello world"


class TestInitParamsForwarded:
    """Verify that init parameters reach the MarkItDown constructor."""

    @patch("am_diag.chunking.markitdown.MarkItDown")
    async def test_enable_plugins_forwarded(self, mock_md_cls: MagicMock):
        MarkItDownLoader(enable_plugins=True)
        mock_md_cls.assert_called_once_with(enable_plugins=True)

    @patch("am_diag.chunking.markitdown.MarkItDown")
    async def test_llm_params_forwarded(self, mock_md_cls: MagicMock):
        client = MagicMock()
        MarkItDownLoader(
            llm_client=client,
            llm_model="gpt-4o",
            llm_prompt="Describe this image",
        )
        mock_md_cls.assert_called_once_with(
            enable_plugins=False,
            llm_client=client,
            llm_model="gpt-4o",
            llm_prompt="Describe this image",
        )

    @patch("am_diag.chunking.markitdown.MarkItDown")
    async def test_azure_params_forwarded(self, mock_md_cls: MagicMock):
        MarkItDownLoader(
            docintel_endpoint="https://example.com",
            docintel_credential="secret",
            cu_endpoint="https://cu.example.com",
            cu_analyzer_id="my-analyzer",
        )
        mock_md_cls.assert_called_once_with(
            enable_plugins=False,
            docintel_endpoint="https://example.com",
            docintel_credential="secret",
            cu_endpoint="https://cu.example.com",
            cu_analyzer_id="my-analyzer",
        )

    @patch("am_diag.chunking.markitdown.MarkItDown")
    async def test_requests_session_forwarded(self, mock_md_cls: MagicMock):
        session = MagicMock()
        MarkItDownLoader(requests_session=session)
        mock_md_cls.assert_called_once_with(
            enable_plugins=False,
            requests_session=session,
        )

    @patch("am_diag.chunking.markitdown.MarkItDown")
    async def test_none_params_not_forwarded(self, mock_md_cls: MagicMock):
        MarkItDownLoader()
        mock_md_cls.assert_called_once_with(enable_plugins=False)
