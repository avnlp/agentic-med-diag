"""MarkItDown-based document loader.

Wraps the ``markitdown`` library to convert files of various formats
(PDF, DOCX, HTML, Excel, etc.) into `Document` objects with full
provenance metadata.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from am_diag.common.data_models.document import Document


class MarkItDownLoader:
    """Convert files to `Document` objects using the MarkItDown library.

    Accepts a list of file paths and returns a list of `Document` objects,
    each with full provenance metadata (file path, name, extension, size).

    Parameters:
        enable_plugins: Enable third-party MarkItDown plugins.
        llm_client: OpenAI-compatible client for image descriptions.
        llm_model: LLM model name for image descriptions.
        llm_prompt: Custom prompt for image descriptions.
        docintel_endpoint: Azure Document Intelligence endpoint.
        docintel_credential: Azure Document Intelligence credential.
        cu_endpoint: Azure Content Understanding endpoint.
        cu_analyzer_id: Azure Content Understanding custom analyzer id.
        requests_session: Custom ``requests.Session`` for HTTP fetching.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        enable_plugins: bool = False,
        llm_client: Any | None = None,
        llm_model: str | None = None,
        llm_prompt: str | None = None,
        docintel_endpoint: str | None = None,
        docintel_credential: str | None = None,
        cu_endpoint: str | None = None,
        cu_analyzer_id: str | None = None,
        requests_session: Any | None = None,
    ) -> None:
        """Initialize the MarkItDown engine with the given parameters.

        See the class docstring for parameter descriptions.
        """
        kwargs: dict[str, Any] = {
            "enable_plugins": enable_plugins,
            "llm_client": llm_client,
            "llm_model": llm_model,
            "llm_prompt": llm_prompt,
            "docintel_endpoint": docintel_endpoint,
            "docintel_credential": docintel_credential,
            "cu_endpoint": cu_endpoint,
            "cu_analyzer_id": cu_analyzer_id,
            "requests_session": requests_session,
        }
        self._md = MarkItDown(**{k: v for k, v in kwargs.items() if v is not None})

    async def load(
        self,
        files: list[str | Path],
        *,
        source: str = "markitdown",
        default_title: str | None = None,
    ) -> list[Document]:
        """Convert files to `Document` objects with full provenance.

        Args:
            files: Paths to files to convert. Supported formats depend on
                installed MarkItDown converters and plugins.
            source: Source identifier stored in ``Document.source``.
            default_title: Fallback title when MarkItDown cannot extract one.

        Returns:
            A `Document` for each successfully converted file, preserving
            input order. Files that fail conversion are skipped.
        """
        return await asyncio.to_thread(
            self._convert, files, source=source, default_title=default_title
        )

    def _convert(
        self,
        files: list[str | Path],
        *,
        source: str = "markitdown",
        default_title: str | None = None,
    ) -> list[Document]:
        documents: list[Document] = []
        for file_path in files:
            path = Path(file_path).resolve()
            try:
                result = self._md.convert(str(path))
            except Exception:  # noqa: BLE001
                continue
            title = result.title or default_title or path.stem
            documents.append(
                Document(
                    text=result.markdown,
                    source=source,
                    external_id=str(path),
                    title=title,
                    properties={
                        "file_path": str(path),
                        "file_name": path.name,
                        "file_extension": path.suffix,
                        "file_size_bytes": path.stat().st_size,
                        "loader": "markitdown",
                    },
                )
            )
        return documents
