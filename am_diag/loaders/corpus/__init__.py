"""Async streaming loaders for knowledge corpora.

Each loader yields batches of the normalized Document objects.
To ingest multiple corpora, chain loaders together.
"""

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.base import CorpusLoader
from am_diag.loaders.corpus.clinical_guidelines import ClinicalGuidelinesCorpusLoader
from am_diag.loaders.corpus.pubmed import PubMedCorpusLoader
from am_diag.loaders.corpus.pubmed_case_reports import PubmedCaseReportsCorpusLoader
from am_diag.loaders.corpus.statpearls import StatPearlsCorpusLoader
from am_diag.loaders.corpus.textbooks import TextbooksCorpusLoader


__all__ = [
    "ClinicalGuidelinesCorpusLoader",
    "CorpusLoader",
    "Document",
    "PubmedCaseReportsCorpusLoader",
    "PubMedCorpusLoader",
    "StatPearlsCorpusLoader",
    "TextbooksCorpusLoader",
]
