"""Helpers for extracting text from CV documents."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from .utils import compact_text


SUPPORTED_SUFFIXES = {".pdf", ".txt"}


class DocumentLoaderError(ValueError):
    """Raised when a CV file cannot be loaded or parsed."""


def _validate_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise DocumentLoaderError(f"Unsupported file type '{suffix}'. Supported types: {allowed}.")
    return suffix


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise DocumentLoaderError("No extractable text was found in the PDF.")
    return compact_text(text)


def load_cv_text_from_bytes(filename: str, file_bytes: bytes) -> str:
    """Extract text from a TXT or PDF CV payload."""

    suffix = _validate_suffix(filename)
    if suffix == ".txt":
        try:
            return compact_text(file_bytes.decode("utf-8"))
        except UnicodeDecodeError:
            return compact_text(file_bytes.decode("latin-1"))
    return _extract_pdf_text(file_bytes)


def load_cv_text_from_path(path: str | Path) -> str:
    """Read a CV file from disk and extract its text."""

    resolved_path = Path(path)
    return load_cv_text_from_bytes(resolved_path.name, resolved_path.read_bytes())

