from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import warnings

import pandas as pd
from pypdf import PdfReader

from rag.embeddings import embed_text
from rag.guardrails import contains_prompt_injection, mask_sensitive


SUPPORTED_TEXT_SUFFIXES = {".txt", ".md"}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str, max_words: int = 150, overlap: int = 30) -> list[str]:
    words = clean_text(text).split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed_dates = pd.to_datetime(series, errors="coerce")
    if parsed_dates.notna().mean() >= 0.8:
        return "date"
    return "text"


def build_csv_metadata(source: str, df: pd.DataFrame, head_limit: int = 5) -> dict[str, Any]:
    column_types = {column: _infer_column_type(df[column]) for column in df.columns}
    numeric_summaries: dict[str, dict[str, Any]] = {}
    for column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            numeric_summaries[column] = {
                "count": int(numeric.count()),
                "min": _json_safe(numeric.min()),
                "max": _json_safe(numeric.max()),
                "mean": _json_safe(round(float(numeric.mean()), 3)),
                "sum": _json_safe(round(float(numeric.sum()), 3)),
            }

    head_rows = []
    for _, row in df.head(head_limit).iterrows():
        head_rows.append({str(key): _json_safe(value) for key, value in row.to_dict().items()})

    return {
        "source": source,
        "row_count": int(len(df)),
        "columns": [str(column) for column in df.columns],
        "column_types": column_types,
        "numeric_summaries": numeric_summaries,
        "head_rows": head_rows,
    }


def load_text_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [
        {
            "source": path.name,
            "doc_type": path.suffix.lstrip(".") or "text",
            "chunk_text": chunk,
            "snippet": mask_sensitive(chunk[:650]),
            "page_or_row": f"chunk-{index + 1}",
            "embedding": embed_text(chunk),
            "access_level": "public",
            "contains_injection": contains_prompt_injection(chunk),
        }
        for index, chunk in enumerate(chunk_text(text))
    ]


def load_pdf_file(path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(path))
    chunks: list[dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for chunk_index, chunk in enumerate(chunk_text(text)):
            chunks.append(
                {
                    "source": path.name,
                    "doc_type": "pdf",
                    "chunk_text": chunk,
                    "snippet": mask_sensitive(chunk[:650]),
                    "page_or_row": f"page-{page_index + 1}-chunk-{chunk_index + 1}",
                    "embedding": embed_text(chunk),
                    "access_level": "public",
                    "contains_injection": contains_prompt_injection(chunk),
                }
            )
    return chunks


def load_csv_file(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    df = pd.read_csv(path)
    metadata = build_csv_metadata(path.name, df)
    structured_records: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for row_index, row in df.iterrows():
        data = {str(key): _json_safe(value) for key, value in row.to_dict().items()}
        structured_records.append(data)
        row_text = "; ".join(f"{key}: {value}" for key, value in data.items())
        chunk = f"CSV row from {path.name}: {row_text}"
        chunks.append(
            {
                "source": path.name,
                "doc_type": "csv",
                "chunk_text": chunk,
                "snippet": mask_sensitive(chunk[:650]),
                "page_or_row": f"row-{row_index + 1}",
                "embedding": embed_text(chunk),
                "access_level": "public",
                "contains_injection": contains_prompt_injection(chunk),
            }
        )
    return chunks, structured_records, metadata


def load_document(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return load_text_file(file_path), [], None
    if suffix == ".pdf":
        return load_pdf_file(file_path), [], None
    if suffix == ".csv":
        return load_csv_file(file_path)
    raise ValueError(f"Unsupported file type: {file_path.name}")
