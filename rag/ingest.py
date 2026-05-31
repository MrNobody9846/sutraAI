from __future__ import annotations

from pathlib import Path

from rag import db
from rag.loaders import load_document


SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


def ingest_file(path: str | Path, replace_existing: bool = True) -> dict[str, int | str]:
    file_path = Path(path)
    chunks, records, metadata = load_document(file_path)
    if replace_existing:
        db.delete_source(file_path.name)
    for chunk in chunks:
        db.insert_chunk(chunk)
    for row_index, record in enumerate(records):
        db.insert_structured_record(file_path.name, row_index, record)
    if metadata:
        db.upsert_table_metadata(file_path.name, metadata)
    return {
        "source": file_path.name,
        "chunks": len(chunks),
        "structured_records": len(records),
        "structured_tables": 1 if metadata else 0,
    }


def ingest_directory(path: str | Path = SAMPLE_DOCS_DIR) -> list[dict[str, int | str]]:
    db.init_db()
    directory = Path(path)
    results = []
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md", ".pdf", ".csv"}:
            results.append(ingest_file(file_path))
    return results
