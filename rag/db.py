from __future__ import annotations

from contextlib import contextmanager
import json
from typing import Any

import psycopg
from psycopg.rows import dict_row

from rag.config import settings
from rag.embeddings import DEFAULT_EMBEDDING_DIM, vector_literal
from rag.guardrails import mask_sensitive


@contextmanager
def get_conn():
    conn = psycopg.connect(settings.database_url, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                snippet TEXT NOT NULL,
                embedding vector({DEFAULT_EMBEDDING_DIM}) NOT NULL,
                page_or_row TEXT,
                access_level TEXT NOT NULL DEFAULT 'public',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS structured_table_metadata (
                source TEXT PRIMARY KEY,
                row_count INTEGER NOT NULL,
                columns JSONB NOT NULL,
                column_types JSONB NOT NULL,
                numeric_summaries JSONB NOT NULL,
                head_rows JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS structured_records (
                id BIGSERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                data JSONB NOT NULL,
                access_level TEXT NOT NULL DEFAULT 'public',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS document_chunks_source_idx
            ON document_chunks (source);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS structured_records_source_idx
            ON structured_records (source);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS structured_records_data_idx
            ON structured_records USING gin (data);
            """
        )


def reset_database() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM document_chunks;")
        conn.execute("DELETE FROM structured_records;")
        conn.execute("DELETE FROM structured_table_metadata;")


def delete_source(source: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM document_chunks WHERE source = %s;", (source,))
        conn.execute("DELETE FROM structured_records WHERE source = %s;", (source,))
        conn.execute("DELETE FROM structured_table_metadata WHERE source = %s;", (source,))


def insert_chunk(chunk: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO document_chunks
                (source, doc_type, chunk_text, snippet, embedding, page_or_row, access_level)
            VALUES (%s, %s, %s, %s, %s::vector, %s, %s);
            """,
            (
                chunk["source"],
                chunk["doc_type"],
                chunk["chunk_text"],
                mask_sensitive(chunk["snippet"]),
                vector_literal(chunk["embedding"]),
                chunk.get("page_or_row"),
                chunk.get("access_level", "public"),
            ),
        )


def insert_structured_record(source: str, row_index: int, data: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO structured_records (source, row_index, data, access_level)
            VALUES (%s, %s, %s::jsonb, 'public');
            """,
            (source, row_index, json.dumps(data)),
        )


def upsert_table_metadata(source: str, metadata: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO structured_table_metadata
                (source, row_count, columns, column_types, numeric_summaries, head_rows, updated_at)
            VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, now())
            ON CONFLICT (source) DO UPDATE SET
                row_count = EXCLUDED.row_count,
                columns = EXCLUDED.columns,
                column_types = EXCLUDED.column_types,
                numeric_summaries = EXCLUDED.numeric_summaries,
                head_rows = EXCLUDED.head_rows,
                updated_at = now();
            """,
            (
                source,
                metadata["row_count"],
                json.dumps(metadata["columns"]),
                json.dumps(metadata["column_types"]),
                json.dumps(metadata["numeric_summaries"]),
                json.dumps(metadata["head_rows"]),
            ),
        )


def search_chunks(query_embedding: list[float], top_k: int | None = None) -> list[dict[str, Any]]:
    query_vector = vector_literal(query_embedding)
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                source,
                doc_type,
                chunk_text,
                snippet,
                page_or_row,
                access_level,
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            WHERE access_level = 'public'
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_vector, query_vector, top_k or settings.top_k),
        ).fetchall()
    for row in rows:
        row["snippet"] = mask_sensitive(row["snippet"])
        row["chunk_text"] = mask_sensitive(row["chunk_text"])
    return rows


def fetch_structured_records() -> list[dict[str, Any]]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT source, row_index, data
            FROM structured_records
            WHERE access_level = 'public'
            ORDER BY source, row_index;
            """
        ).fetchall()


def list_table_metadata() -> list[dict[str, Any]]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT source, row_count, columns, column_types, numeric_summaries, head_rows
            FROM structured_table_metadata
            ORDER BY source;
            """
        ).fetchall()


def fetch_table_metadata(source: str | None = None) -> list[dict[str, Any]]:
    if source is None:
        return list_table_metadata()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT source, row_count, columns, column_types, numeric_summaries, head_rows
            FROM structured_table_metadata
            WHERE source = %s;
            """,
            (source,),
        ).fetchone()
    return [row] if row else []


def count_rows() -> dict[str, int]:
    with get_conn() as conn:
        chunks = conn.execute("SELECT count(*) AS count FROM document_chunks;").fetchone()["count"]
        records = conn.execute("SELECT count(*) AS count FROM structured_records;").fetchone()["count"]
        tables = conn.execute("SELECT count(*) AS count FROM structured_table_metadata;").fetchone()["count"]
    return {"document_chunks": chunks, "structured_records": records, "structured_tables": tables}
