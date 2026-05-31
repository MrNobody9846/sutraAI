from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.tools import tool

from rag import db
from rag.guardrails import mask_sensitive


OPERATIONS = {"sum", "avg", "min", "max", "count"}
FILTER_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains"}
PANDAS_OPERATIONS = {"avg": "mean", "sum": "sum", "min": "min", "max": "max"}


def _records_to_frame(source: str | None = None) -> pd.DataFrame:
    rows = db.fetch_structured_records()
    if source:
        rows = [row for row in rows if row["source"] == source]
    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        record = dict(row["data"])
        record["_source"] = row["source"]
        record["_row_index"] = row["row_index"]
        records.append(record)

    frame = pd.DataFrame(records)
    for column in frame.columns:
        if column.startswith("_"):
            continue
        try:
            frame[column] = pd.to_numeric(frame[column])
        except (TypeError, ValueError):
            pass
    return frame


def _safe_columns(frame: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if not columns:
        return [column for column in frame.columns if not column.startswith("_")]
    return [column for column in columns if column in frame.columns and not column.startswith("_")]


def _citation(source: str, snippet: str, operation: str = "csv_tool") -> dict[str, Any]:
    return {
        "source": source,
        "location": operation,
        "snippet": mask_sensitive(snippet),
    }


def _source_from_metadata(source: str | None = None) -> str | None:
    metadata = db.fetch_table_metadata(source)
    if source:
        return source if metadata else None
    if len(metadata) == 1:
        return metadata[0]["source"]
    if metadata:
        return metadata[0]["source"]
    return None


@tool
def list_csv_sources() -> dict[str, Any]:
    """List CSV sources with row counts, columns, and compact summaries."""
    tables = db.list_table_metadata()
    sources = [
        {
            "source": table["source"],
            "row_count": table["row_count"],
            "columns": table["columns"],
            "numeric_columns": list((table["numeric_summaries"] or {}).keys()),
        }
        for table in tables
    ]
    return {
        "ok": True,
        "sources": sources,
        "citations": [
            _citation(item["source"], f"CSV source has {item['row_count']} rows and columns {item['columns']}.")
            for item in sources
        ],
    }


@tool
def get_csv_table_metadata(source: str | None = None) -> dict[str, Any]:
    """Return CSV table metadata including columns, types, summaries, and head rows."""
    tables = db.fetch_table_metadata(source)
    return {
        "ok": bool(tables),
        "tables": tables,
        "citations": [
            _citation(
                table["source"],
                f"Metadata: {table['row_count']} rows; columns={table['columns']}; head_rows={table['head_rows']}.",
                "table_metadata",
            )
            for table in tables
        ],
    }


@tool
def aggregate_csv(
    metric: str,
    operation: str,
    group_by: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Aggregate a numeric CSV column with optional grouping."""
    operation = operation.lower()
    if operation not in OPERATIONS:
        return {"ok": False, "error": f"Unsupported operation: {operation}"}

    resolved_source = _source_from_metadata(source)
    frame = _records_to_frame(resolved_source)
    if frame.empty:
        return {"ok": False, "error": "No structured CSV records are available."}
    if operation != "count" and metric not in frame.columns:
        return {"ok": False, "error": f"Column not found: {metric}"}
    if group_by and group_by not in frame.columns:
        return {"ok": False, "error": f"Group-by column not found: {group_by}"}

    if operation == "count":
        if group_by:
            result_frame = frame.groupby(group_by, as_index=False).size().rename(columns={"size": "count"})
            rows = result_frame.to_dict(orient="records")
            snippet = f"Counted rows grouped by {group_by}: {rows}."
            return {
                "ok": True,
                "source": resolved_source,
                "operation": "count",
                "metric": metric,
                "group_by": group_by,
                "rows": rows,
                "citations": [_citation(resolved_source or "csv", snippet, "aggregate_csv")],
            }
        value = int(len(frame))
        return {
            "ok": True,
            "source": resolved_source,
            "operation": "count",
            "metric": metric,
            "value": value,
            "citations": [_citation(resolved_source or "csv", f"Counted {value} CSV rows.", "aggregate_csv")],
        }

    numeric = pd.to_numeric(frame[metric], errors="coerce")
    work = frame.copy()
    work[metric] = numeric
    work = work.dropna(subset=[metric])
    if work.empty:
        return {"ok": False, "error": f"Column is not numeric: {metric}"}

    if group_by:
        grouped = getattr(work.groupby(group_by, as_index=False)[metric], PANDAS_OPERATIONS[operation])()
        grouped = grouped.sort_values(metric, ascending=False)
        rows = grouped.to_dict(orient="records")
        snippet = f"Aggregated {operation}({metric}) grouped by {group_by}: {rows}."
        return {
            "ok": True,
            "source": resolved_source,
            "operation": operation,
            "metric": metric,
            "group_by": group_by,
            "rows": rows,
            "citations": [_citation(resolved_source or "csv", snippet, "aggregate_csv")],
        }

    value = getattr(work[metric], PANDAS_OPERATIONS[operation])()
    if hasattr(value, "item"):
        value = value.item()
    snippet = f"Computed {operation}({metric}) across {len(work)} CSV rows: {value}."
    return {
        "ok": True,
        "source": resolved_source,
        "operation": operation,
        "metric": metric,
        "value": value,
        "citations": [_citation(resolved_source or "csv", snippet, "aggregate_csv")],
    }


@tool
def top_n_csv(
    sort_by: str,
    n: int = 5,
    columns: list[str] | None = None,
    source: str | None = None,
    descending: bool = True,
) -> dict[str, Any]:
    """Return the top N CSV rows sorted by a whitelisted column."""
    resolved_source = _source_from_metadata(source)
    frame = _records_to_frame(resolved_source)
    if frame.empty:
        return {"ok": False, "error": "No structured CSV records are available."}
    if sort_by not in frame.columns:
        return {"ok": False, "error": f"Sort column not found: {sort_by}"}

    selected_columns = _safe_columns(frame, columns)
    if sort_by not in selected_columns:
        selected_columns.append(sort_by)
    n = max(1, min(int(n), 25))
    sorted_frame = frame.sort_values(sort_by, ascending=not descending).head(n)
    rows = sorted_frame[selected_columns].to_dict(orient="records")
    snippet = f"Top {n} rows by {sort_by}: {rows}."
    return {
        "ok": True,
        "source": resolved_source,
        "operation": "top_n",
        "sort_by": sort_by,
        "rows": rows,
        "citations": [_citation(resolved_source or "csv", snippet, "top_n_csv")],
    }


@tool
def filter_csv_rows(
    filters: list[dict[str, Any]],
    columns: list[str] | None = None,
    limit: int = 10,
    source: str | None = None,
) -> dict[str, Any]:
    """Filter CSV rows using safe column/operator/value filters."""
    resolved_source = _source_from_metadata(source)
    frame = _records_to_frame(resolved_source)
    if frame.empty:
        return {"ok": False, "error": "No structured CSV records are available."}

    filtered = frame
    for item in filters:
        column = item.get("column")
        operator = item.get("operator")
        value = item.get("value")
        if column not in filtered.columns:
            return {"ok": False, "error": f"Filter column not found: {column}"}
        if operator not in FILTER_OPERATORS:
            return {"ok": False, "error": f"Unsupported filter operator: {operator}"}

        series = filtered[column]
        if operator == "eq":
            filtered = filtered[series == value]
        elif operator == "ne":
            filtered = filtered[series != value]
        elif operator == "gt":
            filtered = filtered[pd.to_numeric(series, errors="coerce") > float(value)]
        elif operator == "gte":
            filtered = filtered[pd.to_numeric(series, errors="coerce") >= float(value)]
        elif operator == "lt":
            filtered = filtered[pd.to_numeric(series, errors="coerce") < float(value)]
        elif operator == "lte":
            filtered = filtered[pd.to_numeric(series, errors="coerce") <= float(value)]
        elif operator == "contains":
            filtered = filtered[series.astype(str).str.contains(str(value), case=False, na=False)]

    limit = max(1, min(int(limit), 50))
    selected_columns = _safe_columns(filtered, columns)
    rows = filtered.head(limit)[selected_columns].to_dict(orient="records")
    snippet = f"Filtered CSV rows with {filters}; returned {len(rows)} rows."
    return {
        "ok": True,
        "source": resolved_source,
        "operation": "filter",
        "rows": rows,
        "citations": [_citation(resolved_source or "csv", snippet, "filter_csv_rows")],
    }
