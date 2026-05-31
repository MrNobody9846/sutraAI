from __future__ import annotations

import pandas as pd

from rag import db
from rag.guardrails import mask_sensitive


def _records_to_frame() -> pd.DataFrame:
    rows = db.fetch_structured_records()
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


def _citation(source: str, snippet: str) -> dict[str, str]:
    return {"source": source, "snippet": mask_sensitive(snippet)}


def try_answer_structured(question: str) -> dict | None:
    q = question.lower()
    if not any(term in q for term in ["sales", "aging", "sku", "branch", "inventory"]):
        return None

    frame = _records_to_frame()
    if frame.empty:
        return None

    if "highest" in q and "sales" in q and "branch" in q:
        grouped = frame.groupby("branch", as_index=False)["sales"].sum()
        top = grouped.sort_values("sales", ascending=False).iloc[0]
        snippet = f"Grouped CSV sales by branch; top branch={top['branch']}, sales={top['sales']}."
        return {
            "answer": f"{top['branch']} has the highest total sales at {top['sales']:,.0f}.",
            "citations": [_citation("operations_kpis.csv", snippet)],
            "confidence": "high",
            "used_structured_data": True,
        }

    if "average" in q and "aging" in q:
        average = frame["aging_days"].mean()
        snippet = f"Computed average aging_days across {len(frame)} CSV rows."
        return {
            "answer": f"The average inventory aging is {average:.1f} days.",
            "citations": [_citation("operations_kpis.csv", snippet)],
            "confidence": "high",
            "used_structured_data": True,
        }

    if ("top" in q or "highest" in q) and "sku" in q and "aging" in q:
        limit = 5 if "5" in q or "five" in q else 3
        top_rows = frame.sort_values("aging_days", ascending=False).head(limit)
        lines = [
            f"{index + 1}. {row['sku']} - {row['aging_days']:.0f} days ({row['branch']})"
            for index, row in top_rows.reset_index(drop=True).iterrows()
        ]
        snippet = "Top SKUs by aging_days: " + "; ".join(lines)
        return {
            "answer": "Top SKUs by aging days:\n" + "\n".join(lines),
            "citations": [_citation("operations_kpis.csv", snippet)],
            "confidence": "high",
            "used_structured_data": True,
        }

    return None
