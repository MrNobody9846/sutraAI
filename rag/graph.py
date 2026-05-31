from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from rag import db
from rag.config import settings
from rag.embeddings import embed_text
from rag.guardrails import blocked_injection_response, contains_prompt_injection, mask_sensitive
from rag.llm import generate_grounded_answer
from rag.tools import aggregate_csv, get_csv_table_metadata, top_n_csv


Route = Literal["csv_tools", "document_rag", "hybrid", "clarify", "abstain", "blocked"]


class AssistantState(TypedDict, total=False):
    question: str
    route: Route
    table_metadata: list[dict[str, Any]]
    tool_request: dict[str, Any]
    tool_result: dict[str, Any]
    contexts: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    confidence: str
    used_structured_data: bool
    blocked: bool


DATA_TERMS = {
    "sales",
    "aging",
    "sku",
    "branch",
    "inventory",
    "table",
    "csv",
    "rows",
    "columns",
    "average",
    "highest",
    "top",
    "performance",
}
DOC_TERMS = {
    "sop",
    "policy",
    "workflow",
    "approval",
    "process",
    "escalation",
    "kpi",
    "explain",
    "guidance",
    "should",
    "why",
    "report",
    "slow-moving",
}


def _blank_response(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "answer": "Ask a question about the uploaded business documents.",
        "citations": [],
        "confidence": "none",
        "used_structured_data": False,
        "route": "abstain",
    }


def _format_number(value: Any, decimals: int = 1) -> str:
    if isinstance(value, float):
        return f"{value:,.{decimals}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _tool_invoke(tool_obj, payload: dict[str, Any]) -> dict[str, Any]:
    return tool_obj.invoke(payload)


def guardrail_node(state: AssistantState) -> AssistantState:
    question = (state.get("question") or "").strip()
    if not question:
        return _blank_response(question)
    if contains_prompt_injection(question):
        return blocked_injection_response() | {"question": question, "route": "blocked", "blocked": True}
    return {"question": question, "blocked": False}


def metadata_node(state: AssistantState) -> AssistantState:
    question = state["question"].lower()
    if any(term in question for term in DATA_TERMS):
        result = _tool_invoke(get_csv_table_metadata, {"source": None})
        return {"table_metadata": result.get("tables", [])}
    return {"table_metadata": []}


def _has_any(question: str, terms: set[str]) -> bool:
    return any(term in question for term in terms)


def route_node(state: AssistantState) -> AssistantState:
    if state.get("blocked"):
        return {"route": "blocked"}
    question = state["question"].lower()
    if not question:
        return {"route": "abstain"}
    has_data = _has_any(question, DATA_TERMS)
    has_docs = _has_any(question, DOC_TERMS)
    has_tables = bool(state.get("table_metadata"))

    if has_data and not has_tables:
        return {"route": "abstain"}
    if _is_csv_question(question):
        return {"route": "csv_tools"}
    if "kpi" in question and "explain" in question and not any(
        term in question
        for term in ["sales", "sku", "branch", "high", "highest", "average", "top", "performance", "should", "why", "slow-moving"]
    ):
        return {"route": "document_rag"}
    if has_data and has_docs:
        return {"route": "hybrid"}
    if has_data and any(term in question for term in ["why", "should", "summarize", "performance", "slow-moving"]):
        return {"route": "hybrid"}
    if any(term in question for term in ["bad", "unclear", "issue"]) and not has_data and not has_docs:
        return {"route": "clarify"}
    return {"route": "document_rag"}


def route_edge(state: AssistantState) -> str:
    return state.get("route", "document_rag")


def _is_csv_question(question: str) -> bool:
    return (
        ("highest" in question and "sales" in question and "branch" in question)
        or ("average" in question and "aging" in question)
        or (("top" in question or "highest" in question) and "sku" in question and "aging" in question)
        or ("list" in question and "columns" in question)
    )


def csv_tool_planner_node(state: AssistantState) -> AssistantState:
    question = state["question"].lower()
    metadata = state.get("table_metadata") or []
    source = metadata[0]["source"] if metadata else None

    if "highest" in question and "sales" in question and "branch" in question:
        return {
            "tool_request": {
                "tool": "aggregate_csv",
                "args": {"metric": "sales", "operation": "sum", "group_by": "branch", "source": source},
            }
        }
    if "average" in question and "aging" in question:
        return {
            "tool_request": {
                "tool": "aggregate_csv",
                "args": {"metric": "aging_days", "operation": "avg", "group_by": None, "source": source},
            }
        }
    if ("top" in question or "highest" in question) and "sku" in question and "aging" in question:
        limit = 5 if "5" in question or "five" in question else 3
        return {
            "tool_request": {
                "tool": "top_n_csv",
                "args": {
                    "sort_by": "aging_days",
                    "n": limit,
                    "columns": ["sku", "aging_days", "branch"],
                    "source": source,
                    "descending": True,
                },
            }
        }
    if "columns" in question or "schema" in question:
        return {"tool_request": {"tool": "get_csv_table_metadata", "args": {"source": source}}}

    return {
        "tool_request": {
            "tool": "top_n_csv",
            "args": {
                "sort_by": "aging_days",
                "n": 5,
                "columns": ["sku", "aging_days", "branch", "inventory_value"],
                "source": source,
                "descending": True,
            },
        }
    }


def csv_tool_node(state: AssistantState) -> AssistantState:
    request = state.get("tool_request") or {}
    tool_name = request.get("tool")
    args = request.get("args", {})
    if tool_name == "aggregate_csv":
        result = _tool_invoke(aggregate_csv, args)
    elif tool_name == "top_n_csv":
        result = _tool_invoke(top_n_csv, args)
    elif tool_name == "get_csv_table_metadata":
        result = _tool_invoke(get_csv_table_metadata, args)
    else:
        result = {"ok": False, "error": "No safe CSV tool matched the question."}

    evidence = []
    if result.get("ok"):
        evidence.append({"kind": "csv_tool", "tool": tool_name, "result": result})
    return {
        "tool_result": result,
        "evidence": evidence,
        "citations": result.get("citations", []),
        "used_structured_data": bool(result.get("ok")),
    }


def document_retrieval_node(state: AssistantState) -> AssistantState:
    contexts = _retrieve_contexts(state["question"])
    evidence = [{"kind": "document", "result": context} for context in contexts]
    citations = [
        {
            "source": item["source"],
            "location": item.get("page_or_row") or "",
            "similarity": round(float(item["similarity"]), 3),
            "snippet": item["snippet"],
        }
        for item in contexts
    ]
    return {"contexts": contexts, "evidence": evidence, "citations": citations}


def hybrid_node(state: AssistantState) -> AssistantState:
    planned = csv_tool_planner_node(state)
    csv_state = dict(state) | planned
    with ThreadPoolExecutor(max_workers=2) as executor:
        csv_future = executor.submit(csv_tool_node, csv_state)
        rag_future = executor.submit(document_retrieval_node, state)
        csv_result = csv_future.result()
        rag_result = rag_future.result()

    return {
        "tool_request": planned.get("tool_request"),
        "tool_result": csv_result.get("tool_result", {}),
        "contexts": rag_result.get("contexts", []),
        "evidence": (csv_result.get("evidence") or []) + (rag_result.get("evidence") or []),
        "citations": _dedupe_citations((csv_result.get("citations") or []) + (rag_result.get("citations") or [])),
        "used_structured_data": csv_result.get("used_structured_data", False),
    }


def merge_evidence_node(state: AssistantState) -> AssistantState:
    if not (state.get("question") or "").strip():
        return {
            "answer": "Ask a question about the uploaded business documents.",
            "citations": [],
            "confidence": "none",
        }
    citations = _dedupe_citations(state.get("citations") or [])
    evidence = state.get("evidence") or []
    if not evidence:
        return {
            "answer": (
                "I do not have enough source-backed evidence to answer that confidently. "
                "The available documents or tables may not cover this topic."
            ),
            "citations": [],
            "confidence": "low",
        }
    return {"citations": citations}


def answer_node(state: AssistantState) -> AssistantState:
    route = state.get("route")
    if route == "blocked":
        return blocked_injection_response() | {"route": "blocked"}
    if route == "clarify":
        return {
            "answer": (
                "I need a bit more detail before answering safely. Which report or table do you mean, "
                "and should I judge it by accuracy, completeness, KPI performance, or business impact?"
            ),
            "citations": [],
            "confidence": "clarification_needed",
            "used_structured_data": False,
        }
    if state.get("answer") and state.get("confidence") == "low":
        return state

    csv_text = _format_csv_answer(state)
    doc_text = _format_doc_answer(state)
    if route == "csv_tools" and csv_text:
        return {
            "answer": csv_text,
            "confidence": "high",
            "used_structured_data": True,
        }
    if route == "hybrid":
        parts = [part for part in [csv_text, doc_text] if part]
        if parts:
            return {
                "answer": "\n\n".join(parts),
                "confidence": "medium",
                "used_structured_data": bool(state.get("used_structured_data")),
            }
    if doc_text:
        return {
            "answer": doc_text,
            "confidence": _doc_confidence(state.get("contexts") or []),
            "used_structured_data": False,
        }

    return {
        "answer": (
            "I do not have enough source-backed evidence to answer that confidently. "
            "The available documents or tables may not cover this topic."
        ),
        "citations": [],
        "confidence": "low",
        "used_structured_data": bool(state.get("used_structured_data")),
    }


def _retrieve_contexts(question: str) -> list[dict[str, Any]]:
    query_embedding = embed_text(question)
    contexts = db.search_chunks(query_embedding, settings.top_k)
    supported = [
        item for item in contexts if float(item["similarity"]) >= settings.min_retrieval_similarity
    ]
    return [
        item for item in supported
        if not contains_prompt_injection(item["chunk_text"])
    ]


def _format_csv_answer(state: AssistantState) -> str:
    result = state.get("tool_result") or {}
    if not result.get("ok"):
        return ""
    operation = result.get("operation")
    metric = result.get("metric")
    group_by = result.get("group_by")

    if operation == "sum" and metric == "sales" and group_by == "branch":
        rows = result.get("rows") or []
        if not rows:
            return ""
        top = rows[0]
        return f"{top['branch']} has the highest total sales at {_format_number(top['sales'], 0)}."

    if operation == "avg" and metric == "aging_days":
        return f"The average inventory aging is {_format_number(result.get('value'), 1)} days."

    if operation == "top_n":
        rows = result.get("rows") or []
        if not rows:
            return ""
        lines = []
        for index, row in enumerate(rows, start=1):
            sku = row.get("sku", "Unknown SKU")
            aging = _format_number(row.get("aging_days"), 0)
            branch = row.get("branch", "unknown branch")
            lines.append(f"{index}. {sku} - {aging} days ({branch})")
        return "Top SKUs by aging days:\n" + "\n".join(lines)

    if result.get("tables"):
        lines = [
            f"{table['source']}: {table['row_count']} rows; columns: {', '.join(table['columns'])}"
            for table in result["tables"]
        ]
        return "Available CSV table metadata:\n" + "\n".join(lines)

    return ""


def _format_doc_answer(state: AssistantState) -> str:
    contexts = state.get("contexts") or []
    if not contexts:
        return ""
    return mask_sensitive(generate_grounded_answer(state["question"], contexts))


def _doc_confidence(contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "low"
    top_score = float(contexts[0]["similarity"])
    return "high" if top_score >= 0.35 else "medium"


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for citation in citations:
        key = (
            citation.get("source"),
            citation.get("location"),
            citation.get("snippet"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(citation)
    return unique


def build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("metadata", metadata_node)
    graph.add_node("route", route_node)
    graph.add_node("csv_tool_planner", csv_tool_planner_node)
    graph.add_node("csv_tool", csv_tool_node)
    graph.add_node("document_retrieval", document_retrieval_node)
    graph.add_node("hybrid", hybrid_node)
    graph.add_node("merge_evidence", merge_evidence_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("guardrail")
    graph.add_edge("guardrail", "metadata")
    graph.add_edge("metadata", "route")
    graph.add_conditional_edges(
        "route",
        route_edge,
        {
            "blocked": "answer",
            "clarify": "answer",
            "abstain": "merge_evidence",
            "csv_tools": "csv_tool_planner",
            "document_rag": "document_retrieval",
            "hybrid": "hybrid",
        },
    )
    graph.add_edge("csv_tool_planner", "csv_tool")
    graph.add_edge("csv_tool", "merge_evidence")
    graph.add_edge("document_retrieval", "merge_evidence")
    graph.add_edge("hybrid", "merge_evidence")
    graph.add_edge("merge_evidence", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


ASSISTANT_GRAPH = build_graph()


def run_graph(question: str) -> dict[str, Any]:
    result = ASSISTANT_GRAPH.invoke(
        {
            "question": question,
            "table_metadata": [],
            "contexts": [],
            "evidence": [],
            "citations": [],
            "used_structured_data": False,
        }
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "confidence": result.get("confidence", "low"),
        "used_structured_data": result.get("used_structured_data", False),
        "route": result.get("route", "unknown"),
        "tool_request": result.get("tool_request"),
    }
