from __future__ import annotations

from rag.graph import run_graph


def answer_question(question: str) -> dict:
    return run_graph(question)
