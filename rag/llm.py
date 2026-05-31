from __future__ import annotations

from rag.config import settings


def _extractive_answer(question: str, contexts: list[dict]) -> str:
    bullets = []
    for context in contexts[:3]:
        snippet = context["snippet"].strip()
        if snippet:
            bullets.append(f"- {snippet}")
    if not bullets:
        return "I could not find enough source-backed information to answer that."
    return (
        "Based on the retrieved sources, here is the supported answer:\n"
        + "\n".join(bullets)
    )


def generate_grounded_answer(question: str, contexts: list[dict]) -> str:
    if not settings.openai_api_key:
        return _extractive_answer(question, contexts)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        context_block = "\n\n".join(
            f"Source: {item['source']} ({item.get('page_or_row')})\nSnippet: {item['snippet']}"
            for item in contexts
        )
        response = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided snippets. If the snippets do not support "
                        "the answer, say that the documents do not contain enough information. "
                        "Keep the answer concise and do not invent details."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nSnippets:\n{context_block}",
                },
            ],
        )
        return response.choices[0].message.content or _extractive_answer(question, contexts)
    except Exception:
        return _extractive_answer(question, contexts)
