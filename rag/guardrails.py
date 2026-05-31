import re


INJECTION_PATTERNS = [
    re.compile(r"\bignore (all )?(previous|prior|above) instructions\b", re.I),
    re.compile(r"\bdo not cite\b", re.I),
    re.compile(r"\bhide (the )?(source|sources|citation|citations)\b", re.I),
    re.compile(r"\breveal (system|developer) prompt\b", re.I),
    re.compile(r"\boverride (the )?(policy|rules|instructions)\b", re.I),
]

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")


def contains_prompt_injection(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in INJECTION_PATTERNS)


def mask_sensitive(text: str) -> str:
    masked = EMAIL_PATTERN.sub("[masked-email]", text or "")
    return PHONE_PATTERN.sub("[masked-phone]", masked)


def blocked_injection_response() -> dict:
    return {
        "answer": (
            "I cannot follow instructions that try to bypass grounding, hide citations, "
            "or override assistant rules. Please ask a document-grounded question."
        ),
        "citations": [],
        "confidence": "blocked",
        "used_structured_data": False,
    }
