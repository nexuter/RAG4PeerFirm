"""Text cleanup helpers for peer-firm search."""

import re
from typing import Iterable


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def join_parts(parts: Iterable[str]) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return normalize_text(" ".join(cleaned))
