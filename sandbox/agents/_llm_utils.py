"""Shared helpers: robust JSON extraction + a system+user call wrapper.

Models sometimes wrap JSON in ```json fences or add preamble; we strip both.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _first_json_object(text: str) -> str:
    """Extract the first balanced { ... } substring."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return text


def parse_json(raw: str, *, fallback: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Parse LLM output as JSON. If it fails, return ``fallback`` (or {})."""
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        return json.loads(_first_json_object(cleaned))
    except Exception:
        return dict(fallback) if fallback else {}


def call_llm(llm, system: str, user: str) -> str:
    """One-shot call. Returns the model's text content."""
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        # Anthropic may return list-of-blocks; join text blocks.
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        content = "".join(parts)
    return str(content)
