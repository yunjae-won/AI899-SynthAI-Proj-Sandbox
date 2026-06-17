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


def _repair_to_object(text: str) -> str | None:
    """Best-effort extraction of a single JSON object, string-aware.

    Handles the dominant real-world failure modes observed from Qwen-class
    models under an instruct/JSON protocol: a leading prose/markdown preamble, a
    stray ```` ``` ```` fence, and&mdash;most commonly&mdash;a **missing outer
    closing brace** (the model writes ``{"action": {...}`` and forgets the final
    ``}``). Scans with string/escape awareness; returns the first balanced object
    if present, otherwise closes any open string and appends the missing braces.
    Returns None if there is no ``{`` at all.
    """
    s = re.sub(r"```$", "", re.sub(r"^```(?:json)?", "", text.strip()).strip()).strip()
    i = s.find("{")
    if i < 0:
        return None
    s = s[i:]
    depth = 0
    in_str = False
    esc = False
    for j, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[: j + 1]          # first complete, balanced object
    if in_str:                                  # unterminated string
        s += '"'
    return s + ("}" * depth) if depth > 0 else s


def parse_json(raw: str, *, fallback: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Parse LLM output as a JSON **object**, always returning a dict.

    Tries, in order: the fence-stripped text, the first balanced ``{...}``, and a
    repaired object (preamble/fence strip + brace-balancing). A bare JSON string,
    array, or number is rejected so callers can always rely on ``.get(...)``
    (avoids ``'str' object has no attribute 'get'``). Returns ``fallback`` (or
    ``{}``) only if every attempt fails.
    """
    cleaned = _strip_fences(raw)
    for candidate in (cleaned, _first_json_object(cleaned), _repair_to_object(raw)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
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
