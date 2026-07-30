"""LM Studio client for NL -> query-args parsing.

Uses the OpenAI-compatible HTTP API exposed by LM Studio at
http://127.0.0.1:1234/v1 (or LMSTUDIO_BASE_URL). Falls back gracefully
when the server is unreachable so the API stays usable offline.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234")
DEFAULT_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen2.5-1.5b-instruct")

_SYSTEM_PROMPT = """You translate natural-language materials-science questions into JSON filters.
You ONLY reply with a single JSON object (no prose, no markdown fences).
Schema (all fields optional except 'limit'):
{
  "elements_any": [string, ...],      // elements that must be present (e.g. ["W","Ti"])
  "elements_all": [string, ...],      // elements ALL required (often same as any)
  "max_formation_energy_per_atom": number,
  "min_formation_energy_per_atom": number,
  "limit": integer (1..50),
  "sort_by": "energy_asc" | "energy_desc"
}
Always set "limit" (default 10). No comments, no extra text."""


def is_available(base_url: str = DEFAULT_BASE_URL, timeout: float = 1.5) -> bool:
    """Lightweight liveness check for LM Studio /v1/models."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def list_models(base_url: str = DEFAULT_BASE_URL, timeout: float = 2.0) -> list[str]:
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return [m.get("id") for m in payload.get("data", []) if m.get("id")]
    except Exception:
        return []


def parse_with_llm(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    """Ask the local LLM to translate a prompt into filter JSON.

    Returns the parsed dict on success, or None on any failure (timeout,
    bad JSON, missing fields, server error) so the caller can fallback to
    the rule-based parser.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 200,
    }
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"].strip()
        # strip code-fence if the model wrapped the JSON
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None
        parsed.setdefault("limit", 10)
        # bound limit to a sane range
        try:
            parsed["limit"] = max(1, min(50, int(parsed["limit"])))
        except (TypeError, ValueError):
            parsed["limit"] = 10
        return parsed
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
            json.JSONDecodeError, KeyError, ValueError, IndexError):
        return None
