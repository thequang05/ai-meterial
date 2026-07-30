"""
Stage 0: NL → Candidate composition → neo4j query → candidates JSON.

Inputs:
  --prompt "natural language query"
  --output candidates.json
  --limit N (max materials to return)
  --model qwen2.5-14b-instruct (LM Studio model id)

Outputs:
  JSON list of {formula, elements, formation_energy_per_atom,
                num_atoms, num_edges, source_uid}

Pipeline position:
  Stage 0 → [Stage 1 GraphVAE → Stage 2 GNN → Stage 3 CHGNet → Stage 4 ML eval]

Design:
  1) Try LM Studio tool-use call (best for Qwen/Llama).
  2) Fall back to rule-based NL parsing if model mis-calls tools.
  3) Both paths converge to a unified query against MaterialsRetriever.

Usage:
  python nl_to_candidates.py --prompt "Find W-Ti-C with E_f below -1.5"
  python nl_to_candidates.py --prompt "top 5 carbides" --limit 5
  python nl_to_candidates.py --prompt "W-V-Zr-Nb-C refractory"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval import MaterialsRetriever  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Tool schema exposed to the LLM
# ──────────────────────────────────────────────────────────────────────
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "find_by_formation_energy",
            "description": (
                "Retrieve materials from the Materials knowledge graph filtered "
                "by formation energy (eV/atom) and chemistry. Lower (more "
                "negative) = more stable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_energy": {
                        "type": "number",
                        "description": (
                            "Upper bound on formation_energy_per_atom. "
                            "For 'low formation energy' use max_energy=-1.0."
                        ),
                    },
                    "min_energy": {
                        "type": "number",
                        "description": "Lower bound on formation_energy_per_atom.",
                    },
                    "include_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Material must contain ALL of these elements.",
                    },
                    "exclude_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Material must contain NONE of these.",
                    },
                    "only_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict to a chemical subsystem (subset).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of materials to return.",
                    },
                    "order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "'asc' = most stable first.",
                    },
                },
                "required": ["limit"],
            },
        },
    }
]


SYSTEM_PROMPT = """You are an assistant for a materials discovery pipeline.
Convert the user's natural-language chemistry request into a single
tool call to `find_by_formation_energy`. Only set the parameters that
match what the user asked for. If they don't specify a value, omit it.

Element symbols are case-sensitive (e.g. 'W', 'Ti', 'C').
Formation energy is in eV/atom; negative = more stable.

Examples:
- "low energy W-Ti-C" -> max_energy=-1.0, include_elements=["W","Ti","C"]
- "top 5 Nb carbides" -> include_elements=["Nb","C"], limit=5
- "stable W-V-Zr-Nb-C refractory" ->
   only_elements=["W","V","Zr","Nb","C"], max_energy=-0.5, limit=20
"""


# ──────────────────────────────────────────────────────────────────────
# LM Studio HTTP call (OpenAI-compatible)
# ──────────────────────────────────────────────────────────────────────
def call_lm_studio(
    prompt: str,
    model: str,
    base_url: str,
    timeout: int = 60,
) -> dict[str, Any] | None:
    """Call LM Studio chat-completions endpoint with tool-calling."""
    import requests

    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "tools": TOOL_SCHEMA,
            "tool_choice": "auto",
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    choice = body["choices"][0]["message"]
    if choice.get("tool_calls"):
        tc = choice["tool_calls"][0]
        if tc["function"]["name"] == "find_by_formation_energy":
            return json.loads(tc["function"]["arguments"])
    return None


# ──────────────────────────────────────────────────────────────────────
# Rule-based fallback for Gemma-4 quantized (tool-use is unreliable)
# ──────────────────────────────────────────────────────────────────────
ELEMENT_PATTERN = re.compile(r"\b([A-Z][a-z]?)\b")
KNOWN_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U",
}

CARBON_KEYWORDS = ["carbide", "carbon", "with c", "with -c", "-c"]
NITRIDE_KEYWORDS = ["nitride", "with n", "-n"]
OXIDE_KEYWORDS = ["oxide", "with o", "-o"]

ENERGY_PATTERNS = [
    (re.compile(r"below\s*(-?\d+\.?\d*)"), "max"),
    (re.compile(r"under\s*(-?\d+\.?\d*)"), "max"),
    (re.compile(r"less\s*than\s*(-?\d+\.?\d*)"), "max"),
    (re.compile(r"<\s*(-?\d+\.?\d*)"), "max"),
    (re.compile(r"above\s*(-?\d+\.?\d*)"), "min"),
    (re.compile(r"greater\s*than\s*(-?\d+\.?\d*)"), "min"),
    (re.compile(r">\s*(-?\d+\.?\d*)"), "min"),
    (re.compile(r"between\s*(-?\d+\.?\d*)\s*and\s*(-?\d+\.?\d*)"), "range"),
]

LIMIT_PATTERNS = [
    re.compile(r"top\s*(\d+)"),
    re.compile(r"first\s*(\d+)"),
    re.compile(r"(\d+)\s*(?:best|most\s+stable|candidates|results)"),
    re.compile(r"limit\s*(?:to\s*)?(\d+)"),
]


def _extract_elements(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in ELEMENT_PATTERN.finditer(text):
        sym = match.group(1)
        if sym in KNOWN_ELEMENTS and sym not in seen:
            found.append(sym)
            seen.add(sym)
    return found


def _extract_energy(text: str) -> tuple[float | None, float | None]:
    text_low = text.lower()
    for pat, kind in ENERGY_PATTERNS:
        m = pat.search(text_low)
        if not m:
            continue
        if kind == "max":
            return None, float(m.group(1))
        if kind == "min":
            return float(m.group(1)), None
        if kind == "range":
            return float(m.group(1)), float(m.group(2))
    return None, None


def _extract_limit(text: str, default: int) -> int:
    text_low = text.lower()
    for pat in LIMIT_PATTERNS:
        m = pat.search(text_low)
        if m:
            return int(m.group(1))
    return default


def _has_keyword(text: str, keywords: list[str]) -> bool:
    text_low = text.lower()
    return any(k in text_low for k in keywords)


def rule_based_parse(prompt: str, default_limit: int = 10) -> dict[str, Any]:
    elements = _extract_elements(prompt)
    min_e, max_e = _extract_energy(prompt)
    limit = _extract_limit(prompt, default_limit)

    args: dict[str, Any] = {"limit": limit, "order": "asc"}

    if elements:
        args["include_elements"] = elements
    if min_e is not None:
        args["min_energy"] = min_e
    if max_e is not None:
        args["max_energy"] = max_e

    if not elements:
        refractory_set = ["W", "Ti", "V", "Nb", "Zr", "Ta", "Hf", "Mo", "Re"]
        if _has_keyword(prompt, refractory_set) or _has_keyword(
            prompt, ["refractory", "high-temperature", "high temp"]
        ):
            args["only_elements"] = refractory_set + ["C", "N"]

    return args


# ──────────────────────────────────────────────────────────────────────
# Stage 0 orchestrator
# ──────────────────────────────────────────────────────────────────────
def stage0(
    prompt: str,
    limit: int,
    model: str,
    base_url: str,
    use_llm: bool,
) -> list[dict[str, Any]]:
    args: dict[str, Any] | None = None
    method = "rule"

    if use_llm:
        try:
            args = call_lm_studio(prompt, model=model, base_url=base_url)
            if args is not None:
                method = "llm_tool_use"
                args.setdefault("limit", limit)
                args.setdefault("order", "asc")
        except Exception as exc:
            print(f"[stage0] LM Studio call failed: {exc!r}", file=sys.stderr)

    if args is None:
        args = rule_based_parse(prompt, default_limit=limit)

    print(f"[stage0] method={method}")
    print(f"[stage0] parsed args={args}")

    try:
        with MaterialsRetriever() as r:
            rows = r.find_by_formation_energy(**args)
    except Exception as exc:
        print(f"[stage0] Neo4j unreachable: {exc!r}", file=sys.stderr)
        print(
            "[stage0] Hint: start Neo4j (brew services start neo4j) "
            "or run without --no-llm offline mode.",
            file=sys.stderr,
        )
        rows = []

    return rows, method, args, model


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="Stage 0: NL → candidates JSON")
    p.add_argument("--prompt", required=True, help="Natural-language query")
    p.add_argument("--output", default="candidates.json", help="Output JSON path")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--model", default=os.environ.get("LMSTUDIO_MODEL", "qwen2.5-1.5b-instruct"))
    p.add_argument("--base-url", default=os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234"))
    p.add_argument("--no-llm", action="store_true", help="Skip LM Studio, use rules only")
    args = p.parse_args()

    rows, method, parsed_args, model_used = stage0(
        prompt=args.prompt,
        limit=args.limit,
        model=args.model,
        base_url=args.base_url,
        use_llm=not args.no_llm,
    )

    candidates = []
    for row in rows:
        candidates.append({
            "uid": row.get("uid"),
            "formula": row.get("formula"),
            "elements": row.get("elements", []),
            "formation_energy_per_atom": row.get("formation_energy_per_atom"),
            "num_atoms": row.get("num_atoms"),
            "num_edges": row.get("num_edges"),
            "graph_index": row.get("graph_index"),
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "prompt": args.prompt,
            "method": method,
            "llm_model": model_used if method == "llm_tool_use" else None,
            "parsed_args": parsed_args,
            "n_candidates": len(candidates),
            "candidates": candidates,
        }, f, indent=2, default=str)

    print(f"[stage0] wrote {len(candidates)} candidates → {out_path}")
    for c in candidates[:5]:
        print(
            f"  {c['formula']:30s}  E_f={c['formation_energy_per_atom']:+.4f} eV/atom"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())