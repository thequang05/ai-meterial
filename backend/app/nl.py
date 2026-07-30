"""NL prompt parser + candidate filter.

Mirrors frontend/src/api/nlQuery.ts mockParse() so the backend
returns the same shape the FE mock would. When LM Studio is
wired in, swap this module for a tool-use call.
"""
from __future__ import annotations

import re
from typing import Any

_KNOWN_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
}


def parse_prompt(prompt: str, default_limit: int = 10) -> dict[str, Any]:
    found: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b([A-Z][a-z]?)\b", prompt):
        sym = match.group(1)
        if sym in _KNOWN_ELEMENTS and sym not in seen:
            found.append(sym)
            seen.add(sym)

    args: dict[str, Any] = {"limit": default_limit, "order": "asc"}
    limit_match = re.search(r"(?:top|first)\s*(\d+)", prompt, re.IGNORECASE)
    if limit_match:
        args["limit"] = int(limit_match.group(1))

    if found:
        args["include_elements"] = found

    max_match = re.search(r"(?:below|under|<)\s*(-?\d+\.?\d*)", prompt, re.IGNORECASE)
    min_match = re.search(r"(?:above|>|greater than)\s*(-?\d+\.?\d*)", prompt, re.IGNORECASE)
    if max_match:
        args["max_energy"] = float(max_match.group(1))
    if min_match:
        args["min_energy"] = float(min_match.group(1))

    return args


# Local MP-style snapshot used to satisfy NL queries end-to-end.
# Same 9 entries the FE mock uses, so both views stay consistent.
_MP_DB: list[dict[str, Any]] = [
    {"uid": "mp-1",  "formula": "WC",        "elements": ["W", "C"],          "formation_energy_per_atom": -0.40},
    {"uid": "mp-2",  "formula": "W2C",       "elements": ["W", "C"],          "formation_energy_per_atom": -0.30},
    {"uid": "mp-3",  "formula": "TiC",       "elements": ["Ti", "C"],         "formation_energy_per_atom": -0.85},
    {"uid": "mp-4",  "formula": "Ti3NbWC5",  "elements": ["Ti", "Nb", "W", "C"], "formation_energy_per_atom": -0.35},
    {"uid": "mp-5",  "formula": "Ti3VWC5",   "elements": ["Ti", "V", "W", "C"],  "formation_energy_per_atom": -0.30},
    {"uid": "mp-6",  "formula": "NbC",       "elements": ["Nb", "C"],         "formation_energy_per_atom": -0.55},
    {"uid": "mp-7",  "formula": "VC",        "elements": ["V", "C"],          "formation_energy_per_atom": -0.45},
    {"uid": "mp-8",  "formula": "ZrC",       "elements": ["Zr", "C"],         "formation_energy_per_atom": -0.90},
    {"uid": "mp-9",  "formula": "TaC",       "elements": ["Ta", "C"],         "formation_energy_per_atom": -0.60},
]


def filter_candidates(args: dict[str, Any]) -> list[dict[str, Any]]:
    include = args.get("include_elements") or []
    max_e = args.get("max_energy")
    min_e = args.get("min_energy")
    limit = args.get("limit", 10)

    results = []
    for c in _MP_DB:
        if include and not all(e in c["elements"] for e in include):
            continue
        if max_e is not None and c["formation_energy_per_atom"] > max_e:
            continue
        if min_e is not None and c["formation_energy_per_atom"] < min_e:
            continue
        results.append(c)
    return results[:limit] if limit else results