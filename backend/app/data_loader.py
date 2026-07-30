"""Load + cache candidates from the CSV report.

Single source of truth is
research/phase_2/reports/ml_thermal_eval_v1/ml_thermal_evaluation.csv
(15 rows = ML-evaluated candidates for the W-C campaign).
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schemas import Candidate

CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "research"
    / "phase_2"
    / "reports"
    / "ml_thermal_eval_v1"
    / "ml_thermal_evaluation.csv"
)

CAMPAIGN_ID = "w_c_dft_campaign_v1"


def _formula_to_elements(formula: str) -> list[str]:
    """Extract unique element symbols from a Hill-style formula string."""
    return list(dict.fromkeys(re.findall(r"[A-Z][a-z]?", formula)))


def _formula_to_num_atoms(formula: str) -> int:
    """Sum the atom counts in a Hill-style formula string."""
    total = 0
    for count in re.findall(r"[A-Z][a-z]?(\d+)", formula):
        total += int(count)
    if total == 0:
        total = len(re.findall(r"[A-Z][a-z]?", formula))
    return total


def _load_csv() -> list[dict[str, Any]]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def _load_rows_cached() -> tuple[dict, ...]:
    return tuple(_load_csv())


def reset_cache() -> None:
    """Clear the row cache (used after CSV regeneration)."""
    _load_rows_cached.cache_clear()


def load_candidates() -> list[Candidate]:
    rows = _load_rows_cached()
    candidates: list[Candidate] = []
    for row in rows:
        candidates.append(
            Candidate(
                id=f"{row['candidate_id']}",
                rank=int(row["ml_rank"]),
                formula=row["formula"],
                elements=_formula_to_elements(row["formula"]),
                num_atoms=_formula_to_num_atoms(row["formula"]),
                ef_post=float(row["gnn_formation_energy_post_chgnet_ev_per_atom"] or 0.0),
                ef_gap_surrogate=float(row["ml_estimated_hull_gap_ev_per_atom"] or 0.0),
                ef_mp_best=float(row["mp_best_formation_energy_per_atom"] or 0.0),
                status=row["thermal_proxy_status"],
            )
        )
    return candidates


def get_candidate_by_id(candidate_id: str) -> Candidate | None:
    for c in load_candidates():
        if c.id == candidate_id:
            return c
    return None