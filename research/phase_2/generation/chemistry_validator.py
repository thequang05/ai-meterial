"""Composition-level screening for generated material candidates.

This is deliberately a conservative screen.  It can establish only that a
composition is absent from the local Materials Project export and that
pymatgen can assign a charge-neutral oxidation-state combination.  It does
not establish synthesizability, a stable phase, or high-temperature service.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pymatgen.core import Composition


CompositionKey = tuple[tuple[str, float], ...]


def composition_key(composition: Composition) -> CompositionKey:
    """Canonical, reduced composition key independent of element order."""
    reduced = composition.reduced_composition
    return tuple(sorted(
        (str(element), round(float(amount), 8))
        for element, amount in reduced.get_el_amt_dict().items()
    ))


@dataclass(frozen=True)
class ChemicalValidation:
    accepted: bool
    reduced_formula: str
    oxidation_states: dict[str, float] | None
    reason: str | None = None


class ChemicalValidator:
    """Reject known or charge-unbalanced compositions before GNN scoring."""

    def __init__(self, known_compositions: Iterable[CompositionKey]):
        self.known_compositions = set(known_compositions)

    @classmethod
    def from_materials_csv(cls, path: Path) -> "ChemicalValidator":
        keys: set[CompositionKey] = set()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                formula = (row.get("formula") or "").strip()
                if not formula:
                    continue
                try:
                    keys.add(composition_key(Composition(formula)))
                except (ValueError, TypeError):
                    # A malformed source formula cannot be used as a novelty
                    # reference, but should not prevent the pipeline starting.
                    continue
        return cls(keys)

    def validate(
        self,
        formula: str,
        required_elements: Iterable[str] = (),
        allowed_elements: Iterable[str] | None = None,
    ) -> ChemicalValidation:
        try:
            composition = Composition(formula)
        except (ValueError, TypeError) as exc:
            return ChemicalValidation(False, formula, None, f"invalid_formula:{exc}")

        reduced = composition.reduced_composition
        reduced_formula = reduced.reduced_formula
        present = {str(element) for element in reduced.elements}
        missing = sorted(set(required_elements) - present)
        if missing:
            return ChemicalValidation(
                False, reduced_formula, None, f"missing_required_elements:{','.join(missing)}"
            )

        if allowed_elements is not None:
            allowed = set(allowed_elements)
            outside_domain = sorted(present - allowed)
            if outside_domain:
                return ChemicalValidation(
                    False,
                    reduced_formula,
                    None,
                    f"outside_allowed_domain:{','.join(outside_domain)}",
                )

        if composition_key(reduced) in self.known_compositions:
            return ChemicalValidation(False, reduced_formula, None, "known_composition")

        # oxi_state_guesses returns only charge-neutral assignments.  Keeping
        # the first result records one plausible assignment for audit output.
        guesses = reduced.oxi_state_guesses(max_sites=-1)
        if not guesses:
            return ChemicalValidation(False, reduced_formula, None, "no_charge_neutral_oxidation_state")
        states = {str(element): float(state) for element, state in guesses[0].items()}
        return ChemicalValidation(True, reduced_formula, states)
