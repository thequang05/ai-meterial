"""Non-mutating chemistry filters for decoded Phase 2 candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pymatgen.core import Element


@dataclass(frozen=True)
class FilterResult:
    passed: bool
    reasons: list[str]
    metadata: dict


def symbols_from_atomic_numbers(atomic_numbers: Iterable[int]) -> list[str]:
    """Map valid atomic numbers to symbols without altering any candidate."""
    symbols: list[str] = []
    for value in atomic_numbers:
        atomic_number = int(value)
        if not 1 <= atomic_number <= 94:
            raise ValueError(f"invalid_atomic_number:{atomic_number}")
        symbols.append(Element.from_Z(atomic_number).symbol)
    return symbols


def validate_decoded_atomic_numbers(
    atomic_numbers: Iterable[int],
    original_atomic_numbers: Iterable[int],
    *,
    required_elements: Iterable[str] = (),
    allowed_elements: Iterable[str] | None = None,
    max_unique_elements: int = 10,
) -> FilterResult:
    """Validate decoded site species; reject instead of repairing them."""
    decoded = [int(value) for value in atomic_numbers]
    original = [int(value) for value in original_atomic_numbers]
    reasons: list[str] = []

    invalid = sorted({value for value in decoded if not 1 <= value <= 94})
    if invalid:
        reasons.append("invalid_atomic_number:" + ",".join(map(str, invalid)))
        return FilterResult(False, reasons, {"num_sites": len(decoded)})
    if len(decoded) != len(original):
        reasons.append(f"candidate_node_count_mismatch:{len(decoded)}!={len(original)}")
    if decoded == original:
        reasons.append("unchanged_from_prototype")

    symbols = symbols_from_atomic_numbers(decoded)
    present = set(symbols)
    if len(present) > max_unique_elements:
        reasons.append(f"too_many_unique_elements:{len(present)}>{max_unique_elements}")
    missing = sorted(set(required_elements) - present)
    if missing:
        reasons.append("missing_required_elements:" + ",".join(missing))
    if allowed_elements is not None:
        outside = sorted(present - set(allowed_elements))
        if outside:
            reasons.append("outside_allowed_domain:" + ",".join(outside))

    return FilterResult(
        not reasons,
        reasons,
        {
            "num_sites": len(decoded),
            "num_unique_elements": len(present),
            "elements": sorted(present),
            "num_substitutions": sum(a != b for a, b in zip(original, decoded)),
        },
    )
