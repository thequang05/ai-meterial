"""Canonical execution-resource provenance for Quantum ESPRESSO jobs."""

from __future__ import annotations

import re
from typing import Any


HASH_RE = re.compile(r"[0-9a-f]{64}")
IDENTITY_FIELDS = (
    "execution_mode",
    "mpi_ranks",
    "omp_threads",
    "mpi_launcher_sha256",
    "mpi_program_version",
)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Execution provenance {label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Execution provenance {label} must be a positive integer"
        ) from exc
    if result < 1 or str(result) != str(value).strip():
        raise ValueError(f"Execution provenance {label} must be a positive integer")
    return result


def validate_execution_provenance(value: Any) -> dict[str, Any]:
    """Return a normalized, fail-closed serial/MPI provenance record."""

    if not isinstance(value, dict):
        raise ValueError("Execution provenance record is missing")
    mode = str(value.get("execution_mode") or "").strip().lower()
    ranks = _positive_int(value.get("mpi_ranks"), "mpi_ranks")
    omp_threads = _positive_int(value.get("omp_threads"), "omp_threads")
    launcher_path_raw = value.get("mpi_launcher_path")
    launcher_sha_raw = value.get("mpi_launcher_sha256")
    version_raw = value.get("mpi_program_version")

    if mode == "serial":
        if ranks != 1:
            raise ValueError("Serial execution provenance requires mpi_ranks=1")
        if any(item not in (None, "") for item in (
            launcher_path_raw, launcher_sha_raw, version_raw,
        )):
            raise ValueError("Serial execution provenance must not name an MPI launcher")
        launcher_path = None
        launcher_sha = None
        version = None
    elif mode == "mpi":
        if ranks < 2:
            raise ValueError("MPI execution provenance requires mpi_ranks>=2")
        launcher_path = str(launcher_path_raw or "").strip()
        launcher_sha = str(launcher_sha_raw or "").strip().lower()
        version = str(version_raw or "").strip()
        if not launcher_path:
            raise ValueError("MPI execution provenance launcher path is missing")
        if not HASH_RE.fullmatch(launcher_sha):
            raise ValueError("MPI execution provenance launcher hash is invalid")
        if not version:
            raise ValueError("MPI execution provenance version is missing")
    else:
        raise ValueError("Execution provenance mode must be 'serial' or 'mpi'")

    return {
        "execution_mode": mode,
        "mpi_ranks": ranks,
        "omp_threads": omp_threads,
        "mpi_launcher_path": launcher_path,
        "mpi_launcher_sha256": launcher_sha,
        "mpi_program_version": version,
    }


def execution_identity(value: Any) -> dict[str, Any]:
    """Return fields that must be identical across one certified campaign.

    The resolved path is intentionally recorded but excluded from identity: an
    immutable launcher may be relocated on another mounted filesystem while its
    binary hash, reported implementation/version, ranks, and thread policy stay
    identical.
    """

    normalized = validate_execution_provenance(value)
    return {field: normalized[field] for field in IDENTITY_FIELDS}


def require_same_execution_provenance(
    observed: Any, expected: Any, *, label: str,
) -> dict[str, Any]:
    normalized = validate_execution_provenance(observed)
    if execution_identity(normalized) != execution_identity(expected):
        raise ValueError(f"Execution provenance differs from {label}")
    return normalized
