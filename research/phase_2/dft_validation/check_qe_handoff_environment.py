"""Read-only readiness check for the Quantum ESPRESSO DFT handoff.

This utility performs only local metadata, file, hash, architecture, and free
space checks.  It never downloads anything and never starts a DFT calculation.
The only optional subprocess probes are fixed ``pw.x -help``,
``mpirun --version``, and ``mpirun -np 1 pw.x -help`` commands.  They run only
when ``--probe-executables`` is supplied explicitly.  The last command checks
that the selected MPI launcher can start the selected QE executable, but it
does not read a QE input or start an SCF/DFT calculation.

Except for an explicitly requested JSON report, the check does not create,
modify, or delete any project or scratch files.  Exit status 0 means the local
handoff prerequisites passed; exit status 2 means one or more blockers remain.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_sssp_manifest import _inspect_upf_header, _normalize_expected_md5


SCHEMA_VERSION = "qe_handoff_environment_v1"
PSEUDO_MANIFEST_SCHEMA = "qe_pseudo_manifest_v1"
DEFAULT_ELEMENTS = ["C", "Ti", "V", "Zr", "Nb", "Ta", "W"]
HASH_RE = re.compile(r"[0-9a-f]{64}")
ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
GIB = 1024**3
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _normalize_architecture(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"arm64", "arm64e", "aarch64"}:
        return "arm64"
    if normalized in {"x86_64", "amd64", "x64"}:
        return "x86_64"
    return normalized or "unknown"


def _architectures_from_description(description: str) -> list[str]:
    lowered = description.lower()
    found: list[str] = []
    if re.search(r"\b(?:arm64e?|aarch64)\b", lowered):
        found.append("arm64")
    if re.search(r"\b(?:x86[_ -]?64|amd64)\b", lowered):
        found.append("x86_64")
    return found


def _describe_executable(path: Path) -> dict[str, Any]:
    file_program = shutil.which("file")
    if not file_program:
        return {
            "description": None,
            "architectures": [],
            "inspection_error": "file_utility_not_found",
        }
    try:
        completed = subprocess.run(
            [file_program, "-b", str(path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "description": None,
            "architectures": [],
            "inspection_error": f"{type(exc).__name__}: {exc}",
        }
    description = completed.stdout.strip()
    error = completed.stderr.strip()
    if completed.returncode != 0 or not description:
        return {
            "description": description or None,
            "architectures": [],
            "inspection_error": error or f"file_exit_{completed.returncode}",
        }
    return {
        "description": description,
        "architectures": _architectures_from_description(description),
        "inspection_error": None,
    }


def _resolve_executable(requested: str) -> Path | None:
    expanded = os.path.expandvars(os.path.expanduser(requested))
    if os.sep in expanded or (os.altsep and os.altsep in expanded):
        candidate = Path(expanded).resolve()
        return candidate if candidate.is_file() else None
    located = shutil.which(expanded)
    return Path(located).resolve() if located else None


def _probe_executable(
    path: Path, kind: str, timeout_seconds: int, omp_threads: int = 1,
) -> dict[str, Any]:
    if kind == "pw":
        command = [str(path), "-help"]
    elif kind == "mpi":
        command = [str(path), "--version"]
    else:  # Internal programming error; do not allow arbitrary probe commands.
        raise ValueError(f"Unsupported executable probe kind: {kind}")
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(omp_threads),
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "status": "timeout",
            "timeout_seconds": timeout_seconds,
            "returncode": None,
            "output": str(exc)[:8000],
        }
    except OSError as exc:
        return {
            "command": command,
            "status": "launch_failed",
            "timeout_seconds": timeout_seconds,
            "returncode": None,
            "output": f"{type(exc).__name__}: {exc}"[:8000],
        }
    combined = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return {
        "command": command,
        "status": "passed" if completed.returncode == 0 else "nonzero_exit",
        "timeout_seconds": timeout_seconds,
        "returncode": completed.returncode,
        "output": combined[:8000],
    }


def _probe_mpi_qe_interoperability(
    mpi_path: Path,
    pw_path: Path,
    timeout_seconds: int,
    omp_threads: int = 1,
) -> dict[str, Any]:
    """Run the one fixed MPI/QE help-only interoperability command."""

    command = [str(mpi_path), "-np", "1", str(pw_path), "-help"]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(omp_threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "status": "timeout",
            "timeout_seconds": timeout_seconds,
            "returncode": None,
            "output": str(exc)[:8000],
        }
    except OSError as exc:
        return {
            "command": command,
            "status": "launch_failed",
            "timeout_seconds": timeout_seconds,
            "returncode": None,
            "output": f"{type(exc).__name__}: {exc}"[:8000],
        }
    combined = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return {
        "command": command,
        "status": "passed" if completed.returncode == 0 else "nonzero_exit",
        "timeout_seconds": timeout_seconds,
        "returncode": completed.returncode,
        "output": combined[:8000],
    }


def _check_mpi_qe_interoperability(
    *,
    pw_check: dict[str, Any],
    mpi_check: dict[str, Any],
    probe: bool,
    probe_timeout_seconds: int,
    omp_threads: int,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "probe_requested": probe,
        "pw_resolved_path": pw_check.get("resolved_path"),
        "mpi_resolved_path": mpi_check.get("resolved_path"),
        "command": None,
        "status": "not_requested" if not probe else "not_run_missing_executable",
        "timeout_seconds": probe_timeout_seconds,
        "returncode": None,
        "output": None,
        "starts_dft_calculation": False,
    }
    if not probe:
        _append_unique(
            warnings,
            "mpi_qe_interoperability:runtime_probe_not_requested",
        )
        return result
    if not pw_check.get("resolved_path") or not mpi_check.get("resolved_path"):
        # The executable-specific checks already identify the missing binary.
        return result

    probe_result = _probe_mpi_qe_interoperability(
        Path(str(mpi_check["resolved_path"])),
        Path(str(pw_check["resolved_path"])),
        timeout_seconds=probe_timeout_seconds,
        omp_threads=omp_threads,
    )
    result.update(probe_result)
    if probe_result["status"] != "passed":
        _append_unique(blockers, "mpi_qe_interoperability:probe_failed")
    return result


def _check_executable(
    *,
    requested: str,
    label: str,
    kind: str,
    host_architecture: str,
    probe: bool,
    probe_timeout_seconds: int,
    omp_threads: int,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    resolved = _resolve_executable(requested)
    result: dict[str, Any] = {
        "requested": requested,
        "resolved_path": str(resolved) if resolved else None,
        "exists": bool(resolved),
        "is_executable": False,
        "sha256": None,
        "file_description": None,
        "architectures": [],
        "host_architecture": host_architecture,
        "native_architecture_verified": False,
        "probe_requested": probe,
        "probe": None,
        "status": "blocked",
    }
    if resolved is None:
        _append_unique(blockers, f"{label}:not_found")
        return result
    result["is_executable"] = os.access(resolved, os.X_OK)
    if not result["is_executable"]:
        _append_unique(blockers, f"{label}:not_executable")
    try:
        result["sha256"] = _sha256(resolved)
    except OSError as exc:
        result["hash_error"] = f"{type(exc).__name__}: {exc}"
        _append_unique(blockers, f"{label}:unreadable")

    inspection = _describe_executable(resolved)
    result["file_description"] = inspection["description"]
    result["architectures"] = inspection["architectures"]
    if inspection["inspection_error"]:
        result["architecture_inspection_error"] = inspection["inspection_error"]
        _append_unique(blockers, f"{label}:architecture_unverifiable")
    elif not inspection["architectures"]:
        _append_unique(blockers, f"{label}:architecture_unverifiable")
    elif host_architecture not in inspection["architectures"]:
        _append_unique(
            blockers,
            f"{label}:not_native_for_{host_architecture}",
        )
    else:
        result["native_architecture_verified"] = True

    if probe:
        result["probe"] = _probe_executable(
            resolved,
            kind=kind,
            timeout_seconds=probe_timeout_seconds,
            omp_threads=omp_threads,
        )
        if result["probe"]["status"] != "passed":
            _append_unique(blockers, f"{label}:probe_failed")
    else:
        _append_unique(
            warnings,
            f"{label}:runtime_probe_not_requested",
        )

    label_prefix = f"{label}:"
    if not any(item.startswith(label_prefix) for item in blockers):
        result["status"] = "passed"
    return result


def _check_volume(
    *,
    path: Path,
    label: str,
    min_free_bytes: int,
    require_writable: bool,
    blockers: list[str],
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
        "is_directory": resolved.is_dir(),
        "writable_and_searchable": False,
        "total_bytes": None,
        "used_bytes": None,
        "free_bytes": None,
        "minimum_free_bytes": min_free_bytes,
        "status": "blocked",
    }
    if not resolved.is_dir():
        _append_unique(blockers, f"{label}:directory_missing")
        return result
    result["writable_and_searchable"] = os.access(
        resolved, os.W_OK | os.X_OK
    )
    if require_writable and not result["writable_and_searchable"]:
        _append_unique(blockers, f"{label}:not_writable")
    try:
        usage = shutil.disk_usage(resolved)
    except OSError as exc:
        result["disk_usage_error"] = f"{type(exc).__name__}: {exc}"
        _append_unique(blockers, f"{label}:disk_usage_unavailable")
        return result
    result.update(
        {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    )
    if usage.free < min_free_bytes:
        _append_unique(blockers, f"{label}:insufficient_free_space")
    prefix = f"{label}:"
    if not any(item.startswith(prefix) for item in blockers):
        result["status"] = "passed"
    return result


def _check_python_environment(blockers: list[str]) -> dict[str, Any]:
    version = platform.python_version()
    result: dict[str, Any] = {
        "executable": str(Path(sys.executable).resolve()),
        "implementation": platform.python_implementation(),
        "version": version,
        "minimum_supported_version": "3.10",
        "numpy_version": None,
        "ase_version": None,
        "pymatgen_version": None,
        "status": "blocked",
    }
    if sys.version_info < (3, 10):
        _append_unique(blockers, "python:version_below_3_10")
    try:
        import numpy as np

        result["numpy_version"] = str(np.__version__)
    except Exception as exc:  # Broken compiled wheels can fail during import.
        result["numpy_import_error"] = f"{type(exc).__name__}: {exc}"
        _append_unique(blockers, "python:numpy_unavailable")
    try:
        import ase
        from ase import Atoms  # noqa: F401

        result["ase_version"] = str(ase.__version__)
    except Exception as exc:
        result["ase_import_error"] = f"{type(exc).__name__}: {exc}"
        _append_unique(blockers, "python:ase_unavailable")
    try:
        from pymatgen.core import Structure  # noqa: F401

        try:
            result["pymatgen_version"] = importlib.metadata.version("pymatgen")
        except importlib.metadata.PackageNotFoundError:
            result["pymatgen_version"] = "installed_version_metadata_unavailable"
    except Exception as exc:  # Import errors can include broken compiled dependencies.
        result["pymatgen_import_error"] = f"{type(exc).__name__}: {exc}"
        _append_unique(blockers, "python:pymatgen_unavailable")
    if not any(item.startswith("python:") for item in blockers):
        result["status"] = "passed"
    return result


def _check_dataset_file(
    path: Path,
    *,
    label: str,
    expected_kind: str,
    blockers: list[str],
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "readable": False,
        "size_bytes": None,
        "lightweight_format_check": False,
        "full_hash_or_scan_performed": False,
        "status": "blocked",
    }
    if not resolved.is_file():
        _append_unique(blockers, f"dataset:{label}_missing")
        return result
    result["readable"] = os.access(resolved, os.R_OK)
    if not result["readable"]:
        _append_unique(blockers, f"dataset:{label}_unreadable")
        return result
    try:
        result["size_bytes"] = resolved.stat().st_size
        with resolved.open("rb") as handle:
            prefix = handle.read(4096)
    except OSError as exc:
        result["read_error"] = f"{type(exc).__name__}: {exc}"
        _append_unique(blockers, f"dataset:{label}_unreadable")
        return result
    if result["size_bytes"] <= 0:
        _append_unique(blockers, f"dataset:{label}_empty")
        return result

    if expected_kind == "json":
        stripped = prefix.lstrip()
        valid = stripped.startswith((b"[", b"{"))
    elif expected_kind == "csv":
        first_line = prefix.splitlines()[0] if prefix.splitlines() else b""
        valid = b"," in first_line and bool(first_line.strip())
    elif expected_kind == "sqlite":
        valid = prefix.startswith(b"SQLite format 3\x00")
    else:
        raise ValueError(f"Unsupported dataset kind: {expected_kind}")
    result["lightweight_format_check"] = valid
    if not valid:
        _append_unique(blockers, f"dataset:{label}_format_mismatch")
    if not any(item.startswith(f"dataset:{label}_") for item in blockers):
        result["status"] = "passed"
    return result


def _safe_manifest_filename(value: Any) -> str | None:
    filename = str(value or "").strip()
    candidate = Path(filename)
    if not filename or candidate.is_absolute() or candidate.parent != Path("."):
        return None
    return filename


def _check_sssp(
    *,
    manifest_path: Path | None,
    pseudo_dir: Path | None,
    metadata_path: Path | None,
    required_elements: list[str],
    blockers: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "manifest_path": str(manifest_path.expanduser().resolve())
        if manifest_path
        else None,
        "manifest_sha256": None,
        "pseudo_dir": None,
        "metadata_path": None,
        "required_elements": required_elements,
        "verified_elements": [],
        "status": "blocked",
    }
    if manifest_path is None:
        _append_unique(blockers, "sssp:manifest_not_provided")
        return result
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        _append_unique(blockers, "sssp:manifest_missing")
        return result
    try:
        result["manifest_sha256"] = _sha256(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["manifest_error"] = f"{type(exc).__name__}: {exc}"
        _append_unique(blockers, "sssp:manifest_invalid")
        return result
    if not isinstance(manifest, dict):
        _append_unique(blockers, "sssp:manifest_invalid")
        return result

    expected_fields = {
        "schema_version": PSEUDO_MANIFEST_SCHEMA,
        "library": "SSSP PBE Precision",
        "functional": "PBE",
        "relativistic": "scalar_relativistic",
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != expected:
            _append_unique(blockers, f"sssp:manifest_{key}_mismatch")
    if not str(manifest.get("library_version") or "").strip():
        _append_unique(blockers, "sssp:library_version_missing")
    if manifest.get("licenses_acknowledged_by_user") is not True:
        _append_unique(blockers, "sssp:licenses_not_acknowledged")

    resolved_pseudo_dir: Path | None = None
    if pseudo_dir is not None:
        resolved_pseudo_dir = pseudo_dir.expanduser().resolve()
    elif manifest.get("pseudo_dir_at_manifest_creation"):
        resolved_pseudo_dir = Path(
            str(manifest["pseudo_dir_at_manifest_creation"])
        ).expanduser().resolve()
    result["pseudo_dir"] = (
        str(resolved_pseudo_dir) if resolved_pseudo_dir else None
    )
    if resolved_pseudo_dir is None:
        _append_unique(blockers, "sssp:pseudo_dir_not_provided")
    elif not resolved_pseudo_dir.is_dir():
        _append_unique(blockers, "sssp:pseudo_dir_missing")

    resolved_metadata: Path | None = None
    official_metadata: dict[str, Any] | None = None
    if metadata_path is not None:
        resolved_metadata = metadata_path.expanduser().resolve()
    elif manifest.get("source_metadata"):
        resolved_metadata = Path(str(manifest["source_metadata"])).expanduser().resolve()
    result["metadata_path"] = str(resolved_metadata) if resolved_metadata else None
    expected_metadata_hash = str(manifest.get("source_metadata_sha256") or "").lower()
    if resolved_metadata is None:
        _append_unique(blockers, "sssp:source_metadata_not_provided")
    elif not resolved_metadata.is_file():
        _append_unique(blockers, "sssp:source_metadata_missing")
    elif not HASH_RE.fullmatch(expected_metadata_hash):
        _append_unique(blockers, "sssp:source_metadata_hash_invalid")
    else:
        try:
            actual_metadata_hash = _sha256(resolved_metadata)
            result["metadata_sha256"] = actual_metadata_hash
            if actual_metadata_hash != expected_metadata_hash:
                _append_unique(blockers, "sssp:source_metadata_hash_mismatch")
            else:
                loaded_metadata = json.loads(
                    resolved_metadata.read_text(encoding="utf-8")
                )
                if not isinstance(loaded_metadata, dict):
                    _append_unique(blockers, "sssp:source_metadata_invalid")
                else:
                    official_metadata = loaded_metadata
        except (OSError, json.JSONDecodeError) as exc:
            result["metadata_hash_error"] = f"{type(exc).__name__}: {exc}"
            _append_unique(blockers, "sssp:source_metadata_invalid")

    entries = manifest.get("elements")
    if not isinstance(entries, dict):
        _append_unique(blockers, "sssp:elements_invalid")
        entries = {}
    per_element: dict[str, Any] = {}
    for symbol in required_elements:
        entry_result: dict[str, Any] = {
            "filename": None,
            "expected_sha256": None,
            "actual_sha256": None,
            "expected_original_md5": None,
            "actual_md5": None,
            "official_identity_verified": False,
            "upf_header_verified": False,
            "cutoffs_valid": False,
            "status": "blocked",
        }
        per_element[symbol] = entry_result
        entry = entries.get(symbol)
        if not isinstance(entry, dict):
            _append_unique(blockers, f"sssp:{symbol}_entry_missing")
            continue
        filename = _safe_manifest_filename(entry.get("filename"))
        expected_hash = str(entry.get("sha256") or "").lower()
        entry_result["filename"] = filename
        entry_result["expected_sha256"] = expected_hash or None
        try:
            ecutwfc = float(entry.get("ecutwfc_ry"))
            ecutrho = float(entry.get("ecutrho_ry"))
            cutoffs_valid = (
                math.isfinite(ecutwfc)
                and math.isfinite(ecutrho)
                and ecutwfc > 0
                and ecutrho >= ecutwfc
            )
        except (TypeError, ValueError):
            cutoffs_valid = False
        entry_result["cutoffs_valid"] = cutoffs_valid
        if not cutoffs_valid:
            _append_unique(blockers, f"sssp:{symbol}_cutoffs_invalid")
        if filename is None:
            _append_unique(blockers, f"sssp:{symbol}_filename_invalid")
            continue
        if not HASH_RE.fullmatch(expected_hash):
            _append_unique(blockers, f"sssp:{symbol}_hash_invalid")
            continue
        if resolved_pseudo_dir is None or not resolved_pseudo_dir.is_dir():
            continue
        upf_path = resolved_pseudo_dir / filename
        entry_result["path"] = str(upf_path)
        if not upf_path.is_file():
            _append_unique(blockers, f"sssp:{symbol}_upf_missing")
            continue
        try:
            actual_hash = _sha256(upf_path)
        except OSError as exc:
            entry_result["hash_error"] = f"{type(exc).__name__}: {exc}"
            _append_unique(blockers, f"sssp:{symbol}_upf_unreadable")
            continue
        entry_result["actual_sha256"] = actual_hash
        if actual_hash != expected_hash:
            _append_unique(blockers, f"sssp:{symbol}_hash_mismatch")
            continue
        expected_md5 = _normalize_expected_md5(entry.get("original_md5"))
        entry_result["expected_original_md5"] = expected_md5 or None
        if not re.fullmatch(r"[0-9a-f]{32}", expected_md5):
            _append_unique(blockers, f"sssp:{symbol}_original_md5_invalid")
        else:
            try:
                actual_md5 = _md5(upf_path)
                entry_result["actual_md5"] = actual_md5
                if actual_md5 != expected_md5:
                    _append_unique(blockers, f"sssp:{symbol}_original_md5_mismatch")
            except OSError as exc:
                entry_result["md5_error"] = f"{type(exc).__name__}: {exc}"
                _append_unique(blockers, f"sssp:{symbol}_upf_unreadable")

        try:
            header = _inspect_upf_header(upf_path, symbol)
            entry_result["upf_header"] = header
            attested_header = {
                "element": str(entry.get("upf_header_element") or "").strip(),
                "functional": str(
                    entry.get("upf_header_functional") or ""
                ).strip(),
                "relativistic": str(
                    entry.get("upf_header_relativistic") or ""
                ).strip(),
            }
            if attested_header != header:
                _append_unique(blockers, f"sssp:{symbol}_header_attestation_mismatch")
            else:
                entry_result["upf_header_verified"] = True
        except (OSError, ValueError) as exc:
            entry_result["upf_header_error"] = f"{type(exc).__name__}: {exc}"
            _append_unique(blockers, f"sssp:{symbol}_upf_header_invalid")

        official = official_metadata.get(symbol) if official_metadata else None
        if not isinstance(official, dict):
            _append_unique(blockers, f"sssp:{symbol}_official_metadata_missing")
        else:
            official_md5 = _normalize_expected_md5(
                official.get("md5") or official.get("checksum")
            )
            try:
                official_ecutwfc = float(official["cutoff"])
                official_ecutrho = (
                    official_ecutwfc * float(official["dual"])
                    if "dual" in official
                    else float(official["ecutrho"])
                )
            except (KeyError, TypeError, ValueError):
                _append_unique(blockers, f"sssp:{symbol}_official_cutoffs_invalid")
            else:
                if (
                    str(official.get("filename") or "").strip() != filename
                    or official_md5 != expected_md5
                    or not math.isclose(
                        ecutwfc, official_ecutwfc, rel_tol=0, abs_tol=1e-10
                    )
                    or not math.isclose(
                        ecutrho, official_ecutrho, rel_tol=0, abs_tol=1e-8
                    )
                ):
                    _append_unique(blockers, f"sssp:{symbol}_official_identity_mismatch")
                else:
                    entry_result["official_identity_verified"] = True

        if cutoffs_valid and not any(
            blocker.startswith(f"sssp:{symbol}_") for blocker in blockers
        ):
            entry_result["status"] = "passed"
            result["verified_elements"].append(symbol)
    result["elements"] = per_element
    if not any(item.startswith("sssp:") for item in blockers):
        result["status"] = "passed"
    return result


def _validate_required_elements(elements: list[str]) -> list[str]:
    if not elements or len(elements) != len(set(elements)):
        raise ValueError("required_elements must be non-empty and unique")
    for symbol in elements:
        if not ELEMENT_RE.fullmatch(symbol):
            raise ValueError(f"Invalid element symbol: {symbol!r}")
    return elements


def run_preflight(
    *,
    project_root: Path,
    scratch_dir: Path,
    min_free_gib: float,
    pw_executable: str,
    mpi_executable: str,
    probe_executables: bool,
    probe_timeout_seconds: int,
    raw_json_path: Path,
    materials_csv_path: Path,
    structure_cache_path: Path,
    pseudo_manifest_path: Path | None,
    pseudo_dir: Path | None,
    sssp_metadata_path: Path | None,
    required_elements: list[str],
    mpi_ranks: int = 2,
    omp_threads: int = 1,
    host_machine: str | None = None,
) -> dict[str, Any]:
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("min_free_gib must be finite and non-negative")
    if not 1 <= probe_timeout_seconds <= 30:
        raise ValueError("probe_timeout_seconds must be between 1 and 30")
    if isinstance(mpi_ranks, bool) or not isinstance(mpi_ranks, int) or mpi_ranks < 1:
        raise ValueError("mpi_ranks must be a positive integer")
    if (
        isinstance(omp_threads, bool)
        or not isinstance(omp_threads, int)
        or omp_threads < 1
    ):
        raise ValueError("omp_threads must be a positive integer")
    required_elements = _validate_required_elements(required_elements)
    project_root = project_root.expanduser().resolve()
    host_machine = host_machine or platform.machine()
    host_architecture = _normalize_architecture(host_machine)
    blockers: list[str] = []
    warnings: list[str] = []
    if host_architecture not in {"arm64", "x86_64"}:
        _append_unique(blockers, f"host:unsupported_architecture_{host_architecture}")

    minimum_free_bytes = int(min_free_gib * GIB)
    pw_check = _check_executable(
        requested=pw_executable,
        label="pw_executable",
        kind="pw",
        host_architecture=host_architecture,
        probe=probe_executables,
        probe_timeout_seconds=probe_timeout_seconds,
        omp_threads=omp_threads,
        blockers=blockers,
        warnings=warnings,
    )
    if mpi_ranks > 1:
        mpi_check = _check_executable(
            requested=mpi_executable,
            label="mpi_executable",
            kind="mpi",
            host_architecture=host_architecture,
            probe=probe_executables,
            probe_timeout_seconds=probe_timeout_seconds,
            omp_threads=omp_threads,
            blockers=blockers,
            warnings=warnings,
        )
        interoperability_check = _check_mpi_qe_interoperability(
            pw_check=pw_check,
            mpi_check=mpi_check,
            probe=probe_executables,
            probe_timeout_seconds=probe_timeout_seconds,
            omp_threads=omp_threads,
            blockers=blockers,
            warnings=warnings,
        )
    else:
        mpi_check = {
            "requested": mpi_executable,
            "resolved_path": None,
            "probe_requested": False,
            "status": "not_required_serial",
        }
        interoperability_check = {
            "probe_requested": False,
            "pw_resolved_path": pw_check.get("resolved_path"),
            "mpi_resolved_path": None,
            "command": None,
            "status": "not_required_serial",
            "timeout_seconds": probe_timeout_seconds,
            "returncode": None,
            "output": None,
            "starts_dft_calculation": False,
        }
    checks = {
        "storage": {
            "project_volume": _check_volume(
                path=project_root,
                label="project_volume",
                min_free_bytes=minimum_free_bytes,
                require_writable=True,
                blockers=blockers,
            ),
            "scratch": _check_volume(
                path=scratch_dir,
                label="scratch",
                min_free_bytes=minimum_free_bytes,
                require_writable=True,
                blockers=blockers,
            ),
        },
        "python": _check_python_environment(blockers),
        "executables": {
            "pw": pw_check,
            "mpi": mpi_check,
            "mpi_qe_interoperability": interoperability_check,
        },
        "datasets": {
            "raw_mp_json": _check_dataset_file(
                raw_json_path,
                label="raw_mp_json",
                expected_kind="json",
                blockers=blockers,
            ),
            "materials_csv": _check_dataset_file(
                materials_csv_path,
                label="materials_csv",
                expected_kind="csv",
                blockers=blockers,
            ),
            "structure_cache": _check_dataset_file(
                structure_cache_path,
                label="structure_cache",
                expected_kind="sqlite",
                blockers=blockers,
            ),
        },
        "sssp": _check_sssp(
            manifest_path=pseudo_manifest_path,
            pseudo_dir=pseudo_dir,
            metadata_path=sssp_metadata_path,
            required_elements=required_elements,
            blockers=blockers,
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready" if not blockers else "blocked",
        "ready_for_qe_handoff": not blockers,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety_mode": {
            "network_access": False,
            "downloads": False,
            "dft_calculations": False,
            "filesystem_mutation": "optional_json_report_only",
            "executable_probe_requested": probe_executables,
            "mpi_qe_interoperability_probe": (
                "mpirun -np 1 pw.x -help (resolved executable paths)"
                if probe_executables and mpi_ranks > 1
                else "not_required_serial"
                if mpi_ranks == 1
                else "not_requested"
            ),
            "probe_starts_dft_calculation": False,
        },
        "project_root": str(project_root),
        "host": {
            "platform": platform.platform(),
            "machine_reported": host_machine,
            "native_architecture": host_architecture,
        },
        "execution_policy": {
            "mode": "serial" if mpi_ranks == 1 else "mpi",
            "mpi_ranks": mpi_ranks,
            "omp_threads": omp_threads,
            "mpi_launcher_required": mpi_ranks > 1,
        },
        "thresholds": {
            "minimum_free_gib_per_project_and_scratch_volume": min_free_gib,
            "minimum_free_bytes": minimum_free_bytes,
            "probe_timeout_seconds": probe_timeout_seconds,
        },
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "Dataset checks read only a small header and file metadata; the full MP snapshot audit is a later explicit workflow stage.",
            "Executable hashes identify this local installation but do not replace the QE hashes captured by each executed job.",
            "The MPI/QE interoperability probe is required only for mpi_ranks > 1 and invokes pw.x -help through one MPI rank; it supplies no QE input and performs no SCF calculation.",
            "A ready result is an environment prerequisite, not a DFT or thermodynamic validation result.",
        ],
    }


def _elements_arg(value: str) -> list[str]:
    elements = [part.strip() for part in value.split(",") if part.strip()]
    try:
        return _validate_required_elements(elements)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--scratch-dir", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument("--pw-executable", default="pw.x")
    parser.add_argument("--mpi-executable", default="mpirun")
    parser.add_argument(
        "--mpi-ranks", type=int, default=2,
        help="Planned QE MPI rank count; use 1 for an explicitly serial handoff.",
    )
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument(
        "--probe-executables",
        action="store_true",
        help=(
            "Run only fixed pw.x -help, mpirun --version, and "
            "mpirun -np 1 pw.x -help probes; no QE input is supplied."
        ),
    )
    parser.add_argument("--probe-timeout-seconds", type=int, default=5)
    parser.add_argument(
        "--raw-json", type=Path, default=PROJECT_ROOT / "data/mp.2019.04.01.json"
    )
    parser.add_argument(
        "--materials-csv",
        type=Path,
        default=PROJECT_ROOT / "research/phase_2/data/processed/materials.csv",
    )
    parser.add_argument(
        "--structure-cache",
        type=Path,
        default=PROJECT_ROOT / "research/phase_3/cache/structures.sqlite",
    )
    parser.add_argument("--pseudo-manifest", type=Path)
    parser.add_argument("--pseudo-dir", type=Path)
    parser.add_argument(
        "--sssp-metadata",
        type=Path,
        help="Override a handoff-stale source_metadata path recorded in the manifest.",
    )
    parser.add_argument(
        "--required-elements", type=_elements_arg, default=DEFAULT_ELEMENTS
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_preflight(
        project_root=args.project_root,
        scratch_dir=args.scratch_dir,
        min_free_gib=args.min_free_gib,
        pw_executable=args.pw_executable,
        mpi_executable=args.mpi_executable,
        probe_executables=args.probe_executables,
        probe_timeout_seconds=args.probe_timeout_seconds,
        raw_json_path=args.raw_json,
        materials_csv_path=args.materials_csv,
        structure_cache_path=args.structure_cache,
        pseudo_manifest_path=args.pseudo_manifest,
        pseudo_dir=args.pseudo_dir,
        sssp_metadata_path=args.sssp_metadata,
        required_elements=args.required_elements,
        mpi_ranks=args.mpi_ranks,
        omp_threads=args.omp_threads,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ready_for_qe_handoff"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
