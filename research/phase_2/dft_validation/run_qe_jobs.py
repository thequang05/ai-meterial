"""Safely plan or sequentially execute prepared Quantum ESPRESSO jobs.

Execution is opt-in via ``--execute``. The default only writes a run plan.
The runner never launches candidates concurrently and defaults to one job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from qe_output import summarize_qe_output
from qe_convergence_certificate import verify_convergence_certificate
from qe_execution_provenance import (
    execution_identity,
    require_same_execution_provenance,
    validate_execution_provenance,
)


RUNNER_VERSION = "qe_sequential_runner_v1"
GIB = 1024 ** 3


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_queue(path: Path) -> list[dict[str, str]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("DFT queue is empty")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_blockers(
    preflight: dict[str, Any], queue: list[dict[str, str]]
) -> list[str]:
    blockers: list[str] = []
    queue_path = Path(str(preflight.get("queue_manifest") or "")).resolve()
    expected_queue_sha = str(preflight.get("queue_manifest_sha256") or "").lower()
    if not queue_path.is_file():
        blockers.append("queue_manifest_missing")
    elif len(expected_queue_sha) != 64 or _sha256(queue_path) != expected_queue_sha:
        blockers.append("queue_manifest_hash_mismatch")
    pseudos = preflight.get("bundled_pseudopotentials")
    if not isinstance(pseudos, list) or not pseudos:
        blockers.append("bundled_pseudopotential_hash_inventory_missing")
    else:
        for entry in pseudos:
            path = Path(str(entry.get("path") or "")).resolve()
            expected = str(entry.get("sha256") or "").lower()
            label = str(entry.get("element") or entry.get("filename") or path.name)
            if not path.is_file():
                blockers.append(f"bundled_pseudopotential_missing:{label}")
            elif len(expected) != 64 or _sha256(path) != expected:
                blockers.append(f"bundled_pseudopotential_hash_mismatch:{label}")
    for row in queue:
        candidate_id = row.get("candidate_id", "unknown")
        input_path = Path(str(row.get("qe_input") or "")).resolve()
        expected = str(row.get("qe_input_sha256") or "").lower()
        if not input_path.is_file():
            blockers.append(f"qe_input_missing:{candidate_id}")
        elif len(expected) != 64:
            blockers.append(f"qe_input_hash_missing:{candidate_id}")
        elif _sha256(input_path) != expected:
            blockers.append(f"qe_input_hash_mismatch:{candidate_id}")
    return blockers


def _required_absolute_path(value: Any, *, label: str) -> Path:
    """Resolve one recorded artifact path without accepting CWD-relative input."""

    text = str(value or "").strip()
    if not text:
        raise ValueError(f"QE queue path is missing: {label}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"QE queue path must be absolute: {label}={text!r}")
    return path.resolve()


def _validated_queue_manifest_path(
    *, preflight_path: Path, recorded_path: Any,
) -> Path:
    """Return the queue only when it is beside the selected preflight."""

    campaign_dir = Path(preflight_path).resolve().parent
    queue_path = _required_absolute_path(
        recorded_path, label="preflight.queue_manifest"
    )
    if queue_path.parent != campaign_dir:
        raise ValueError(
            "QE queue manifest must be a direct child of its preflight campaign: "
            f"queue={queue_path}, campaign={campaign_dir}"
        )
    return queue_path


def _validate_queue_path_containment(
    *,
    preflight_path: Path,
    preflight: dict[str, Any],
    queue_path: Path,
    queue: list[dict[str, str]],
) -> None:
    """Reject a queue that could read/write outside its prepared namespaces.

    A queue is hash-bound by its preflight, but the preflight itself is an
    operator-supplied artifact rather than a signed trust anchor.  Containment
    therefore remains mandatory even when all recorded hashes match.
    """

    campaign_dir = Path(preflight_path).resolve().parent
    expected_queue_path = _validated_queue_manifest_path(
        preflight_path=preflight_path,
        recorded_path=preflight.get("queue_manifest"),
    )
    if Path(queue_path).resolve() != expected_queue_path:
        raise ValueError("Loaded QE queue differs from its preflight path")

    jobs_root = (campaign_dir / "jobs").resolve()
    if jobs_root.parent != campaign_dir or not jobs_root.is_dir():
        raise ValueError(f"QE campaign jobs directory is missing: {jobs_root}")

    scratch_root_raw = preflight.get("scratch_root")
    scratch_namespace_raw = preflight.get("scratch_campaign_namespace")
    scratch_campaign_raw = preflight.get("scratch_campaign_dir")
    explicit_scratch_values = (
        scratch_root_raw,
        scratch_namespace_raw,
        scratch_campaign_raw,
    )
    has_explicit_scratch = any(value not in (None, "") for value in explicit_scratch_values)
    if has_explicit_scratch:
        if any(value in (None, "") for value in explicit_scratch_values):
            raise ValueError(
                "QE preflight has an incomplete explicit scratch namespace"
            )
        scratch_root = _required_absolute_path(
            scratch_root_raw, label="preflight.scratch_root"
        )
        namespace = Path(str(scratch_namespace_raw))
        if (
            namespace.is_absolute()
            or len(namespace.parts) != 2
            or any(part in {"", ".", ".."} for part in namespace.parts)
        ):
            raise ValueError(
                "QE preflight scratch campaign namespace is invalid: "
                f"{scratch_namespace_raw!r}"
            )
        scratch_campaign_dir = _required_absolute_path(
            scratch_campaign_raw, label="preflight.scratch_campaign_dir"
        )
        expected_scratch_campaign = (scratch_root / namespace).resolve()
        if (
            scratch_campaign_dir != expected_scratch_campaign
            or not scratch_campaign_dir.is_relative_to(scratch_root)
            or not scratch_campaign_dir.is_dir()
        ):
            raise ValueError(
                "QE preflight scratch campaign is outside or differs from its "
                "declared root/namespace"
            )
    else:
        scratch_root = None
        scratch_campaign_dir = None

    seen_job_dirs: set[Path] = set()
    seen_artifact_paths: set[Path] = set()
    seen_scratch_dirs: set[Path] = set()
    for index, row in enumerate(queue, start=1):
        row_label = str(row.get("candidate_id") or f"row_{index}")
        input_path = _required_absolute_path(
            row.get("qe_input"), label=f"{row_label}.qe_input"
        )
        job_dir = input_path.parent
        if job_dir.parent != jobs_root:
            raise ValueError(
                "QE input/job directory escaped the campaign jobs directory: "
                f"{row_label}:{input_path}"
            )
        if job_dir in seen_job_dirs:
            raise ValueError(f"QE queue reuses a job directory: {job_dir}")
        seen_job_dirs.add(job_dir)

        output_path = _required_absolute_path(
            row.get("qe_output"), label=f"{row_label}.qe_output"
        )
        record_path = _required_absolute_path(
            row.get("run_record"), label=f"{row_label}.run_record"
        )
        if output_path.parent != job_dir or record_path.parent != job_dir:
            raise ValueError(
                "QE output/run record escaped its prepared job directory: "
                f"{row_label}"
            )
        artifacts = (input_path, output_path, record_path)
        if len(set(artifacts)) != len(artifacts):
            raise ValueError(f"QE job artifact paths overlap: {row_label}")
        for artifact in artifacts:
            if artifact in seen_artifact_paths:
                raise ValueError(f"QE queue reuses an artifact path: {artifact}")
            seen_artifact_paths.add(artifact)

        scratch_text = str(row.get("qe_scratch_outdir") or "").strip()
        if not scratch_text:
            raise ValueError(f"QE scratch path is missing: {row_label}")
        scratch_path = Path(scratch_text).expanduser()
        if scratch_campaign_dir is None:
            if scratch_path.is_absolute():
                raise ValueError(
                    "Job-local QE scratch must be relative: "
                    f"{row_label}:{scratch_text!r}"
                )
            resolved_scratch = (job_dir / scratch_path).resolve()
            expected_local_scratch = (job_dir / "tmp").resolve()
            if (
                resolved_scratch != expected_local_scratch
                or not resolved_scratch.is_relative_to(job_dir)
            ):
                raise ValueError(
                    "Job-local QE scratch must resolve to the job's ./tmp: "
                    f"{row_label}:{scratch_text!r}"
                )
        else:
            if not scratch_path.is_absolute():
                raise ValueError(
                    "Explicit campaign QE scratch must be absolute: "
                    f"{row_label}:{scratch_text!r}"
                )
            resolved_scratch = scratch_path.resolve()
            if (
                resolved_scratch.parent != scratch_campaign_dir
                or not resolved_scratch.is_relative_to(scratch_campaign_dir)
                or not resolved_scratch.is_dir()
            ):
                raise ValueError(
                    "QE scratch directory escaped or differs from its prepared "
                    f"campaign namespace: {row_label}:{resolved_scratch}"
                )
        if resolved_scratch in seen_scratch_dirs:
            raise ValueError(
                f"QE queue reuses a scratch directory: {resolved_scratch}"
            )
        seen_scratch_dirs.add(resolved_scratch)


def _settings_hash_from_preflight(preflight: dict[str, Any]) -> str:
    return str(
        preflight.get("convergence_settings_hash")
        or preflight.get("static_settings_hash")
        or preflight.get("relax_input_settings_hash")
        or preflight.get("relax_settings_hash")
        or ""
    )


def _completed_record_matches(
    *,
    record_path: Path,
    job: dict[str, Any],
    output_path: Path,
    preflight_sha256: str,
    queue_manifest_sha256: str,
    preflight_settings_hash: str,
    pw_executable_sha256: str,
    execution_provenance: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    if not record_path.is_file() or not output_path.is_file():
        return False, None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None
    if record.get("run_status") != "completed_requires_collection":
        return False, record
    expected_identity = {
        key: job[key]
        for key in ("entry_id", "entry_role", "source_id", "candidate_id", "formula")
    }
    recorded_identity = {key: record.get(key) for key in expected_identity}
    checks = [
        recorded_identity == expected_identity,
        record.get("qe_input_sha256") == job.get("qe_input_sha256"),
        record.get("qe_output_sha256") == _sha256(output_path),
        record.get("preflight_sha256") == preflight_sha256,
        record.get("queue_manifest_sha256") == queue_manifest_sha256,
        record.get("preflight_settings_hash") == preflight_settings_hash,
        record.get("pw_executable_sha256") == pw_executable_sha256,
    ]
    try:
        checks.append(
            execution_identity(record.get("execution_provenance"))
            == execution_identity(execution_provenance)
        )
    except ValueError:
        checks.append(False)
    return all(checks), record


def _resolve_executable(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        path = path.resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    resolved = shutil.which(value)
    if resolved:
        return resolved
    raise FileNotFoundError(f"Executable not found: {value}")


def _nearest_existing_ancestor(path: Path) -> Path:
    """Return ``path`` or its closest existing parent for filesystem checks."""

    current = Path(path).expanduser().resolve()
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                f"No existing ancestor is available for disk check: {path}"
            )
        current = parent
    return current


def _disk_space_observation(
    *, label: str, requested_path: Path, minimum_free_bytes: int,
) -> dict[str, Any]:
    """Measure one target filesystem and return a serializable fail-closed result."""

    requested = Path(requested_path).expanduser().resolve()
    observation: dict[str, Any] = {
        "label": label,
        "requested_path": str(requested),
        "minimum_free_bytes": minimum_free_bytes,
        "minimum_free_gib": minimum_free_bytes / GIB,
        "sufficient": False,
    }
    try:
        checked = _nearest_existing_ancestor(requested)
        usage = shutil.disk_usage(checked)
        observation.update({
            "checked_path": str(checked),
            "filesystem_device_id": int(checked.stat().st_dev),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "free_gib": float(usage.free / GIB),
            "sufficient": int(usage.free) >= minimum_free_bytes,
            "check_error": None,
        })
    except (OSError, ValueError) as exc:
        observation["check_error"] = f"{type(exc).__name__}: {exc}"
    return observation


def _disk_space_gate(
    *, job_dir: Path, qe_scratch_outdir: str, min_free_gib: float,
) -> dict[str, Any]:
    """Check both the job and QE scratch filesystems immediately before launch."""

    minimum_free_bytes = int(min_free_gib * GIB)
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    if qe_scratch_outdir:
        scratch = Path(qe_scratch_outdir).expanduser()
        if not scratch.is_absolute():
            scratch = resolved_job_dir / scratch
        scratch = scratch.resolve()
        scratch_observation = _disk_space_observation(
            label="qe_scratch_filesystem",
            requested_path=scratch,
            minimum_free_bytes=minimum_free_bytes,
        )
    else:
        scratch_observation = {
            "label": "qe_scratch_filesystem",
            "requested_path": None,
            "minimum_free_bytes": minimum_free_bytes,
            "minimum_free_gib": min_free_gib,
            "sufficient": False,
            "check_error": "qe_scratch_outdir_missing_from_queue",
        }
    observations = [
        _disk_space_observation(
            label="project_job_filesystem",
            requested_path=resolved_job_dir,
            minimum_free_bytes=minimum_free_bytes,
        ),
        scratch_observation,
    ]
    passed = all(item.get("sufficient") is True for item in observations)
    has_check_error = any(item.get("check_error") for item in observations)
    return {
        "status": (
            "passed"
            if passed
            else "blocked_check_failed"
            if has_check_error
            else "blocked_insufficient_free_space"
        ),
        "passed": passed,
        "minimum_free_bytes_per_filesystem": minimum_free_bytes,
        "minimum_free_gib_per_filesystem": min_free_gib,
        "observations": observations,
        "checked_unix_time": time.time(),
    }


def _mpi_program_version(path: str, *, timeout_seconds: int = 10) -> str:
    """Read a fixed launcher version string without allowing arbitrary probes."""

    try:
        completed = subprocess.run(
            [path, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Unable to inspect MPI launcher version: {exc}") from exc
    rendered = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode != 0 or not rendered:
        raise RuntimeError(
            "Unable to inspect MPI launcher version: "
            f"exit={completed.returncode}, output={rendered[:500]!r}"
        )
    return rendered.splitlines()[0].strip()


def _execution_provenance(
    *, mpi_ranks: int, omp_threads: int, mpi_path: str,
) -> dict[str, Any]:
    if mpi_ranks == 1:
        return validate_execution_provenance({
            "execution_mode": "serial",
            "mpi_ranks": 1,
            "omp_threads": omp_threads,
            "mpi_launcher_path": None,
            "mpi_launcher_sha256": None,
            "mpi_program_version": None,
        })
    launcher = Path(mpi_path).resolve()
    return validate_execution_provenance({
        "execution_mode": "mpi",
        "mpi_ranks": mpi_ranks,
        "omp_threads": omp_threads,
        "mpi_launcher_path": str(launcher),
        "mpi_launcher_sha256": _sha256(launcher),
        "mpi_program_version": _mpi_program_version(str(launcher)),
    })


def _command(
    *,
    pw_executable: str,
    input_name: str,
    mpi_ranks: int,
    mpi_executable: str,
) -> list[str]:
    if mpi_ranks < 1:
        raise ValueError("mpi_ranks must be at least 1")
    if mpi_ranks == 1:
        return [pw_executable, "-in", input_name]
    return [mpi_executable, "-np", str(mpi_ranks), pw_executable, "-in", input_name]


def run_jobs(
    *,
    preflight_path: Path,
    execute: bool = False,
    candidate_ids: list[str] | None = None,
    max_jobs: int = 1,
    mpi_ranks: int = 2,
    omp_threads: int = 1,
    mpi_executable: str = "mpirun",
    timeout_seconds: int = 21600,
    min_free_gib: float = 1.0,
    force: bool = False,
    override_resource_policy: bool = False,
) -> dict[str, Any]:
    if max_jobs < 1:
        raise ValueError("max_jobs must be at least 1")
    if (
        isinstance(mpi_ranks, bool)
        or not isinstance(mpi_ranks, int)
        or mpi_ranks < 1
    ):
        raise ValueError("mpi_ranks must be a positive integer")
    if omp_threads < 1:
        raise ValueError("omp_threads must be at least 1")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 1
    ):
        raise ValueError("timeout_seconds must be a positive integer")
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("min_free_gib must be finite and non-negative")
    preflight_path = Path(preflight_path).resolve()
    preflight = _load_json(preflight_path)
    preflight_sha256 = _sha256(preflight_path)
    queue_manifest_sha256 = str(preflight.get("queue_manifest_sha256") or "")
    preflight_settings_hash = _settings_hash_from_preflight(preflight)
    allowed_plan_statuses = {"runnable_not_started", "inputs_ready_engine_missing"}
    if preflight.get("status") not in allowed_plan_statuses:
        raise ValueError(
            "DFT preflight has no complete inputs: "
            f"{preflight.get('status')}"
        )
    if execute and preflight.get("status") != "runnable_not_started":
        raise ValueError(
            "Execution requires preflight status runnable_not_started; got "
            f"{preflight.get('status')}"
        )
    queue_path = _validated_queue_manifest_path(
        preflight_path=preflight_path,
        recorded_path=preflight.get("queue_manifest"),
    )
    queue = _load_queue(queue_path)
    _validate_queue_path_containment(
        preflight_path=preflight_path,
        preflight=preflight,
        queue_path=queue_path,
        queue=queue,
    )
    if candidate_ids:
        wanted = set(candidate_ids)
        queue = [row for row in queue if row.get("candidate_id") in wanted]
        missing = wanted - {row.get("candidate_id") for row in queue}
        if missing:
            raise ValueError(f"Candidate IDs not found in queue: {sorted(missing)}")
    queue = sorted(queue, key=lambda row: int(row["rank"]))

    policy = preflight.get("resource_policy") or {}
    max_mpi = policy.get("recommended_mpi_ranks_on_this_mac")
    max_omp = policy.get("omp_threads_per_rank")
    if not override_resource_policy:
        if max_mpi is not None and mpi_ranks > int(max_mpi):
            raise ValueError(
                f"mpi_ranks={mpi_ranks} exceeds locked local policy {max_mpi}; "
                "use --override-resource-policy only on an appropriate machine"
            )
        if max_omp is not None and omp_threads > int(max_omp):
            raise ValueError(
                f"omp_threads={omp_threads} exceeds locked local policy {max_omp}; "
                "use --override-resource-policy only on an appropriate machine"
            )

    requested_pw = str(
        preflight.get("pw_executable_resolved")
        or preflight.get("pw_executable_requested")
        or "pw.x"
    )
    # A plan is allowed without installed executables. Execution always
    # resolves them again so a stale preflight path cannot be launched.
    pw_executable = _resolve_executable(requested_pw) if execute else requested_pw
    mpi_path = (
        _resolve_executable(mpi_executable)
        if execute and mpi_ranks > 1 else mpi_executable if mpi_ranks > 1 else ""
    )
    execution_provenance = (
        _execution_provenance(
            mpi_ranks=mpi_ranks, omp_threads=omp_threads, mpi_path=mpi_path,
        )
        if execute else None
    )
    provenance_blockers = _provenance_blockers(preflight, queue)
    production_workflows = {
        "qe_pbe_candidate_relax_v1", "qe_pbe_candidate_static_v1",
    }
    if preflight.get("workflow_version") in production_workflows:
        if preflight.get("production_settings_certified") is not True:
            provenance_blockers.append(
                "production_settings_not_convergence_certified"
            )
        else:
            try:
                certificate = verify_convergence_certificate(
                    Path(str(preflight.get("convergence_certificate") or "")),
                    required_elements=preflight.get("required_elements") or [],
                    config_path=Path(preflight["config"]),
                    pseudo_manifest_path=Path(preflight["pseudo_manifest"]),
                    pw_executable_path=(Path(pw_executable) if execute else None),
                )
                if (
                    preflight.get("convergence_certificate_sha256")
                    != certificate["certificate_sha256"]
                    or preflight.get("convergence_certificate_payload_sha256")
                    != certificate["certificate_payload_sha256"]
                ):
                    provenance_blockers.append(
                        "convergence_certificate_lineage_mismatch"
                    )
                if execute:
                    try:
                        require_same_execution_provenance(
                            execution_provenance,
                            certificate["execution_provenance"],
                            label="convergence certificate",
                        )
                    except ValueError as exc:
                        provenance_blockers.append(
                            f"certified_execution_provenance_mismatch:{exc}"
                        )
            except (FileNotFoundError, ValueError, OSError) as exc:
                provenance_blockers.append(
                    f"convergence_certificate_verification_failed:{exc}"
                )
    if execute and preflight.get("stage") == "confirmation":
        actual_pw_sha = _sha256(Path(pw_executable))
        if actual_pw_sha != preflight.get("baseline_pw_executable_sha256"):
            provenance_blockers.append(
                "confirmation_executable_differs_from_sweep"
            )
        try:
            require_same_execution_provenance(
                execution_provenance,
                preflight.get("baseline_execution_provenance"),
                label="convergence sweep",
            )
        except ValueError as exc:
            provenance_blockers.append(
                f"confirmation_execution_provenance_mismatch:{exc}"
            )
    if execute and provenance_blockers:
        raise ValueError(
            "Execution provenance verification failed: "
            + "; ".join(provenance_blockers)
        )
    planned: list[dict[str, Any]] = []
    for row in queue:
        input_path = Path(row["qe_input"]).resolve()
        blockers = json.loads(row.get("blockers") or "[]")
        if execute and blockers:
            raise ValueError(f"Candidate {row['candidate_id']} still has blockers: {blockers}")
        job_dir = input_path.parent
        output_path = (
            Path(row["qe_output"]).resolve()
            if row.get("qe_output")
            else job_dir / "vc-relax.out"
        )
        record_path = (
            Path(row["run_record"]).resolve()
            if row.get("run_record")
            else job_dir / "job_run.json"
        )
        command = _command(
            pw_executable=pw_executable,
            input_name=input_path.name,
            mpi_ranks=mpi_ranks,
            mpi_executable=mpi_path,
        )
        planned.append({
            "entry_id": row.get("entry_id") or f"candidate:{row['candidate_id']}",
            "entry_role": row.get("entry_role") or "candidate",
            "source_id": row.get("source_id") or row["candidate_id"],
            "candidate_id": row["candidate_id"],
            "rank": int(row["rank"]),
            "formula": row["formula"],
            "job_dir": str(job_dir),
            "input": str(input_path),
            "output": str(output_path),
            "record": str(record_path),
            "qe_scratch_outdir": str(row.get("qe_scratch_outdir") or ""),
            "command": command,
            "preflight_blockers": blockers,
            "qe_input_sha256": str(row.get("qe_input_sha256") or ""),
        })

    summary_path = preflight_path.parent / "qe_run_summary.json"
    if not execute:
        planned = planned[:max_jobs]
        summary = {
            "runner_version": RUNNER_VERSION,
            "status": "planned_not_executed",
            "execute_requested": False,
            "jobs": planned,
            "resource_limits": {
                "max_parallel_jobs": 1,
                "max_jobs_this_invocation": max_jobs,
                "mpi_ranks": mpi_ranks,
                "omp_threads_per_rank": omp_threads,
                "timeout_seconds_per_job": timeout_seconds,
                "minimum_free_gib_per_project_and_scratch_filesystem": min_free_gib,
            },
            "disk_space_policy": {
                "minimum_free_gib_per_project_and_scratch_filesystem": min_free_gib,
                "check_timing": "immediately_before_each_actual_launch",
                "plan_only_disk_checks_performed": False,
            },
            "dft_validated_count": 0,
            "provenance_blockers": provenance_blockers,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Planned {len(planned)} job(s); nothing executed.")
        print(f"Plan: {summary_path}")
        return summary

    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = str(omp_threads)
    env["OMP_DYNAMIC"] = "FALSE"
    pw_executable_sha256 = _sha256(Path(pw_executable))

    results: list[dict[str, Any]] = []
    launched_count = 0
    for index, job in enumerate(planned, start=1):
        output_path = Path(job["output"])
        record_path = Path(job["record"])
        existing = summarize_qe_output(output_path)
        binding_matches, existing_record = _completed_record_matches(
            record_path=record_path,
            job=job,
            output_path=output_path,
            preflight_sha256=preflight_sha256,
            queue_manifest_sha256=queue_manifest_sha256,
            preflight_settings_hash=preflight_settings_hash,
            pw_executable_sha256=pw_executable_sha256,
            execution_provenance=execution_provenance,
        )
        if (
            existing["job_done"]
            and not existing["electronic_convergence_failed"]
            and not existing["fatal_error_detected"]
            and binding_matches
            and not force
        ):
            results.append({
                **job,
                "run_status": "skipped_existing_complete_output",
                "return_code": 0,
                "elapsed_seconds": 0.0,
                "qe_output_summary": existing,
                "completed_record": existing_record,
            })
            continue

        if launched_count >= max_jobs:
            break

        disk_space_gate = _disk_space_gate(
            job_dir=Path(job["job_dir"]),
            qe_scratch_outdir=job["qe_scratch_outdir"],
            min_free_gib=min_free_gib,
        )
        if not disk_space_gate["passed"]:
            blocked_at = time.time()
            gate_id = f"disk_gate_{time.time_ns()}"
            attempts_dir = record_path.parent / "attempts"
            attempts_dir.mkdir(parents=True, exist_ok=True)
            gate_record_path = attempts_dir / f"{gate_id}.json"
            if record_path.is_file():
                shutil.copy2(
                    record_path,
                    attempts_dir / f"stale_record_{gate_id}.json",
                )
            blocked_status = (
                "blocked_disk_space_check_failed"
                if disk_space_gate["status"] == "blocked_check_failed"
                else "blocked_insufficient_disk_space"
            )
            blocked_record = {
                "runner_version": RUNNER_VERSION,
                **job,
                "run_status": blocked_status,
                "return_code": None,
                "launched": False,
                "blocked_unix_time": blocked_at,
                "attempt_id": gate_id,
                "attempt_record": str(gate_record_path),
                "preflight_sha256": preflight_sha256,
                "queue_manifest_sha256": queue_manifest_sha256,
                "preflight_settings_hash": preflight_settings_hash,
                "pw_executable": pw_executable,
                "pw_executable_sha256": pw_executable_sha256,
                "execution_provenance": execution_provenance,
                "disk_space_gate": disk_space_gate,
                "resource_limits": {
                    "mpi_ranks": mpi_ranks,
                    "omp_threads_per_rank": omp_threads,
                    "timeout_seconds": timeout_seconds,
                    "minimum_free_gib_per_project_and_scratch_filesystem": min_free_gib,
                },
            }
            rendered = json.dumps(blocked_record, indent=2)
            record_path.write_text(rendered, encoding="utf-8")
            gate_record_path.write_text(rendered, encoding="utf-8")
            results.append(blocked_record)
            print(
                f"[{index}/{len(planned)}] Blocked {job['candidate_id']}: "
                f"{disk_space_gate['status']}"
            )
            # Stop the invocation after a disk gate failure. Later jobs may use
            # the same full filesystem, and no automatic launch is safe here.
            break
        launched_count += 1

        started = time.time()
        attempt_id = f"{time.time_ns()}"
        attempts_dir = record_path.parent / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        attempt_record_path = attempts_dir / f"attempt_{attempt_id}.json"
        if output_path.is_file():
            stale_output = attempts_dir / f"stale_output_{attempt_id}.out"
            output_path.replace(stale_output)
        if record_path.is_file():
            stale_record = attempts_dir / f"stale_record_{attempt_id}.json"
            shutil.copy2(record_path, stale_record)
        running_record = {
            "runner_version": RUNNER_VERSION,
            **job,
            "run_status": "running",
            "started_unix_time": started,
            "attempt_id": attempt_id,
            "attempt_record": str(attempt_record_path),
            "qe_input_sha256": job["qe_input_sha256"],
            "preflight_sha256": preflight_sha256,
            "queue_manifest_sha256": queue_manifest_sha256,
            "preflight_settings_hash": preflight_settings_hash,
            "pw_executable": pw_executable,
            "pw_executable_sha256": pw_executable_sha256,
            "execution_provenance": execution_provenance,
            "disk_space_gate": disk_space_gate,
            "resource_limits": {
                "mpi_ranks": mpi_ranks,
                "omp_threads_per_rank": omp_threads,
                "timeout_seconds": timeout_seconds,
                "minimum_free_gib_per_project_and_scratch_filesystem": min_free_gib,
            },
        }
        record_path.write_text(json.dumps(running_record, indent=2), encoding="utf-8")
        attempt_record_path.write_text(
            json.dumps(running_record, indent=2), encoding="utf-8"
        )
        print(f"[{index}/{len(planned)}] Running {job['candidate_id']} sequentially")
        timed_out = False
        timeout_cleanup = "not_needed"
        with output_path.open("w", encoding="utf-8") as output_handle:
            process = subprocess.Popen(
                job["command"],
                cwd=job["job_dir"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                timeout_cleanup = "sigterm_sent_to_process_group"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    return_code = process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    return_code = process.poll()
                group_still_alive = False
                try:
                    os.killpg(process.pid, 0)
                    group_still_alive = True
                except ProcessLookupError:
                    pass
                if group_still_alive:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                        timeout_cleanup = "sigkill_sent_to_remaining_process_group"
                    except ProcessLookupError:
                        pass
                if process.poll() is None:
                    return_code = process.wait()
        parsed = summarize_qe_output(output_path)
        output_sha256 = _sha256(output_path) if output_path.is_file() else ""
        if timed_out:
            run_status = "timed_out"
        elif return_code != 0:
            run_status = "process_failed"
        elif not parsed["job_done"]:
            run_status = "incomplete_output"
        elif parsed["electronic_convergence_failed"] or parsed["fatal_error_detected"]:
            run_status = "qe_failed"
        else:
            run_status = "completed_requires_collection"
        result = {
            **job,
            "run_status": run_status,
            "return_code": return_code,
            "timed_out": timed_out,
            "timeout_cleanup": timeout_cleanup,
            "attempt_id": attempt_id,
            "attempt_record": str(attempt_record_path),
            "elapsed_seconds": time.time() - started,
            "qe_output_summary": parsed,
            "qe_output_sha256": output_sha256,
            "qe_program_version": parsed.get("program_version"),
            "preflight_sha256": preflight_sha256,
            "queue_manifest_sha256": queue_manifest_sha256,
            "preflight_settings_hash": preflight_settings_hash,
            "pw_executable": pw_executable,
            "pw_executable_sha256": pw_executable_sha256,
            "execution_provenance": execution_provenance,
            "disk_space_gate": disk_space_gate,
        }
        record_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        attempt_record_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)

    disk_blocked = any(
        item.get("run_status") in {
            "blocked_insufficient_disk_space",
            "blocked_disk_space_check_failed",
        }
        for item in results
    )
    summary = {
        "runner_version": RUNNER_VERSION,
        "status": (
            "execution_blocked_by_disk_space_gate"
            if disk_blocked else "execution_finished_requires_collection"
        ),
        "execute_requested": True,
        "jobs": results,
        "dft_validated_count": 0,
        "execution_provenance": execution_provenance,
        "disk_space_policy": {
            "minimum_free_gib_per_project_and_scratch_filesystem": min_free_gib,
            "check_timing": "immediately_before_each_actual_launch",
        },
        "disk_space_checks": [
            {
                "candidate_id": item["candidate_id"],
                "run_status": item["run_status"],
                "disk_space_gate": item["disk_space_gate"],
            }
            for item in results
            if item.get("disk_space_gate") is not None
        ],
        "scientific_limit": (
            "Process completion alone is not DFT validation. Collect final "
            "structures, convergence metrics, and consistent static energies next."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Run summary: {summary_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--mpi-ranks", type=int, default=2)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--mpi-executable", default="mpirun")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=1.0,
        help=(
            "Minimum free GiB required on both the job and QE scratch "
            "filesystems immediately before each launch (default: 1)."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--override-resource-policy", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_jobs(
        preflight_path=args.preflight,
        execute=args.execute,
        candidate_ids=args.candidate_id,
        max_jobs=args.max_jobs,
        mpi_ranks=args.mpi_ranks,
        omp_threads=args.omp_threads,
        mpi_executable=args.mpi_executable,
        timeout_seconds=args.timeout_seconds,
        min_free_gib=args.min_free_gib,
        force=args.force,
        override_resource_policy=args.override_resource_policy,
    )
