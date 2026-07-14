"""Safely plan or sequentially execute prepared Quantum ESPRESSO jobs.

Execution is opt-in via ``--execute``. The default only writes a run plan.
The runner never launches candidates concurrently and defaults to one job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from qe_output import summarize_qe_output


RUNNER_VERSION = "qe_sequential_runner_v1"


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


def _settings_hash_from_preflight(preflight: dict[str, Any]) -> str:
    return str(
        preflight.get("static_settings_hash")
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
    force: bool = False,
    override_resource_policy: bool = False,
) -> dict[str, Any]:
    if max_jobs < 1:
        raise ValueError("max_jobs must be at least 1")
    if omp_threads < 1:
        raise ValueError("omp_threads must be at least 1")
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
    queue_path = Path(preflight["queue_manifest"]).resolve()
    queue = _load_queue(queue_path)
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
    provenance_blockers = _provenance_blockers(preflight, queue)
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
            "resource_limits": {
                "mpi_ranks": mpi_ranks,
                "omp_threads_per_rank": omp_threads,
                "timeout_seconds": timeout_seconds,
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
        }
        record_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        attempt_record_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)

    summary = {
        "runner_version": RUNNER_VERSION,
        "status": "execution_finished_requires_collection",
        "execute_requested": True,
        "jobs": results,
        "dft_validated_count": 0,
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
        force=args.force,
        override_resource_policy=args.override_resource_policy,
    )
