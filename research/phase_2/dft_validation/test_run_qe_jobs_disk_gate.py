from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_qe_jobs import build_arg_parser, run_jobs


GIB = 1024 ** 3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_runnable_preflight(root: Path) -> tuple[Path, Path, Path, Path]:
    job_dir = root / "jobs" / "candidate_1"
    job_dir.mkdir(parents=True)
    input_path = job_dir / "scf.in"
    input_path.write_text("&CONTROL\n/\n", encoding="utf-8")
    output_path = job_dir / "scf.out"
    record_path = job_dir / "job_run.json"

    scratch_root = root / "scratch-volume"
    scratch_namespace = "qe_disk_gate_test_v1/campaign_test"
    scratch_campaign = scratch_root / scratch_namespace
    scratch_outdir = scratch_campaign / "candidate_1"
    scratch_outdir.mkdir(parents=True)

    queue_path = root / "dft_queue.csv"
    fieldnames = [
        "rank",
        "entry_id",
        "entry_role",
        "source_id",
        "candidate_id",
        "formula",
        "qe_input",
        "qe_input_sha256",
        "qe_output",
        "run_record",
        "qe_scratch_outdir",
        "blockers",
    ]
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "rank": 1,
            "entry_id": "candidate:candidate_1",
            "entry_role": "candidate",
            "source_id": "candidate_1",
            "candidate_id": "candidate_1",
            "formula": "TiC",
            "qe_input": str(input_path),
            "qe_input_sha256": _sha(input_path),
            "qe_output": str(output_path),
            "run_record": str(record_path),
            "qe_scratch_outdir": str(scratch_outdir),
            "blockers": "[]",
        })

    pseudo = root / "C.UPF"
    pseudo.write_text("synthetic pseudo", encoding="utf-8")
    preflight_path = root / "dft_preflight.json"
    preflight_path.write_text(json.dumps({
        "status": "runnable_not_started",
        "workflow_version": "qe_disk_gate_test_v1",
        "queue_manifest": str(queue_path),
        "queue_manifest_sha256": _sha(queue_path),
        "pw_executable_resolved": "/usr/bin/true",
        "scratch_root": str(scratch_root),
        "scratch_campaign_namespace": scratch_namespace,
        "scratch_campaign_dir": str(scratch_campaign),
        "resource_policy": {
            "recommended_mpi_ranks_on_this_mac": 1,
            "omp_threads_per_rank": 1,
        },
        "bundled_pseudopotentials": [{
            "element": "C",
            "path": str(pseudo),
            "sha256": _sha(pseudo),
        }],
    }), encoding="utf-8")
    return preflight_path, job_dir, scratch_outdir, record_path


def _rewrite_first_queue_row(
    preflight_path: Path, *, updates: dict[str, str]
) -> Path:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    queue_path = Path(preflight["queue_manifest"])
    with queue_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0].update(updates)
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    preflight["queue_manifest_sha256"] = _sha(queue_path)
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    return queue_path


class QeRunnerDiskGateTests(unittest.TestCase):
    def test_cli_default_is_one_gib(self):
        args = build_arg_parser().parse_args(["--preflight", "placeholder.json"])
        self.assertEqual(args.min_free_gib, 1.0)

    def test_invalid_threshold_is_rejected_before_preflight_access(self):
        for invalid in (-0.01, math.inf, -math.inf, math.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError, "min_free_gib must be finite and non-negative"
                ):
                    run_jobs(
                        preflight_path=Path("does-not-exist.json"),
                        min_free_gib=invalid,
                    )

    def test_invalid_rank_and_timeout_values_are_rejected_before_preflight(self):
        for field, invalid_values, expected in (
            (
                "mpi_ranks",
                (0, -1, 1.5, True),
                "mpi_ranks must be a positive integer",
            ),
            (
                "timeout_seconds",
                (0, -1, 1.5, True),
                "timeout_seconds must be a positive integer",
            ),
        ):
            for invalid in invalid_values:
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, expected):
                        run_jobs(
                            preflight_path=Path("does-not-exist.json"),
                            **{field: invalid},
                        )

    def test_low_scratch_space_blocks_launch_and_records_both_filesystems(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight, job_dir, scratch_outdir, record_path = (
                _build_runnable_preflight(root)
            )

            def fake_disk_usage(path: str | Path):
                checked = Path(path).resolve()
                free = GIB // 2 if checked == scratch_outdir.resolve() else 8 * GIB
                return shutil._ntuple_diskusage(10 * GIB, 10 * GIB - free, free)

            with patch(
                "run_qe_jobs.shutil.disk_usage", side_effect=fake_disk_usage
            ), patch("run_qe_jobs.subprocess.Popen") as popen:
                summary = run_jobs(
                    preflight_path=preflight,
                    execute=True,
                    max_jobs=1,
                    mpi_ranks=1,
                    omp_threads=1,
                    min_free_gib=1.0,
                )

            popen.assert_not_called()
            self.assertEqual(
                summary["status"], "execution_blocked_by_disk_space_gate"
            )
            self.assertEqual(
                summary["disk_space_policy"][
                    "minimum_free_gib_per_project_and_scratch_filesystem"
                ],
                1.0,
            )
            self.assertEqual(len(summary["disk_space_checks"]), 1)

            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertFalse(record["launched"])
            self.assertEqual(
                record["run_status"], "blocked_insufficient_disk_space"
            )
            gate = record["disk_space_gate"]
            self.assertFalse(gate["passed"])
            self.assertEqual(gate["minimum_free_gib_per_filesystem"], 1.0)
            observations = {item["label"]: item for item in gate["observations"]}
            project = observations["project_job_filesystem"]
            scratch = observations["qe_scratch_filesystem"]
            self.assertEqual(Path(project["checked_path"]), job_dir.resolve())
            self.assertEqual(project["free_bytes"], 8 * GIB)
            self.assertTrue(project["sufficient"])
            self.assertEqual(Path(scratch["checked_path"]), scratch_outdir.resolve())
            self.assertEqual(
                Path(scratch["requested_path"]),
                scratch_outdir.resolve(),
            )
            self.assertEqual(scratch["free_bytes"], GIB // 2)
            self.assertFalse(scratch["sufficient"])

    def test_disk_measurement_error_fails_closed_without_launch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight, _job_dir, _scratch_parent, record_path = (
                _build_runnable_preflight(root)
            )
            with patch(
                "run_qe_jobs.shutil.disk_usage",
                side_effect=OSError("disk query unavailable"),
            ), patch("run_qe_jobs.subprocess.Popen") as popen:
                summary = run_jobs(
                    preflight_path=preflight,
                    execute=True,
                    max_jobs=1,
                    mpi_ranks=1,
                    omp_threads=1,
                    min_free_gib=1.0,
                )
            popen.assert_not_called()
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["run_status"], "blocked_disk_space_check_failed")
            self.assertEqual(
                record["disk_space_gate"]["status"], "blocked_check_failed"
            )
            self.assertEqual(
                summary["status"], "execution_blocked_by_disk_space_gate"
            )

    def test_queue_manifest_must_be_sibling_of_preflight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight_path, _job_dir, _scratch, _record = (
                _build_runnable_preflight(root)
            )
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            original_queue = Path(preflight["queue_manifest"])
            outside_dir = root / "outside"
            outside_dir.mkdir()
            outside_queue = outside_dir / original_queue.name
            shutil.copy2(original_queue, outside_queue)
            preflight["queue_manifest"] = str(outside_queue)
            preflight["queue_manifest_sha256"] = _sha(outside_queue)
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

            with patch("run_qe_jobs.subprocess.Popen") as popen:
                with self.assertRaisesRegex(
                    ValueError, "queue manifest must be a direct child"
                ):
                    run_jobs(
                        preflight_path=preflight_path,
                        execute=True,
                        mpi_ranks=1,
                    )
            popen.assert_not_called()

    def test_job_artifact_and_scratch_paths_cannot_escape_campaign(self):
        cases = (
            ("qe_input", "input/job directory escaped"),
            ("qe_output", "output/run record escaped"),
            ("run_record", "output/run record escaped"),
            ("qe_scratch_outdir", "scratch directory escaped or differs"),
        )
        for field, expected_error in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                preflight_path, _job_dir, _scratch, _record = (
                    _build_runnable_preflight(root)
                )
                outside_dir = root / "outside"
                outside_dir.mkdir()
                if field == "qe_input":
                    escaped = outside_dir / "escaped.in"
                    escaped.write_text("&CONTROL\n/\n", encoding="utf-8")
                    updates = {
                        field: str(escaped),
                        "qe_input_sha256": _sha(escaped),
                    }
                elif field == "qe_scratch_outdir":
                    escaped = outside_dir / "escaped_scratch"
                    escaped.mkdir()
                    updates = {field: str(escaped)}
                else:
                    escaped = outside_dir / f"victim_{field}.txt"
                    escaped.write_text("preserve me", encoding="utf-8")
                    updates = {field: str(escaped)}
                _rewrite_first_queue_row(preflight_path, updates=updates)

                with patch("run_qe_jobs.subprocess.Popen") as popen:
                    with self.assertRaisesRegex(ValueError, expected_error):
                        run_jobs(
                            preflight_path=preflight_path,
                            execute=True,
                            mpi_ranks=1,
                        )
                popen.assert_not_called()
                if field in {"qe_output", "run_record"}:
                    self.assertEqual(escaped.read_text(encoding="utf-8"), "preserve me")

    def test_job_local_tmp_symlink_cannot_escape_job_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight_path, job_dir, _scratch, _record = _build_runnable_preflight(root)
            outside = root / "outside_local_scratch"
            outside.mkdir()
            (job_dir / "tmp").symlink_to(outside, target_is_directory=True)
            _rewrite_first_queue_row(
                preflight_path, updates={"qe_scratch_outdir": "./tmp"}
            )
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            for key in (
                "scratch_root",
                "scratch_campaign_namespace",
                "scratch_campaign_dir",
            ):
                preflight[key] = None
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

            with patch("run_qe_jobs.subprocess.Popen") as popen:
                with self.assertRaisesRegex(
                    ValueError, "Job-local QE scratch must resolve"
                ):
                    run_jobs(
                        preflight_path=preflight_path,
                        execute=True,
                        mpi_ranks=1,
                    )
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
