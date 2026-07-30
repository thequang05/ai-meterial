from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import check_qe_handoff_environment as handoff


class QEHandoffEnvironmentTests(unittest.TestCase):
    def _fixtures(self, root: Path) -> dict[str, Path]:
        raw_json = root / "mp.2019.04.01.json"
        raw_json.write_text("{}", encoding="utf-8")
        materials_csv = root / "materials.csv"
        materials_csv.write_text("material_id,formula\nmp-1,TiC\n", encoding="utf-8")
        structure_cache = root / "structures.sqlite"
        structure_cache.write_bytes(b"SQLite format 3\x00" + b"\x00" * 128)

        pw = root / "pw.x"
        mpi = root / "mpirun"
        pw.write_bytes(b"synthetic-pw-binary")
        mpi.write_bytes(b"synthetic-mpi-binary")
        pw.chmod(0o755)
        mpi.chmod(0o755)

        pseudo_dir = root / "pseudos"
        pseudo_dir.mkdir()
        upf = pseudo_dir / "C.test.UPF"
        upf.write_text(
            '<UPF><PP_HEADER element="C" functional="PBE" '
            'relativistic="scalar"/></UPF>',
            encoding="utf-8",
        )
        upf_md5 = hashlib.md5(upf.read_bytes()).hexdigest()
        metadata = root / "SSSP_precision_metadata.json"
        metadata.write_text(
            json.dumps({
                "C": {
                    "filename": upf.name,
                    "md5": upf_md5,
                    "cutoff": 45,
                    "dual": 8,
                }
            }),
            encoding="utf-8",
        )
        manifest = root / "sssp_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "qe_pseudo_manifest_v1",
                    "library": "SSSP PBE Precision",
                    "library_version": "1.3.0-test",
                    "functional": "PBE",
                    "relativistic": "scalar_relativistic",
                    "source_metadata": str(metadata),
                    "source_metadata_sha256": hashlib.sha256(
                        metadata.read_bytes()
                    ).hexdigest(),
                    "pseudo_dir_at_manifest_creation": str(pseudo_dir),
                    "licenses_acknowledged_by_user": True,
                    "elements": {
                        "C": {
                            "filename": upf.name,
                            "sha256": hashlib.sha256(upf.read_bytes()).hexdigest(),
                            "original_md5": upf_md5,
                            "ecutwfc_ry": 45,
                            "ecutrho_ry": 360,
                            "upf_header_element": "C",
                            "upf_header_functional": "PBE",
                            "upf_header_relativistic": "scalar",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "raw_json": raw_json,
            "materials_csv": materials_csv,
            "structure_cache": structure_cache,
            "pw": pw,
            "mpi": mpi,
            "pseudo_dir": pseudo_dir,
            "metadata": metadata,
            "manifest": manifest,
            "upf": upf,
        }

    def _run(
        self,
        root: Path,
        files: dict[str, Path],
        *,
        probe_executables: bool = False,
        mpi_ranks: int = 2,
        omp_threads: int = 1,
    ) -> dict:
        return handoff.run_preflight(
            project_root=root,
            scratch_dir=root,
            min_free_gib=0,
            pw_executable=str(files["pw"]),
            mpi_executable=str(files["mpi"]),
            probe_executables=probe_executables,
            probe_timeout_seconds=5,
            raw_json_path=files["raw_json"],
            materials_csv_path=files["materials_csv"],
            structure_cache_path=files["structure_cache"],
            pseudo_manifest_path=files["manifest"],
            pseudo_dir=files["pseudo_dir"],
            sssp_metadata_path=files["metadata"],
            required_elements=["C"],
            mpi_ranks=mpi_ranks,
            omp_threads=omp_threads,
            host_machine="arm64",
        )

    @patch.object(
        handoff,
        "_check_python_environment",
        return_value={"status": "passed", "version": "test"},
    )
    @patch.object(
        handoff,
        "_describe_executable",
        return_value={
            "description": "Mach-O 64-bit executable arm64",
            "architectures": ["arm64"],
            "inspection_error": None,
        },
    )
    def test_ready_report_is_hash_bound_and_never_probes_by_default(
        self, _describe, _python
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = self._fixtures(root)
            with patch.object(handoff, "_probe_executable") as probe:
                report = self._run(root, files)

            self.assertEqual(report["status"], "ready")
            self.assertTrue(report["ready_for_qe_handoff"])
            self.assertEqual(report["blockers"], [])
            self.assertEqual(
                report["checks"]["executables"]["pw"]["sha256"],
                hashlib.sha256(files["pw"].read_bytes()).hexdigest(),
            )
            self.assertEqual(report["checks"]["sssp"]["verified_elements"], ["C"])
            self.assertTrue(
                report["checks"]["sssp"]["elements"]["C"][
                    "official_identity_verified"
                ]
            )
            self.assertFalse(
                report["checks"]["datasets"]["raw_mp_json"][
                    "full_hash_or_scan_performed"
                ]
            )
            probe.assert_not_called()

    def test_python_environment_imports_and_reports_required_packages(self):
        blockers: list[str] = []
        result = handoff._check_python_environment(blockers)

        self.assertEqual(blockers, [])
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["numpy_version"])
        self.assertTrue(result["ase_version"])
        self.assertTrue(result["pymatgen_version"])

    def test_interoperability_probe_uses_only_fixed_help_command_and_safe_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mpi = root / "mpirun"
            pw = root / "pw.x"
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="QE help", stderr=""
            )
            with patch.object(
                handoff.subprocess, "run", return_value=completed
            ) as run:
                result = handoff._probe_mpi_qe_interoperability(
                    mpi, pw, timeout_seconds=7
                )

            self.assertEqual(
                run.call_args.args[0],
                [str(mpi), "-np", "1", str(pw), "-help"],
            )
            self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
            self.assertTrue(run.call_args.kwargs["capture_output"])
            self.assertEqual(run.call_args.kwargs["timeout"], 7)
            self.assertFalse(run.call_args.kwargs["check"])
            for variable in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ):
                self.assertEqual(run.call_args.kwargs["env"][variable], "1")
            self.assertEqual(result["status"], "passed")

    def test_qe_pwscf_banner_with_exit_one_is_classified_as_passed(self):
        # QE 7.x prints the PWSCF banner, waits for input, then exits with
        # status 1 when ``pw.x -help`` is invoked with empty stdin. A healthy
        # binary must therefore be classified as "passed" even with non-zero
        # return code, both for the standalone probe and the MPI/QE probe.
        banner = (
            "Program PWSCF v.7.5 starts on 14Jul2026 at 20:40: 6\n"
            "This program is part of the open-source Quantum ESPRESSO suite\n"
            "Error in routine read_namelists (2):\n"
            " could not find namelist &control\n"
            "stopping ...\n"
        )
        standalone = handoff._classify_probe_result(
            kind="pw",
            returncode=1,
            combined_output=banner,
        )
        self.assertEqual(standalone, "passed")
        interoperability = handoff._classify_probe_result(
            kind="pw",
            returncode=1,
            combined_output=banner,
        )
        self.assertEqual(interoperability, "passed")

    def test_non_banner_nonzero_exit_is_classified_as_failed(self):
        # A non-QE failure (e.g. a wrapper script that exits with code 1 and
        # no PWSCF banner) must still fail closed and be reported as
        # ``nonzero_exit`` regardless of probe kind.
        for kind in ("pw", "mpi"):
            with self.subTest(kind=kind):
                status = handoff._classify_probe_result(
                    kind=kind,
                    returncode=1,
                    combined_output="synthetic wrapper failure: command not found",
                )
                self.assertEqual(status, "nonzero_exit")

    @patch.object(
        handoff,
        "_check_python_environment",
        return_value={"status": "passed", "version": "test"},
    )
    @patch.object(
        handoff,
        "_describe_executable",
        return_value={
            "description": "Mach-O 64-bit executable arm64",
            "architectures": ["arm64"],
            "inspection_error": None,
        },
    )
    def test_interoperability_nonzero_and_timeout_fail_closed(
        self, _describe, _python
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = self._fixtures(root)
            individual_pass = {
                "command": ["fixed-individual-probe"],
                "status": "passed",
                "timeout_seconds": 5,
                "returncode": 0,
                "output": "help",
            }
            for failure_status in ("nonzero_exit", "timeout"):
                interoperability_failure = {
                    "command": [
                        str(files["mpi"]),
                        "-np",
                        "1",
                        str(files["pw"]),
                        "-help",
                    ],
                    "status": failure_status,
                    "timeout_seconds": 5,
                    "returncode": 1 if failure_status == "nonzero_exit" else None,
                    "output": "synthetic failure",
                }
                with self.subTest(status=failure_status), patch.object(
                    handoff,
                    "_probe_executable",
                    return_value=individual_pass,
                ), patch.object(
                    handoff,
                    "_probe_mpi_qe_interoperability",
                    return_value=interoperability_failure,
                ):
                    report = self._run(root, files, probe_executables=True)

                self.assertEqual(report["status"], "blocked")
                self.assertFalse(report["ready_for_qe_handoff"])
                self.assertIn(
                    "mpi_qe_interoperability:probe_failed", report["blockers"]
                )
                self.assertEqual(
                    report["checks"]["executables"]["mpi_qe_interoperability"][
                        "status"
                    ],
                    failure_status,
                )

    @patch.object(
        handoff,
        "_check_python_environment",
        return_value={"status": "passed", "version": "test"},
    )
    @patch.object(
        handoff,
        "_describe_executable",
        return_value={
            "description": "Mach-O 64-bit executable arm64",
            "architectures": ["arm64"],
            "inspection_error": None,
        },
    )
    def test_interoperability_probe_is_not_attempted_without_both_binaries(
        self, _describe, _python
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = self._fixtures(root)
            files["mpi"].unlink()
            with patch.object(
                handoff,
                "_probe_executable",
                return_value={"status": "passed"},
            ), patch.object(
                handoff, "_probe_mpi_qe_interoperability"
            ) as interoperability:
                report = self._run(root, files, probe_executables=True)

            interoperability.assert_not_called()
            self.assertIn("mpi_executable:not_found", report["blockers"])
            self.assertEqual(
                report["checks"]["executables"]["mpi_qe_interoperability"][
                    "status"
                ],
                "not_run_missing_executable",
            )

    @patch.object(
        handoff,
        "_check_python_environment",
        return_value={"status": "passed", "version": "test"},
    )
    @patch.object(
        handoff,
        "_describe_executable",
        return_value={
            "description": "Mach-O 64-bit executable arm64",
            "architectures": ["arm64"],
            "inspection_error": None,
        },
    )
    def test_explicit_serial_handoff_does_not_require_or_probe_mpi(
        self, _describe, _python
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = self._fixtures(root)
            files["mpi"].unlink()
            passed_probe = {
                "command": [str(files["pw"]), "-help"],
                "status": "passed",
                "timeout_seconds": 5,
                "returncode": 0,
                "output": "help",
            }
            with patch.object(
                handoff, "_probe_executable", return_value=passed_probe
            ) as executable_probe, patch.object(
                handoff, "_probe_mpi_qe_interoperability"
            ) as interoperability_probe:
                report = self._run(
                    root,
                    files,
                    probe_executables=True,
                    mpi_ranks=1,
                    omp_threads=2,
                )

            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["execution_policy"]["mode"], "serial")
            self.assertEqual(report["execution_policy"]["mpi_ranks"], 1)
            self.assertEqual(report["execution_policy"]["omp_threads"], 2)
            self.assertEqual(
                report["checks"]["executables"]["mpi"]["status"],
                "not_required_serial",
            )
            executable_probe.assert_called_once()
            interoperability_probe.assert_not_called()

    @patch.object(
        handoff,
        "_check_python_environment",
        return_value={"status": "passed", "version": "test"},
    )
    @patch.object(
        handoff,
        "_describe_executable",
        return_value={
            "description": "Mach-O 64-bit executable x86_64",
            "architectures": ["x86_64"],
            "inspection_error": None,
        },
    )
    def test_blocks_x86_only_executables_on_arm64(self, _describe, _python):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = self._fixtures(root)
            report = self._run(root, files)

            self.assertEqual(report["status"], "blocked")
            self.assertIn("pw_executable:not_native_for_arm64", report["blockers"])
            self.assertIn("mpi_executable:not_native_for_arm64", report["blockers"])

    @patch.object(
        handoff,
        "_check_python_environment",
        return_value={"status": "passed", "version": "test"},
    )
    @patch.object(
        handoff,
        "_describe_executable",
        return_value={
            "description": "Mach-O 64-bit executable arm64",
            "architectures": ["arm64"],
            "inspection_error": None,
        },
    )
    def test_blocks_tampered_pseudopotential(self, _describe, _python):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = self._fixtures(root)
            files["upf"].write_text("tampered", encoding="utf-8")
            report = self._run(root, files)

            self.assertEqual(report["status"], "blocked")
            self.assertIn("sssp:C_hash_mismatch", report["blockers"])

    @patch.object(
        handoff,
        "_check_python_environment",
        return_value={"status": "passed", "version": "test"},
    )
    @patch.object(
        handoff,
        "_describe_executable",
        return_value={
            "description": "Mach-O 64-bit executable arm64",
            "architectures": ["arm64"],
            "inspection_error": None,
        },
    )
    def test_blocks_manifest_that_no_longer_matches_official_sssp_metadata(
        self, _describe, _python
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = self._fixtures(root)
            metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
            metadata["C"]["cutoff"] = 46
            files["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            manifest["source_metadata_sha256"] = hashlib.sha256(
                files["metadata"].read_bytes()
            ).hexdigest()
            files["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

            report = self._run(root, files)

            self.assertEqual(report["status"], "blocked")
            self.assertIn("sssp:C_official_identity_mismatch", report["blockers"])

    @patch.object(
        handoff,
        "_check_python_environment",
        return_value={"status": "passed", "version": "test"},
    )
    @patch.object(
        handoff,
        "_describe_executable",
        return_value={
            "description": "Mach-O 64-bit executable arm64",
            "architectures": ["arm64"],
            "inspection_error": None,
        },
    )
    def test_accepts_sssp_v1_3_0_cutoff_wfc_cutoff_rho_metadata(
        self, _describe, _python,
    ):
        # SSSP 1.3.0 publishes independent ``cutoff_wfc`` and ``cutoff_rho``
        # instead of the legacy ``cutoff``+``dual`` pair. The checker must
        # accept this layout without flagging ``official_cutoffs_invalid``.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = self._fixtures(root)
            upf = files["upf"]
            upf_md5 = hashlib.md5(upf.read_bytes()).hexdigest()
            metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
            metadata["C"] = {
                "filename": upf.name,
                "md5": upf_md5,
                "cutoff_wfc": 45,
                "cutoff_rho": 360,
            }
            files["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            manifest["source_metadata_sha256"] = hashlib.sha256(
                files["metadata"].read_bytes()
            ).hexdigest()
            files["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

            report = self._run(root, files)

            self.assertEqual(report["status"], "ready")
            self.assertTrue(report["ready_for_qe_handoff"])
            self.assertEqual(report["blockers"], [])
            self.assertTrue(
                report["checks"]["sssp"]["elements"]["C"][
                    "official_identity_verified"
                ]
            )

    def test_rejects_unsafe_probe_timeout_before_any_probe(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 30"):
            handoff.run_preflight(
                project_root=Path("."),
                scratch_dir=Path("."),
                min_free_gib=0,
                pw_executable="pw.x",
                mpi_executable="mpirun",
                probe_executables=True,
                probe_timeout_seconds=31,
                raw_json_path=Path("missing"),
                materials_csv_path=Path("missing"),
                structure_cache_path=Path("missing"),
                pseudo_manifest_path=None,
                pseudo_dir=None,
                sssp_metadata_path=None,
                required_elements=["C"],
            )

    def test_rejects_invalid_execution_resource_counts(self):
        with self.assertRaisesRegex(ValueError, "mpi_ranks"):
            handoff.run_preflight(
                project_root=Path("."),
                scratch_dir=Path("."),
                min_free_gib=0,
                pw_executable="pw.x",
                mpi_executable="mpirun",
                probe_executables=False,
                probe_timeout_seconds=5,
                raw_json_path=Path("missing"),
                materials_csv_path=Path("missing"),
                structure_cache_path=Path("missing"),
                pseudo_manifest_path=None,
                pseudo_dir=None,
                sssp_metadata_path=None,
                required_elements=["C"],
                mpi_ranks=0,
            )

    def test_cli_writes_json_and_returns_two_for_blockers(self):
        blocked = {
            "schema_version": "qe_handoff_environment_v1",
            "status": "blocked",
            "ready_for_qe_handoff": False,
            "blockers": ["pw_executable:not_found"],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "reports" / "environment.json"
            with patch.object(handoff, "run_preflight", return_value=blocked):
                with redirect_stdout(io.StringIO()):
                    exit_code = handoff.main(["--output", str(output)])

            self.assertEqual(exit_code, 2)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), blocked)


if __name__ == "__main__":
    unittest.main()
