from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure

from collect_qe_static import collect_static
from collect_qe_convergence import collect_convergence
from collect_qe_convergence_confirmation import collect_confirmation
from prepare_qe_jobs import prepare_jobs
from prepare_qe_convergence_confirmation import prepare_confirmation_jobs
from prepare_qe_static_jobs import prepare_static_jobs
from run_qe_jobs import run_jobs
from test_qe_convergence_workflow import (
    SERIAL_EXECUTION,
    _prepare_convergence,
    _write_confirmation_outputs,
    _write_synthetic_outputs,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StaticWorkflowTests(unittest.TestCase):
    def test_prepares_plans_and_collects_static_without_running_qe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            structure = Structure(
                Lattice.cubic(4.3), ["Ti", "C"], [[0, 0, 0], [0.5, 0.5, 0.5]]
            )
            cif = root / "tic.cif"
            structure.to(filename=cif)
            report = root / "report.json"
            _write_json(report, {
                "status": "ml_structural_validation_complete",
                "recommended_dft_queue": [{
                    "rank": 1,
                    "candidate_id": "tic_candidate",
                    "formula": "TiC",
                    "relaxed_cif": str(cif),
                }],
            })
            config = root / "config.json"
            _write_json(config, {
                "schema_version": "qe_dft_config_v1",
                "input_dft": "PBE",
                "kpoint_spacing_inv_angstrom": 0.3,
                "occupations": "smearing",
                "smearing": "mv",
                "degauss_ry": 0.02,
                "conv_thr": 1e-8,
                "electron_maxstep": 100,
                "mixing_beta": 0.3,
                "diagonalization": "david",
                "nstep": 100,
                "etot_conv_thr_ry": 1e-5,
                "forc_conv_thr_ry_per_bohr": 8e-4,
                "press_conv_thr_kbar": 0.5,
                "cell_dofree": "all",
                "static_kpoint_spacing_inv_angstrom": 0.2,
                "static_degauss_ry": 0.01,
                "static_conv_thr": 1e-10,
                "resource_policy": {
                    "max_parallel_jobs": 1,
                    "recommended_mpi_ranks_on_this_mac": 2,
                    "omp_threads_per_rank": 1,
                },
            })
            certificate_root = root / "certificate_campaign"
            certificate_root.mkdir()
            sweep_path, sweep = _prepare_convergence(certificate_root)
            _write_synthetic_outputs(sweep_path, sweep)
            sweep_collected = certificate_root / "sweep_collected"
            collect_convergence(
                preflight_path=sweep_path, output_dir=sweep_collected
            )
            confirmation_dir = certificate_root / "confirmation"
            confirmation = prepare_confirmation_jobs(
                provisional_summary_path=sweep_collected / "convergence_summary.json",
                sweep_preflight_path=sweep_path,
                output_dir=confirmation_dir,
                pw_executable="/usr/bin/true",
            )
            confirmation_path = confirmation_dir / "confirmation_preflight.json"
            _write_confirmation_outputs(confirmation_path, confirmation)
            confirmation_collected = certificate_root / "confirmation_collected"
            collect_confirmation(
                preflight_path=confirmation_path,
                output_dir=confirmation_collected,
            )
            certificate_path = (
                confirmation_collected / "qe_convergence_certificate.json"
            )
            source = json.loads(
                (certificate_root / "source_relax" / "dft_preflight.json").read_text(
                    encoding="utf-8"
                )
            )
            config = Path(source["config_source"])
            pseudo_manifest = Path(source["pseudo_manifest_source"])
            pseudo_dir = certificate_root / "source_pseudos"
            relax_dir = root / "relax"
            relax_preflight = prepare_jobs(
                validation_report_path=report,
                config_path=config,
                output_dir=relax_dir,
                top_n=1,
                pseudo_manifest_path=pseudo_manifest,
                pseudo_dir=pseudo_dir,
                pw_executable="/usr/bin/true",
                convergence_certificate_path=certificate_path,
            )
            relax_results = root / "relax_results.csv"
            with relax_results.open("w", newline="", encoding="utf-8") as handle:
                fields = [
                    "rank", "entry_id", "entry_role", "source_id",
                    "candidate_id", "formula", "relax_gate_status",
                    "final_cif", "initial_source_cif_sha256", "final_cif_sha256",
                    "relax_input_settings_hash", "settings_hash",
                    "qe_program_version", "pw_executable_sha256",
                    "convergence_certificate_payload_sha256",
                    "execution_provenance",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                input_settings_hash = relax_preflight["relax_settings_hash"]
                effective_relax_hash = hashlib.sha256(
                    json.dumps({
                        "relax_input_settings_hash": input_settings_hash,
                        "qe_program_version": "7.4",
                        "pw_executable_sha256": _sha(Path("/usr/bin/true")),
                    }, sort_keys=True).encode("utf-8")
                ).hexdigest()
                writer.writerow({
                    "rank": 1,
                    "entry_id": "candidate:tic_candidate",
                    "entry_role": "candidate",
                    "source_id": "tic_candidate",
                    "candidate_id": "tic_candidate",
                    "formula": "TiC",
                    "relax_gate_status": "dft_relax_converged",
                    "final_cif": str(cif),
                    "initial_source_cif_sha256": _sha(cif),
                    "final_cif_sha256": _sha(cif),
                    "relax_input_settings_hash": input_settings_hash,
                    "settings_hash": effective_relax_hash,
                    "qe_program_version": "7.4",
                    "pw_executable_sha256": _sha(Path("/usr/bin/true")),
                    "convergence_certificate_payload_sha256": relax_preflight[
                        "convergence_certificate_payload_sha256"
                    ],
                    "execution_provenance": json.dumps(
                        SERIAL_EXECUTION, sort_keys=True
                    ),
                })

            relax_preflight_path = relax_dir / "dft_preflight.json"
            relax_summary = root / "qe_relaxation_summary.json"
            _write_json(relax_summary, {
                "collector_version": "qe_relax_collector_v1",
                "source_preflight": str(relax_preflight_path.resolve()),
                "source_preflight_sha256": _sha(relax_preflight_path),
                "source_queue_manifest": relax_preflight["queue_manifest"],
                "source_queue_manifest_sha256": relax_preflight[
                    "queue_manifest_sha256"
                ],
                "relax_input_settings_hash": input_settings_hash,
                "convergence_certificate": relax_preflight[
                    "convergence_certificate"
                ],
                "convergence_certificate_sha256": relax_preflight[
                    "convergence_certificate_sha256"
                ],
                "convergence_certificate_id": relax_preflight[
                    "convergence_certificate_id"
                ],
                "convergence_certificate_payload_sha256": relax_preflight[
                    "convergence_certificate_payload_sha256"
                ],
                "execution_provenance": SERIAL_EXECUTION,
                "settings_hash": effective_relax_hash,
                "settings_hashes": [effective_relax_hash],
                "candidate_count": 1,
                "status_counts": {"dft_relax_converged": 1},
                "dft_relax_converged_count": 1,
                "results_manifest": str(relax_results.resolve()),
                "results_manifest_sha256": _sha(relax_results),
            })

            original_results = relax_results.read_bytes()
            relax_results.write_bytes(original_results + b"\n")
            with self.assertRaisesRegex(ValueError, "results manifest hash mismatch"):
                prepare_static_jobs(
                    relaxation_results_path=relax_results,
                    relaxation_summary_path=relax_summary,
                    relax_preflight_path=relax_preflight_path,
                    output_dir=root / "static_tampered",
                    pw_executable="/usr/bin/true",
                )
            relax_results.write_bytes(original_results)

            static_dir = root / "static"
            static_scratch = root / "static_scratch"
            static_preflight = prepare_static_jobs(
                relaxation_results_path=relax_results,
                relaxation_summary_path=relax_summary,
                relax_preflight_path=relax_preflight_path,
                output_dir=static_dir,
                pw_executable="/usr/bin/true",
                scratch_root=static_scratch,
            )
            self.assertEqual(static_preflight["status"], "runnable_not_started")
            static_input = next(static_dir.glob("jobs/*/static-scf.in"))
            input_text = static_input.read_text(encoding="utf-8")
            self.assertIn("calculation = 'scf'", input_text)
            self.assertNotIn("vc-relax", input_text)
            self.assertIn("degauss = 0.01", input_text)

            plan = run_jobs(
                preflight_path=static_dir / "static_preflight.json",
                execute=False,
                max_jobs=1,
                mpi_ranks=2,
                omp_threads=1,
                mpi_executable="definitely_not_installed_mpirun",
            )
            self.assertEqual(plan["status"], "planned_not_executed")
            self.assertEqual(plan["provenance_blockers"], [])

            queue_path = Path(static_preflight["queue_manifest"])
            with queue_path.open(newline="", encoding="utf-8") as handle:
                queue_row = next(csv.DictReader(handle))
            self.assertEqual(
                Path(static_preflight["scratch_root"]), static_scratch.resolve()
            )
            self.assertTrue(Path(queue_row["qe_scratch_outdir"]).is_dir())
            self.assertIn(queue_row["qe_scratch_outdir"], input_text)
            qe_output = Path(queue_row["qe_output"])
            qe_output.write_text(
                """
Program PWSCF v.7.4 starts
convergence has been achieved in 8 iterations
!    total energy = -10.00000000 Ry
JOB DONE.
""",
                encoding="utf-8",
            )
            run_record = Path(queue_row["run_record"])
            run_record.write_text(json.dumps({
                "run_status": "completed_requires_collection",
                "entry_id": queue_row["entry_id"],
                "entry_role": queue_row["entry_role"],
                "source_id": queue_row["source_id"],
                "candidate_id": queue_row["candidate_id"],
                "formula": queue_row["formula"],
                "qe_input_sha256": queue_row["qe_input_sha256"],
                "qe_output_sha256": _sha(qe_output),
                "preflight_sha256": _sha(static_dir / "static_preflight.json"),
                "queue_manifest_sha256": static_preflight["queue_manifest_sha256"],
                "preflight_settings_hash": static_preflight["static_settings_hash"],
                "qe_program_version": "7.4",
                "pw_executable_sha256": _sha(Path("/usr/bin/true")),
                "execution_provenance": SERIAL_EXECUTION,
            }), encoding="utf-8")
            summary = collect_static(
                preflight_path=static_dir / "static_preflight.json",
                output_dir=root / "collected",
            )
            self.assertEqual(summary["dft_static_converged_count"], 1)
            self.assertEqual(len(summary["static_settings_hash"]), 64)
            self.assertNotEqual(
                summary["static_settings_hash"],
                summary["static_input_settings_hash"],
            )
            self.assertTrue(
                (root / "collected" / "candidate_static_energies.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
