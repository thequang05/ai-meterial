from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure

from collect_qe_static import collect_static
from prepare_qe_jobs import prepare_jobs
from prepare_sssp_manifest import create_manifest
from prepare_qe_static_jobs import prepare_static_jobs
from run_qe_jobs import run_jobs


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
            pseudo_dir = root / "source_pseudos"
            pseudo_dir.mkdir()
            pseudo_entries = {}
            for symbol in ("C", "Ti"):
                pseudo = pseudo_dir / f"{symbol}.UPF"
                pseudo.write_text(
                    f'<UPF><PP_HEADER element="{symbol}" functional="PBE" '
                    'relativistic="scalar"/></UPF>',
                    encoding="utf-8",
                )
                pseudo_entries[symbol] = {
                    "filename": pseudo.name,
                    "md5": hashlib.md5(pseudo.read_bytes()).hexdigest(),
                    "cutoff": 50,
                    "dual": 8,
                    "pseudopotential": "synthetic-test-family",
                }
            metadata_path = root / "official_pseudo_metadata.json"
            _write_json(metadata_path, pseudo_entries)
            pseudo_manifest = root / "pseudo.json"
            create_manifest(
                metadata_path=metadata_path,
                pseudo_dir=pseudo_dir,
                output_path=pseudo_manifest,
                library_name="SSSP PBE Precision",
                library_version="1",
                elements=["C", "Ti"],
                acknowledge_original_licenses=True,
            )
            relax_dir = root / "relax"
            relax_preflight = prepare_jobs(
                validation_report_path=report,
                config_path=config,
                output_dir=relax_dir,
                top_n=1,
                pseudo_manifest_path=pseudo_manifest,
                pseudo_dir=pseudo_dir,
                pw_executable="/usr/bin/true",
            )
            relax_results = root / "relax_results.csv"
            with relax_results.open("w", newline="", encoding="utf-8") as handle:
                fields = [
                    "rank", "entry_id", "entry_role", "source_id",
                    "candidate_id", "formula", "relax_gate_status",
                    "final_cif", "initial_source_cif_sha256", "final_cif_sha256",
                    "relax_input_settings_hash", "settings_hash",
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
                })

            static_dir = root / "static"
            static_preflight = prepare_static_jobs(
                relaxation_results_path=relax_results,
                relax_preflight_path=relax_dir / "dft_preflight.json",
                output_dir=static_dir,
                pw_executable="/usr/bin/true",
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
