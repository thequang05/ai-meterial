from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from collect_qe_convergence import collect_convergence
from collect_qe_convergence_confirmation import collect_confirmation
from collect_qe_relaxations import collect_relaxations
from collect_qe_static import collect_static
from compute_qe_hull import compute_hulls
from qe_output_directory import require_fresh_output_dir


class QeOutputDirectoryGuardTests(unittest.TestCase):
    def test_helper_allows_absent_and_existing_empty_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            absent = root / "absent"
            self.assertEqual(require_fresh_output_dir(absent), absent.resolve())
            self.assertFalse(absent.exists())

            empty = root / "empty"
            empty.mkdir()
            self.assertEqual(require_fresh_output_dir(empty), empty.resolve())

    def test_every_collector_rejects_any_nonempty_output_directory_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing.json"

            calls = [
                lambda output: collect_convergence(
                    preflight_path=missing, output_dir=output
                ),
                lambda output: collect_confirmation(
                    preflight_path=missing, output_dir=output
                ),
                lambda output: collect_relaxations(
                    preflight_path=missing, output_dir=output
                ),
                lambda output: collect_static(
                    preflight_path=missing, output_dir=output
                ),
                lambda output: compute_hulls(
                    candidate_entries_path=missing,
                    candidate_summary_path=missing,
                    reference_entries_path=missing,
                    reference_summary_path=missing,
                    reference_inventory_path=missing,
                    coverage_manifest_path=missing,
                    output_dir=output,
                ),
            ]
            for index, call in enumerate(calls):
                output = root / f"nonempty_{index}"
                output.mkdir()
                (output / "unrelated-stale-artifact.txt").write_text(
                    "stale", encoding="utf-8"
                )
                with self.subTest(index=index):
                    with self.assertRaisesRegex(
                        FileExistsError, "non-empty output directory"
                    ):
                        call(output)

    def test_helper_rejects_a_file_output_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "not-a-directory"
            output.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not a directory"):
                require_fresh_output_dir(output)


if __name__ == "__main__":
    unittest.main()
