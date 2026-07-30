from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_mp_reference_inventory import build_inventory


class MPReferenceInventoryGuardTests(unittest.TestCase):
    def test_build_refuses_nonempty_output_before_reading_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "inventory"
            output.mkdir()
            stale = output / "operator-owned.txt"
            stale.write_text("do not overwrite", encoding="utf-8")
            missing = root / "missing"

            with self.assertRaisesRegex(FileExistsError, "non-empty output"):
                build_inventory(
                    candidate_entries_path=missing,
                    snapshot_manifest_path=missing,
                    materials_csv_path=missing,
                    structure_cache_path=missing,
                    output_dir=output,
                )
            self.assertEqual(stale.read_text(encoding="utf-8"), "do not overwrite")


if __name__ == "__main__":
    unittest.main()
