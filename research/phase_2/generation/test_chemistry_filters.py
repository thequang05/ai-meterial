from __future__ import annotations

import unittest

from chemistry_filters import validate_decoded_atomic_numbers


class ChemistryFilterTests(unittest.TestCase):
    def test_valid_refractory_carbide_substitution_passes_without_mutation(self):
        original = [73, 22, 74, 6, 6, 6]  # TaTiWC3
        decoded = [22, 22, 74, 6, 6, 6]   # Ti2WC3
        result = validate_decoded_atomic_numbers(
            decoded,
            original,
            required_elements=["W", "C"],
            allowed_elements=["C", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"],
        )
        self.assertTrue(result.passed)
        self.assertEqual(decoded, [22, 22, 74, 6, 6, 6])
        self.assertEqual(result.metadata["num_substitutions"], 1)

    def test_outside_domain_is_rejected_not_replaced(self):
        decoded = [73, 22, 74, 9, 6, 6]  # F must remain F for the rejection reason
        result = validate_decoded_atomic_numbers(
            decoded,
            [73, 22, 74, 6, 6, 6],
            required_elements=["W", "C"],
            allowed_elements=["C", "Ti", "Ta", "W"],
        )
        self.assertFalse(result.passed)
        self.assertIn("outside_allowed_domain:F", result.reasons)
        self.assertEqual(decoded[3], 9)

    def test_unchanged_and_invalid_atomic_number_are_rejected(self):
        unchanged = validate_decoded_atomic_numbers([74, 6], [74, 6])
        self.assertFalse(unchanged.passed)
        self.assertIn("unchanged_from_prototype", unchanged.reasons)

        invalid = validate_decoded_atomic_numbers([74, 0], [74, 6])
        self.assertFalse(invalid.passed)
        self.assertIn("invalid_atomic_number:0", invalid.reasons)


if __name__ == "__main__":
    unittest.main()
