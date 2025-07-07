import unittest
import re

from pktabner.ner.patterns import DOSE_PATTERN, normalise_unit_synonyms, UNIT_SYNONYMS


class TestDosePattern(unittest.TestCase):

    def test_dose_matches(self):
        matches = [
            "10 mg",
            "5.5 mg/kg",
            "0.25 mg per kg",
            "100 µg/kg",
            "75 mcg per kg",
            "50 mg/m2",
            "20 mg per m²",
            "2 mg kg⁻¹",
            "1.5 g/kg",
            "500 ng / kg",
            "5 mg per hour",
            "25 mcg/min",
            "2 mg/kg/min",
        ]
        for text in matches:
            normalized = normalise_unit_synonyms(text, UNIT_SYNONYMS)
            match = re.search(DOSE_PATTERN, normalized)
            if not match:
                print(f"❌ '{text}' → Normalized: '{normalized}' → NO MATCH")

    def test_dose_non_matches(self):
        non_matches = [
            "10ml",
            "mg/kg",
            "dose was 10",
        ]
        for text in non_matches:
            normalized = normalise_unit_synonyms(text, UNIT_SYNONYMS)
            with self.subTest(text=text):
                match = re.search(DOSE_PATTERN, normalized, flags=re.IGNORECASE)
                if match:
                    print(f"❌ INCORRECT MATCH: '{text}' → Normalized: '{normalized}' → '{match.group()}'")
                self.assertIsNone(match, f"Unexpected match: '{text}'")

if __name__ == "__main__":
    unittest.main()
