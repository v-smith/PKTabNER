import re
import unittest

from pktabner.ner.patterns import normalise_unit_synonyms, UNIT_SYNONYMS, UNIT_SPAN_PATTERN


class TestPKUnitPattern(unittest.TestCase):

    def test_unit_regex_matches(self):
        examples = [
            "μmol/l", "mmol/l*min", "pmol/min/l", "min", "nmol/l", "l/hr",
            "hr × ug/ml", "1/(nm · d)", "l/m2", "min μmol/l", "iu/l·hr",
            "hour−1", "lh−1kg−1", "1/hour", "min/l", "l/nmol/h",
            "nmol⋅l-1", "ml min-1", "μmol/l·h",
            "ng/mL", "mg/L", "µg/mL", "ng/L",
            "ng·h/mL", "µg·hr/L", "mg·min/L", "ng/mL/mg",
            "L/h", "mL/min", "mL/hr/kg", "L/kg", "L/hr/kg", "mL/min/kg",
            "hr", "min", "days",
        ]

        for ex in examples:
            norm = normalise_unit_synonyms(ex, UNIT_SYNONYMS)
            with self.subTest(text=ex):
                match = re.search(UNIT_SPAN_PATTERN, norm, re.IGNORECASE)
                self.assertIsNotNone(match, f"Should match: {ex}")


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)