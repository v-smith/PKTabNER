import unittest
import re

from pktabner.ner.patterns import WEIGHT_PATTERNS


class TestWeightPatterns(unittest.TestCase):

    def test_weight_patterns(self):
        examples = {
            "weight_value": [
                "70 kg", "55 kilograms", "88.5kg", "100.0 kilograms"
            ],
            "weight_range": [
                "65 – 75 kg", "50-60 kilograms", "55.0 - 63.5 kg", "70 – 80 kilograms"
            ],
            "weight_header": [
                "Body weight", "body-weight", "body wt", "BWT", "wt", "weights"
            ]
        }

        for label, texts in examples.items():
            pattern = re.compile(WEIGHT_PATTERNS[label], re.IGNORECASE)
            for text in texts:
                with self.subTest(label=label, text=text):
                    match = pattern.search(text)
                    if not match:
                        print(f"❌ Pattern '{label}' failed to match: '{text}'")
                    self.assertIsNotNone(match, f"Pattern '{label}' failed to match: '{text}'")

