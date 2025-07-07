import unittest
import re

from pktabner.ner.patterns import MEASURE_PATTERNS


class TestDataTypePatterns(unittest.TestCase):

    def test_data_type_patterns(self):
        examples = {
            "ratios": [
                "geometric mean ratio", "GMR", "ratio of geometric means", "ratios", "ratio"
            ],
            "geometric_mean": [
                "geometric mean", "geom mean", "G mean", "gmean", "GM", "geo. mean"
            ],
            "mean": [
                "mean", "arithmetic mean", "average", "mean values"
            ],
            "typical_value": [
                "typical value", "TV", "tv", "t.v."
            ],
            "median": [
                "median"
            ],
            "range": [
                "range", "min, max", "min–max"
            ],
            "interquartile_range": [
                "interquartile range", "25–75th percentiles", "iqr"
            ],
            "confidence_interval": [
                "90% CI", "95.0%ci", "CI interval range", "confidence intervals", "CI"
            ],
            "standard_deviation": [
                "SD", "s.d.", "standard deviation"
            ],
            "standard_error": [
                "SE", "standard error", "s.e.m", "s.e."
            ],
            "coefficient_of_variation": [
                "CV%", "CV %", "CV", "geoCV"
            ],
            "variance": [
                "var", "variance"
            ]
        }

        for label, texts in examples.items():
            pattern = re.compile(MEASURE_PATTERNS[label], re.IGNORECASE)
            for text in texts:
                with self.subTest(label=label, text=text):
                    match = pattern.search(text)
                    if not match:
                        print(f"❌ Pattern '{label}' failed to match: '{text}'")
                    self.assertIsNotNone(match, f"Pattern '{label}' failed to match: '{text}'")
