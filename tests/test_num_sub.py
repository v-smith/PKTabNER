import unittest
import re

from pktabner.ner.patterns import N_SUB_PATTERNS


class TestNSubPatterns(unittest.TestCase):

    def test_n_sub_inline(self):
        pattern = N_SUB_PATTERNS["n_sub_inline_patterns"]
        positive = [
            "n = 42",
            "N=100",
            "n= 5",
            " n =123 ",
        ]
        negative = [
            "Number of patients",
            "Subjects enrolled",
            "study size: 20",
            "naive subjects"
        ]
        for text in positive:
            self.assertIsNotNone(re.search(pattern, text, re.IGNORECASE), f"Should match: {text}")
        for text in negative:
            self.assertIsNone(re.search(pattern, text, re.IGNORECASE), f"Should NOT match: {text}")

    def test_n_sub_headers(self):
        pattern = N_SUB_PATTERNS["n_sub_headers"]
        positive = [
            "Number of patients",
            "no. of subjects",
            "n of animals enrolled",
            "study size",
            "patients enrolled",
            "subjects n° 20",
            "n"
        ]
        negative = [
            "patient characteristics",
            "blood concentration",
            "trial outcome",
            "this is not related to n"
        ]
        for text in positive:
            self.assertIsNotNone(re.search(pattern, text, re.IGNORECASE), f"Should match: {text}")
        for text in negative:
            self.assertIsNone(re.search(pattern, text, re.IGNORECASE), f"Should NOT match: {text}")

