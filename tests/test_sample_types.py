import unittest
import re

from pktabner.ner.patterns import SAMPLE_TYPE_PATTERNS


class TestSampleTypePatterns(unittest.TestCase):
    def test_sample_type_matching(self):
        matched_texts = [
            "spleen", "kidney", "synovial fluid", "plasma", "salivary",
            "foetal plasma", "whole blood", "ileum content", "brain",
            "urinary", "synovium", "conjunctiva", "cornea", "heart",
            "umbilical vein", "plasma free", "isf", "lung tissue", "urine",
            "eyelid", "csf", "liver microsomes", "perilymph", "lung",
            "synovial lavage fluid", "serum", "breast milk", "feces",
            "blood/plasma", "hepatic", "liver", "blood", "lymphatic fluid"
        ]

        for text in matched_texts:
            norm = text.lower().strip()
            matched = any(
                re.search(pattern, norm, re.IGNORECASE)
                for pattern in SAMPLE_TYPE_PATTERNS.values()
            )
            self.assertTrue(matched, f"❌ Should match but didn't: '{text}'")

    def test_sample_type_non_matching(self):
        negative_texts = [
            "muscle", "bone marrow", "skin", "nerves", "tissue",
            "cells", "DNA", "RNA", "study population", "healthy volunteers",
            "baseline sample", "trial arm", "timepoint", "plasmid",
            "plasmacytoid", "blood pressure"
        ]

        for text in negative_texts:
            norm = text.lower().strip()
            matched = any(
                re.search(pattern, norm, re.IGNORECASE)
                for pattern in SAMPLE_TYPE_PATTERNS.values()
            )
            self.assertFalse(matched, f"❌ Should NOT match: '{text}'")

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)