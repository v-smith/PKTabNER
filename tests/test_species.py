import unittest
import re

from pktabner.ner.patterns import SPECIES_PATTERNS


class TestSpeciesPatterns(unittest.TestCase):

    def test_species_matches(self):
        test_cases = {
            "animals": [
                "rats", "mouse", "mice", "rabbit", "hamsters", "monkey", "dogs",
                "goat", "sheep", "macaque", "chimpanzee", "animal model"
            ],
            "humans": [
                "human subjects", "healthy volunteers", "patients",
                "study participants", "human trial"
            ]
        }

        for label, examples in test_cases.items():
            pattern = SPECIES_PATTERNS[label]
            for text in examples:
                with self.subTest(label=label, text=text):
                    self.assertRegex(text, pattern, f"❌ Failed to match '{text}' as {label}")

    def test_species_negatives(self):
        negatives = [
            "cell line", "in vitro", "tissue sample", "study group", "clinical data",
            "enzymes", "bacteria", "simulation"
        ]
        matched_labels = []

        for text in negatives:
            matched = [label for label, pattern in SPECIES_PATTERNS.items() if re.search(pattern, text, re.IGNORECASE)]
            if matched:
                matched_labels.append((text, matched))
            self.assertFalse(matched, f"❌ Unexpected match for '{text}': {matched}")

        if matched_labels:
            print("\n🔍 False positives:")
            for text, labels in matched_labels:
                print(f"  - '{text}' matched as {labels}")

if __name__ == "__main__":
    unittest.main()