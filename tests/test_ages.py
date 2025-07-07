import unittest
import re

from pktabner.ner.patterns import AGE_PATTERNS


class TestAgePatterns(unittest.TestCase):

    def test_age_group_matching(self):
        test_cases = {
            "neonates": [
                "newborn", "neonates", "1-week-old", "age < 1 month", "<1 mo old"
            ],
            "infants": [
                "infants", "18 months old", "age ≤ 24 months", "0 to 2 years", "subjects < 2 years"
            ],
            "children": [
                "children", "older children", "≥2 years to <12 years", "5 yrs old", "subjects < 12 yrs"
            ],
            "adolescents": [
                "adolescents", "≥12 years to <18 years", "age 13 to 17 years"
            ],
            "pediatric_patients": [
                "paediatric patients", "children and adolescents", "age 0–17 years", "subjects < 18 yrs"
            ],
            "adults": [
                "adults", "aged 18 to 64", "age ≥ 18 years to < 65 years"
            ],
            "elderly_patients": [
                "elderly", "seniors", "aged 65 and older", "geriatric population", "age ≥ 65 years"
            ]
        }

        for label, examples in test_cases.items():
            pattern = AGE_PATTERNS[label]
            for example in examples:
                with self.subTest(label=label, example=example):
                    if not re.search(pattern, example, flags=re.IGNORECASE):
                        print(f"\n🔍 Pattern failed for: '{example}' with pattern: {pattern}")
                    self.assertRegex(example, pattern, msg=f"Failed to match {example} for {label}")

    def test_age_group_non_matches(self):
        non_matches = [
            "study subjects", "healthy volunteers", "middle-aged", "baseline", "sample size",
            "randomised cohort", "ageing process", "mature subjects", "height 130 cm",
            "age group 4", "age group 1"
        ]
        for text in non_matches:
            matched_patterns = [
                label for label, pattern in AGE_PATTERNS.items()
                if re.search(pattern, text, re.IGNORECASE)
            ]

            if matched_patterns:
                print(f"🔍 Pattern(s) matched for '{text}': {matched_patterns}")

            self.assertFalse(matched_patterns, f"Unexpected match for: {text}")


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
