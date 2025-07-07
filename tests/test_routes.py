import unittest
import re

from pktabner.ner.patterns import ROUTE_PATTERNS


class TestRoutePatterns(unittest.TestCase):

    def test_route_matches(self):
        test_cases = {
            "intravenous": ["IV", "i.v.", "IV infusion", "intravenous", "bolus", "i.v. infusion"],
            "intramuscular": ["IM", "i.m.", "intramuscular injection"],
            "subcutaneous": ["subcutaneous", "sc", "sq", "sub-cut", "subq", "s.c."],
            "oral": ["oral", "PO", "per os", "p.o.", "orally"],
            "sublingual": ["sublingual", "SL", "s.l.", "sub lingual", "sublingually"],
            "buccal": ["buccal", "buc", "b.u.c."],
            "inhaled": ["inhalation", "via inhalation", "inhaled", "pulmonary route"],
            "intranasal": ["intranasal", "nasal spray", "intra-nasal", "i.n.", "nasally"],
            "topical": ["topical", "cutaneous", "applied to skin"],
            "transdermal": ["transdermal", "patch", "td", "t.d."],
            "intradermal": ["intradermal", "i.d.", "id"],
            "rectal": ["rectal", "per rectum", "p.r.", "suppository", "pr"],
            "vaginal": ["vaginal", "pv", "p.v.", "per vaginam", "vaginally"],
            "intrauterine": ["intrauterine", "i.u.", "iu"],
            "intrathecal": ["intrathecal", "i.t.", "it"],
            "epidural": ["epidural", "epi", "e.p.i."],
        }

        for label, examples in test_cases.items():
            pattern = ROUTE_PATTERNS[label]
            print(f"\n🔎 Testing route: {label}")
            for text in examples:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if not match:
                    print(f"  ❌ FAILED: {text}")
                with self.subTest(route=label, text=text):
                    self.assertIsNotNone(match, f"❌ Failed to match '{text}' for route '{label}'")

    def test_route_negatives(self):
        negatives = [
            "cell culture", "IVT method", "baseline", "clinical outcome",
            "observed at 2h", "study arm", "device applied", "cohort",
        ]
        print("\n🔎 Testing negatives:")
        for text in negatives:
            matched_labels = [label for label, pattern in ROUTE_PATTERNS.items()
                              if re.search(pattern, text, flags=re.IGNORECASE)]
            if matched_labels:
                print(f"  ❌ UNEXPECTED MATCH: '{text}' ➜ matched as {matched_labels}")
            with self.subTest(text=text):
                self.assertFalse(matched_labels, f"❌ Unexpected match for '{text}' as {matched_labels}")

if __name__ == "__main__":
    unittest.main()
