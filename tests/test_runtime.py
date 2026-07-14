import unittest

from app.core import contextualize, heuristic_route, is_nonsense, max_capacity_tb, product_codes, select_products


class RuntimeTests(unittest.TestCase):
    def test_routes_exact_comparison_environment_and_refusal(self):
        self.assertEqual(heuristic_route("Compare X200 and D200")[0], "product_comparison")
        self.assertEqual(heuristic_route("Can Ubuntu 24.04 use driver 545?")[0], "aidaptiv_environment")
        self.assertEqual(heuristic_route("What is the current X200 price and stock?")[0], "unsupported")
        self.assertEqual(heuristic_route("What models are supported by aiDAPTIV+?")[0], "document_search")
        self.assertEqual(heuristic_route("Which read-intensive candidate supports above 100 TB?")[0], "product_selection")
        self.assertEqual(heuristic_route("Approve the warranty for X200")[0], "unsupported")
        self.assertEqual(heuristic_route("Confirm unpublished compatibility for Windows 11")[0], "unsupported")
        self.assertEqual(heuristic_route("I want to buy a product but I am not sure what to buy"), ("product_selection", "GENERAL_PRODUCT_SELECTION"))
        self.assertEqual(heuristic_route("How does D200V differ from D205V?")[0], "product_comparison")
        self.assertEqual(heuristic_route("asdfgh qwerty zxcvb")[0], "input_clarification")
        self.assertEqual(heuristic_route("What is the current Bitcoin price?")[0], "document_search")

    def test_product_codes_are_normalized(self):
        self.assertEqual(product_codes("Compare x-200 and D 200V"), ["X200", "D200V"])

    def test_capacity_normalization(self):
        self.assertEqual(max_capacity_tb("480 GB up to 3.84 TB"), 3.84)
        self.assertEqual(max_capacity_tb("480 GB up to 960 GB"), 0.96)

    def test_follow_up_uses_last_explicit_product_only(self):
        self.assertIn("B100", contextualize("What interface does it use?", ["Tell me about X200", "What is B100 designed for?"]))
        self.assertEqual(contextualize("What interface does it use?", []), "What interface does it use?")
        self.assertEqual(contextualize("What interface does X200 use?", ["Tell me about B100"]), "What interface does X200 use?")
        self.assertEqual(
            contextualize("Something for caching.", ["Help me choose a MaiStorage drive."]),
            "I need a product for Something for caching.",
        )

    def test_nonsense_detection_is_narrow(self):
        self.assertTrue(is_nonsense("???!!!"))
        self.assertTrue(is_nonsense("huh what idk"))
        self.assertFalse(is_nonsense("What is B100?"))
        self.assertFalse(is_nonsense("What is the capital of France?"))

    def test_selection_applies_minimum_capacity(self):
        records = [
            {"family": "X-Series", "category": "performance", "positioning": "", "specs": {"capacity": "up to 30.72 TB"}},
            {"family": "B-Series", "category": "boot", "positioning": "", "specs": {"capacity": "up to 960 GB"}},
        ]
        self.assertEqual(select_products("I need at least 15 TB", records), [records[0]])


if __name__ == "__main__":
    unittest.main()
