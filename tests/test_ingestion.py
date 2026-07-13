import unittest

from ingestion.crawl import MaiStorageSpider, extract_html, is_allowed_url, parse_products, source_diff


class IngestionTests(unittest.TestCase):
    def test_url_allowlist_is_narrow(self):
        self.assertTrue(is_allowed_url("https://www.maistorage.com/enterprise-x-series/"))
        self.assertTrue(is_allowed_url("https://maistorage.com/wp-content/uploads/2025/guide.pdf"))
        self.assertFalse(is_allowed_url("http://www.maistorage.com/enterprise/"))
        self.assertFalse(is_allowed_url("https://www.maistorage.com/wp-admin/"))
        self.assertFalse(is_allowed_url("https://example.com/enterprise/"))

    def test_product_parser_keeps_documented_fields(self):
        products = parse_products(
            [
                "Performance X-Series",
                "X200",
                "Feature-rich enterprise storage",
                "Interface Specifications",
                "PCIe Gen5x4 Dual Port",
                "Capacity",
                "1.6 TB up to 30.72 TB",
                "SSD Power Consumption",
                "Active: 25W",
                "Performance",
                "Seq Read (up to)",
                "14.8 GB/s",
            ]
        )
        self.assertEqual(products[0]["name"], "X200")
        self.assertIn("PCIe Gen5x4", products[0]["interface"])
        self.assertEqual(products[0]["capacity"], "1.6 TB up to 30.72 TB")
        self.assertIn("14.8 GB/s", products[0]["performance"])

    def test_html_extraction_ignores_navigation(self):
        document = b"""<html><head><title>X Series</title></head><body>
        <nav>X999</nav><main id='content'><h2>X200</h2><p>Storage product</p>
        <h3>Capacity</h3><p>30.72 TB</p></main></body></html>"""
        result = extract_html(document, "https://www.maistorage.com/enterprise-x-series/")
        self.assertEqual(result["products"][0]["name"], "X200")
        self.assertNotIn("X999", result["text"])

    def test_html_extraction_decodes_utf8_punctuation(self):
        document = "<main><h2>B100</h2><p>industry’s latest NAND</p></main>".encode("utf-8")
        result = extract_html(document, "https://www.maistorage.com/enterprise-b-series/")
        self.assertIn("industry’s", result["products"][0]["description"])
        self.assertNotIn("â", result["text"])

    def test_source_diff_is_hash_based(self):
        old = [{"url": "https://a", "content_hash": "1"}, {"url": "https://b", "content_hash": "1"}]
        new = [{"url": "https://a", "content_hash": "1"}, {"url": "https://c", "content_hash": "2"}]
        diff = source_diff(old, new)
        self.assertEqual(diff["unchanged"], ["https://a"])
        self.assertEqual(diff["added"], ["https://c"])
        self.assertEqual(diff["removed"], ["https://b"])

    def test_source_diff_detects_parser_output_changes(self):
        old = [{"url": "https://a", "content_hash": "1", "processed_hash": "old"}]
        new = [{"url": "https://a", "content_hash": "1", "processed_hash": "new"}]
        self.assertEqual(source_diff(old, new)["changed"], ["https://a"])

    def test_sitemap_filter_keeps_child_maps_and_approved_pages(self):
        entries = [
            {"loc": "https://www.maistorage.com/page-sitemap.xml"},
            {"loc": "https://www.maistorage.com/enterprise-x-series/"},
            {"loc": "https://www.maistorage.com/contact/"},
        ]
        kept = [entry["loc"] for entry in MaiStorageSpider().sitemap_filter(entries)]
        self.assertEqual(kept, [entries[0]["loc"], entries[1]["loc"]])


if __name__ == "__main__":
    unittest.main()
