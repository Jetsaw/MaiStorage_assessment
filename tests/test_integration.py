import os
import unittest

import psycopg

from app.core import DATABASE_URL, ask


def database_ready():
    try:
        with psycopg.connect(DATABASE_URL) as connection:
            return connection.execute("SELECT EXISTS (SELECT 1 FROM active_corpus)").fetchone()[0]
    except psycopg.Error:
        return False


@unittest.skipUnless(database_ready(), "indexed PostgreSQL corpus is not running")
class IntegrationTests(unittest.TestCase):
    def test_indexed_counts(self):
        with psycopg.connect(DATABASE_URL) as connection:
            counts = connection.execute(
                "SELECT (SELECT count(*) FROM source s JOIN active_corpus ac ON ac.version=s.corpus_version), "
                "(SELECT count(*) FROM product p JOIN active_corpus ac ON ac.version=p.corpus_version), "
                "(SELECT count(*) FROM document_chunk dc JOIN source_section ss ON ss.id=dc.source_section_id "
                "JOIN source s ON s.id=ss.source_id JOIN active_corpus ac ON ac.version=s.corpus_version)"
            ).fetchone()
        self.assertEqual(counts[:2], (27, 10))
        self.assertGreaterEqual(counts[2], 150)

    def test_agent_routes_are_grounded(self):
        cases = [
            ("What is the X200 capacity?", "product_lookup", "passed"),
            ("Compare B100 and D100", "product_comparison", "passed"),
            ("Can Ubuntu 24.04 use NVIDIA driver 545?", "aidaptiv_environment", "passed"),
            ("What should I do when Pro Suite services behave abnormally?", "document_search", "passed"),
            ("What is the current X200 price and stock?", "unsupported", "not_required"),
        ]
        for question, route, citation_status in cases:
            with self.subTest(question=question):
                result = ask(question)
                self.assertEqual(result["route"], route)
                self.assertEqual(result["citation_status"], citation_status)

    def test_out_of_scope_questions_are_refused_without_evidence(self):
        for question in (
            "What is the capital of France?",
            "How do I bake a chocolate cake?",
            "I want to buy a product but I am not sure what to buy",
        ):
            with self.subTest(question=question):
                result = ask(question)
                self.assertEqual(result["route"], "document_search")
                self.assertEqual(result["citation_status"], "not_required")
                self.assertEqual(result["evidence"], [])
                self.assertIn("do not contain enough evidence", result["answer"])


if __name__ == "__main__":
    unittest.main()
