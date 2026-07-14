import unittest

from streamlit.testing.v1 import AppTest


class UIRegressionTests(unittest.TestCase):
    def test_redesigned_workspace_renders_core_navigation(self):
        app = AppTest.from_file("app/chat_ui.py").run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Streaming Chat", "Product Explorer", "Environment Check", "Source Library", "Evaluation Lab"],
        )
        self.assertIn("New chat", [button.label for button in app.button])
        self.assertEqual([(item.label, item.value) for item in app.selectbox], [("Model provider", "DeepSeek")])
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("Streaming LLM Chat", rendered)
        self.assertIn("Stream inspector", rendered)
        self.assertIn("Persisted messages", rendered)
        self.assertEqual([item.label for item in app.multiselect], ["Products"])
        self.assertEqual([item.label for item in app.number_input], ["NVIDIA driver version"])
        self.assertEqual(len(app.dataframe), 3)
        self.assertEqual(
            [(item.label, item.value) for item in app.metric],
            [("Strict accuracy", "100.0%"), ("Passed", "100/100"), ("Corpus", "3826d75c")],
        )

        next(
            item for item in app.text_input if item.label == "Available ports (comma-separated)"
        ).set_value("bad-port")
        next(
            button for button in app.button if button.label == "Validate documented environment"
        ).click()
        app.run(timeout=20)
        self.assertIn("Ports must be comma-separated numbers.", [error.value for error in app.error])


if __name__ == "__main__":
    unittest.main()
