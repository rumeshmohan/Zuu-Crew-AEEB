import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOCK_PROFILES = {
    "customer_001": {
        "recipients": {
            "mom":    {"allergies": ["nuts", "dairy"]},
            "friend": {"allergies": []},
        }
    }
}

if "agents.reflection_loop" in sys.modules:
    del sys.modules["agents.reflection_loop"]

with patch("builtins.open", mock_open(read_data=json.dumps(MOCK_PROFILES))):
    import agents.reflection_loop


class TestLoadRecipientProfile(unittest.TestCase):
    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_PROFILES)))
    def test_returns_profile_for_known_recipient(self):
        result = agents.reflection_loop.load_recipient_profile("customer_001", "mom")
        self.assertEqual(result["allergies"], ["nuts", "dairy"])

    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_PROFILES)))
    def test_returns_empty_dict_for_unknown_customer(self):
        result = agents.reflection_loop.load_recipient_profile("unknown_id", "mom")
        self.assertEqual(result, {})

    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_PROFILES)))
    def test_returns_empty_dict_for_unknown_recipient(self):
        result = agents.reflection_loop.load_recipient_profile("customer_001", "stranger")
        self.assertEqual(result, {})

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_raises_file_not_found(self, _):
        with self.assertRaises(FileNotFoundError):
            agents.reflection_loop.load_recipient_profile("customer_001", "mom")


class TestRunReflection(unittest.TestCase):
    def _make_mock_llm(self, response_text):
        llm = MagicMock()
        llm.generate.return_value = response_text
        return llm

    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_PROFILES)))
    @patch("agents.reflection_loop.get_llm")
    def test_returns_draft_when_no_allergies(self, mock_get_llm):
        draft = "We recommend the Milk Chocolate Box."
        result = agents.reflection_loop.run_reflection("gift for friend", draft, "customer_001", "friend")
        mock_get_llm.assert_not_called()
        self.assertEqual(result, draft)

    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("agents.reflection_loop.get_llm")
    def test_returns_draft_when_profiles_missing(self, mock_get_llm, _):
        draft = "We recommend the Almond Cake."
        result = agents.reflection_loop.run_reflection("gift for mom", draft, "customer_001", "mom")
        mock_get_llm.assert_not_called()
        self.assertEqual(result, draft)

    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_PROFILES)))
    @patch("agents.reflection_loop.get_llm")
    def test_calls_llm_when_allergies_present(self, mock_get_llm):
        mock_get_llm.return_value = self._make_mock_llm("We recommend the Fruit Basket.")
        agents.reflection_loop.run_reflection("gift for mom", "Almond Cake.", "customer_001", "mom")
        mock_get_llm.assert_called_once_with(tier="strong")

    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_PROFILES)))
    @patch("agents.reflection_loop.get_llm")
    def test_returns_revised_response_when_unsafe(self, mock_get_llm):
        revised = "REVISED: Sorry, this item contains nuts which your recipient is allergic to."
        mock_get_llm.return_value = self._make_mock_llm(revised)
        result = agents.reflection_loop.run_reflection("gift for mom", "Almond Cake.", "customer_001", "mom")
        self.assertTrue(result.startswith("REVISED:"))

    @patch("builtins.print")
    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_PROFILES)))
    @patch("agents.reflection_loop.get_llm")
    def test_returns_draft_on_llm_exception(self, mock_get_llm, _mock_print):
        mock_get_llm.return_value = self._make_mock_llm("")
        mock_get_llm.return_value.generate.side_effect = Exception("LLM down")
        draft = "We recommend the Almond Cake."
        result = agents.reflection_loop.run_reflection("gift for mom", draft, "customer_001", "mom")
        self.assertEqual(result, draft)

    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_PROFILES)))
    @patch("agents.reflection_loop.get_llm")
    def test_prompt_contains_allergies_and_draft(self, mock_get_llm):
        mock_llm = self._make_mock_llm("We recommend the Fruit Basket.")
        mock_get_llm.return_value = mock_llm
        draft = "We recommend the Cheese Gift Box."
        agents.reflection_loop.run_reflection("gift for mom", draft, "customer_001", "mom")
        call_args = mock_llm.generate.call_args
        prompt = call_args.kwargs.get("prompt", "") or call_args.args[0]
        self.assertIn("nuts", prompt)
        self.assertIn(draft, prompt)


if __name__ == "__main__":
    unittest.main()