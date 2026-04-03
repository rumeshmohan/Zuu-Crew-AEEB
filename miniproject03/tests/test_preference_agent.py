import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOCK_PROFILES = {
    "customer_001": {
        "recipients": {
            "mom": {"allergies": ["nuts"], "likes": ["chocolate"]}
        }
    }
}

EXTRACTED_DATA = {
    "recipient": "mom",
    "updates": {"allergies": ["dairy"], "likes": ["fruit"]},
}


class TestLoadProfiles(unittest.TestCase):
    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_PROFILES)))
    def test_loads_existing_profiles(self):
        from agents.preference_agent import load_profiles
        result = load_profiles()
        self.assertIn("customer_001", result)

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_returns_empty_dict_when_file_missing(self, _):
        from agents.preference_agent import load_profiles
        self.assertEqual(load_profiles(), {})


class TestSaveProfiles(unittest.TestCase):
    @patch("json.dump")
    @patch("builtins.open", mock_open())
    def test_saves_profiles(self, mock_dump):
        from agents.preference_agent import save_profiles
        save_profiles(MOCK_PROFILES)
        mock_dump.assert_called_once()


class TestExtractPreferenceData(unittest.TestCase):
    @patch("agents.preference_agent.get_llm")
    def test_extracts_valid_json(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(EXTRACTED_DATA)
        mock_get_llm.return_value = mock_llm
        from agents.preference_agent import extract_preference_data
        result = extract_preference_data("My mom is allergic to dairy and likes fruit")
        self.assertEqual(result["recipient"], "mom")
        self.assertIn("dairy", result["updates"]["allergies"])

    @patch("agents.preference_agent.get_llm")
    def test_strips_markdown_fences(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = f"```json\n{json.dumps(EXTRACTED_DATA)}\n```"
        mock_get_llm.return_value = mock_llm
        from agents.preference_agent import extract_preference_data
        result = extract_preference_data("My mom is allergic to dairy")
        self.assertEqual(result["recipient"], "mom")

    @patch("agents.preference_agent.get_llm")
    def test_raises_on_invalid_json(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "not valid json"
        mock_get_llm.return_value = mock_llm
        from agents.preference_agent import extract_preference_data
        with self.assertRaises(json.JSONDecodeError):
            extract_preference_data("some query")


class TestUpdateProfile(unittest.TestCase):
    @patch("agents.preference_agent.save_profiles")
    @patch("agents.preference_agent.load_profiles", return_value=json.loads(json.dumps(MOCK_PROFILES)))
    def test_merges_new_allergies(self, _, mock_save):
        from agents.preference_agent import update_profile
        update_profile("customer_001", "mom", {"allergies": ["dairy"]})
        saved = mock_save.call_args[0][0]
        allergies = saved["customer_001"]["recipients"]["mom"]["allergies"]
        self.assertIn("dairy", allergies)
        self.assertIn("nuts", allergies)

    @patch("agents.preference_agent.save_profiles")
    @patch("agents.preference_agent.load_profiles", return_value={})
    def test_creates_new_customer_and_recipient(self, _, mock_save):
        from agents.preference_agent import update_profile
        update_profile("new_customer", "friend", {"likes": ["flowers"]})
        saved = mock_save.call_args[0][0]
        self.assertIn("new_customer", saved)
        self.assertIn("friend", saved["new_customer"]["recipients"])

    @patch("agents.preference_agent.save_profiles")
    @patch("agents.preference_agent.load_profiles", return_value=json.loads(json.dumps(MOCK_PROFILES)))
    def test_does_not_duplicate_existing_values(self, _, mock_save):
        from agents.preference_agent import update_profile
        update_profile("customer_001", "mom", {"allergies": ["nuts"]})
        saved = mock_save.call_args[0][0]
        allergies = saved["customer_001"]["recipients"]["mom"]["allergies"]
        self.assertEqual(allergies.count("nuts"), 1)


class TestHandlePreferenceQuery(unittest.TestCase):
    @patch("agents.preference_agent.get_llm")
    @patch("agents.preference_agent.extract_preference_data", return_value=EXTRACTED_DATA)
    @patch("agents.preference_agent.update_profile")
    def test_returns_confirmation_on_success(self, _update, _extract, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Got it! I have noted those preferences."
        mock_get_llm.return_value = mock_llm
        from agents.preference_agent import handle_preference_query
        result = handle_preference_query("My mom is allergic to dairy", "customer_001")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    @patch("agents.preference_agent.extract_preference_data", return_value={"recipient": "mom", "updates": {}})
    def test_returns_message_when_no_updates_extracted(self, _):
        from agents.preference_agent import handle_preference_query
        result = handle_preference_query("My mom is great", "customer_001")
        self.assertIn("didn't catch", result)

    @patch("agents.preference_agent.extract_preference_data", side_effect=json.JSONDecodeError("", "", 0))
    def test_returns_rephrase_message_on_json_error(self, _):
        from agents.preference_agent import handle_preference_query
        result = handle_preference_query("gibberish input", "customer_001")
        self.assertIn("rephrase", result)

    @patch("agents.preference_agent.extract_preference_data", side_effect=Exception("LLM down"))
    def test_returns_fallback_on_unexpected_exception(self, _):
        from agents.preference_agent import handle_preference_query
        result = handle_preference_query("any query", "customer_001")
        self.assertIn("try again", result)


if __name__ == "__main__":
    unittest.main()