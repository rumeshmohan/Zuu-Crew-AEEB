import sys
import unittest
from unittest.mock import patch, MagicMock

with patch("utils.config.get_api_key", return_value="mock-key"):
    if "agents.router" in sys.modules:
        del sys.modules["agents.router"]
    import agents.router


class TestRouteQueryKeywords(unittest.TestCase):
    # Chitchat
    def test_chitchat_hi(self):
        self.assertEqual(agents.router.route_query("hi"), "[CHITCHAT]")

    def test_chitchat_hello(self):
        self.assertEqual(agents.router.route_query("Hello there!"), "[CHITCHAT]")

    def test_chitchat_thanks(self):
        self.assertEqual(agents.router.route_query("Thanks a lot!"), "[CHITCHAT]")

    def test_chitchat_how_are_you(self):
        self.assertEqual(agents.router.route_query("How are you doing?"), "[CHITCHAT]")

    def test_chitchat_ayubowan(self):
        self.assertEqual(agents.router.route_query("Ayubowan!"), "[CHITCHAT]")

    # Logistics
    def test_logistics_delivery(self):
        self.assertEqual(agents.router.route_query("What are the delivery options?"), "[LOGISTICS]")

    def test_logistics_kandy(self):
        self.assertEqual(agents.router.route_query("Do you deliver to Kandy?"), "[LOGISTICS]")

    def test_logistics_how_much(self):
        self.assertEqual(agents.router.route_query("How much does shipping cost?"), "[LOGISTICS]")

    def test_logistics_fee(self):
        self.assertEqual(agents.router.route_query("What is the fee for Colombo?"), "[LOGISTICS]")

    # Preference
    def test_preference_allergic(self):
        self.assertEqual(agents.router.route_query("She is allergic to nuts"), "[PREFERENCE]")

    def test_preference_allergy(self):
        self.assertEqual(agents.router.route_query("He has a dairy allergy"), "[PREFERENCE]")

    def test_preference_likes(self):
        self.assertEqual(agents.router.route_query("She likes chocolate"), "[PREFERENCE]")
    
    def test_preference_loves(self):
        self.assertEqual(agents.router.route_query("My son loves puzzles"), "[PREFERENCE]")
    
    def test_preference_hates(self):
        self.assertEqual(agents.router.route_query("My daughter hates chocolate"), "[PREFERENCE]")

    def test_preference_dislikes(self):
        self.assertEqual(agents.router.route_query("My son dislikes chocolate"), "[PREFERENCE]")


class TestRouteQueryLLMFallback(unittest.TestCase):
    def _make_mock_response(self, intent):
        mock_choice = MagicMock()
        mock_choice.message.content = intent
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    @patch("agents.router.client")
    def test_llm_fallback_returns_catalog(self, mock_client):
        mock_client.chat.completions.create.return_value = self._make_mock_response("[CATALOG]")
        result = agents.router.route_query("I want to send a birthday gift")
        self.assertEqual(result, "[CATALOG]")

    @patch("agents.router.client")
    def test_llm_fallback_returns_default_on_exception(self, mock_client):
        mock_client.chat.completions.create.side_effect = Exception("API error")
        result = agents.router.route_query("something completely ambiguous")
        self.assertEqual(result, "[CATALOG]")

    @patch("agents.router.client")
    def test_llm_not_called_for_keyword_match(self, mock_client):
        agents.router.route_query("hi there")
        mock_client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()