import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open

if "agents.logistics_agent" in sys.modules:
    del sys.modules["agents.logistics_agent"]

with patch("builtins.open", mock_open(read_data="Delivery takes 1-3 days. Fee: Rs. 400.")):
    import agents.logistics_agent


class TestLoadLogisticsKnowledge(unittest.TestCase):
    @patch("builtins.open", mock_open(read_data="Delivery takes 1-3 days. Fee: Rs. 400."))
    def test_reads_policy_file(self):
        result = agents.logistics_agent.load_logistics_knowledge()
        self.assertIn("Rs. 400", result)

    @patch("builtins.print")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_returns_fallback_when_file_missing(self, _, __):
        result = agents.logistics_agent.load_logistics_knowledge()
        self.assertIn("1-5 days", result)
        self.assertIn("Rs. 400", result)


class TestHandleLogisticsQuery(unittest.TestCase):
    def _make_mock_llm(self, response_text="Delivery to Kandy takes 2-3 days and costs Rs. 400."):
        llm = MagicMock()
        llm.generate.return_value = response_text
        return llm

    @patch("agents.logistics_agent.get_llm")
    def test_returns_string_on_success(self, mock_get_llm):
        mock_get_llm.return_value = self._make_mock_llm()
        result = agents.logistics_agent.handle_logistics_query("How long does delivery take?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    @patch("agents.logistics_agent.get_llm")
    def test_query_included_in_prompt(self, mock_get_llm):
        mock_llm = self._make_mock_llm()
        mock_get_llm.return_value = mock_llm
        agents.logistics_agent.handle_logistics_query("What is the return policy?")
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args
        prompt = call_args.kwargs.get("prompt", "") or call_args.args[0]
        self.assertIn("What is the return policy?", prompt)

    @patch("agents.logistics_agent.get_llm")
    def test_system_prompt_included_in_call(self, mock_get_llm):
        mock_llm = self._make_mock_llm()
        mock_get_llm.return_value = mock_llm
        agents.logistics_agent.handle_logistics_query("Delivery to Galle?")
        call_args = mock_llm.generate.call_args
        prompt = call_args.kwargs.get("prompt", "") or call_args.args[0]
        self.assertIn("KNOWLEDGE BASE", prompt)

    @patch("agents.logistics_agent.get_llm")
    def test_returns_error_message_on_exception(self, mock_get_llm):
        mock_get_llm.side_effect = Exception("Service down")
        result = agents.logistics_agent.handle_logistics_query("How much is shipping?")
        self.assertIn("unavailable", result.lower())
        self.assertIn("Service down", result)


if __name__ == "__main__":
    unittest.main()