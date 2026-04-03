import sys
import unittest
from unittest.mock import patch, MagicMock

with patch("utils.config.get_api_key", return_value="mock-key"), \
     patch("memory.vector_db.retrieve_products", return_value=[]):
    if "agents.catalog_agent" in sys.modules:
        del sys.modules["agents.catalog_agent"]
    import agents.catalog_agent


class TestExtractFilters(unittest.TestCase):
    def test_returns_none_for_plain_query(self):
        self.assertIsNone(agents.catalog_agent.extract_filters("show me cakes"))

    def test_extracts_single_tag(self):
        result = agents.catalog_agent.extract_filters("I want an eggless cake")
        self.assertEqual(result["tags"], ["eggless"])

    def test_extracts_multiple_tags(self):
        result = agents.catalog_agent.extract_filters("eggless and sugar-free options please")
        self.assertIn("eggless", result["tags"])
        self.assertIn("sugar-free", result["tags"])

    def test_extracts_budget_with_hyphen(self):
        result = agents.catalog_agent.extract_filters("Budget: LKR 1000-5000")
        self.assertEqual(result["min_price"], 1000)
        self.assertEqual(result["max_price"], 5000)

    def test_extracts_budget_with_dash(self):
        result = agents.catalog_agent.extract_filters("Budget: LKR 500–3000")
        self.assertEqual(result["min_price"], 500)
        self.assertEqual(result["max_price"], 3000)

    def test_extracts_tags_and_budget_together(self):
        result = agents.catalog_agent.extract_filters("vegetarian gifts Budget: LKR 1000-4000")
        self.assertIn("vegetarian", result["tags"])
        self.assertEqual(result["min_price"], 1000)
        self.assertEqual(result["max_price"], 4000)


class TestBuildQdrantFilter(unittest.TestCase):
    def test_returns_none_for_empty_filters(self):
        self.assertIsNone(agents.catalog_agent.build_qdrant_filter({}))

    def test_builds_filter_for_tags(self):
        self.assertIsNotNone(agents.catalog_agent.build_qdrant_filter({"tags": ["eggless"]}))

    def test_builds_filter_for_price_range(self):
        self.assertIsNotNone(agents.catalog_agent.build_qdrant_filter({"min_price": 500, "max_price": 3000}))

    def test_builds_filter_for_tags_and_price(self):
        self.assertIsNotNone(agents.catalog_agent.build_qdrant_filter({
            "tags": ["vegetarian"], "min_price": 500, "max_price": 3000,
        }))


class TestHandleCatalogQuery(unittest.TestCase):
    def _make_mock_response(self, text="I recommend the Chocolate Cake for Rs.1500."):
        mock_choice = MagicMock()
        mock_choice.message.content = text
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    @patch("agents.catalog_agent.retrieve_products")
    @patch("agents.catalog_agent.client")
    def test_returns_response_on_success(self, mock_client, mock_retrieve):
        mock_retrieve.return_value = [{"name": "Chocolate Cake", "price": 1500}]
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        result = agents.catalog_agent.handle_catalog_query("I want a cake")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    @patch("agents.catalog_agent.retrieve_products")
    @patch("agents.catalog_agent.client")
    def test_returns_error_string_on_exception(self, mock_client, mock_retrieve):
        mock_retrieve.return_value = []
        mock_client.chat.completions.create.side_effect = Exception("API timeout")
        result = agents.catalog_agent.handle_catalog_query("show me flowers")
        self.assertTrue(result.startswith("Catalog error:"))

    @patch("agents.catalog_agent.retrieve_products")
    @patch("agents.catalog_agent.client")
    def test_passes_filter_when_tag_detected(self, mock_client, mock_retrieve):
        mock_retrieve.return_value = []
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        agents.catalog_agent.handle_catalog_query("eggless cake please")
        _, kwargs = mock_retrieve.call_args
        self.assertIsNotNone(kwargs.get("search_filter"))

    @patch("agents.catalog_agent.retrieve_products")
    @patch("agents.catalog_agent.client")
    def test_no_filter_for_plain_query(self, mock_client, mock_retrieve):
        mock_retrieve.return_value = []
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        agents.catalog_agent.handle_catalog_query("show me something nice")
        _, kwargs = mock_retrieve.call_args
        self.assertIsNone(kwargs.get("search_filter"))


if __name__ == "__main__":
    unittest.main()