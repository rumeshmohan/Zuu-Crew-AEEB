import unittest
from unittest.mock import patch, MagicMock
from agents.router import route_query
from agents.catalog_agent import handle_catalog_query

class TestIntegrationFlow(unittest.TestCase):
    @patch("agents.catalog_agent.retrieve_products")
    @patch("agents.catalog_agent.client.chat.completions.create")
    def test_end_to_end_gift_search_with_safety(self, mock_llm, mock_retrieve):
        # 1. Test Router correctly identifies a gift search
        query = "I want a chocolate cake for my wife"
        intent = route_query(query)
        self.assertEqual(intent, "[CATALOG]")

        # 2. Mock RAG results and LLM response
        mock_retrieve.return_value = [{"name": "Java Chocolate Cake", "price": 5000}]
        
        # Simulate the LLM providing a safe recommendation
        mock_choice = MagicMock()
        mock_choice.message.content = "I recommend the Java Chocolate Cake (Rs. 5000) as it is nut-free."
        mock_llm.return_value.choices = [mock_choice]

        # 3. Run the Catalog Agent
        response = handle_catalog_query(query, customer_id="CUS_001")
        
        # 4. Verify the response contains product data and safety context
        self.assertIn("Java Chocolate Cake", response)
        self.assertIn("Rs. 5000", response)

if __name__ == "__main__":
    unittest.main()