import unittest
import json
import os

class TestDataValidation(unittest.TestCase):
    def test_catalog_schema(self):
        catalog_path = os.path.join("data", "catalog.json")
        if not os.path.exists(catalog_path):
            self.skipTest("catalog.json not found")
            
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIsInstance(data, list)
            for item in data:
                self.assertIn("name", item)
                self.assertIn("price", item)
                self.assertIsInstance(item["price"], (int, float))

    def test_profiles_schema(self):
        profiles_path = os.path.join("data", "profiles.json")
        if not os.path.exists(profiles_path):
            self.skipTest("profiles.json not found")

        with open(profiles_path, "r") as f:
            data = json.load(f)
            self.assertIsInstance(data, dict)
            for cust_id, info in data.items():
                self.assertTrue(cust_id.startswith("CUS_") or cust_id == "default")
                self.assertIn("recipients", info)

if __name__ == "__main__":
    unittest.main()