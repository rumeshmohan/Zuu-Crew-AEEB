"""Unit tests for the SQL Validator safety layer."""
import unittest
from src.engine.sql_validator import SQLValidator

class TestSQLValidator(unittest.TestCase):
    """Unit tests for the SQL Validator safety layer — covers valid queries, destructive commands, and syntax errors."""
    def setUp(self):
        self.validator = SQLValidator()

    def test_valid_select_query(self):
        """Test that a standard SELECT query passes."""
        result = self.validator.validate_query("SELECT * FROM patients;")
        self.assertTrue(result["is_valid"])

    def test_reject_drop_table(self):
        """Test that destructive DROP commands are blocked."""
        result = self.validator.validate_query("DROP TABLE patients;")
        self.assertFalse(result["is_valid"])
        self.assertIn("Destructive command", result["message"])

    def test_reject_unbalanced_parentheses(self):
        """Test that basic syntax errors are caught."""
        result = self.validator.validate_query("SELECT COUNT(*) FROM patients WHERE (active = true;")
        self.assertFalse(result["is_valid"])
        self.assertIn("Unbalanced parentheses", result["message"])

if __name__ == "__main__":
    unittest.main()