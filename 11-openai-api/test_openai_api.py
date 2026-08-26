import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-function-calling"))
from functions import execute_tool, TOOLS

class TestOpenAIFunctionCalling(unittest.TestCase):
    def test_calculator_execution(self):
        res = execute_tool("calculator", {"expression": "sqrt(144) + 10"})
        self.assertEqual(res, "22.0")

    def test_unit_convert_execution(self):
        res = execute_tool("unit_convert", {"value": 100, "from_unit": "km", "to_unit": "miles"})
        self.assertIn("62.1371 miles", res)

    def test_weather_execution(self):
        res = execute_tool("get_weather", {"city": "Tokyo", "unit": "celsius"})
        self.assertIn("18°C", res)

    def test_tools_schema_validity(self):
        names = [t["function"]["name"] for t in TOOLS]
        self.assertIn("calculator", names)
        self.assertIn("get_datetime", names)
        self.assertIn("get_weather", names)

if __name__ == "__main__":
    unittest.main()
