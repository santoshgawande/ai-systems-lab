"""
Lab 03 — Mini MCP Server

A tiny MCP server with 3 tools to simulate real-world tool overhead.
Run this as a subprocess; the token overhead lab will inspect its tool schemas.

Tools:
  get_weather(city)         — fake weather lookup
  calculate(expression)     — safe math eval
  query_database(sql)       — fake DB query

Start this server with:
  python 03_mini_mcp_server.py

Or use it via stdio transport for programmatic introspection in lab 04.
"""

import json
import math
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "token-lab-mcp",
    instructions="You are a helpful assistant with access to weather, math, and database tools.",
)


@mcp.tool()
def get_weather(city: str, units: str = "celsius") -> dict:
    """
    Get current weather conditions for a city.

    Args:
        city: The city name to get weather for (e.g. 'San Francisco', 'Tokyo')
        units: Temperature units, either 'celsius' or 'fahrenheit'

    Returns:
        Current weather data including temperature, humidity, wind speed,
        UV index, and a short description of conditions.
    """
    fake_data = {
        "city": city,
        "temperature": 22 if units == "celsius" else 71,
        "units": units,
        "humidity_pct": 65,
        "wind_speed_kmh": 14,
        "uv_index": 4,
        "condition": "Partly cloudy",
        "feels_like": 20 if units == "celsius" else 68,
    }
    return fake_data


@mcp.tool()
def calculate(expression: str) -> dict:
    """
    Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, sqrt(), sin(), cos(), log(), abs(), round()
    Does NOT support: arbitrary Python, imports, or string operations.

    Args:
        expression: A mathematical expression string, e.g. '2 * (3 + 4) / sqrt(9)'

    Returns:
        Result of the calculation with the original expression echoed back.
    """
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return {"expression": expression, "result": result, "error": None}
    except Exception as e:
        return {"expression": expression, "result": None, "error": str(e)}


@mcp.tool()
def query_database(sql: str, database: str = "production") -> dict:
    """
    Execute a read-only SQL query against the specified database.

    Only SELECT statements are permitted. Mutations (INSERT, UPDATE, DELETE,
    DROP) are rejected immediately. Queries are limited to 1000 rows.

    Args:
        sql: A valid SELECT SQL statement
        database: Target database name ('production', 'staging', 'analytics')

    Returns:
        Query results as a list of row dicts, plus row_count and execution_time_ms.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT statements allowed", "rows": [], "row_count": 0}

    fake_rows = [
        {"id": 1, "name": "Alice", "department": "Engineering", "salary": 95000},
        {"id": 2, "name": "Bob", "department": "Product", "salary": 88000},
        {"id": 3, "name": "Carol", "department": "Engineering", "salary": 102000},
    ]
    return {
        "database": database,
        "sql": sql,
        "rows": fake_rows,
        "row_count": len(fake_rows),
        "execution_time_ms": 12,
    }


if __name__ == "__main__":
    print("Starting mini MCP server on stdio transport...")
    print("Tools registered: get_weather, calculate, query_database")
    mcp.run(transport="stdio")
