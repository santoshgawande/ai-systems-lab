"""
Lab 05 — Cost Optimisation Strategies

Demonstrates concrete techniques to reduce MCP token overhead.
Each strategy is measured and compared against the bloated baseline.
"""

import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from token_utils import count_tokens, count_tokens_in_tools, token_report

console = Console()

# ── Baseline: fat tool definitions (what most people write first) ──────────

BLOATED_TOOLS = [
    {
        "type": "function",
        "name": "get_current_weather_data_for_city",
        "description": (
            "This tool retrieves the current weather conditions for any given city around "
            "the world. It returns detailed meteorological data including temperature in "
            "the requested unit system, relative humidity as a percentage, wind speed "
            "and direction, UV index, atmospheric pressure, visibility, and a human-readable "
            "condition description. The data is sourced from real-time weather stations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": "The full name of the city for which you want to retrieve "
                                   "weather data. This should be the city name in English. "
                                   "Examples: 'San Francisco', 'New York City', 'Tokyo', 'London'"
                },
                "temperature_units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit", "kelvin"],
                    "description": "The unit system to use for temperature values in the response. "
                                   "Use 'celsius' for metric, 'fahrenheit' for imperial, or "
                                   "'kelvin' for scientific applications. Defaults to celsius."
                },
            },
            "required": ["city_name"],
        },
    },
    {
        "type": "function",
        "name": "perform_mathematical_calculation",
        "description": (
            "This tool safely evaluates mathematical expressions provided as strings. "
            "It supports all standard arithmetic operations as well as trigonometric "
            "functions, logarithms, square roots, and other common mathematical operations. "
            "It does NOT support arbitrary code execution, string manipulation, or imports."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mathematical_expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate as a string. "
                                   "Supported: +, -, *, /, **, sqrt(), sin(), cos(), tan(), "
                                   "log(), log10(), abs(), round(), floor(), ceil(). "
                                   "Example: '2 * (3 + 4) / sqrt(9)' returns 4.667"
                },
            },
            "required": ["mathematical_expression"],
        },
    },
]

# ── Strategy 1: Trim descriptions ─────────────────────────────────────────

TRIMMED_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
    {
        "type": "function",
        "name": "calculate",
        "description": "Evaluate a math expression. Supports +,-,*,/,**,sqrt(),sin(),cos(),log().",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
    },
]

# ── Strategy 2: Dynamic tool loading (only register what's needed) ─────────

WEATHER_ONLY = [TRIMMED_TOOLS[0]]
CALC_ONLY = [TRIMMED_TOOLS[1]]

# ── Strategy 3: Compress tool results before returning ────────────────────

VERBOSE_TOOL_RESULT = {
    "city": "San Francisco",
    "country": "United States",
    "coordinates": {"latitude": 37.7749, "longitude": -122.4194},
    "temperature": {"value": 18, "unit": "celsius", "feels_like": 16},
    "humidity": {"percentage": 72, "description": "Moderate humidity"},
    "wind": {"speed_kmh": 15, "direction_degrees": 270, "description": "West wind"},
    "uv_index": {"value": 3, "description": "Moderate UV"},
    "condition": {"code": "PARTLY_CLOUDY", "description": "Partly cloudy skies"},
    "last_updated": "2025-05-06T10:00:00Z",
    "data_source": "WeatherStation_SFO_001",
}

COMPRESSED_TOOL_RESULT = {
    "city": "San Francisco",
    "temp_c": 18,
    "humidity": 72,
    "wind_kmh": 15,
    "condition": "Partly cloudy",
}

# ── Strategy 4: Summarise long tool results before re-feeding ─────────────

LONG_DB_RESULT = {
    "rows": [
        {"id": i, "name": f"Employee {i}", "dept": "Engineering",
         "salary": 90000 + i * 1000, "joined": f"2020-0{(i%9)+1}-01",
         "manager_id": i // 5, "location": "SF", "status": "active"}
        for i in range(1, 21)
    ],
    "row_count": 20,
}

SUMMARISED_DB_RESULT = {
    "row_count": 20,
    "dept": "Engineering",
    "salary_range": "90000-109000",
    "location": "SF",
}


def measure(label: str, tools: list, tool_result: dict = None,
            strategy_note: str = "") -> dict:
    tool_tokens = count_tokens_in_tools(tools) if tools else 0
    result_tokens = count_tokens(json.dumps(tool_result)) if tool_result else 0
    # simulate a real request: system + user + tools + result
    system = count_tokens("You are a helpful assistant.")
    user = count_tokens("What is the weather in San Francisco?")
    total_input = system + user + tool_tokens + result_tokens
    total_output = count_tokens("It's 18°C and partly cloudy in San Francisco.")
    return {
        "label": label,
        "tool_tokens": tool_tokens,
        "result_tokens": result_tokens,
        "total_input": total_input,
        "total_output": total_output,
        "note": strategy_note,
    }


def main():
    console.rule("[bold blue]Lab 05 — Cost Optimisation Strategies")

    baseline = measure(
        "Baseline (bloated tools, verbose result)",
        BLOATED_TOOLS, VERBOSE_TOOL_RESULT,
    )

    strategies = [
        baseline,
        measure("Strategy 1: Trim descriptions",
                TRIMMED_TOOLS, VERBOSE_TOOL_RESULT,
                "Remove redundant prose from descriptions"),
        measure("Strategy 2: Dynamic loading (1 tool only)",
                WEATHER_ONLY, VERBOSE_TOOL_RESULT,
                "Only register tools relevant to the request"),
        measure("Strategy 2 + Strategy 1",
                WEATHER_ONLY, VERBOSE_TOOL_RESULT,
                "Trimmed + only load needed tool"),
        measure("Strategy 3: Compress tool result",
                WEATHER_ONLY, COMPRESSED_TOOL_RESULT,
                "Return minimal fields from tool"),
        measure("All 3 combined (optimal)",
                WEATHER_ONLY, COMPRESSED_TOOL_RESULT,
                "Trimmed defs + dynamic load + compressed result"),
    ]

    table = Table(show_header=True, header_style="bold magenta",
                  title="Cost Optimisation: Input Token Comparison")
    table.add_column("Strategy", max_width=38)
    table.add_column("Tool Def\nTokens", justify="right")
    table.add_column("Result\nTokens", justify="right")
    table.add_column("Total\nInput", justify="right")
    table.add_column("Saving\nvs Baseline", justify="right")
    table.add_column("Note", max_width=35, style="dim")

    baseline_total = baseline["total_input"]
    for s in strategies:
        saving_pct = (baseline_total - s["total_input"]) / baseline_total * 100
        if saving_pct <= 0:
            color = "white"
        elif saving_pct < 30:
            color = "yellow"
        else:
            color = "green"
        table.add_row(
            s["label"],
            str(s["tool_tokens"]),
            str(s["result_tokens"]),
            str(s["total_input"]),
            f"[{color}]{saving_pct:.0f}%[/{color}]",
            s["note"],
        )

    console.print(table)

    # Show the DB result compression example separately
    console.rule("[bold blue]Bonus: Large Tool Result Compression")

    verbose_tokens = count_tokens(json.dumps(LONG_DB_RESULT))
    summary_tokens = count_tokens(json.dumps(SUMMARISED_DB_RESULT))
    console.print(f"  Verbose DB result (20 rows, all fields): [red]{verbose_tokens} tokens[/red]")
    console.print(f"  Summarised result (aggregated):          [green]{summary_tokens} tokens[/green]")
    console.print(f"  Saving: [bold green]{(verbose_tokens - summary_tokens)/verbose_tokens*100:.0f}%[/bold green]\n")
    console.print("[dim]Summarise inside the tool function before returning — never return raw DB rows.[/dim]")

    console.print(Panel(
        "[bold]Optimisation Playbook:[/bold]\n\n"
        "1. [green]Trim tool descriptions[/green] — The model doesn't need an essay.\n"
        "   One clear sentence per parameter is enough.\n\n"
        "2. [green]Dynamic tool loading[/green] — Don't register all 20 tools on every request.\n"
        "   Route the intent first, then load only the relevant tool(s).\n\n"
        "3. [green]Compress tool results[/green] — Return only the fields the model needs.\n"
        "   Never return raw DB rows or full API responses.\n\n"
        "4. [green]Summarise before re-feeding[/green] — For multi-turn chains, summarise\n"
        "   intermediate results rather than keeping full JSON in history.\n\n"
        "5. [green]Batch tool calls[/green] — If the model needs A and B, ask it to call both\n"
        "   in parallel (one turn) rather than A → result → B → result (two turns).\n\n"
        "6. [green]Prompt caching[/green] — If tool defs are static, cache the system prompt\n"
        "   + tool defs block. Claude API supports this natively (cache_control).",
        title="Playbook", border_style="green"
    ))


if __name__ == "__main__":
    main()
