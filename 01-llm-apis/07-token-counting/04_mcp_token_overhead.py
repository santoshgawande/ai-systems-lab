"""
Lab 04 — MCP Token Overhead (THE KEY LAB)

Shows exactly where MCP adds tokens compared to a plain prompt.

Token budget breakdown for a single MCP tool call:

  PLAIN PROMPT
  ┌─────────────────────────────────────┐
  │ system prompt          ~50 tokens   │
  │ user message           ~15 tokens   │
  │                        ──────────   │
  │ INPUT TOTAL            ~65 tokens   │
  │ output (response)      ~20 tokens   │
  └─────────────────────────────────────┘

  WITH MCP (same question)
  ┌─────────────────────────────────────┐
  │ system prompt          ~50 tokens   │
  │ tool definitions      ~300 tokens   │  ← NEW (per registered tool)
  │ user message           ~15 tokens   │
  │ tool_call JSON         ~30 tokens   │  ← counted as OUTPUT, then becomes
  │ tool_result JSON       ~50 tokens   │  ← INPUT on next turn
  │ final response         ~20 tokens   │
  │                        ──────────   │
  │ INPUT TOTAL           ~445 tokens   │  6.8x more expensive
  │ OUTPUT TOTAL           ~50 tokens   │  2.5x more expensive
  └─────────────────────────────────────┘
"""

import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from token_utils import count_tokens, count_tokens_in_tools, token_report

console = Console()

# ── Simulated tool definitions (what MCP injects into context) ─────────────

TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather conditions for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name e.g. 'San Francisco'"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"],
                          "description": "Temperature units"},
            },
            "required": ["city"],
        },
    },
    {
        "type": "function",
        "name": "calculate",
        "description": "Evaluate a mathematical expression safely. Supports +,-,*,/,**,sqrt(),sin(),cos(),log().",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string",
                               "description": "Math expression e.g. '2 * (3 + 4) / sqrt(9)'"},
            },
            "required": ["expression"],
        },
    },
    {
        "type": "function",
        "name": "query_database",
        "description": "Execute a read-only SQL SELECT query. Mutations are rejected. Max 1000 rows.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A valid SELECT SQL statement"},
                "database": {"type": "string", "enum": ["production", "staging", "analytics"],
                             "description": "Target database"},
            },
            "required": ["sql"],
        },
    },
]

SYSTEM_PROMPT = "You are a helpful assistant."
USER_QUESTION = "What is the weather in San Francisco?"


def scenario_plain_prompt() -> dict:
    """No tools — just a system prompt + user question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_QUESTION},
    ]
    simulated_response = "I don't have access to real-time weather data. Please check a weather service."

    input_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(USER_QUESTION) + 8  # msg framing
    output_tokens = count_tokens(simulated_response)

    return token_report("Plain prompt (no tools)", input_tokens, output_tokens)


def scenario_mcp_tools_registered_not_called() -> dict:
    """
    Tools are registered in context but the model answers without calling them.
    You still pay for the tool definitions even if no tool is invoked.
    """
    tool_tokens = count_tokens_in_tools(TOOLS)
    base_input = count_tokens(SYSTEM_PROMPT) + count_tokens(USER_QUESTION) + 8
    input_tokens = base_input + tool_tokens

    simulated_response = "I don't have real-time data, but let me check... (no tool called)"
    output_tokens = count_tokens(simulated_response)

    return token_report(
        "MCP tools registered (not called)",
        input_tokens, output_tokens,
        note=f"+{tool_tokens} tokens just for tool defs"
    )


def scenario_mcp_one_tool_called() -> dict:
    """
    Model calls get_weather, gets a result, then responds.
    Turn 1 output = tool_call JSON.
    Turn 2 input = everything above + tool_result JSON.
    """
    tool_tokens = count_tokens_in_tools(TOOLS)
    base_input = count_tokens(SYSTEM_PROMPT) + count_tokens(USER_QUESTION) + 8

    # Turn 1: model decides to call a tool
    tool_call_json = json.dumps({
        "tool": "get_weather",
        "arguments": {"city": "San Francisco", "units": "celsius"}
    })
    turn1_output_tokens = count_tokens(tool_call_json)

    # Turn 2: tool result fed back as input
    tool_result_json = json.dumps({
        "city": "San Francisco", "temperature": 18, "units": "celsius",
        "humidity_pct": 72, "wind_speed_kmh": 15, "condition": "Partly cloudy",
        "feels_like": 16, "uv_index": 3
    })
    tool_result_tokens = count_tokens(tool_result_json)

    # Turn 2 input = all previous context + tool_call (now history) + tool_result
    turn2_input_tokens = base_input + tool_tokens + turn1_output_tokens + tool_result_tokens

    final_response = "The current weather in San Francisco is 18°C and partly cloudy with 72% humidity."
    turn2_output_tokens = count_tokens(final_response)

    total_input = turn2_input_tokens
    total_output = turn1_output_tokens + turn2_output_tokens

    return token_report(
        "MCP: one tool called",
        total_input, total_output,
        note=f"tool_defs={tool_tokens} tool_call={turn1_output_tokens} tool_result={tool_result_tokens}"
    )


def scenario_mcp_chained_tools() -> dict:
    """
    Model calls get_weather THEN calculate — two tool calls in one conversation.
    Each turn the context grows because tool calls and results accumulate.
    """
    tool_tokens = count_tokens_in_tools(TOOLS)
    base_input = count_tokens(SYSTEM_PROMPT) + count_tokens(USER_QUESTION) + 8

    # Tool call 1: get_weather
    tc1 = json.dumps({"tool": "get_weather", "arguments": {"city": "San Francisco"}})
    tr1 = json.dumps({"city": "San Francisco", "temperature": 18, "condition": "Partly cloudy"})

    # Tool call 2: calculate (e.g. convert to Fahrenheit)
    tc2 = json.dumps({"tool": "calculate", "arguments": {"expression": "18 * 9/5 + 32"}})
    tr2 = json.dumps({"expression": "18 * 9/5 + 32", "result": 64.4})

    tc1_tokens = count_tokens(tc1)
    tr1_tokens = count_tokens(tr1)
    tc2_tokens = count_tokens(tc2)
    tr2_tokens = count_tokens(tr2)

    # Context grows with every round trip
    final_input = base_input + tool_tokens + tc1_tokens + tr1_tokens + tc2_tokens + tr2_tokens
    final_response = "It's 18°C (64.4°F) and partly cloudy in San Francisco."
    final_output = tc1_tokens + tc2_tokens + count_tokens(final_response)

    return token_report(
        "MCP: 2 chained tool calls",
        final_input, final_output,
        note=f"tool_defs={tool_tokens} history grows each round"
    )


def print_overhead_breakdown():
    console.rule("[bold blue]Token Overhead Breakdown Per Component")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("Who pays?")
    table.add_column("When?")

    rows = [
        ("System prompt", count_tokens(SYSTEM_PROMPT), "Input", "Every request"),
        ("User message", count_tokens(USER_QUESTION), "Input", "Every request"),
        ("--- MCP additions ---", "", "", ""),
        (f"Tool defs ({len(TOOLS)} tools)", count_tokens_in_tools(TOOLS), "Input", "Every request"),
        ("Per-tool avg", count_tokens_in_tools(TOOLS) // len(TOOLS), "Input", "Every request"),
        ("Tool call JSON (output)", count_tokens('{"tool":"get_weather","arguments":{"city":"SF"}}'), "Output→Input", "When tool is called"),
        ("Tool result JSON (input)", count_tokens('{"city":"SF","temperature":18,"condition":"Partly cloudy"}'), "Input", "Next turn after call"),
        ("Final response", count_tokens("It's 18°C and partly cloudy in San Francisco."), "Output", "End of chain"),
    ]

    for row in rows:
        if row[0].startswith("---"):
            table.add_row("", "", "", "")
            table.add_row(f"[bold yellow]{row[0]}[/bold yellow]", "", "", "")
        else:
            table.add_row(row[0], str(row[1]), row[2], row[3])

    console.print(table)


def main():
    console.rule("[bold blue]Lab 04 — MCP Token Overhead")

    scenarios = [
        scenario_plain_prompt(),
        scenario_mcp_tools_registered_not_called(),
        scenario_mcp_one_tool_called(),
        scenario_mcp_chained_tools(),
    ]

    # Print comparison table
    table = Table(show_header=True, header_style="bold magenta", title="Token Cost Comparison")
    table.add_column("Scenario", max_width=35)
    table.add_column("Input\nTokens", justify="right")
    table.add_column("Output\nTokens", justify="right")
    table.add_column("Total\nTokens", justify="right")
    table.add_column("vs Plain\n(multiplier)", justify="right")
    table.add_column("Note", max_width=35, style="dim")

    baseline_total = scenarios[0]["total_tokens"]
    for s in scenarios:
        multiplier = s["total_tokens"] / baseline_total
        color = "green" if multiplier <= 1.1 else ("yellow" if multiplier < 4 else "red")
        table.add_row(
            s["label"],
            str(s["input_tokens"]),
            str(s["output_tokens"]),
            str(s["total_tokens"]),
            f"[{color}]{multiplier:.1f}x[/{color}]",
            s["note"],
        )

    console.print(table)

    print_overhead_breakdown()

    console.print(Panel(
        "[bold]Key Takeaways:[/bold]\n\n"
        "1. [yellow]Tool definitions are always in context[/yellow] — even if no tool is called.\n"
        "   3 tools with detailed descriptions = ~300 extra input tokens per request.\n\n"
        "2. [yellow]Tool results become input on the next turn[/yellow] — the model has to\n"
        "   re-read them. Large tool results (big JSON) are expensive.\n\n"
        "3. [yellow]Chained tool calls multiply the cost[/yellow] — each round trip adds\n"
        "   tool_call + tool_result to the growing context.\n\n"
        "4. [yellow]The model pays for tool defs even if it answers directly[/yellow] —\n"
        "   MCP overhead is per-request, not per-tool-invocation.",
        title="Summary", border_style="blue"
    ))


if __name__ == "__main__":
    main()
