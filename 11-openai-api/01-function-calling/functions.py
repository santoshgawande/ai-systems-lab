"""
OpenAI function calling: define tools, dispatch calls, feed results back.
Demonstrates parallel tool calls — model invokes multiple tools in one turn.
Requires: OPENAI_API_KEY
"""
import os
import json
import math
import datetime

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    print("OPENAI_API_KEY not set. Showing function calling mechanics.\n")
    LIVE = False
else:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    LIVE = True

MODEL = "gpt-4o-mini"

# ─── Tool definitions (JSON schema) ──────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, log.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. 'sqrt(144)', '2**10 + 5'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city (simulated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unit_convert",
            "description": "Convert between units (km/miles, kg/lbs, celsius/fahrenheit).",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    }
]


# ─── Tool implementations ─────────────────────────────────────────────────────

WEATHER_DB = {
    "tokyo": {"celsius": 18, "fahrenheit": 64, "condition": "partly cloudy"},
    "paris": {"celsius": 15, "fahrenheit": 59, "condition": "rainy"},
    "new york": {"celsius": 22, "fahrenheit": 72, "condition": "sunny"},
    "london": {"celsius": 12, "fahrenheit": 54, "condition": "overcast"},
    "sydney": {"celsius": 25, "fahrenheit": 77, "condition": "clear"},
}

CONVERSIONS = {
    ("km", "miles"): lambda v: v * 0.621371,
    ("miles", "km"): lambda v: v * 1.60934,
    ("kg", "lbs"): lambda v: v * 2.20462,
    ("lbs", "kg"): lambda v: v * 0.453592,
    ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
    ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
}


def execute_tool(name: str, args: dict) -> str:
    if name == "calculator":
        try:
            safe = {k: v for k, v in vars(math).items() if not k.startswith("_")}
            result = eval(args["expression"], {"__builtins__": {}}, safe)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    elif name == "get_datetime":
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elif name == "get_weather":
        city = args["city"].lower()
        unit = args.get("unit", "celsius")
        data = WEATHER_DB.get(city)
        if not data:
            return f"No weather data for {args['city']}"
        temp = data[unit]
        symbol = "°C" if unit == "celsius" else "°F"
        return f"{args['city']}: {temp}{symbol}, {data['condition']}"

    elif name == "unit_convert":
        key = (args["from_unit"].lower(), args["to_unit"].lower())
        fn = CONVERSIONS.get(key)
        if not fn:
            return f"No conversion for {args['from_unit']} → {args['to_unit']}"
        result = fn(args["value"])
        return f"{args['value']} {args['from_unit']} = {result:.4f} {args['to_unit']}"

    return f"Unknown tool: {name}"


# ─── Tool loop ────────────────────────────────────────────────────────────────

def chat_with_tools(user_message: str, verbose: bool = True) -> str:
    messages = [{"role": "user", "content": user_message}]
    if verbose:
        print(f"User: {user_message}")

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        choice = response.choices[0]
        msg = choice.message

        # Append assistant message (may contain tool_calls)
        messages.append(msg)

        if choice.finish_reason == "tool_calls":
            if verbose:
                print(f"\n  [Model requesting {len(msg.tool_calls)} tool call(s)]")
            # Execute all tool calls (parallel!)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                if verbose:
                    print(f"  Tool: {tc.function.name}({args}) → {result}")
                # Feed result back as tool message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
        elif choice.finish_reason == "stop":
            answer = msg.content or ""
            if verbose:
                print(f"\nAssistant: {answer}\n")
            return answer
        else:
            break

    return ""


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== OPENAI FUNCTION CALLING DEMO ===\n")

    if not LIVE:
        print("Function calling API shape:\n")
        print("""
# 1. Single tool call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's sqrt(144)?"}],
    tools=TOOLS,
    tool_choice="auto"
)

# Check if model wants a tool
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    name = tool_call.function.name         # "calculator"
    args = json.loads(tool_call.function.arguments)  # {"expression": "sqrt(144)"}
    result = execute_tool(name, args)      # "12.0"

    # Feed result back
    messages.append(response.choices[0].message)  # assistant with tool_calls
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result
    })

    # Get final answer
    final = client.chat.completions.create(model=model, messages=messages, tools=tools)


# 2. Parallel tool calls — model asks for multiple tools at once
# user: "What's the weather in Tokyo AND Paris?"
# → model returns tool_calls = [get_weather(Tokyo), get_weather(Paris)]
# → execute both, append two tool messages, continue
""")
        print("Key points:")
        print("  - finish_reason='tool_calls' means model wants tools (not 'stop')")
        print("  - Parallel calls: multiple items in tool_calls[] array")
        print("  - tool_call_id must match when returning results")
        print("  - tool_choice='required' forces at least one tool call")
    else:
        DEMO_QUERIES = [
            "What is sqrt(256) + 2**8?",
            "What's the weather in Tokyo and Paris right now?",
            "Convert 100 km to miles and 70 kg to lbs.",
            "What time is it, and what's 15% of 847?",
        ]
        for query in DEMO_QUERIES:
            print("-" * 60)
            chat_with_tools(query)
