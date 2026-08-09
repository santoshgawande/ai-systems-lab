import sys
import json
import math
import os
import httpx

OLLAMA = "http://localhost:11434"
MODEL = "llama3.3:70b"

SYSTEM = """You solve tasks step by step using tools. ALWAYS use this exact format:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <JSON arguments>

When finished:
Thought: I have the answer.
Final Answer: <your complete answer>

Available tools:
- calculator  {"expression": "..."} — evaluate math
- read_file   {"path": "..."}       — read a file
- write_file  {"path": "...", "content": "..."} — write a file
- list_dir    {"path": "..."}       — list directory contents
- run_bash    {"command": "..."}    — run a shell command

Never skip the Thought line. Never use a tool not listed above."""


def calculator(expression: str) -> str:
    try:
        safe = {k: v for k, v in vars(math).items() if not k.startswith("_")}
        return str(eval(expression, {"__builtins__": {}}, safe))
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str) -> str:
    try:
        return open(path).read()[:3000]
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"


def list_dir(path: str) -> str:
    try:
        return "\n".join(os.listdir(path)) or "(empty)"
    except Exception as e:
        return f"Error: {e}"


def run_bash(command: str) -> str:
    import subprocess
    blocked = ["rm -rf", "mkfs", ":(){:", "dd if="]
    if any(b in command for b in blocked):
        return f"Blocked: unsafe command"
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
        return (r.stdout + r.stderr)[:2000] or "(no output)"
    except Exception as e:
        return f"Error: {e}"


TOOLS = {
    "calculator": lambda a: calculator(a["expression"]),
    "read_file":  lambda a: read_file(a["path"]),
    "write_file": lambda a: write_file(a["path"], a["content"]),
    "list_dir":   lambda a: list_dir(a["path"]),
    "run_bash":   lambda a: run_bash(a["command"]),
}


def parse(text: str) -> dict:
    result = {"thought": "", "action": None, "input": None, "final": None}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Thought:"):
            result["thought"] = s[8:].strip()
        elif s.startswith("Action:"):
            result["action"] = s[7:].strip()
        elif s.startswith("Action Input:"):
            raw = s[13:].strip()
            try:
                result["input"] = json.loads(raw)
            except json.JSONDecodeError:
                result["input"] = {"raw": raw}
        elif s.startswith("Final Answer:"):
            result["final"] = s[13:].strip()
    return result


def run(task: str, max_steps: int = 8):
    print(f"Task: {task}\n{'='*60}")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} ---")
        r = httpx.post(f"{OLLAMA}/api/chat",
                       json={"model": MODEL, "messages": messages, "stream": False},
                       timeout=60)
        r.raise_for_status()
        reply = r.json()["message"]["content"]
        messages.append({"role": "assistant", "content": reply})
        print(reply)

        parsed = parse(reply)

        if parsed["final"]:
            print(f"\n✓ Final Answer: {parsed['final']}")
            return

        if parsed["action"] and parsed["input"] is not None:
            fn = TOOLS.get(parsed["action"])
            observation = fn(parsed["input"]) if fn else f"Unknown tool: {parsed['action']}"
            print(f"\nObservation: {observation[:300]}")
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            messages.append({"role": "user", "content": "Continue using Thought/Action/Action Input format."})

    print("\n⚠ Max steps reached.")


task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
    "Calculate the area of a circle with radius 7, then write the result to /tmp/circle_area.txt"
run(task)
