#!/usr/bin/env python3
"""mini-claude-code: a minimal CLI agent that reads/writes files and runs shell commands."""

import sys
import json
import math
import subprocess
import httpx
from pathlib import Path

OLLAMA = "http://localhost:11434"
MODEL = "llama3.3:70b"

SYSTEM = """You are a coding assistant with access to the local filesystem and shell.

ALWAYS respond in this exact format:
Thought: <reasoning about what to do next>
Action: <tool_name>
Action Input: <valid JSON arguments>

When the task is complete:
Thought: The task is complete.
Final Answer: <summary of what was done>

Available tools:
- read_file   {"path": "..."}                    — read a file (first 3000 chars)
- write_file  {"path": "...", "content": "..."}  — write text to a file
- list_dir    {"path": "..."}                    — list directory contents
- run_bash    {"command": "..."}                 — execute a shell command

Safety rules:
- Never run destructive commands (rm -rf, format, drop database)
- Only write to /tmp unless the user specifies otherwise
- Always verify your work by reading back what you wrote"""


def read_file(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore")[:3000]
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"


def list_dir(path: str) -> str:
    try:
        entries = sorted(Path(path).iterdir())
        lines = [f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries]
        return "\n".join(lines) or "(empty directory)"
    except Exception as e:
        return f"Error: {e}"


def run_bash(command: str) -> str:
    blocked = ["rm -rf", "mkfs", ":(){:", "dd if=", "DROP TABLE", "truncate /"]
    if any(b.lower() in command.lower() for b in blocked):
        return "Blocked: command contains a potentially destructive pattern"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        output = (result.stdout + result.stderr).strip()
        return output[:2000] if output else "(no output, exit code: " + str(result.returncode) + ")"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


TOOLS = {
    "read_file":  lambda a: read_file(a["path"]),
    "write_file": lambda a: write_file(a["path"], a["content"]),
    "list_dir":   lambda a: list_dir(a["path"]),
    "run_bash":   lambda a: run_bash(a["command"]),
}


def parse(text: str) -> dict:
    r = {"thought": "", "action": None, "input": None, "final": None}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Thought:"):
            r["thought"] = s[8:].strip()
        elif s.startswith("Action:"):
            r["action"] = s[7:].strip()
        elif s.startswith("Action Input:"):
            raw = s[13:].strip()
            try:
                r["input"] = json.loads(raw)
            except json.JSONDecodeError:
                r["input"] = {"raw": raw}
        elif s.startswith("Final Answer:"):
            r["final"] = s[13:].strip()
    return r


def stream(messages: list) -> str:
    parts = []
    with httpx.stream("POST", f"{OLLAMA}/api/chat",
                      json={"model": MODEL, "messages": messages, "stream": True},
                      timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    parts.append(token)
    print()
    return "".join(parts)


def run(task: str, max_steps: int = 10):
    print(f"Task: {task}\n{'='*60}")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n[Step {step}] ", end="")
        reply = stream(messages)
        messages.append({"role": "assistant", "content": reply})

        p = parse(reply)

        if p["final"]:
            print(f"\n{'='*60}")
            print(f"Done: {p['final']}")
            return

        if p["action"] and p["input"] is not None:
            fn = TOOLS.get(p["action"])
            if fn:
                print(f"\n[{p['action']}({json.dumps(p['input'])[:80]})]")
                obs = fn(p["input"])
            else:
                obs = f"Unknown tool: {p['action']}"
            print(f"→ {obs[:300]}")
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        else:
            messages.append({"role": "user", "content": "Continue using Thought/Action/Action Input format."})

    print("\n⚠ Max steps reached.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        print("mini-claude-code")
        print("Usage: python agent.py <task>")
        print()
        print("Examples:")
        print('  python agent.py "list all python files in /tmp"')
        print('  python agent.py "write a hello world script to /tmp/hello.py and run it"')
        print('  python agent.py "read /etc/hosts and tell me how many entries it has"')
