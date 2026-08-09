# Section 17 — Claude Code SDK

Claude Code is more than a CLI — it's an extensible AI development platform you can program.

## What you learn

- CLAUDE.md — project-level and user-level instructions that persist across sessions
- Hooks — shell commands that run before/after Claude's tool calls
- MCP server development — build and wire custom tools into Claude Code

## Labs

| Lab | What it covers |
|---|---|
| 01-claude-md | CLAUDE.md structure, memory files, project vs user-level settings |
| 02-hooks | Pre/post-tool hooks, blocking bad commands, custom automation |
| 03-mcp-server-dev | Build a real MCP server, wire it into Claude Code |

## Setup

```bash
pip install mcp httpx
# Claude Code: claude.ai/code or the Claude CLI
```

## CLAUDE.md hierarchy

```
~/.claude/CLAUDE.md          # user-level: applies everywhere
project-root/CLAUDE.md       # project-level: applies to this project
project-root/subdir/CLAUDE.md # subdirectory-level
```

Claude reads all of these and merges instructions (project takes precedence).

## Hook event types

```json
{
  "hooks": {
    "PreToolUse": [
      {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo pre-bash"}]}
    ],
    "PostToolUse": [
      {"matcher": "Write", "hooks": [{"type": "command", "command": "prettier --write $FILE"}]}
    ],
    "Notification": [
      {"hooks": [{"type": "command", "command": "terminal-notifier -message 'Claude needs input'"}]}
    ]
  }
}
```

## Claude Code extension points

| Mechanism | What it does | Where configured |
|---|---|---|
| CLAUDE.md | Persistent instructions | Files in project/home |
| Hooks | Shell automation around tool calls | settings.json |
| MCP servers | Custom tools available to Claude | settings.json mcpServers |
| Slash commands | Custom /commands | settings.json |
| Permissions | Allow/deny specific tool patterns | settings.json |
