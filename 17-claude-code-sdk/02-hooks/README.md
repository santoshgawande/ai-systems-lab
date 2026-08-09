# Lab 02 — Claude Code Hooks

Shell commands that run automatically before or after Claude's tool calls.

## What you learn

- Hook event types: PreToolUse, PostToolUse, Notification, Stop
- How to block dangerous commands (PreToolUse exit code 2)
- How to auto-format files after Claude writes them (PostToolUse)
- How to audit every tool call for security and compliance
- Environment variables available to hook scripts

## Run

```bash
python hooks.py
# Generates hooks configurations and test scripts
```

## Hook event types

| Event | When | Exit code 2 effect |
|---|---|---|
| `PreToolUse` | Before Claude calls a tool | BLOCKS the tool call |
| `PostToolUse` | After Claude calls a tool | No blocking (already done) |
| `Notification` | When Claude needs user input | N/A |
| `Stop` | When Claude finishes a turn | N/A |

## Configuration in ~/.claude/settings.json

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "/path/to/pre-bash-guard.sh"
        }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [{
          "type": "command",
          "command": "prettier --write \"$CLAUDE_FILE_PATH\" 2>/dev/null; true"
        }]
      }
    ]
  }
}
```

## Environment variables in hooks

| Variable | Value |
|---|---|
| `$CLAUDE_TOOL_NAME` | "Bash", "Write", "Edit", "Read", etc. |
| `$CLAUDE_TOOL_INPUT` | Full JSON input to the tool |
| `$CLAUDE_FILE_PATH` | File path (for Read/Write/Edit) |
| `$CLAUDE_BASH_COMMAND` | Shell command (for Bash) |

## Hook use cases

| Hook | Event | What it does |
|---|---|---|
| Bash guard | PreToolUse(Bash) | Block `rm -rf`, `sudo`, `curl \| bash` |
| Audit log | PreToolUse(*) | Log every tool call to file |
| Auto-format | PostToolUse(Write) | Run prettier/black after file write |
| Git stage | PostToolUse(Write) | Auto-stage modified files |
| Notify | Notification | Send macOS notification when Claude needs input |
| Test runner | PostToolUse(Write) | Run pytest on modified test files |
