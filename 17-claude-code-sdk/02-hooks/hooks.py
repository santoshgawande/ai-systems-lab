"""
Claude Code hooks: generate hook scripts and show configuration patterns.
Creates working hook scripts you can drop into ~/.claude/settings.json.
"""
import os
import json
import stat

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated-hooks")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Hook scripts ─────────────────────────────────────────────────────────────

BASH_GUARD_SCRIPT = '''#!/bin/bash
# PreToolUse(Bash) — blocks dangerous shell commands
# Exit code 2 = block the tool call; any other exit = allow

COMMAND="${CLAUDE_BASH_COMMAND}"

# Patterns to block
BLOCKED=(
    "rm -rf"
    "rm -f /"
    "dd if="
    "> /dev/sda"
    "mkfs"
    "chmod 777 /"
    "curl.*|.*bash"
    "wget.*|.*sh"
    "sudo rm"
    "sudo dd"
)

for pattern in "${BLOCKED[@]}"; do
    if echo "$COMMAND" | grep -qiE "$pattern"; then
        echo "BLOCKED: dangerous pattern '$pattern' in command: $COMMAND" >&2
        exit 2   # exit 2 = block this tool call
    fi
done

# Log allowed commands
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) BASH: $COMMAND" >> /tmp/claude-bash-audit.log
exit 0   # allow
'''

WRITE_FORMAT_SCRIPT = '''#!/bin/bash
# PostToolUse(Write/Edit) — auto-format Python files after Claude writes them
# Exit code doesn't block (already done), but can signal issues

FILE="${CLAUDE_FILE_PATH}"

if [[ -z "$FILE" ]]; then
    exit 0
fi

# Auto-format Python files
if [[ "$FILE" == *.py ]]; then
    if command -v black &>/dev/null; then
        black --quiet "$FILE" 2>/dev/null && echo "Formatted: $FILE"
    elif command -v autopep8 &>/dev/null; then
        autopep8 --in-place "$FILE" 2>/dev/null
    fi
fi

# Auto-format JS/TS/JSON files
if [[ "$FILE" =~ \\.(js|ts|tsx|jsx|json)$ ]]; then
    if command -v prettier &>/dev/null; then
        prettier --write "$FILE" 2>/dev/null && echo "Formatted: $FILE"
    fi
fi

exit 0
'''

AUDIT_LOG_SCRIPT = '''#!/bin/bash
# PreToolUse(*) — log every tool call for audit/compliance
# Always allows — only logs

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TOOL="${CLAUDE_TOOL_NAME}"
INPUT="${CLAUDE_TOOL_INPUT}"

# Sanitize for single-line logging
INPUT_SAFE=$(echo "$INPUT" | tr '\\n' ' ' | head -c 200)

echo "${TIMESTAMP} TOOL=${TOOL} INPUT=${INPUT_SAFE}" >> /tmp/claude-audit.log

exit 0   # always allow
'''

NOTIFY_SCRIPT = '''#!/bin/bash
# Notification — alert when Claude needs user input (macOS)

if command -v terminal-notifier &>/dev/null; then
    terminal-notifier -title "Claude Code" -message "Waiting for your input" -sound default
elif command -v osascript &>/dev/null; then
    osascript -e 'display notification "Claude Code is waiting for input" with title "Claude Code"'
fi

exit 0
'''

GIT_STAGE_SCRIPT = '''#!/bin/bash
# PostToolUse(Write) — auto-stage files Claude writes
# Useful for reviewing Claude's changes with git diff --staged

FILE="${CLAUDE_FILE_PATH}"
if [[ -n "$FILE" ]] && git -C "$(dirname "$FILE")" rev-parse --git-dir &>/dev/null; then
    git -C "$(dirname "$FILE")" add "$(basename "$FILE")" 2>/dev/null
fi

exit 0
'''

TEST_RUNNER_SCRIPT = '''#!/bin/bash
# PostToolUse(Write) — run tests if Claude modified a test file
# Reports pass/fail but doesn't block

FILE="${CLAUDE_FILE_PATH}"
if [[ "$FILE" == */test_*.py ]] || [[ "$FILE" == *_test.py ]]; then
    echo "Running tests for: $FILE"
    cd "$(git -C "$(dirname "$FILE")" rev-parse --show-toplevel)" 2>/dev/null || cd "$(dirname "$FILE")"
    python -m pytest "$FILE" -q 2>&1 | tail -5
fi

exit 0
'''

SCRIPTS = {
    "bash-guard.sh": BASH_GUARD_SCRIPT,
    "write-format.sh": WRITE_FORMAT_SCRIPT,
    "audit-log.sh": AUDIT_LOG_SCRIPT,
    "notify.sh": NOTIFY_SCRIPT,
    "git-stage.sh": GIT_STAGE_SCRIPT,
    "test-runner.sh": TEST_RUNNER_SCRIPT,
}

# ─── settings.json with all hooks wired ──────────────────────────────────────

def settings_json(scripts_dir: str) -> dict:
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{
                        "type": "command",
                        "command": f"{scripts_dir}/bash-guard.sh"
                    }]
                },
                {
                    "matcher": ".*",   # all tools
                    "hooks": [{
                        "type": "command",
                        "command": f"{scripts_dir}/audit-log.sh"
                    }]
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "Write",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{scripts_dir}/write-format.sh"
                        },
                        {
                            "type": "command",
                            "command": f"{scripts_dir}/git-stage.sh"
                        },
                        {
                            "type": "command",
                            "command": f"{scripts_dir}/test-runner.sh"
                        }
                    ]
                }
            ],
            "Notification": [
                {
                    "hooks": [{
                        "type": "command",
                        "command": f"{scripts_dir}/notify.sh"
                    }]
                }
            ]
        }
    }


# ─── Generate scripts ─────────────────────────────────────────────────────────

print("=== CLAUDE CODE HOOKS DEMO ===\n")
print(f"Writing hook scripts to: {OUTPUT_DIR}\n")

for filename, content in SCRIPTS.items():
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  Created: {filename}")

settings = settings_json(OUTPUT_DIR)
settings_path = os.path.join(OUTPUT_DIR, "settings-with-hooks.json")
with open(settings_path, "w") as f:
    json.dump(settings, f, indent=2)
print(f"  Created: settings-with-hooks.json\n")

print("To activate these hooks:")
print(f"  cp {settings_path} ~/.claude/settings.json")
print(f"  # or merge the 'hooks' section into your existing settings.json\n")

print("─── Hook scripts summary ───\n")
hooks_summary = [
    ("bash-guard.sh",    "PreToolUse(Bash)",   "Blocks rm -rf, sudo, pipe-to-bash patterns"),
    ("audit-log.sh",     "PreToolUse(*)",       "Logs every tool call to /tmp/claude-audit.log"),
    ("write-format.sh",  "PostToolUse(Write)",  "Auto-runs black/prettier after file writes"),
    ("git-stage.sh",     "PostToolUse(Write)",  "Auto-stages modified files with git add"),
    ("test-runner.sh",   "PostToolUse(Write)",  "Runs pytest when Claude modifies test files"),
    ("notify.sh",        "Notification",        "macOS notification when Claude needs input"),
]

for script, event, description in hooks_summary:
    print(f"  {script:<22} [{event:<22}] {description}")

print()
print("Environment variables available in hook scripts:")
env_vars = [
    ("$CLAUDE_TOOL_NAME",    "'Bash', 'Write', 'Edit', 'Read', ..."),
    ("$CLAUDE_TOOL_INPUT",   "Full JSON input to the tool"),
    ("$CLAUDE_FILE_PATH",    "File path (Write/Edit/Read only)"),
    ("$CLAUDE_BASH_COMMAND", "Shell command string (Bash only)"),
]
for var, desc in env_vars:
    print(f"  {var:<25} {desc}")

print()
print("Exit codes:")
print("  PreToolUse exit 2  → BLOCK the tool call (shows error to Claude)")
print("  PreToolUse exit 0  → ALLOW the tool call")
print("  PostToolUse exit * → Ignored (action already taken)")
