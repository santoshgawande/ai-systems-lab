#!/usr/bin/env python3
"""
Autonomous Claude Code Lab Runner

Reads tasks from tasks.md, runs each with `claude -p`,
detects quota exhaustion, sleeps until reset, then continues.
Leave your Mac running overnight — it uses your full quota every day.

Usage:
  python runner.py                  # run all pending tasks
  python runner.py --dry-run        # preview without running
  python runner.py --list           # show task status
  python runner.py --reset-failed   # mark failed tasks as pending again
  python runner.py --one            # run exactly one task then stop

Environment vars:
  QUOTA_RESET_HOUR=5        hour (0-23) when Claude quota resets (default: 5am)
  TASK_TIMEOUT=600          max seconds per task (default: 10 min)
  TASK_BUDGET_USD=0.50      max $ per task via --max-budget-usd (default: 0.50)
  CLAUDE_MODEL=sonnet       model alias: sonnet | opus | haiku (default: sonnet)
"""

import argparse
import datetime
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent.resolve()          # ai-systems-lab root
TASKS_FILE = HERE / "tasks.md"
STATE_FILE = HERE / "state.json"
LOGS_DIR = HERE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────

QUOTA_RESET_HOUR = int(os.environ.get("QUOTA_RESET_HOUR", "5"))
TASK_TIMEOUT_SEC = int(os.environ.get("TASK_TIMEOUT", "600"))
TASK_BUDGET_USD = os.environ.get("TASK_BUDGET_USD", "0.50")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "sonnet")
INTER_TASK_SLEEP = 5   # seconds between tasks

# Text patterns that indicate the usage quota is exhausted
QUOTA_SIGNALS = [
    "usage limit reached",
    "usage limit has been reached",
    "claude code usage limit",
    "daily usage limit",
    "rate limit exceeded",
    "quota exceeded",
    "too many requests",
    "please try again tomorrow",
    "please try again later",
    "upgrade your plan",
    "limit will reset",
    "usage will reset",
    "usagelimitexceeded",
    "overloaded_error",          # anthropic error type when servers are saturated
]

# Exit codes that claude CLI emits on quota/auth failure
QUOTA_EXIT_CODES = {1}  # may expand — test with your account type

CONTEXT_PREAMBLE = f"""You are working autonomously in the ai-systems-lab repository.
Repository root: {REPO_ROOT}

This is a structured learning lab for production AI systems engineering.
Each numbered directory (01-llm-apis … 28-autonomous-runner) is a concept with working code.
Pattern for every lab:
  - <section>/README.md          — what you learn + run instructions
  - <section>/<nn>-<name>/       — one lab per concept
  - <section>/<nn>-<name>/<name>.py — working Python with graceful API-key fallbacks
  - <section>/<nn>-<name>/README.md — what the lab covers

Rules:
1. Work completely autonomously — make reasonable decisions, never ask for clarification.
2. Follow existing code patterns: API key checks at top, fallback demo when key missing.
3. Every .py file must be runnable standalone (python <file>.py).
4. Add READMEs that include: what you learn, how to run, key code snippet, trade-off table.
5. Keep code concise — no unnecessary comments, no docstring novels.

"""


# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "runner.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("runner")


# ─── Task file parser/updater ─────────────────────────────────────────────────
#
# Format in tasks.md:
#   - [ ] pending task
#   - [x] done task
#   - [!] failed task
#   - [~] skipped task
#

TASK_RE = re.compile(r"^- \[([ x!~])\] (.+)$", re.MULTILINE)
STATUS_CHAR = {"pending": " ", "done": "x", "failed": "!", "skipped": "~"}
CHAR_STATUS = {v: k for k, v in STATUS_CHAR.items()}


def load_tasks() -> list[dict]:
    text = TASKS_FILE.read_text(encoding="utf-8")
    tasks = []
    for m in TASK_RE.finditer(text):
        char, desc = m.group(1), m.group(2).strip()
        tasks.append({
            "description": desc,
            "status": CHAR_STATUS.get(char, "pending"),
        })
    return tasks


def set_task_status(description: str, status: str) -> None:
    char = STATUS_CHAR[status]
    text = TASKS_FILE.read_text(encoding="utf-8")
    escaped = re.escape(description)
    pattern = re.compile(r"^(- \[)[ x!~](\] " + escaped + r")$", re.MULTILINE)
    new_text = pattern.sub(rf"\g<1>{char}\g<2>", text, count=1)
    if new_text == text:
        log.warning(f"  Could not find task line to update: {description[:60]}")
    TASKS_FILE.write_text(new_text, encoding="utf-8")


# ─── State ────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"runs": [], "completed": [], "failed": []}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ─── Quota detection ──────────────────────────────────────────────────────────

def is_quota_signal(text: str) -> bool:
    lower = text.lower()
    return any(sig in lower for sig in QUOTA_SIGNALS)


# ─── Task runner ─────────────────────────────────────────────────────────────

def run_one_task(description: str, index: int, total: int, dry_run: bool = False) -> str:
    """
    Run a single task with `claude -p`.
    Returns: "done" | "failed" | "quota"
    """
    bar = f"[{index}/{total}]"
    log.info(f"{bar} ▶  {description[:80]}")

    if dry_run:
        log.info(f"{bar}    [DRY RUN] would run claude -p for this task")
        return "done"

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^\w]+", "_", description[:40]).lower().strip("_")
    log_file = LOGS_DIR / f"{ts}_{slug}.log"

    prompt = CONTEXT_PREAMBLE + "Task: " + description

    cmd = [
        "claude",
        "-p", prompt,
        "--dangerously-skip-permissions",
        "--model", CLAUDE_MODEL,
        "--max-budget-usd", TASK_BUDGET_USD,
    ]

    quota_hit = False
    output_lines: list[str] = []

    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"Task:    {description}\n")
            lf.write(f"Model:   {CLAUDE_MODEL}\n")
            lf.write(f"Budget:  ${TASK_BUDGET_USD}\n")
            lf.write(f"Started: {datetime.datetime.now().isoformat()}\n")
            lf.write("=" * 70 + "\n\n")

            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            try:
                for line in iter(proc.stdout.readline, ""):
                    sys.stdout.write(f"  {line}")
                    sys.stdout.flush()
                    lf.write(line)
                    output_lines.append(line)

                    if is_quota_signal(line):
                        log.warning(f"\n{bar} ⚡ Quota limit detected in output")
                        quota_hit = True
                        proc.terminate()
                        break

            except KeyboardInterrupt:
                proc.terminate()
                raise

            exit_code = proc.wait(timeout=30)

        all_output = "".join(output_lines)

        if quota_hit or is_quota_signal(all_output):
            return "quota"

        if exit_code == 0:
            log.info(f"{bar} ✓  Done  (log: logs/{log_file.name})")
            return "done"
        else:
            log.warning(f"{bar} ✗  Failed (exit={exit_code}  log: logs/{log_file.name})")
            return "failed"

    except subprocess.TimeoutExpired:
        log.error(f"{bar} ✗  Timed out after {TASK_TIMEOUT_SEC}s")
        try:
            proc.kill()
        except Exception:
            pass
        return "failed"

    except FileNotFoundError:
        log.error("'claude' CLI not found. Run: npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

    except KeyboardInterrupt:
        log.info(f"\n{bar} Interrupted — saving progress")
        raise

    except Exception as e:
        log.error(f"{bar} ✗  Unexpected error: {e}")
        return "failed"


# ─── Quota sleep ──────────────────────────────────────────────────────────────

def sleep_until_reset() -> None:
    now = datetime.datetime.now()
    reset = now.replace(hour=QUOTA_RESET_HOUR, minute=5, second=0, microsecond=0)
    if reset <= now:
        reset += datetime.timedelta(days=1)

    wait_sec = (reset - now).total_seconds()
    hours = int(wait_sec // 3600)
    minutes = int((wait_sec % 3600) // 60)

    print(f"\n{'━' * 60}")
    print(f"  ⚡ QUOTA EXHAUSTED")
    print(f"  Resuming at:  {reset.strftime('%A, %b %d at %I:%M %p')}")
    print(f"  Sleep time:   {hours}h {minutes}m")
    print(f"  Safe to leave this running — it will wake automatically.")
    print(f"{'━' * 60}\n")

    # Log to file so you can verify it's running
    log.info(f"Sleeping until {reset.isoformat()} ({hours}h {minutes}m)")

    # Countdown loop — updates every minute
    while True:
        remaining = (reset - datetime.datetime.now()).total_seconds()
        if remaining <= 0:
            break
        h = int(remaining // 3600)
        m = int((remaining % 3600) // 60)
        print(f"\r  ⏳ Resuming in {h:02d}h {m:02d}m …  ", end="", flush=True)
        time.sleep(min(60, remaining))

    print(f"\n\n  ✅ Quota reset — resuming tasks …\n")
    log.info("Quota reset — resuming")


# ─── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Claude Code lab runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would run, don't execute")
    parser.add_argument("--list", action="store_true", help="Print task status and exit")
    parser.add_argument("--one", action="store_true", help="Run exactly one pending task then stop")
    parser.add_argument("--reset-failed", action="store_true", help="Reset failed tasks → pending")
    args = parser.parse_args()

    if not TASKS_FILE.exists():
        log.error(f"tasks.md not found at {TASKS_FILE}")
        log.error("Create it with lines like:  - [ ] Create a lab on LangGraph basics")
        sys.exit(1)

    # ── Reset failed tasks ──
    if args.reset_failed:
        tasks = load_tasks()
        n = sum(1 for t in tasks if t["status"] == "failed")
        for t in tasks:
            if t["status"] == "failed":
                set_task_status(t["description"], "pending")
        log.info(f"Reset {n} failed tasks → pending")
        return

    # ── List mode ──
    if args.list:
        tasks = load_tasks()
        icons = {"pending": "○", "done": "✓", "failed": "✗", "skipped": "–"}
        counts = {s: 0 for s in icons}
        print()
        for t in tasks:
            s = t["status"]
            counts[s] += 1
            print(f"  {icons[s]}  [{s:<8}]  {t['description'][:72]}")
        print()
        print(f"  Total: {len(tasks)}  |  " + "  ".join(f"{icons[s]} {counts[s]} {s}" for s in icons))
        print()
        return

    # ── Run mode ──
    tasks = load_tasks()
    pending = [t for t in tasks if t["status"] == "pending"]
    done_count = sum(1 for t in tasks if t["status"] == "done")

    print(f"\n{'━' * 60}")
    print(f"  Claude Code Autonomous Runner")
    print(f"  Repo:    {REPO_ROOT}")
    print(f"  Model:   {CLAUDE_MODEL}  |  Budget: ${TASK_BUDGET_USD}/task")
    print(f"  Tasks:   {len(pending)} pending  {done_count} done  {len(tasks)-len(pending)-done_count} other")
    print(f"  Quota resets at: {QUOTA_RESET_HOUR:02d}:05 daily")
    print(f"{'━' * 60}\n")

    if not pending:
        log.info("No pending tasks. Add tasks to tasks.md and re-run.")
        return

    state = load_state()
    run_record = {
        "started": datetime.datetime.now().isoformat(),
        "completed": 0,
        "failed": 0,
        "quota_waits": 0,
    }

    def handle_interrupt(sig, frame):
        print("\n\nInterrupted. Progress saved in tasks.md and state.json.\nRe-run to continue.\n")
        state["runs"].append(run_record)
        save_state(state)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    # If --one, only process the first pending task
    if args.one:
        pending = pending[:1]

    total = len(pending)
    i = 0

    while i < len(pending):
        task = pending[i]
        desc = task["description"]

        result = run_one_task(desc, i + 1, total, dry_run=args.dry_run)

        if result == "quota":
            run_record["quota_waits"] += 1
            # Don't mark the task failed — it didn't run
            sleep_until_reset()
            # Re-read tasks in case user edited the file while we slept
            pending = [t for t in load_tasks() if t["status"] == "pending"]
            total = len(pending)
            # Retry current task (don't advance i)
            result = run_one_task(desc, i + 1, total, dry_run=args.dry_run)
            if result == "quota":
                log.error("Still hitting quota immediately after reset. Check your plan.")
                state["runs"].append(run_record)
                save_state(state)
                sys.exit(1)

        # Update tasks.md (skip in dry-run — read-only mode)
        if not args.dry_run:
            if result == "done":
                set_task_status(desc, "done")
                state["completed"].append({"task": desc, "ts": datetime.datetime.now().isoformat()})
                run_record["completed"] += 1
            elif result == "failed":
                set_task_status(desc, "failed")
                state["failed"].append({"task": desc, "ts": datetime.datetime.now().isoformat()})
                run_record["failed"] += 1

        save_state(state)
        i += 1

        # Pause between tasks (skip after last)
        if not args.dry_run and i < len(pending):
            time.sleep(INTER_TASK_SLEEP)

    run_record["finished"] = datetime.datetime.now().isoformat()
    state["runs"].append(run_record)
    save_state(state)

    print(f"\n{'━' * 60}")
    print(f"  Run complete")
    print(f"  Completed: {run_record['completed']}  Failed: {run_record['failed']}  Quota waits: {run_record['quota_waits']}")
    print(f"{'━' * 60}\n")


if __name__ == "__main__":
    main()
