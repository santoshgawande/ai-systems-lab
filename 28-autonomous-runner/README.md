# 28 — Autonomous Lab Runner

Leave your Mac running overnight. It reads a task queue, runs each task with Claude Code, waits when the quota is exhausted, and resumes automatically when it resets.

## What it does

```
tasks.md  →  runner.py  →  claude -p "task"  →  updates tasks.md
                  ↓
            quota hit? → sleep until 5am → resume
```

Every task runs `claude -p` (non-interactive mode) with `--dangerously-skip-permissions` so Claude can read/write files without prompts.

## Quick start

```bash
cd 28-autonomous-runner

# 1. Preview what would run (safe — no claude calls)
python runner.py --list
python runner.py --dry-run

# 2. Run all pending tasks
python runner.py

# 3. Run just one task to test
python runner.py --one
```

## Add your own tasks

Edit `tasks.md` — add any line with `- [ ]`:

```markdown
- [ ] Create a LangGraph lab in section 29 with a basic agent workflow
- [ ] Add semantic chunking to 03-rag/02-indexing using sentence boundaries
- [ ] Build a cost dashboard for 05-system-design that shows per-model spend
```

**Be specific.** Claude Code works best with concrete instructions:
- Bad: `- [ ] improve the RAG section`
- Good: `- [ ] Add HNSW index tuning to 14-vector-databases/01-qdrant: benchmark ef_construct values 50/100/200 against recall@10 and insert speed`

## Run it overnight (automatic)

### Option A — leave terminal open

```bash
python runner.py
# It sleeps when quota hits, wakes at 5am, continues
```

### Option B — macOS launchd (runs even if terminal is closed)

```bash
# Install (runs at 5:10am daily, after quota resets)
cp com.ailabs.runner.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ailabs.runner.plist

# Watch it
tail -f /tmp/ailabs-runner.log

# Remove when done
launchctl unload ~/Library/LaunchAgents/com.ailabs.runner.plist
```

### Option C — cron

```bash
# Run at 5:10am every day
crontab -e
# Add:
10 5 * * * cd /Users/santosh/workspace/github/ai-systems-lab && python 28-autonomous-runner/runner.py >> /tmp/ailabs-runner.log 2>&1
```

## Configuration

```bash
CLAUDE_MODEL=sonnet        # sonnet | opus | haiku  (default: sonnet)
TASK_BUDGET_USD=0.50       # max $ per task (default: $0.50)
QUOTA_RESET_HOUR=5         # hour quota resets (default: 5am)
TASK_TIMEOUT=600           # max seconds per task (default: 10min)
```

```bash
CLAUDE_MODEL=opus TASK_BUDGET_USD=2.00 python runner.py
```

## Commands

```bash
python runner.py              # run all pending tasks
python runner.py --list       # show status of all tasks
python runner.py --dry-run    # preview without running
python runner.py --one        # run exactly one task
python runner.py --reset-failed  # retry failed tasks
```

## How it works

```python
for task in pending_tasks:
    result = run_claude(task)   # subprocess: claude -p "..." --dangerously-skip-permissions

    if result == "quota":
        sleep_until(5am)        # countdown display, wakes automatically
        result = run_claude(task)  # retry after reset

    update_tasks_md(task, result)   # marks [x] done or [!] failed
    save_state()                    # state.json for audit trail
```

### Quota detection

The runner watches `claude` CLI output in real-time for strings like:
- `"usage limit reached"`
- `"limit will reset"`
- `"please try again later"`

When detected: current task is re-queued (not marked failed), runner sleeps.

### Per-task logs

Every task creates a timestamped log in `logs/`:
```
logs/20260507_055132_add_langchain_lab.log
logs/20260507_060415_create_section_29.log
logs/runner.log   ← summary across all runs
```

## Task status markers

| Marker | Meaning |
|--------|---------|
| `- [ ]` | Pending — will be run |
| `- [x]` | Done — completed successfully |
| `- [!]` | Failed — check the log in `logs/` |
| `- [~]` | Skipped — won't be run |

## What's safe

Claude Code with `--dangerously-skip-permissions` can:
- Read and write files in the repo
- Run `git` commands
- Run `python` scripts

It **cannot** (no credentials in env):
- Push to GitHub (no git credentials in the launchd env)
- Call cloud APIs (no API keys in launchd env unless you add them)
- Delete files outside the repo

If you want Claude to also call OpenAI/Anthropic APIs while building labs, add keys to the plist `EnvironmentVariables` section.

## Typical overnight session

```
05:10  launchd starts runner
05:11  [1/12] ▶  Create section 29-langgraph with basic workflow...
05:18  [1/12] ✓  Done  (logs/20260508_051132_create_section_29.log)
05:18  [2/12] ▶  Add streaming lab to 01-llm-apis...
...
08:45  [7/12] ▶  Create mini-memory-agent in 06-build-your-own...
09:02  ⚡ QUOTA EXHAUSTED
       Resuming at: Thursday, May 9 at 05:05 AM  (19h 58m)
...
(next morning)
05:05  ✅ Quota reset — resuming tasks
05:05  [7/12] ▶  Create mini-memory-agent (retry)...
```
