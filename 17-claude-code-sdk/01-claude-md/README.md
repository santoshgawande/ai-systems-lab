# Lab 01 — CLAUDE.md

Persistent instructions that survive across every Claude Code session.

## What you learn

- The CLAUDE.md file format and what Claude reads from it
- Hierarchy: user-level vs project-level vs subdirectory-level
- What belongs in CLAUDE.md vs what belongs in settings.json
- The auto-memory system: how Claude builds its own memory files
- Writing CLAUDE.md files that actually change behavior

## Run

```bash
python claude_md.py
# Generates example CLAUDE.md files for different use cases
```

## CLAUDE.md hierarchy

```
~/.claude/CLAUDE.md                  ← user-level (all projects)
~/.claude/projects/<project>/        ← auto-memory directory (per project)
    MEMORY.md                        ← memory index (auto-generated)
    user_profile.md                  ← user preferences
    feedback_*.md                    ← feedback memories
    project_*.md                     ← project context memories

/your-project/CLAUDE.md              ← project-level (this project)
/your-project/src/CLAUDE.md          ← subdirectory-level (src/ only)
```

## What to put in CLAUDE.md

```markdown
# Project instructions

## Tech stack
- Python 3.12, FastAPI, SQLAlchemy 2.0
- PostgreSQL on proxmox1 (192.168.0.111:5432)
- Ollama on localhost:11434

## Code conventions
- Always use httpx (not requests) for HTTP calls
- No print() in production code — use structlog
- Type hints required on all public functions

## Testing
- pytest, not unittest
- Test files go in tests/ mirroring src/ structure
- Use fixtures, not setUp/tearDown

## What NOT to do
- Never install packages globally (always in virtualenv)
- Never commit .env files
- Don't add TODO comments — create a GitHub issue instead
```

## What NOT to put in CLAUDE.md

- Secrets, API keys, passwords
- Highly volatile information (sprint tasks, current bugs) → use conversation
- Personal preferences that don't affect this project
