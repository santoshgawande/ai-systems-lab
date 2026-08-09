"""
CLAUDE.md generator: creates example CLAUDE.md files for different project types.
Shows how to structure persistent instructions for different use cases.
No dependencies needed — pure Python.
"""
import os
import json

# ─── CLAUDE.md templates ─────────────────────────────────────────────────────

TEMPLATES = {
    "python-backend": """# Python Backend Project

## Tech stack
- Python 3.12, FastAPI 0.115+, SQLAlchemy 2.0 async
- PostgreSQL 16 on proxmox1 (192.168.0.111:5432)
- Redis for caching/sessions on proxmox1 (192.168.0.111:6379)
- Ollama on localhost:11434 for local LLM inference
- httpx (not requests) for all HTTP calls
- structlog for structured logging (no print() in production code)

## Project structure
```
src/
  api/          API routers (one file per resource)
  models/       SQLAlchemy models
  schemas/      Pydantic schemas
  services/     Business logic
  deps.py       FastAPI dependency injection
tests/          Mirror of src/ structure
alembic/        Database migrations
```

## Code conventions
- Type hints required on all public functions
- Pydantic v2 models for all API schemas
- Async/await throughout (no sync database calls)
- Use `Annotated[Type, Depends(...)]` for FastAPI dependencies
- No bare `except:` — always catch specific exceptions

## Database
- Alembic for migrations — never edit existing migration files
- All queries use SQLAlchemy ORM (no raw SQL strings)
- Parameterized queries only — no f-string SQL

## Testing
- pytest with pytest-asyncio
- Use TestClient for endpoint tests, not real HTTP
- One fixture file per module in tests/conftest.py
- Target: 80% coverage on services/, 100% on models/

## Do NOT
- Install packages globally — use virtualenv
- Commit .env files — use .env.example
- Use print() — use logger = structlog.get_logger()
- Write synchronous database code
""",

    "data-science": """# Data Science / ML Project

## Environment
- Python 3.11, Jupyter Lab on proxmox2 (192.168.0.112:8888)
- Key libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- Ollama on localhost:11434 for LLM-assisted analysis

## Conventions
- All notebooks start with a "## Setup" cell with imports
- Never commit notebooks with executed cells — clear output before committing
- Data files go in data/raw/ (never modified) and data/processed/
- Models saved to models/ with version date: model_YYYYMMDD.pkl

## Code style
- Functions over notebooks for reusable logic
- Docstrings on all functions with Args/Returns
- Magic numbers in a CONFIG dict at top of file

## Analysis workflow
1. Explore in notebook (data/notebooks/XX-title.ipynb)
2. Refactor reusable code to src/
3. Write tests for non-trivial transformations

## Do NOT
- Store PII in git (anonymize first)
- Use random seeds that vary — always set random_state=42
- Merge notebooks with conflicts — restart and re-run from scratch
""",

    "frontend-react": """# React Frontend Project

## Tech stack
- React 18, TypeScript 5, Vite
- Tailwind CSS + shadcn/ui
- TanStack Query for server state
- Zod for schema validation
- Vitest for unit tests, Playwright for E2E

## Code conventions
- Functional components only (no class components)
- Props interfaces named `{ComponentName}Props`
- Custom hooks prefixed with `use` in hooks/
- All API calls through TanStack Query — no raw fetch in components
- No any types — use unknown and narrow

## File structure
```
src/
  components/     Reusable UI components (no business logic)
  features/       Feature-specific components + hooks
  hooks/          Shared custom hooks
  lib/            Utilities, API client, Zod schemas
  pages/          Route-level components
```

## Testing
- Vitest + Testing Library for component tests
- Mock API with MSW (Mock Service Worker)
- Playwright for critical user flows

## Do NOT
- Use inline styles — Tailwind classes only
- Import from parent directories (../../../) — use path aliases (@/)
- Store sensitive data in localStorage — use httpOnly cookies
""",
}

# ─── settings.json examples ──────────────────────────────────────────────────

SETTINGS_EXAMPLES = {
    "basic-mcp": {
        "mcpServers": {
            "lab-server": {
                "command": "python",
                "args": ["/Users/santosh/workspace/github/ai-systems-lab/10-claude-api/04-mcp-concepts/mcp_demo.py"],
                "env": {}
            }
        }
    },
    "with-hooks": {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{
                        "type": "command",
                        "command": "echo \"[hook] bash command incoming\" >> /tmp/claude-audit.log"
                    }]
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "Write",
                    "hooks": [{
                        "type": "command",
                        "command": "cd \"$(dirname $CLAUDE_FILE_PATH)\" && git add \"$(basename $CLAUDE_FILE_PATH)\" 2>/dev/null; true"
                    }]
                }
            ]
        }
    },
    "with-permissions": {
        "permissions": {
            "allow": [
                "Bash(git status)",
                "Bash(git log*)",
                "Bash(pytest*)",
                "Read(*)",
                "Edit(*)",
                "Write(*)"
            ],
            "deny": [
                "Bash(rm -rf*)",
                "Bash(sudo*)",
                "Bash(curl * | bash*)"
            ]
        }
    }
}

# ─── Auto-memory format ───────────────────────────────────────────────────────

MEMORY_EXAMPLES = {
    "user_profile.md": """---
name: User Profile
description: Sr. software engineer learning AI systems engineering
type: user
---

Senior software engineer (10+ years). Deep backend expertise (Java, Python, Go).
Learning AI systems engineering — embeddings, RAG, agents, LLM APIs.
Mac Studio M4 Max 64GB + Proxmox homelab (proxmox1/2 on 192.168.0.111/112).
Prefers working code over theory. Learns best by building.
""",
    "feedback_code_style.md": """---
name: Code style feedback
description: Prefers terse, no-comment code — no docstrings unless asked
type: feedback
---

Keep code comments minimal — only add when WHY is non-obvious.
No multi-line docstrings. No trailing summary at end of response.

**Why:** User explicitly said "I can read the code" — over-commenting is noise.
**How to apply:** Default to 0 comments. Add one if there's a non-obvious constraint or workaround.
""",
}


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== CLAUDE.md STRUCTURE DEMO ===\n")

print("CLAUDE.md hierarchy:")
print("  ~/.claude/CLAUDE.md                     ← applies to ALL projects")
print("  ~/.claude/projects/<hash>/MEMORY.md     ← auto-memory index")
print("  /project-root/CLAUDE.md                 ← this project only")
print("  /project-root/subdir/CLAUDE.md          ← that subdirectory only")
print()

print("Example templates generated:\n")
for name, content in TEMPLATES.items():
    print(f"  {name}/CLAUDE.md ({len(content)} chars, {len(content.splitlines())} lines)")

print()
print("─── Example: python-backend CLAUDE.md ───\n")
print(TEMPLATES["python-backend"])

print("─── settings.json examples ───\n")
for name, config in SETTINGS_EXAMPLES.items():
    print(f"  {name}:")
    print(json.dumps(config, indent=4))
    print()

print("─── Auto-memory file format ───\n")
for filename, content in MEMORY_EXAMPLES.items():
    print(f"  ~/.claude/projects/<hash>/{filename}:")
    print(content)

print("Key rules for effective CLAUDE.md:")
rules = [
    "Be specific: 'use httpx not requests' beats 'use good libraries'",
    "Include context: WHY matters as much as WHAT",
    "Project conventions over personal preferences",
    "Negative rules ('do NOT do X') are as important as positive ones",
    "Keep it under 200 lines — Claude reads all of it on every turn",
    "Update it when you correct Claude's behavior — prevents re-explaining",
]
for r in rules:
    print(f"  ✓ {r}")
