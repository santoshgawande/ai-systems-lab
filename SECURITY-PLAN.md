# Security Plan — ai-systems-lab

> A collection of runnable AI examples (LLM APIs, RAG, agents, evals,
> guardrails, AI-security). It already has `09-guardrails` and `15-ai-security`
> modules, so this plan is about making the *whole lab* safe to run and turning
> the security modules into the reference the other examples follow.

## 1. What we're protecting

~20 numbered example projects, several of which call paid APIs, run agents,
build RAG pipelines, and do fine-tuning. The cross-cutting risks are **leaked
API keys across many small examples** and **inconsistent injection/guardrail
handling** between modules.

| Asset | Where | Risk |
|---|---|---|
| API keys (Claude/OpenAI/Gemini) | env across many examples | One committed `.env` leaks everything |
| Agent tool actions | `04-agents`, `17-claude-code-sdk` | Injection → unintended tool use |
| RAG inputs | `03-rag`, `14-vector-databases`, `18-hybrid-search` | Indirect injection from documents |
| Guardrail/security modules | `09-guardrails`, `15-ai-security` | Should be the canonical, correct reference |

## 2. Trust boundaries

```
example ──► LLM API (paid) / local Ollama
   ▲ key handling (shared across 20 dirs)
agent/RAG examples ──► tools / retrieved docs ── untrusted content re-enters prompt
```

## 3. Findings & gaps

- **Central key hygiene.** With ~20 examples, the risk is one stray `.env` or a
  key pasted into a notebook/cell getting committed. Enforce a single ignored
  `.env` pattern repo-wide; scan history for keys.
- **Guardrails as the reference.** `09-guardrails` and `15-ai-security` should
  contain the canonical input/output filtering, prompt-injection defense, and
  PII redaction that the agent/RAG examples import — not each example rolling its
  own (or none).
- **Agent examples.** `04-agents` / `17-claude-code-sdk` — any tool execution
  should be gated/sandboxed; treat model output as untrusted before acting.
- **RAG examples.** Demonstrate indirect prompt injection and its mitigation
  (delimit retrieved content) — this is a teaching lab, so show the attack + fix.
- **Batch/fine-tuning.** `13-fine-tuning`, `16-batch-processing` — ensure
  training/batch inputs aren't unsanitized untrusted data; note data-poisoning.
- **No exposed services expected**, but any example that starts a server should
  bind localhost.

## 4. Roadmap

### P0 — Keys across the lab
- [ ] One repo-wide `.gitignore` for `.env`/keys; scan git history with
      `sec-auditor secrets` in Docker; rotate anything found.
- [ ] Add a short `SECURITY.md`/README note: never hardcode keys; use env.

### P1 — Make the security modules canonical
- [ ] Have `09-guardrails` export reusable input/output guards + PII redaction;
      refactor `04-agents`/`03-rag` examples to use them.
- [ ] Add an indirect-prompt-injection demo + mitigation in the RAG examples.

### P2 — Per-example safety
- [ ] Agent examples: gate/sandbox any tool exec; never `eval` model output.
- [ ] Any example server binds `127.0.0.1`; caps on generation.
- [ ] `15-ai-security`: expand to a checklist the other repos in this workspace
      can reuse (this plan cross-references it).

## 5. Verification

```bash
make -C ../sec-auditor scan TARGET=$(pwd)     # secrets sweep across all examples
grep -rn "sk-\|api_key *= *[\"']" --include=*.py --include=*.ipynb .  # no hardcoded keys
```
