"""
Human-in-the-loop: patterns for pausing agent execution to get human input.
Covers: uncertainty detection, clarification requests, approval gates, and
interrupt-and-resume for long-running agents.
"""
import os
import json
import httpx

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")

# Confidence threshold below which agent asks for clarification
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))


def _llm(messages: list[dict], max_tokens: int = 400, json_mode: bool = False) -> str:
    extra = {"response_format": {"type": "json_object"}} if json_mode and OPENAI_KEY else {}
    if OPENAI_KEY:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json={"model": "gpt-4o-mini", "messages": messages, "max_tokens": max_tokens, **extra},
            timeout=30,
        )
        return r.json()["choices"][0]["message"]["content"].strip()

    if ANTHROPIC_KEY:
        sys = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = [m for m in messages if m["role"] != "system"]
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": max_tokens,
                  "system": sys or "You are a helpful assistant.", "messages": user},
            timeout=30,
        )
        return r.json()["content"][0]["text"].strip()

    prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system")
    r = httpx.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=60,
    )
    return r.json()["response"].strip()


# ─── Pattern 1: Uncertainty detection ────────────────────────────────────────

def check_uncertainty(task: str) -> dict:
    """Ask the LLM to self-assess confidence and identify missing info."""
    prompt = f"""Analyse this task and assess your confidence in completing it correctly.

Task: {task}

Respond in JSON:
{{
  "confidence": <float 0.0-1.0>,
  "can_proceed": <bool>,
  "missing_info": [<list of missing details if any>],
  "clarifying_questions": [<list of questions to ask the user, if needed>],
  "reasoning": "<brief explanation>"
}}"""

    text = _llm([
        {"role": "system", "content": "You are a careful AI agent. Always be honest about uncertainty."},
        {"role": "user", "content": prompt},
    ], json_mode=True)

    try:
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {"confidence": 0.5, "can_proceed": True, "missing_info": [], "clarifying_questions": [], "reasoning": text}


# ─── Pattern 2: Approval gate ─────────────────────────────────────────────────

def approval_gate(action: str, action_details: dict, auto_approve: bool = False) -> bool:
    """
    Present a proposed action to the user and require explicit approval.
    In production: this integrates with Slack, email, or a web UI.
    In this demo: auto_approve=True for non-interactive runs.
    """
    print(f"\n{'⚠'*3} APPROVAL REQUIRED {'⚠'*3}")
    print(f"Action:  {action}")
    for k, v in action_details.items():
        print(f"  {k}: {v}")

    if auto_approve:
        print("(auto-approved in demo mode)")
        return True

    try:
        answer = input("Approve? [y/N]: ").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ─── Pattern 3: Interrupt-and-resume ─────────────────────────────────────────

class InterruptibleAgent:
    """
    Agent that checkpoints state so it can pause and resume.
    In production: state is serialised to Redis/DB.
    """
    def __init__(self, goal: str):
        self.goal = goal
        self.steps_completed: list[str] = []
        self.pending_approval: dict | None = None
        self.result: str | None = None
        self.status = "pending"  # pending | awaiting_approval | completed | failed

    def plan(self) -> list[str]:
        """Break goal into steps."""
        response = _llm([
            {"role": "system", "content": "You are a task planner. Break tasks into 3-5 concrete steps."},
            {"role": "user", "content": f"Break this goal into steps: {self.goal}\nList only step descriptions, one per line."},
        ])
        return [line.strip().lstrip("0123456789.-) ") for line in response.strip().split("\n") if line.strip()]

    def execute_step(self, step: str, step_num: int, auto_approve: bool = True) -> bool:
        """Execute one step, pausing for approval on sensitive actions."""
        sensitive_keywords = ["delete", "drop", "modify", "send", "deploy", "publish", "write"]
        is_sensitive = any(kw in step.lower() for kw in sensitive_keywords)

        if is_sensitive:
            approved = approval_gate(
                f"Step {step_num}: {step}",
                {"risk": "medium", "reversible": "unknown"},
                auto_approve=auto_approve,
            )
            if not approved:
                self.status = "failed"
                return False

        # Simulate step execution
        result = _llm([
            {"role": "user", "content": f"Briefly describe the successful outcome of: {step} (1 sentence)"}
        ], max_tokens=60)
        self.steps_completed.append(f"Step {step_num}: {step} → {result}")
        return True

    def run(self, auto_approve: bool = True) -> str:
        self.status = "running"
        steps = self.plan()
        print(f"\n[Agent] Plan ({len(steps)} steps):")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")
        print()

        for i, step in enumerate(steps, 1):
            print(f"[Agent] Executing step {i}/{len(steps)}: {step}")
            ok = self.execute_step(step, i, auto_approve=auto_approve)
            if not ok:
                self.result = f"Stopped at step {i}: {step}"
                return self.result

        self.status = "completed"
        # Final summary
        self.result = _llm([
            {"role": "user", "content": f"Goal: {self.goal}\nCompleted steps:\n" + "\n".join(self.steps_completed) + "\n\nWrite a 2-sentence success summary."}
        ], max_tokens=100)
        return self.result

    def checkpoint(self) -> dict:
        """Serialisable state for pause/resume."""
        return {
            "goal": self.goal,
            "steps_completed": self.steps_completed,
            "status": self.status,
        }


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== HUMAN-IN-THE-LOOP PATTERNS DEMO ===\n")

try:
    httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
    llm_ok = True
except Exception:
    llm_ok = OPENAI_KEY or ANTHROPIC_KEY

if not llm_ok:
    print("No LLM available. Start Ollama or set OPENAI_API_KEY.\n")
    print("""
Three HITL patterns:

1. UNCERTAINTY DETECTION
   confidence = llm("Rate 0-1 how confident you are about: {task}")
   if confidence < 0.7:
       question = llm("What do you need to proceed?")
       user_answer = input(question)
       # continue with answer

2. APPROVAL GATE
   proposed_action = agent.plan_next_action()
   if proposed_action.is_risky:
       approved = present_to_user(proposed_action)
       if not approved:
           agent.stop()

3. INTERRUPT-AND-RESUME
   # Agent serialises state at each step
   checkpoint = agent.get_checkpoint()
   save_to_db(checkpoint)        # can resume later

   # To resume:
   agent = Agent.from_checkpoint(load_from_db(checkpoint_id))
   agent.continue()

Production integrations:
  Slack: post approval request → wait for reaction/reply
  Email: send "approve this action" link with JWT
  Web UI: pop-up approval dialog with diff view
  LangGraph: built-in interrupt/resume primitives
  Temporal.io: durable workflow with human signal
""")
    raise SystemExit(0)

print("─── Pattern 1: Uncertainty detection ───\n")
tasks = [
    "Sort this list: [3, 1, 4, 1, 5, 9]",
    "Refactor our entire authentication system to use OAuth2.",
    "What is the CEO's personal phone number?",
]
for task in tasks:
    result = check_uncertainty(task)
    conf = result.get("confidence", 0)
    can_proceed = result.get("can_proceed", True)
    questions = result.get("clarifying_questions", [])
    print(f"  Task: {task[:60]!r}")
    print(f"  Confidence: {conf:.2f}  |  Can proceed: {can_proceed}")
    if questions:
        print(f"  Clarifying questions:")
        for q in questions[:2]:
            print(f"    - {q}")
    print()

print("─── Pattern 3: Interrupt-and-resume agent ───\n")
agent = InterruptibleAgent("Set up a PostgreSQL database with user authentication tables")
outcome = agent.run(auto_approve=True)
print(f"\n[Agent] Completed. Result: {outcome[:200]}")
print(f"[Agent] Checkpoint: {json.dumps(agent.checkpoint(), indent=2)[:300]}")
