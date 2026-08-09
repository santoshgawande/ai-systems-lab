"""
LLM-as-judge: use a strong model to score another model's outputs.
This is how PromptFoo, Braintrust, and most eval frameworks grade open-ended responses.
"""
import json
import httpx
from dataclasses import dataclass

OLLAMA = "http://localhost:11434"
JUDGE_MODEL = "llama3.3:70b"   # strongest available — be the harsh grader
STUDENT_MODEL = "phi4"          # model being evaluated

JUDGE_SYSTEM = """You are an expert AI evaluator. Grade the response using the rubric.

Respond ONLY with JSON — no other text:
{"score": <1-5>, "reasoning": "<one sentence>", "verdict": "<PASS|FAIL>"}

Scoring guide:
5 = Excellent: accurate, complete, well-formatted
4 = Good: mostly correct, minor gaps
3 = Acceptable: correct but incomplete or verbose (PASS threshold)
2 = Poor: partially wrong or missing key info
1 = Fail: wrong, harmful, or refuses valid request"""


def ask(model: str, system: str, user: str) -> str:
    r = httpx.post(f"{OLLAMA}/api/chat", json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }, timeout=60)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def judge(question: str, response: str, rubric: str) -> dict:
    prompt = f"Question: {question}\n\nResponse:\n{response}\n\nRubric: {rubric}"
    result = ask(JUDGE_MODEL, JUDGE_SYSTEM, prompt)
    try:
        s, e = result.find("{"), result.rfind("}") + 1
        return json.loads(result[s:e]) if s >= 0 else {"score": 0, "reasoning": "parse error", "verdict": "FAIL"}
    except Exception:
        return {"score": 0, "reasoning": "parse error", "verdict": "FAIL"}


@dataclass
class TestCase:
    question: str
    student_system: str
    rubric: str


CASES = [
    TestCase(
        question="Explain what a database index is.",
        student_system="You are a helpful technical assistant. Be concise.",
        rubric="Must explain: what an index is, what problem it solves, and one trade-off. Under 150 words.",
    ),
    TestCase(
        question="What is the difference between TCP and UDP?",
        student_system="Explain networking concepts clearly.",
        rubric="Must mention: connection-oriented vs connectionless, reliability guarantee, and a use case for each. Under 200 words.",
    ),
    TestCase(
        question="Write a Python function to reverse a string.",
        student_system="You are a Python expert. Provide working code.",
        rubric="Must: provide working Python code, handle edge cases (empty string), and be idiomatic Python.",
    ),
    TestCase(
        question="What is 15% of 340?",
        student_system="Answer math questions.",
        rubric="Must give the correct answer: 51. No explanation needed unless helpful.",
    ),
    TestCase(
        question="Explain microservices to a junior developer in 3 bullet points.",
        student_system="You explain technical concepts to junior developers.",
        rubric="Must: use exactly 3 bullet points, avoid jargon, and give a concrete benefit in each point.",
    ),
]

print(f"Evaluating {STUDENT_MODEL} responses with {JUDGE_MODEL} as judge\n")
print(f"{'Case':<45} {'Score':>6}  {'Verdict':<7}  Reasoning")
print("-" * 90)

total_score = 0
for case in CASES:
    response = ask(STUDENT_MODEL, case.student_system, case.question)
    grade = judge(case.question, response, case.rubric)

    score = grade.get("score", 0)
    verdict = grade.get("verdict", "?")
    reasoning = grade.get("reasoning", "")
    total_score += score

    icon = "✓" if verdict == "PASS" else "✗"
    q_short = case.question[:43]
    print(f"  {icon} {q_short:<43}  {score}/5    {verdict:<7}  {reasoning[:50]}")

print("-" * 90)
avg = total_score / len(CASES)
print(f"\nAverage score: {avg:.1f}/5.0  ({len(CASES)} cases)")
print(f"Student model: {STUDENT_MODEL}  |  Judge: {JUDGE_MODEL}")
