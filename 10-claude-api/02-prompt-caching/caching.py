"""
Anthropic prompt caching: mark large system prompts for caching and measure savings.
This is the key feature behind Claude Code's economics.
Requires: ANTHROPIC_API_KEY
"""
import os
import time

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not API_KEY:
    print("ANTHROPIC_API_KEY not set. Showing cache mechanics explanation.\n")
    LIVE = False
else:
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY)
    LIVE = True

MODEL = "claude-sonnet-4-6"

# Simulate a large knowledge base (like Claude Code's system prompt)
KNOWLEDGE_BASE = """
COMPANY ENGINEERING HANDBOOK — Version 12.3

## Architecture Principles

1. Services must be stateless. Session state lives in Redis, not in-process.
2. All inter-service communication uses gRPC. HTTP is for external APIs only.
3. Database per service. No cross-service database joins.
4. Circuit breakers required for all external calls (Resilience4j).
5. Every service exposes /health and /metrics endpoints.

## Code Standards

Java 21+ with virtual threads. Spring Boot 3.x. Gradle builds.
All new code requires 80% unit test coverage measured by JaCoCo.
Integration tests use Testcontainers, not H2 in-memory.
Feature flags via LaunchDarkly. No hard-coded feature toggles.

## Incident Response

P0 (service down): page on-call immediately, 15-min SLA to acknowledge.
P1 (degraded service): page on-call, 1-hour SLA.
P2 (minor issues): create ticket, fix in next sprint.
RCA required for all P0 and P1 incidents within 5 business days.

## Database Guidelines

PostgreSQL 16+ for transactional data. Redis for cache and sessions.
Migrations managed by Flyway. Never modify existing migration files.
All queries must use parameterized statements. No string concatenation in SQL.
Index all foreign keys and columns used in WHERE clauses.

## Security Requirements

OWASP Top 10 compliance required. Pen test annually.
All secrets in Vault. Never in code, environment variables, or config files.
JWT expiry: 15 minutes (access), 7 days (refresh). Rotate signing keys quarterly.
Rate limiting on all public endpoints: 100 req/min per IP.

## Deployment

All deployments via ArgoCD. No manual kubectl apply in production.
Blue-green deployments for all stateful services.
Canary deployments for high-traffic services (10% → 50% → 100%).
Rollback SLA: 5 minutes for any deployment.
""" * 3  # Repeat to make it larger (simulating real-world system prompt size)

QUESTIONS = [
    "What database should I use for session storage?",
    "What is the incident response SLA for a P0?",
    "How should I handle secrets in the application?",
]


def cost_usd(input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0) -> float:
    """Sonnet pricing per 1M tokens (approximate)."""
    return (
        (input_tokens / 1_000_000) * 3.00 +
        (output_tokens / 1_000_000) * 15.00 +
        (cache_write / 1_000_000) * 3.75 +   # 1.25x input price
        (cache_read / 1_000_000) * 0.30       # 0.10x input price
    )


if __name__ == "__main__":
    print("=== PROMPT CACHING DEMO ===\n")
    print(f"Knowledge base size: {len(KNOWLEDGE_BASE)} chars (~{len(KNOWLEDGE_BASE)//4} tokens)\n")

    if not LIVE:
        print("How caching works:")
        print("""
# Mark the large prefix with cache_control
system = [
    {
        "type": "text",
        "text": KNOWLEDGE_BASE,
        "cache_control": {"type": "ephemeral"}  # ← cache this
    }
]

# First call: writes to cache (priced at 1.25x input)
response = client.messages.create(model=model, system=system, ...)
print(response.usage.cache_creation_input_tokens)  # e.g. 2400
print(response.usage.cache_read_input_tokens)      # 0 (first call)

# Second call within 5 minutes: reads from cache (priced at 0.10x)
response = client.messages.create(model=model, system=system, ...)
print(response.usage.cache_creation_input_tokens)  # 0 (cache hit)
print(response.usage.cache_read_input_tokens)      # 2400 (90% cheaper!)

# Cost comparison for 100 calls with 2400-token system prompt:
# Without cache: 100 × 2400 × $3/1M    = $0.72
# With cache:    1 write + 99 reads
#   write:  2400 × $3.75/1M             = $0.009
#   reads:  99 × 2400 × $0.30/1M        = $0.071
#   total:                               = $0.080  ← 89% cheaper
""")
    else:
        # First call — expect cache_creation_input_tokens > 0
        print("Call 1 (cold — writing to cache):")
        start = time.time()
        r1 = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=[
                {"type": "text", "text": KNOWLEDGE_BASE, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Answer only from the handbook above. Be concise."},
            ],
            messages=[{"role": "user", "content": QUESTIONS[0]}]
        )
        t1 = time.time() - start
        u1 = r1.usage
        c1 = cost_usd(u1.input_tokens, u1.output_tokens, u1.cache_creation_input_tokens, u1.cache_read_input_tokens)
        print(f"  Input: {u1.input_tokens}  Output: {u1.output_tokens}  Cache write: {u1.cache_creation_input_tokens}  Cache read: {u1.cache_read_input_tokens}")
        print(f"  Latency: {t1:.2f}s  Cost: ${c1:.6f}")
        print(f"  Answer: {r1.content[0].text[:100]}\n")

        # Second call — expect cache_read_input_tokens > 0
        print("Call 2 (warm — reading from cache):")
        start = time.time()
        r2 = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=[
                {"type": "text", "text": KNOWLEDGE_BASE, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Answer only from the handbook above. Be concise."},
            ],
            messages=[{"role": "user", "content": QUESTIONS[1]}]
        )
        t2 = time.time() - start
        u2 = r2.usage
        c2 = cost_usd(u2.input_tokens, u2.output_tokens, u2.cache_creation_input_tokens, u2.cache_read_input_tokens)
        print(f"  Input: {u2.input_tokens}  Output: {u2.output_tokens}  Cache write: {u2.cache_creation_input_tokens}  Cache read: {u2.cache_read_input_tokens}")
        print(f"  Latency: {t2:.2f}s  Cost: ${c2:.6f}")
        print(f"  Answer: {r2.content[0].text[:100]}\n")

        if u2.cache_read_input_tokens > 0:
            savings = (1 - c2 / c1) * 100
            print(f"✓ Cache hit confirmed! Cost reduced by ~{savings:.0f}%")
        else:
            print("Cache miss (may need to call faster or check TTL)")
