"""
OpenAI structured outputs: guarantee schema-valid JSON with json_schema response_format.
Compares json_object vs json_schema mode, shows extraction and refusal handling.
Requires: OPENAI_API_KEY
"""
import os
import json

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    print("OPENAI_API_KEY not set. Showing structured output mechanics.\n")
    LIVE = False
else:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    LIVE = True

MODEL = "gpt-4o-mini"

# ─── Schema definitions ───────────────────────────────────────────────────────

REVIEW_SCHEMA = {
    "name": "product_review",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral", "mixed"]
            },
            "score": {
                "type": "number",
                "description": "Sentiment score 1-10"
            },
            "summary": {
                "type": "string",
                "description": "One sentence summary"
            },
            "pros": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of positive points"
            },
            "cons": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of negative points"
            },
            "recommend": {
                "type": "boolean"
            }
        },
        "required": ["sentiment", "score", "summary", "pros", "cons", "recommend"],
        "additionalProperties": False
    }
}

CONTACT_SCHEMA = {
    "name": "contact_info",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": ["string", "null"]},
            "phone": {"type": ["string", "null"]},
            "company": {"type": ["string", "null"]},
            "role": {"type": ["string", "null"]}
        },
        "required": ["name", "email", "phone", "company", "role"],
        "additionalProperties": False
    }
}

RECIPE_SCHEMA = {
    "name": "recipe",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "servings": {"type": "integer"},
            "prep_minutes": {"type": "integer"},
            "cook_minutes": {"type": "integer"},
            "ingredients": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "amount": {"type": "string"}
                    },
                    "required": ["item", "amount"],
                    "additionalProperties": False
                }
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"}
            },
            "dietary_tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["vegan", "vegetarian", "gluten-free", "dairy-free", "keto", "none"]
                }
            }
        },
        "required": ["name", "servings", "prep_minutes", "cook_minutes",
                     "ingredients", "steps", "dietary_tags"],
        "additionalProperties": False
    }
}


# ─── Helper ───────────────────────────────────────────────────────────────────

def extract_structured(prompt: str, schema: dict, system: str = "") -> dict | None:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": schema}
    )
    choice = response.choices[0]

    # Refusal: model declined (e.g. harmful content)
    if choice.finish_reason == "refusal":
        print(f"  [REFUSAL] {choice.message.refusal}")
        return None

    return json.loads(choice.message.content)


def extract_json_object(prompt: str, system: str) -> dict | None:
    """json_object mode: valid JSON but no schema enforcement."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== OPENAI STRUCTURED OUTPUTS DEMO ===\n")

if not LIVE:
    print("json_schema mode — model returns schema-valid JSON:\n")
    print("""
schema = {
    "name": "product_review",
    "strict": True,   # ← enforces schema strictly
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed"]},
            "score": {"type": "number"},
            "pros": {"type": "array", "items": {"type": "string"}},
            "cons": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["sentiment", "score", "pros", "cons"],
        "additionalProperties": False   # ← no extra keys allowed
    }
}

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": review_text}],
    response_format={"type": "json_schema", "json_schema": schema}
)

result = json.loads(response.choices[0].message.content)
# Guaranteed: result["sentiment"] is one of ["positive", "negative", "mixed"]
# Guaranteed: result["score"] is a number
# Guaranteed: result["pros"] and result["cons"] are string arrays
""")
    print("Rules for strict schemas:")
    print("  1. All properties must be in 'required'")
    print("  2. 'additionalProperties' must be False")
    print("  3. Nested objects must also follow rules 1+2")
    print("  4. Use null union types for optional fields: {\"type\": [\"string\", \"null\"]}")
    print("\nWhen to use json_schema vs json_object:")
    print("  json_schema: when you need exact fields and types (data pipelines, APIs)")
    print("  json_object: when you just want parseable JSON (simpler prompts, exploration)")
else:
    # Demo 1: Review analysis with json_schema
    print("--- Demo 1: Product review analysis (json_schema) ---\n")
    reviews = [
        "I bought this laptop last month. The speed is incredible and the battery lasts all day. "
        "Build quality feels premium. However, the keyboard is a bit shallow and it gets warm under load. "
        "Overall I'm happy with it.",
        "Absolutely terrible. Arrived broken, customer service was useless, waste of money.",
        "It's okay. Does what it says on the tin, nothing special.",
    ]
    for review in reviews:
        print(f"Review: {review[:80]}...")
        result = extract_structured(
            f"Analyze this product review:\n\n{review}",
            REVIEW_SCHEMA,
            system="You are a review analysis assistant. Analyze objectively."
        )
        if result:
            print(f"  Sentiment: {result['sentiment']} ({result['score']}/10)")
            print(f"  Summary: {result['summary']}")
            print(f"  Pros: {result['pros']}")
            print(f"  Cons: {result['cons']}")
            print(f"  Recommend: {result['recommend']}")
        print()

    # Demo 2: Contact extraction with json_schema
    print("--- Demo 2: Contact info extraction (json_schema with nullables) ---\n")
    texts = [
        "Hi, I'm Sarah Chen, CTO at Acme Corp. Reach me at sarah@acme.com or 555-0142.",
        "John from marketing here — john.doe@company.org.",
        "Dr. Martinez, no email please, call 555-9876.",
    ]
    for text in texts:
        print(f"Text: {text}")
        result = extract_structured(
            f"Extract contact information from this text:\n\n{text}",
            CONTACT_SCHEMA
        )
        if result:
            print(f"  {result}")
        print()

    # Demo 3: json_object mode (less strict)
    print("--- Demo 3: json_object mode (no schema enforcement) ---\n")
    text = "The sprint has 12 open tickets, 5 in progress, and 8 completed. Velocity is 23 points."
    result = extract_json_object(
        f"Extract sprint metrics as JSON:\n\n{text}",
        system="Respond with valid JSON only. Extract all numeric metrics."
    )
    print(f"Input: {text}")
    print(f"Extracted: {json.dumps(result, indent=2)}\n")

    # Demo 4: Recipe generation with complex nested schema
    print("--- Demo 4: Structured recipe generation (nested schema) ---\n")
    result = extract_structured(
        "Give me a simple pasta recipe.",
        RECIPE_SCHEMA,
        system="You are a chef. Generate a recipe following the exact schema provided."
    )
    if result:
        print(f"Recipe: {result['name']}")
        print(f"  Servings: {result['servings']} | Prep: {result['prep_minutes']}m | Cook: {result['cook_minutes']}m")
        print(f"  Ingredients: {len(result['ingredients'])} items")
        print(f"  Steps: {len(result['steps'])} steps")
        print(f"  Tags: {result['dietary_tags']}")
