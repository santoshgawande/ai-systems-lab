# 11 — OpenAI API (ChatGPT)

ChatGPT-specific features: the Chat Completions API, function calling, structured outputs, and the Assistants API.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Get an API key: https://platform.openai.com/api-keys

```bash
export OPENAI_API_KEY=sk-...
```

> Labs fall back to Ollama if `OPENAI_API_KEY` is not set. Assistants API (lab 03) requires a real key — it has no local equivalent.

## Labs

| Dir | What you learn | Run |
|---|---|---|
| `01-function-calling/` | Define tools, parse parallel tool calls, feed results back | `python functions.py` |
| `02-structured-output/` | `response_format: json_schema` — guaranteed valid JSON matching your schema | `python structured.py` |
| `03-assistants/` | Threads, runs, file search, code interpreter — stateful persistent sessions | `python assistants.py` |

## ChatGPT API architecture

```
Chat Completions API (stateless)
  → You manage the messages array
  → Each request is independent
  → Used by: ChatGPT web, most production apps

Assistants API (stateful)
  → OpenAI manages thread state for you
  → Persistent file storage (code interpreter, file search)
  → Used by: GPT builders, long-running tasks
```

## Structured outputs vs JSON mode

| Feature | JSON mode | Structured outputs |
|---|---|---|
| Guarantee | Valid JSON | Valid JSON matching your schema exactly |
| Schema | None | Full JSON Schema |
| Availability | All models | gpt-4o and newer |
| Fail on violation | No | Yes — model will never violate the schema |
