# Lab 01 — Hello Ollama

Raw HTTP call to a local model. No SDK, no framework — just the bare API.

## What you learn

- Ollama's `/api/chat` endpoint shape (request + response JSON)
- The `messages` array format (same as OpenAI)
- How to read `response.message.content` and usage fields

## Run

```bash
pip install requests
# Pull a model first if you haven't:
ollama pull llama3.2
python hello.py
```

## Request shape

```json
POST http://localhost:11434/api/chat
{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "Your prompt here"}],
  "stream": false
}
```

## Response shape

```json
{
  "model": "llama3.2",
  "message": {"role": "assistant", "content": "The answer..."},
  "done": true,
  "prompt_eval_count": 15,
  "eval_count": 42,
  "eval_duration": 1230000000
}
```

## Key insight

Every LLM API (OpenAI, Anthropic, Gemini, Ollama) uses the same conceptual shape:
`messages in → message out`. The differences are in auth, model names, and field names.
