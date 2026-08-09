# Lab 03 — OpenAI Assistants API

The Assistants API manages threads, runs, and built-in tools server-side — no conversation management in your code.

## What you learn

- The four objects: **Assistant**, **Thread**, **Message**, **Run**
- How to create an assistant with instructions + tools
- How to poll a Run until it completes (or fails)
- Tool execution within a Run (requires action)
- When to use Assistants API vs Chat Completions API

## Run

```bash
export OPENAI_API_KEY=sk-...
python assistants.py
```

## Object model

```
Assistant (persistent config: model, instructions, tools)
    └── Thread (conversation history, managed by OpenAI)
            └── Messages (user + assistant turns)
            └── Run (a single execution: Thread + Assistant → response)
                    └── RunStep (tool calls, message creation)
```

## Lifecycle

```
1. Create assistant (once, reuse by ID)
2. Create thread (per conversation)
3. Add user message to thread
4. Create run (attach assistant to thread)
5. Poll run status: queued → in_progress → completed | requires_action | failed
6. If requires_action: submit tool outputs, run resumes
7. Read assistant messages from thread
```

## API shape

```python
# 1. Create assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="Help students with math step by step.",
    model="gpt-4o-mini",
    tools=[{"type": "function", "function": {...}}]
)

# 2. Create thread
thread = client.beta.threads.create()

# 3. Add message
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="What is the derivative of x^3?"
)

# 4. Run
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# 5. Poll
import time
while run.status in ("queued", "in_progress"):
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

# 6. Read response
messages = client.beta.threads.messages.list(thread_id=thread.id)
print(messages.data[0].content[0].text.value)  # latest assistant message
```

## Chat Completions vs Assistants

| | Chat Completions | Assistants |
|---|---|---|
| History management | You manage `messages[]` | OpenAI manages Threads |
| Streaming | Native | Via streaming runs |
| Built-in tools | No | Code interpreter, file search |
| Persistence | Stateless | Thread + message storage |
| Use for | Stateless APIs, chatbots | Long convos, file processing |
