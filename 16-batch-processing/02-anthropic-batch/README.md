# Lab 02 — Anthropic Message Batches API

Process up to 10,000 Claude requests per batch at 50% lower cost.

## What you learn

- Anthropic's Message Batches API — same pattern as OpenAI, different SDK shape
- `custom_id` → `MessageBatchRequestParam` mapping
- Streaming results back once the batch completes
- Differences between OpenAI and Anthropic batch approaches

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python batch.py
```

## API shape

```python
import anthropic

client = anthropic.Anthropic()

# Create batch
batch = client.messages.batches.create(
    requests=[
        anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            custom_id="doc-001",
            params={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Classify: positive/negative/neutral — 'Love it!'"}],
            }
        )
    ]
)

# Poll
while batch.processing_status == "in_progress":
    time.sleep(10)
    batch = client.messages.batches.retrieve(batch.id)

# Stream results
for result in client.messages.batches.results(batch.id):
    if result.result.type == "succeeded":
        print(result.custom_id, result.result.message.content[0].text)
    elif result.result.type == "errored":
        print(result.custom_id, "ERROR:", result.result.error.error.message)
```

## Differences from OpenAI Batch

| | Anthropic | OpenAI |
|---|---|---|
| Input format | SDK objects | JSONL file |
| Max requests | 10,000/batch | 50,000/batch |
| Completion window | 24h | 24h |
| Result retrieval | Stream via SDK | Download file |
| Cost saving | 50% | 50% |
