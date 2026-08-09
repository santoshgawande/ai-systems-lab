# Weekend Plan — May 3–4, 2025

First hands-on weekend. Goal: get the local stack running and complete the Phase 1 labs.

---

## Saturday — Local Stack + Labs (3–4 hrs)

### Step 1 — Pull models (start this first, runs in background)

```bash
ollama pull llama3.3:70b
ollama pull deepseek-r1
ollama pull phi4
ollama pull nomic-embed-text
ollama pull llava
ollama pull moondream
```

### Step 2 — Run the labs in order

```bash
cd ~/workspace/github/ai-systems-lab/01-llm-apis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] `python 01-hello-ollama/hello.py` — see raw JSON request/response
- [ ] `python 02-streaming/stream.py` — watch tokens arrive one by one
- [ ] `python 03-multi-model-compare/compare.py "What is a transformer?"`
- [ ] `python 04-model-router/router.py "Write a Python function to reverse a string"`
- [ ] `python 05-benchmark/benchmark.py --models phi4 llama3.3:70b --trials 2`

### Step 3 — Watch while models download

- [ ] [Karpathy — Intro to Large Language Models (1 hr)](https://www.youtube.com/watch?v=zjkBMFhNj_g)

---

## Sunday — Vision + First Cloud API Call (2–3 hrs)

### Step 1 — Vision lab

Drop any screenshot or photo into `01-llm-apis/06-vision/samples/`, then:

- [ ] `python 06-vision/image_qa.py 06-vision/samples/yourphoto.jpg "Describe this"`
- [ ] `python 06-vision/compare_models.py 06-vision/samples/yourphoto.jpg "What do you see?"`

### Step 2 — First real Anthropic API call

- [ ] Read [Anthropic Docs — Quickstart](https://docs.anthropic.com/en/docs/quickstart)
- [ ] Read [Anthropic Docs — Messages API](https://docs.anthropic.com/en/docs/messages)
- [ ] Make a real API call to claude-sonnet-4-6 (costs a few cents)

---

## Skip this weekend

Embeddings, pgvector, RAG — these need Proxmox services running. Save for a focused weekday session.

---

## After the weekend — mark these off in learning-plan.md

- [ ] OpenAI API basics (done via local Ollama equivalent)
- [ ] Anthropic API basics
- [ ] SSE streaming
- [ ] Tokenization basics
