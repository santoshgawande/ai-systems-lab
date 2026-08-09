# mini-rag

A complete document Q&A system in a single file — ingest local documents, ask questions, get grounded answers.

## What you build

- **Ingest mode**: read files → chunk → embed → store in pgvector
- **Query mode**: embed question → similarity search → generate grounded answer
- Full RAG pipeline in ~120 lines of Python

## Run

```bash
# Step 1: Ingest a file or directory
python app.py ingest sample.txt
python app.py ingest ./my-docs/

# Step 2: Query
python app.py query "what is RAG?"
python app.py query "how does chunking affect quality?"
```

## Requirements

- Ollama with `nomic-embed-text` and `llama3.3:70b`
- PostgreSQL + pgvector at `proxmox1:5432`

## What this teaches

You're building the same thing that powers Perplexity, ChatGPT file uploads, and Notion AI.
The entire pipeline is: embed → store → retrieve → generate.
Once you understand these 4 steps, every RAG framework (LangChain, LlamaIndex) is just abstraction on top.
