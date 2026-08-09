# Retrieval-Augmented Generation (RAG) Basics

RAG is a technique that grounds a language model's answers in an external
knowledge base instead of relying only on what the model memorized during
training. A RAG pipeline has three core stages.

## 1. Ingestion
Documents are split into smaller chunks. Each chunk is converted into a vector
(an embedding) by an embedding model and stored in a vector database. Chunk size
matters: chunks that are too large dilute relevance, while chunks that are too
small lose context. A common starting point is 500-800 tokens with a small
overlap so ideas that straddle a boundary are not lost.

## 2. Retrieval
When a user asks a question, the question is embedded with the same model and the
vector store returns the most similar chunks, usually ranked by cosine
similarity. The number of chunks returned is the "top_k" parameter.

## 3. Generation
The retrieved chunks are inserted into the prompt as context, and the language
model writes an answer grounded in that context. Good RAG systems ask the model
to cite which chunk each claim came from, which makes answers verifiable.

## Why RAG helps
RAG reduces hallucination, lets you update knowledge without retraining, and
keeps private data out of the model's weights. Its main failure mode is poor
retrieval: if the right chunk is not retrieved, the model cannot use it.
