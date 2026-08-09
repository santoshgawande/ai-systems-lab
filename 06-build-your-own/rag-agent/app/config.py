"""Central configuration, all overridable via environment variables (.env)."""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # Which backend the LLM calls go to: "anthropic" or "ollama"
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")

    # Anthropic (cloud)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # Ollama (local, free)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Embeddings + vector store (always local, runs offline)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    collection: str = os.getenv("COLLECTION", "docs")

    # Chunking + retrieval
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("TOP_K", "4"))


config = Config()
