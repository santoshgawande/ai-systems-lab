"""Provider-agnostic LLM layer.

Both backends expose the same `complete(prompt, system)` method so the rest of
the app never cares which one is active. Toggle with LLM_PROVIDER=anthropic|ollama.
"""
from __future__ import annotations

import json
import urllib.request

from .config import config


class BaseLLM:
    name: str = "base"

    def complete(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError

    def complete_json(self, prompt: str, system: str = "") -> dict:
        """Call the model and parse the first JSON object out of the reply.

        Local models love to wrap JSON in prose or ```json fences, so we are
        forgiving here rather than assuming clean output.
        """
        raw = self.complete(prompt, system)
        return _extract_json(raw)


class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self):
        # Imported lazily so the app still runs in ollama-only mode without the SDK.
        from anthropic import Anthropic

        if not config.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        self.client = Anthropic(api_key=config.anthropic_api_key)
        self.model = config.anthropic_model

    def complete(self, prompt: str, system: str = "") -> str:
        # cache_control on the system block: repeated agent turns reuse the same
        # system prompt, so caching it trims latency and cost on every call.
        system_blocks = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if system
            else []
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_blocks,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self):
        self.base_url = config.ollama_base_url.rstrip("/")
        self.model = config.ollama_model

    def complete(self, prompt: str, system: str = "") -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["response"]


def get_llm() -> BaseLLM:
    if config.llm_provider == "anthropic":
        return AnthropicLLM()
    return OllamaLLM()


def _extract_json(text: str) -> dict:
    """Best-effort: pull the first {...} block and parse it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return json.loads(text[start : end + 1])
