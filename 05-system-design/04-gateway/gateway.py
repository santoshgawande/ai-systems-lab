import time
import json
import uuid
import threading
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

OLLAMA = "http://localhost:11434"

API_KEYS = {
    "sk-lab-key-1": "user1",
    "sk-lab-key-2": "user2",
}

ROUTE_MAP = {
    "gpt-4":        "llama3.3:70b",
    "gpt-4o":       "llama3.3:70b",
    "gpt-4o-mini":  "phi4",
    "claude-3-5":   "llama3.3:70b",
    "claude-haiku": "phi4",
    "default":      "llama3.2",
}

# Token bucket: 10 requests per minute per user
_buckets: dict[str, tuple[float, int]] = {}
_lock = threading.Lock()
RATE_LIMIT = 10
RATE_WINDOW = 60

app = FastAPI(title="Mini LLM Gateway")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    max_tokens: Optional[int] = None


def auth(api_key: str) -> str:
    user = API_KEYS.get(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user


def rate_check(user: str):
    now = time.time()
    with _lock:
        reset_at, count = _buckets.get(user, (now, 0))
        if now - reset_at > RATE_WINDOW:
            reset_at, count = now, 0
        if count >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail=f"Rate limit: {RATE_LIMIT} req/{RATE_WINDOW}s")
        _buckets[user] = (reset_at, count + 1)


def resolve_model(name: str) -> str:
    return ROUTE_MAP.get(name, ROUTE_MAP["default"])


def log(trace_id: str, user: str, requested: str, routed: str, latency_ms: float):
    print(json.dumps({
        "trace_id": trace_id,
        "user": user,
        "requested": requested,
        "routed_to": routed,
        "latency_ms": round(latency_ms, 1),
    }))


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, authorization: str = Header(...)):
    user = auth(authorization.replace("Bearer ", ""))
    rate_check(user)

    trace_id = uuid.uuid4().hex[:8]
    model = resolve_model(req.model)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    start = time.time()

    if req.stream:
        def stream():
            with httpx.stream("POST", f"{OLLAMA}/api/chat",
                              json={"model": model, "messages": messages, "stream": True},
                              timeout=120) as r:
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
                        if chunk.get("done"):
                            yield "data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    r = httpx.post(f"{OLLAMA}/api/chat",
                   json={"model": model, "messages": messages, "stream": False},
                   timeout=120)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    log(trace_id, user, req.model, model, (time.time() - start) * 1000)

    return {
        "id": f"chatcmpl-{trace_id}",
        "object": "chat.completion",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
def health():
    return {"status": "ok", "routes": ROUTE_MAP, "rate_limit": f"{RATE_LIMIT}/min"}


if __name__ == "__main__":
    print("Mini LLM Gateway — http://localhost:8080")
    print("Auth:  Authorization: Bearer sk-lab-key-1")
    print("Route: gpt-4o → llama3.3:70b | gpt-4o-mini → phi4\n")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
