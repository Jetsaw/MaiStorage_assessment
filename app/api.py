import json
import time
import uuid
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app import llm_chat
from app.core import active_corpus_version, ask, connect, product_codes, query_products


app = FastAPI(title="MaiStorage Technical Solutions Copilot", version="0.1.0")


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=4000)
    history: list[str] = Field(default_factory=list, max_length=20)


class LLMChatRequest(BaseModel):
    session_id: UUID
    message: str = Field(min_length=1, max_length=4000)
    provider: Literal["deepseek", "openai"] = "deepseek"

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()


class ProductRequest(BaseModel):
    products: list[str] = Field(min_length=1, max_length=10)


class ProductSearchRequest(BaseModel):
    requirements: str = Field(min_length=2, max_length=2000)


class EnvironmentRequest(BaseModel):
    operating_system: str = Field(min_length=2, max_length=100)
    nvidia_driver: int | None = Field(default=None, ge=0)
    available_ports: list[int] = Field(default_factory=list, max_length=50)


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    try:
        return {"status": "ready", "corpus_version": active_corpus_version()}
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/v1/chat")
def chat(request: ChatRequest):
    if llm_chat.prompt_injection_reason(request.message):
        raise HTTPException(status_code=400, detail="Prompt-injection attempt blocked")
    result = ask(request.message, request.history)
    return {key: result.get(key) for key in ("run_id", "route", "reason_code", "answer", "evidence", "trace", "citation_status")}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def chat_events(request: LLMChatRequest):
    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    try:
        if reason := llm_chat.prompt_injection_reason(request.message):
            yield sse("error", {
                "code": "prompt_injection",
                "message": "I can't follow instructions to override safety controls or reveal protected configuration.",
                "reason_code": reason,
            })
            return

        messages = llm_chat.model_messages(request.session_id, request.message)
        settings = llm_chat.provider_settings(request.provider)
        answer_parts = []
        yield sse("meta", {
            "request_id": request_id,
            "session_id": request.session_id,
            "provider": request.provider,
            "model": settings["model"],
        })
        for text in llm_chat.stream_llm(request.provider, messages):
            answer_parts.append(text)
            yield sse("token", {"text": text})
        answer = "".join(answer_parts).strip()
        if not answer:
            raise RuntimeError("empty_provider_response")
        message_id = llm_chat.save_turn(
            request.session_id, request.message, answer, request.provider, settings["model"]
        )
        yield sse("done", {
            "message_id": message_id,
            "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
    except GeneratorExit:
        return
    except Exception as error:
        not_configured = str(error) == "provider_not_configured"
        yield sse("error", {
            "code": "provider_not_configured" if not_configured else "provider_error",
            "message": "Configure a rotated API key for this provider." if not_configured else "The model provider is temporarily unavailable.",
        })


@app.post("/api/v1/chat/stream")
def chat_stream(request: LLMChatRequest):
    return StreamingResponse(
        chat_events(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v1/chat/{session_id}")
def chat_history(session_id: UUID):
    return {"session_id": session_id, "messages": llm_chat.load_messages(session_id)}


@app.post("/api/v1/products/compare")
def compare(request: ProductRequest):
    codes = product_codes(" ".join(request.products))
    return {"products": query_products(codes)}


@app.get("/api/v1/products")
def products():
    return {"products": query_products()}


@app.post("/api/v1/products/search")
def product_search(request: ProductSearchRequest):
    return chat(ChatRequest(message=request.requirements))


@app.post("/api/v1/aidaptiv/validate-environment")
def validate_environment(request: EnvironmentRequest):
    driver = f", NVIDIA driver {request.nvidia_driver}" if request.nvidia_driver is not None else ""
    ports = f", ports {','.join(map(str, request.available_ports))}" if request.available_ports else ""
    return chat(ChatRequest(message=f"Check aiDAPTIV+ environment: {request.operating_system}{driver}{ports}"))


@app.get("/api/v1/sources")
def sources():
    with connect() as connection:
        return list(connection.execute(
            "SELECT s.canonical_url, s.title, s.kind, s.content_hash, s.retrieved_at "
            "FROM source s JOIN active_corpus ac ON ac.version=s.corpus_version ORDER BY s.canonical_url"
        ).fetchall())


@app.get("/api/v1/evaluations/latest")
def latest_evaluation():
    path = Path(__file__).parents[1] / "evaluations" / "results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run the evaluation suite first")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/v1/runs/{run_id}")
def run_detail(run_id: str):
    with connect() as connection:
        row = connection.execute("SELECT * FROM agent_run WHERE id = %s", (run_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return row
