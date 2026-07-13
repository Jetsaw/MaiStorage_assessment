"""Run the Question 2 live-provider smoke test through the public FastAPI contract."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROMPTS = [
    "In one sentence, explain what an API is.",
    "Give three concise benefits of token streaming.",
    "Remember the word ORBIT for my next question.",
    "What word did I ask you to remember?",
    "Summarize this conversation in two short bullet points.",
]


def events(base_url: str, session_id: str, provider: str, message: str):
    request = urllib.request.Request(
        f"{base_url}/api/v1/chat/stream",
        data=json.dumps({"session_id": session_id, "message": message, "provider": provider}).encode(),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
    )
    event_name, data_lines = None, []
    with urllib.request.urlopen(request, timeout=90) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif not line and event_name:
                yield event_name, json.loads("\n".join(data_lines))
                event_name, data_lines = None, []


def run_prompt(base_url: str, session_id: str, provider: str, prompt: str) -> dict:
    started = time.perf_counter()
    first_token_at = None
    token_count = 0
    result = {
        "prompt": prompt,
        "success": False,
        "persisted": False,
        "provider_match": False,
        "error_code": None,
    }
    try:
        for event_name, payload in events(base_url, session_id, provider, prompt):
            if event_name == "meta":
                result["model"] = payload["model"]
                result["observed_provider"] = payload["provider"]
                result["provider_match"] = payload["provider"] == provider
            elif event_name == "token":
                first_token_at = first_token_at or time.perf_counter()
                token_count += 1
            elif event_name == "done":
                result.update(persisted=True, message_id=payload["message_id"])
            elif event_name == "error":
                result.update(error_code=payload["code"], error_message=payload["message"])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        result.update(error_code="client_error", error_message=type(error).__name__)
    finished = time.perf_counter()
    result["incremental_stream"] = token_count > 1
    result["success"] = result["persisted"] and result["provider_match"] and token_count > 0
    result.update(
        token_event_count=token_count,
        empty_response=token_count == 0,
        ttft_ms=round((first_token_at - started) * 1000, 2) if first_token_at else None,
        total_latency_ms=round((finished - started) * 1000, 2),
    )
    return result


def summarize(cases: list[dict]) -> dict:
    ttft = [case["ttft_ms"] for case in cases if case["ttft_ms"] is not None]
    latency = [case["total_latency_ms"] for case in cases]
    return {
        "cases": len(cases),
        "success_rate": round(sum(case["success"] for case in cases) / len(cases), 3),
        "persistence_rate": round(sum(case["persisted"] for case in cases) / len(cases), 3),
        "provider_match_rate": round(sum(case["provider_match"] for case in cases) / len(cases), 3),
        "incremental_stream_rate": round(sum(case["incremental_stream"] for case in cases) / len(cases), 3),
        "empty_response_count": sum(case["empty_response"] for case in cases),
        "median_ttft_ms": round(statistics.median(ttft), 2) if ttft else None,
        "median_total_latency_ms": round(statistics.median(latency), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", choices=["deepseek", "openai", "both"], default="both")
    parser.add_argument("--output", default="evaluations/q2_live_results.json")
    arguments = parser.parse_args()
    providers = ["deepseek", "openai"] if arguments.provider == "both" else [arguments.provider]
    report = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "base_url": arguments.base_url,
        "providers": {},
        "note": "Measured live results; errors remain results and are not replaced with estimates.",
    }
    for provider in providers:
        session = str(uuid.uuid4())
        cases = [run_prompt(arguments.base_url, session, provider, prompt) for prompt in PROMPTS]
        report["providers"][provider] = {"session_id": session, "summary": summarize(cases), "cases": cases}
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
