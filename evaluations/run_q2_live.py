"""Run the frozen QA dataset through the public streaming API."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).parents[1]
KNOWN_CODES = {"B100", "BA50", "D100", "D200", "D200V", "D205V", "SA50", "X100", "X200", "X200Z"}
UNIT_FACT = re.compile(r"\b\d+(?:[,.]\d+)*\s*(?:TB|GB|MB/s|GB/s|W|KIOPS?|IOPS?|DWPD)\b", re.IGNORECASE)
URL = re.compile(r"https?://[^\s)]+")
SESSION_GROUPS = {
    "PRODUCT_002": "multi-b100", "MULTI_001": "multi-b100",
    "PRODUCT_015": "multi-d100", "MULTI_002": "multi-d100",
    "SELECT_009": "selection-boot", "SELECT_014": "selection-boot",
    "SELECT_010": "selection-cache", "SELECT_015": "selection-cache",
}


def category(case: dict) -> str:
    prefix = case["id"].split("_", 1)[0]
    if prefix == "MULTI":
        return "PARA"
    if prefix in {"SECURITY", "NOISE"}:
        return "NOISE"
    if prefix in {"ENV", "DOC"}:
        return "AIDAPTIV"
    if prefix in {"REFUSE", "OUT"}:
        return "UNSUPPORTED"
    return prefix


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


def api_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def normalized_evidence(text: str) -> str:
    """Normalize equivalent published unit spellings for claim checks."""
    text = re.sub(r"drive\s+writes?\s+per\s+day\s*\(\s*dwpd\s*\)", "DWPD", text, flags=re.IGNORECASE)
    return normalized(text)


def expected_mode(route: str | None, citation: str | None) -> str | None:
    if citation == "passed":
        return "grounded_llm"
    if route in {"input_clarification", "product_lookup"}:
        return "deterministic_clarification"
    return "deterministic_refusal" if route else None


def run_prompt(
    base_url: str,
    session_id: str,
    provider: str,
    prompt: str,
    contains: list[str] | None = None,
    forbidden: list[str] | None = None,
    expect_policy: bool = False,
    expected_route: str | None = None,
    expected_citation: str | None = None,
    expected_error: str | None = None,
    case_id: str | None = None,
    case_category: str | None = None,
    group: str | None = None,
) -> dict:
    started = time.perf_counter()
    first_token_at = None
    answer_parts, event_order = [], []
    result = {
        "id": case_id, "category": case_category, "group": group, "prompt": prompt,
        "success": False, "persisted": False, "provider_match": False, "error_code": None,
    }
    try:
        for event_name, payload in events(base_url, session_id, provider, prompt):
            event_order.append(event_name)
            if event_name == "meta":
                result.update(model=payload["model"], observed_provider=payload["provider"], mode=payload.get("mode"))
            elif event_name == "evidence":
                result.update(run_id=payload.get("run_id"), observed_route=payload.get("route"), sources=payload.get("sources", []))
            elif event_name == "token":
                first_token_at = first_token_at or time.perf_counter()
                answer_parts.append(payload["text"])
            elif event_name == "done":
                result.update(persisted=True, message_id=payload["message_id"])
            elif event_name == "error":
                result.update(error_code=payload["code"], error_message=payload["message"])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        result.update(error_code="client_error", error_message=type(error).__name__)

    answer = "".join(answer_parts)
    result.update(answer=answer, event_order=event_order, token_event_count=len(answer_parts))
    finished = time.perf_counter()
    result.update(
        incremental_stream=len(answer_parts) > 1,
        empty_response=not answer_parts,
        ttft_ms=round((first_token_at - started) * 1000, 2) if first_token_at else None,
        total_latency_ms=round((finished - started) * 1000, 2),
    )

    if expected_error:
        correct_error = result["error_code"] == expected_error and not result["persisted"]
        result.update(
            provider_match=True, route_ok=correct_error, mode_ok=correct_error, persistence_ok=correct_error,
            facts_ok=correct_error, forbidden_ok=True, codes_ok=True, numeric_ok=True,
            citation_ok=correct_error, urls_ok=True, hallucination_free=True, quality_pass=correct_error,
            success=correct_error,
        )
        return result

    expected_provider = "policy" if expect_policy or (expected_citation is not None and expected_citation != "passed") else provider
    result["provider_match"] = result.get("observed_provider") == expected_provider
    result["route_ok"] = expected_route is None or result.get("observed_route") == expected_route
    result["mode_ok"] = expected_mode(expected_route, expected_citation) in {None, result.get("mode")}
    result["persistence_ok"] = result["persisted"]
    compact_answer = normalized(answer)
    result["facts_ok"] = all(normalized(term) in compact_answer for term in contains or [])
    result["forbidden_ok"] = not any(normalized(term) in compact_answer for term in forbidden or [])

    run = {}
    if result.get("run_id"):
        try:
            run = api_json(f'{base_url}/api/v1/runs/{result["run_id"]}')
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
    allowed_text = " ".join((prompt, str(run.get("answer", "")), json.dumps(run.get("evidence", []), default=str)))
    allowed_codes = {code for code in KNOWN_CODES if re.search(rf"\b{code}\b", allowed_text, re.IGNORECASE)}
    observed_codes = {code for code in KNOWN_CODES if re.search(rf"\b{code}\b", answer, re.IGNORECASE)}
    result["codes_ok"] = observed_codes.issubset(allowed_codes)
    numeric_claims = set(UNIT_FACT.findall(answer))
    normalized_allowed = normalized_evidence(allowed_text)
    result["unsupported_numeric_claims"] = [
        claim for claim in sorted(numeric_claims) if normalized_evidence(claim) not in normalized_allowed
    ]
    result["numeric_ok"] = not result["unsupported_numeric_claims"]

    valid_citations = {source["citation"] for source in result.get("sources", [])}
    used_citations = set(re.findall(r"\[(S\d+)\]", answer))
    valid_urls = {source["url"] for source in result.get("sources", []) if source.get("url")}
    used_urls = set(URL.findall(answer))
    result["citation_ok"] = expected_citation != "passed" or bool(used_citations) and used_citations.issubset(valid_citations)
    result["urls_ok"] = used_urls.issubset(valid_urls)
    result["hallucination_free"] = all(result[key] for key in ("forbidden_ok", "codes_ok", "numeric_ok", "citation_ok", "urls_ok"))
    result["quality_pass"] = result["facts_ok"] and result["hallucination_free"]
    result["success"] = all(result[key] for key in (
        "provider_match", "route_ok", "mode_ok", "persistence_ok", "quality_pass", "incremental_stream",
    ))
    return result


def summarize(cases: list[dict]) -> dict:
    ttft = sorted(case["ttft_ms"] for case in cases if case["ttft_ms"] is not None)
    latency = sorted(case["total_latency_ms"] for case in cases)
    expected_streams = [case for case in cases if not case.get("error_code")]
    groups = {}
    for name in sorted({case["group"] for case in cases if case.get("group")}):
        items = [case for case in cases if case.get("group") == name]
        groups[name] = all(item["success"] for item in items) and len({item.get("observed_route") for item in items}) == 1
    categories = {
        name: {"cases": len(items), "passed": sum(item["success"] for item in items)}
        for name in sorted({case["category"] for case in cases})
        for items in [[case for case in cases if case["category"] == name]]
    }
    return {
        "cases": len(cases),
        "passed": sum(case["success"] for case in cases),
        "strict_pass_rate": round(sum(case["success"] for case in cases) / len(cases), 3),
        "hallucination_free_rate": round(sum(case["hallucination_free"] for case in cases) / len(cases), 3),
        "critical_hallucinations": sum(not case["hallucination_free"] for case in cases),
        "route_accuracy": round(sum(case["route_ok"] for case in cases) / len(cases), 3),
        "fact_accuracy": round(sum(case["facts_ok"] for case in cases) / len(cases), 3),
        "citation_accuracy": round(sum(case["citation_ok"] for case in cases) / len(cases), 3),
        "persistence_rate": round(sum(case["persistence_ok"] for case in cases) / len(cases), 3),
        "incremental_stream_rate": round(sum(case["incremental_stream"] for case in expected_streams) / len(expected_streams), 3),
        "semantic_consistency": round(sum(groups.values()) / len(groups), 3) if groups else 1.0,
        "empty_response_count": sum(case["empty_response"] for case in expected_streams),
        "median_ttft_ms": round(statistics.median(ttft), 2) if ttft else None,
        "p95_ttft_ms": ttft[min(len(ttft) - 1, int(len(ttft) * 0.95))] if ttft else None,
        "median_total_latency_ms": round(statistics.median(latency), 2),
        "p95_total_latency_ms": latency[min(len(latency) - 1, int(len(latency) * 0.95))],
        "categories": categories,
        "semantic_groups": groups,
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# MaiStorage 100-Question QA Report", "",
        f'- Executed: {report["executed_at"]}',
        f'- Dataset SHA-256: `{report["dataset_sha256"]}`', "",
        "| Provider | Passed | Strict | Hallucination-free | Route | Facts | Citations | Median TTFT | p95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider, data in report["providers"].items():
        summary = data["summary"]
        lines.append(
            f'| {provider} | {summary["passed"]}/{summary["cases"]} | {summary["strict_pass_rate"]:.1%} | '
            f'{summary["hallucination_free_rate"]:.1%} | {summary["route_accuracy"]:.1%} | '
            f'{summary["fact_accuracy"]:.1%} | {summary["citation_accuracy"]:.1%} | '
            f'{summary["median_ttft_ms"]} ms | {summary["p95_total_latency_ms"]} ms |'
        )
    lines += [
        "", "## Coverage", "",
        "Each provider receives the same 100 frozen questions. Multi-turn cases reuse an isolated "
        "session; all other cases use separate sessions.", "",
        "| Category | Questions | DeepSeek passed | OpenAI passed |",
        "|---|---:|---:|---:|",
    ]
    category_names = sorted(next(iter(report["providers"].values()))["summary"]["categories"])
    for name in category_names:
        counts = [report["providers"][provider]["summary"]["categories"][name] for provider in report["providers"]]
        provider_passes = [str(item["passed"]) for item in counts]
        while len(provider_passes) < 2:
            provider_passes.append("-")
        lines.append(f'| {name} | {counts[0]["cases"]} | {provider_passes[0]} | {provider_passes[1]} |')
    lines += [
        "", "## Checks", "",
        "A case passes only when provider identity, route, response mode, persistence, required facts, "
        "forbidden claims, product codes, unit-bearing numeric claims, citations, URLs, and incremental "
        "SSE delivery all pass. The hallucination-free score covers forbidden claims, unsupported product "
        "codes or numbers, fabricated citations, and unapproved URLs.", "",
        "The result is evidence for this versioned public corpus and frozen dataset, not proof that every "
        "possible user question is safe or correct.",
    ]
    lines += ["", "## Failures", ""]
    failures = [(provider, case) for provider, data in report["providers"].items() for case in data["cases"] if not case["success"]]
    if not failures:
        lines.append("No failures.")
    for provider, case in failures:
        failed_checks = [key for key in ("provider_match", "route_ok", "mode_ok", "persistence_ok", "facts_ok", "forbidden_ok", "codes_ok", "numeric_ok", "citation_ok", "urls_ok", "incremental_stream") if not case.get(key, True)]
        lines += [f'### {provider} - {case["id"]}', "", f'- Question: {case["prompt"]}', f'- Failed checks: {", ".join(failed_checks)}', f'- Answer: {case["answer"][:800]}', ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", choices=["deepseek", "openai", "both"], default="both")
    parser.add_argument("--dataset", default="evaluations/gold.jsonl")
    parser.add_argument("--output", default="evaluations/qa_100_live_results.json")
    parser.add_argument("--report", default="evaluations/QA_100_REPORT.md")
    arguments = parser.parse_args()
    dataset = ROOT / arguments.dataset
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    providers = ["deepseek", "openai"] if arguments.provider == "both" else [arguments.provider]
    report = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "base_url": arguments.base_url,
        "providers": {},
        "note": "Measured results. Failed cases are retained and are not replaced with estimates.",
    }
    for provider in providers:
        sessions = {}
        provider_results = []
        for case in cases:
            group = SESSION_GROUPS.get(case["id"], case["id"])
            session = sessions.setdefault(group, str(uuid.uuid4()))
            provider_results.append(run_prompt(
                arguments.base_url, session, provider, case["question"],
                contains=case.get("contains"), forbidden=case.get("forbidden"),
                expected_route=case.get("route"), expected_citation=case.get("citation"),
                expected_error=case.get("live_error"), case_id=case["id"],
                case_category=category(case), group=case.get("group"),
            ))
        report["providers"][provider] = {"summary": summarize(provider_results), "cases": provider_results}
    output = ROOT / arguments.output
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (ROOT / arguments.report).write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({provider: data["summary"] for provider, data in report["providers"].items()}, indent=2))
    return 0 if all(data["summary"]["strict_pass_rate"] >= 0.95 and not data["summary"]["critical_hallucinations"] for data in report["providers"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
