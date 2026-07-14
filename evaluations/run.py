import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core import active_corpus_version, ask
from app.llm_chat import prompt_injection_reason


ROOT = Path(__file__).parents[1]
KNOWN_CODES = {"B100", "BA50", "D100", "D200", "D200V", "D205V", "SA50", "X100", "X200", "X200Z"}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluations/gold.jsonl")
    parser.add_argument("--output", default="evaluations/results.json")
    arguments = parser.parse_args()
    dataset = ROOT / arguments.dataset
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    results = []
    for case in cases:
        started = time.perf_counter()
        if case.get("live_error"):
            blocked = prompt_injection_reason(case["question"]) is not None
            results.append({
                "id": case["id"], "category": category(case), "group": case.get("group"), "passed": blocked,
                "route_ok": blocked, "facts_ok": blocked, "forbidden_ok": True, "codes_ok": True,
                "citation_ok": blocked, "observed_route": "api_security" if blocked else "not_blocked",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2), "answer": "blocked" if blocked else "not blocked",
            })
            continue
        observed = ask(case["question"], case.get("history"))
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        answer = observed["answer"].casefold()
        route_ok = observed["route"] == case["route"]
        facts_ok = all(value.casefold() in answer for value in case["contains"])
        forbidden_ok = not any(value.casefold() in answer for value in case.get("forbidden", []))
        expected_text = " ".join(case["contains"])
        allowed_codes = {code for code in KNOWN_CODES if re.search(rf"\b{code}\b", expected_text, re.IGNORECASE)}
        observed_codes = {code for code in KNOWN_CODES if re.search(rf"\b{code}\b", observed["answer"], re.IGNORECASE)}
        codes_ok = category(case) == "SELECT" or observed_codes.issubset(allowed_codes)
        citation_ok = observed["citation_status"] == case["citation"]
        passed = route_ok and facts_ok and forbidden_ok and codes_ok and citation_ok
        results.append({
            "id": case["id"], "category": category(case), "group": case.get("group"), "passed": passed,
            "route_ok": route_ok, "facts_ok": facts_ok, "forbidden_ok": forbidden_ok,
            "codes_ok": codes_ok, "citation_ok": citation_ok, "observed_route": observed["route"],
            "latency_ms": latency_ms, "answer": observed["answer"],
        })
    latencies = sorted(result["latency_ms"] for result in results)
    categories = {
        name: {"cases": len(items), "passed": sum(item["passed"] for item in items)}
        for name in sorted({result["category"] for result in results})
        for items in [[result for result in results if result["category"] == name]]
    }
    groups = {}
    for name in sorted({result["group"] for result in results if result["group"]}):
        items = [result for result in results if result["group"] == name]
        groups[name] = {"cases": len(items), "consistent": all(item["passed"] for item in items) and len({item["observed_route"] for item in items}) == 1}
    report = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "corpus_version": active_corpus_version(),
        "cases": len(results),
        "passed": sum(result["passed"] for result in results),
        "strict_accuracy": sum(result["passed"] for result in results) / len(results),
        "route_accuracy": sum(result["route_ok"] for result in results) / len(results),
        "fact_accuracy": sum(result["facts_ok"] for result in results) / len(results),
        "hallucination_free_rate": sum(result["forbidden_ok"] and result["codes_ok"] for result in results) / len(results),
        "citation_accuracy": sum(result["citation_ok"] for result in results) / len(results),
        "semantic_consistency": sum(group["consistent"] for group in groups.values()) / len(groups) if groups else 1.0,
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "categories": categories,
        "semantic_groups": groups,
        "results": results,
    }
    (ROOT / arguments.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
