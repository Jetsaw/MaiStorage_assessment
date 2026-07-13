import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core import active_corpus_version, ask


ROOT = Path(__file__).parents[1]


def main():
    cases = [json.loads(line) for line in (ROOT / "evaluations" / "gold.jsonl").read_text(encoding="utf-8").splitlines() if line]
    results = []
    for case in cases:
        started = time.perf_counter()
        observed = ask(case["question"], case.get("history"))
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        answer = observed["answer"].casefold()
        route_ok = observed["route"] == case["route"]
        facts_ok = all(value.casefold() in answer for value in case["contains"])
        citation_ok = observed["citation_status"] == case["citation"]
        results.append({"id": case["id"], "passed": route_ok and facts_ok and citation_ok, "route_ok": route_ok, "facts_ok": facts_ok, "citation_ok": citation_ok, "observed_route": observed["route"], "latency_ms": latency_ms})
    latencies = sorted(result["latency_ms"] for result in results)
    report = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "corpus_version": active_corpus_version(),
        "cases": len(results),
        "passed": sum(result["passed"] for result in results),
        "strict_accuracy": sum(result["passed"] for result in results) / len(results),
        "route_accuracy": sum(result["route_ok"] for result in results) / len(results),
        "fact_accuracy": sum(result["facts_ok"] for result in results) / len(results),
        "citation_accuracy": sum(result["citation_ok"] for result in results) / len(results),
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "results": results,
    }
    (ROOT / "evaluations" / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
