# MaiStorage 100-Question QA Report

- Executed: 2026-07-14T03:23:16.078931+00:00
- Dataset SHA-256: `234f7ba2a19d6cbf1636df4a7e030b0051b2e5aec958a97b62f06e2124351862`

| Provider | Passed | Strict | Hallucination-free | Route | Facts | Citations | Median TTFT | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek | 100/100 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 544.07 ms | 2841.95 ms |
| openai | 100/100 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 957.4 ms | 4720.76 ms |

## Coverage

Each provider received the same 100 frozen questions. Multi-turn cases reused an isolated session; all other cases used separate sessions.

| Category | Questions | DeepSeek passed | OpenAI passed |
|---|---:|---:|---:|
| AIDAPTIV | 15 | 15 | 15 |
| COMPANY | 5 | 5 | 5 |
| COMPARE | 10 | 10 | 10 |
| NOISE | 10 | 10 | 10 |
| PARA | 15 | 15 | 15 |
| PRODUCT | 20 | 20 | 20 |
| SELECT | 15 | 15 | 15 |
| UNSUPPORTED | 10 | 10 | 10 |

## Checks

A case passed only when provider identity, route, response mode, persistence, required facts, forbidden claims, product codes, unit-bearing numeric claims, citations, URLs, and incremental SSE delivery all passed. The hallucination-free score covers forbidden claims, unsupported product codes or numbers, fabricated citations, and unapproved URLs.

The result is evidence for this versioned public corpus and frozen dataset, not proof that every possible user question is safe or correct.

## Failures

No failures.
