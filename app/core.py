from __future__ import annotations

import json
import os
import re
import urllib.request
import uuid
from functools import lru_cache
from typing import TypedDict

import psycopg
from fastembed import TextEmbedding
from langgraph.graph import END, START, StateGraph
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://maistorage:maistorage@127.0.0.1:5432/maistorage")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR")
PRODUCT_RE = re.compile(r"\b(?:X|D|B|BA|SA)\s*-?\s*\d{2,3}[A-Z]?\b", re.IGNORECASE)
UNSUPPORTED = re.compile(r"\b(price|pricing|stock|inventory|warranty|roadmap|confidential|discount|unpublished compatibility)\b", re.IGNORECASE)
ENVIRONMENT_TERMS = re.compile(r"\b(ubuntu|nvidia driver|ports?|environment|compatible|install|installation)\b", re.IGNORECASE)


class AgentState(TypedDict, total=False):
    run_id: str
    question: str
    working_question: str
    route: str
    reason_code: str
    attempts: int
    tool: str
    records: list[dict]
    evidence: list[dict]
    adequate: bool
    answer: str
    citation_status: str
    trace: list[dict]


def connect():
    connection = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(connection)
    return connection


@lru_cache(maxsize=1)
def embedding_model():
    return TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=EMBEDDING_CACHE_DIR)


def product_codes(question: str) -> list[str]:
    return list(dict.fromkeys(re.sub(r"[\s-]", "", match.group()).upper() for match in PRODUCT_RE.finditer(question)))


def contextualize(question: str, history: list[str] | None = None) -> str:
    if product_codes(question) or not re.search(r"\b(it|that product|that drive|this product)\b", question, re.IGNORECASE):
        return question
    for previous in reversed(history or []):
        codes = product_codes(previous)
        if codes:
            return f'{question} ("it" refers to {codes[-1]})'
    return question


def heuristic_route(question: str) -> tuple[str, str]:
    lower = question.casefold()
    codes = product_codes(question)
    if UNSUPPORTED.search(question):
        return "unsupported", "UNSUPPORTED_COMMERCIAL_OR_PRIVATE"
    if ENVIRONMENT_TERMS.search(question):
        return "aidaptiv_environment", "INSTALL_ENVIRONMENT_CHECK"
    if codes and any(word in lower for word in ("compare", "versus", " vs ", "difference")):
        return "product_comparison", "MULTI_PRODUCT_COMPARISON"
    if any(word in lower for word in ("candidate", "which product", "which drive", "i need", "recommend")):
        return "product_selection", "REQUIREMENT_FILTER"
    if codes:
        return "product_lookup", "EXACT_PRODUCT_CODE"
    if any(word in lower for word in ("who is maistorage", "about maistorage", "company")):
        return "company_information", "COMPANY_INFORMATION"
    return "document_search", "TECHNICAL_DOCUMENT_SEARCH"


def optional_llm_route(question: str, fallback: tuple[str, str]) -> tuple[str, str]:
    base = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    if not base or not model or fallback[0] != "document_search":
        return fallback
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Classify into company_information, product_lookup, product_comparison, product_selection, aidaptiv_environment, document_search, or unsupported. Return JSON with route and reason_code only."},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }
    request = urllib.request.Request(
        f'{base.rstrip("/")}/chat/completions',
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f'Bearer {os.getenv("LLM_API_KEY", "local")}'},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            decision = json.loads(json.loads(response.read())["choices"][0]["message"]["content"])
        route = decision.get("route")
        if route in {"company_information", "product_lookup", "product_comparison", "product_selection", "aidaptiv_environment", "document_search", "unsupported"}:
            return route, str(decision.get("reason_code", "MODEL_ROUTE"))[:80]
    except Exception:
        pass
    return fallback


def query_products(codes: list[str] | None = None) -> list[dict]:
    where = "AND p.name = ANY(%s)" if codes else ""
    params = (codes,) if codes else ()
    sql = f"""
        SELECT p.name, p.family, p.category, p.positioning,
               jsonb_object_agg(ps.field, ps.text_value) AS specs,
               ss.id::text AS evidence_id, s.title AS source_title, s.canonical_url AS source_url
        FROM product p
        JOIN active_corpus ac ON ac.version = p.corpus_version
        JOIN product_spec ps ON ps.product_id = p.id
        JOIN source_section ss ON ss.id = p.source_section_id
        JOIN source s ON s.id = ss.source_id
        WHERE true {where}
        GROUP BY p.id, ss.id, s.title, s.canonical_url
        ORDER BY p.name
    """
    with connect() as connection:
        return list(connection.execute(sql, params).fetchall())


def max_capacity_tb(value: str) -> float:
    values = []
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(TB|GB)", value or "", re.IGNORECASE):
        values.append(float(number) if unit.casefold() == "tb" else float(number) / 1000)
    return max(values, default=0)


def select_products(question: str, records: list[dict]) -> list[dict]:
    lower = question.casefold()
    minimum = 0.0
    match = re.search(r"(?:at least|above|more than|minimum)\s*(\d+(?:\.\d+)?)\s*tb", lower)
    if match:
        minimum = float(match.group(1))
    selected = []
    for record in records:
        searchable = " ".join([record["family"], record["category"], record.get("positioning") or "", json.dumps(record["specs"])]).casefold()
        if minimum and max_capacity_tb(record["specs"].get("capacity", "")) < minimum:
            continue
        required = [term for term in ("gen5", "read-intensive", "caching", "boot", "sata") if term in lower]
        if required and not all(term.replace("gen5", "gen 5") in searchable.replace("gen5", "gen 5") for term in required):
            continue
        selected.append(record)
    return selected


def product_tool(state: AgentState) -> AgentState:
    codes = product_codes(state["working_question"])
    records = query_products(codes or None)
    if state["route"] == "product_selection":
        records = select_products(state["working_question"], records)
    return {
        "tool": "query_product_catalog", "records": records, "evidence": records,
        "attempts": state.get("attempts", 0) + 1,
        "trace": state.get("trace", []) + [{"event": "tool_completed", "tool": "query_product_catalog", "result_count": len(records)}],
    }


def environment_tool(state: AgentState) -> AgentState:
    question = state["working_question"]
    with connect() as connection:
        rules = list(connection.execute(
            "SELECT cr.field, cr.operator, cr.expected_value, cr.product_version, ss.id::text AS evidence_id, "
            "ss.page, s.title AS source_title, s.canonical_url AS source_url FROM compatibility_rule cr "
            "JOIN active_corpus ac ON ac.version=cr.corpus_version JOIN source_section ss ON ss.id=cr.source_section_id "
            "JOIN source s ON s.id=ss.source_id ORDER BY cr.field"
        ).fetchall())
    lower = question.casefold()
    ubuntu = re.search(r"ubuntu\s*(\d+(?:\.\d+)?)", lower)
    driver = re.search(r"(?:nvidia\s*)?driver(?:\s+version)?\s*(\d+)", lower)
    ports = {int(value) for value in re.findall(r"\b\d{4}\b", question)}
    checks = []
    for rule in rules:
        if rule["field"] == "operating_system":
            actual = f'Ubuntu {ubuntu.group(1)}' if ubuntu else None
            passed = bool(ubuntu and ubuntu.group(1) == "22.04")
        elif rule["field"] == "nvidia_driver":
            actual = int(driver.group(1)) if driver else None
            passed = actual is not None and actual >= int(rule["expected_value"])
        else:
            actual = sorted(ports) if ports else None
            passed = actual is not None and set(rule["expected_value"]).issubset(ports)
        checks.append({"field": rule["field"], "actual": actual, "expected": rule["expected_value"], "status": "matches" if passed else "not_provided" if actual is None else "mismatch"})
    evidence = [rules[0]] if rules else []
    return {
        "tool": "validate_aidaptiv_environment", "records": checks, "evidence": evidence,
        "attempts": state.get("attempts", 0) + 1,
        "trace": state.get("trace", []) + [{"event": "tool_completed", "tool": "validate_aidaptiv_environment", "result_count": len(checks)}],
    }


def search_documents(question: str, limit: int = 8) -> list[dict]:
    vector = next(embedding_model().embed([question]))
    with connect() as connection:
        lexical = list(connection.execute(
            "SELECT dc.id::text AS chunk_id, dc.content, ss.id::text AS evidence_id, ss.heading_path, ss.page, "
            "s.title AS source_title, s.canonical_url AS source_url, "
            "ts_rank_cd(dc.textsearch, websearch_to_tsquery('english', %s)) AS lexical_score "
            "FROM document_chunk dc JOIN source_section ss ON ss.id=dc.source_section_id JOIN source s ON s.id=ss.source_id "
            "JOIN active_corpus ac ON ac.version=s.corpus_version "
            "WHERE dc.textsearch @@ websearch_to_tsquery('english', %s) "
            "ORDER BY lexical_score DESC LIMIT 12",
            (question, question),
        ).fetchall())
        semantic = list(connection.execute(
            "SELECT dc.id::text AS chunk_id, dc.content, ss.id::text AS evidence_id, ss.heading_path, ss.page, "
            "s.title AS source_title, s.canonical_url AS source_url, dc.embedding <=> %s AS vector_distance "
            "FROM document_chunk dc JOIN source_section ss ON ss.id=dc.source_section_id JOIN source s ON s.id=ss.source_id "
            "JOIN active_corpus ac ON ac.version=s.corpus_version ORDER BY dc.embedding <=> %s LIMIT 12",
            (vector, vector),
        ).fetchall())
    merged: dict[str, dict] = {}
    for rank, row in enumerate(lexical, start=1):
        merged[row["chunk_id"]] = dict(row) | {"rrf": 1 / (60 + rank), "lexical_rank": rank}
    for rank, row in enumerate(semantic, start=1):
        item = merged.setdefault(row["chunk_id"], dict(row) | {"rrf": 0})
        item.update({key: value for key, value in row.items() if key not in item or item[key] is None})
        item["rrf"] += 1 / (60 + rank)
        item["semantic_rank"] = rank
    return sorted(merged.values(), key=lambda item: item["rrf"], reverse=True)[:limit]


def document_tool(state: AgentState) -> AgentState:
    records = search_documents(state["working_question"])
    return {
        "tool": "search_technical_documents", "records": records, "evidence": records,
        "attempts": state.get("attempts", 0) + 1,
        "trace": state.get("trace", []) + [{"event": "tool_completed", "tool": "search_technical_documents", "result_count": len(records)}],
    }


def route_node(state: AgentState) -> AgentState:
    route, reason = optional_llm_route(state["working_question"], heuristic_route(state["working_question"]))
    trace = state.get("trace", []) + [{"event": "route_selected", "route": route, "reason_code": reason}]
    return {"route": route, "reason_code": reason, "trace": trace}


def grade_node(state: AgentState) -> AgentState:
    records = state.get("records", [])
    adequate = bool(records)
    if state["route"] in {"document_search", "company_information"} and records:
        adequate = records[0].get("vector_distance", 1) < 0.35
    trace = state.get("trace", []) + [{"event": "evidence_graded", "adequate": adequate, "count": len(records)}]
    return {"adequate": adequate, "trace": trace}


def rewrite_node(state: AgentState) -> AgentState:
    rewritten = re.sub(r"\bai\s*daptiv\b", "aiDAPTIV", state["working_question"], flags=re.IGNORECASE)
    rewritten = re.sub(r"\b([XDBS])\s*-\s*(\d)", r"\1\2", rewritten, flags=re.IGNORECASE)
    return {"working_question": rewritten, "route": "document_search", "trace": state.get("trace", []) + [{"event": "query_rewritten", "query": rewritten}]}


def label_evidence(evidence: list[dict]) -> list[dict]:
    result, seen = [], set()
    for item in evidence:
        key = item.get("evidence_id")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(dict(item) | {"citation": f"S{len(result) + 1}"})
    return result


def answer_node(state: AgentState) -> AgentState:
    route = state["route"]
    evidence = label_evidence(state.get("evidence", []))
    labels = {item["evidence_id"]: item["citation"] for item in evidence}
    records = state.get("records", [])
    if route == "aidaptiv_environment":
        lines = ["Environment check against the documented Pro Suite 2.0.5 guide:"]
        for check in records:
            lines.append(f'- {check["field"]}: {check["status"]}; supplied={check["actual"]}; documented={check["expected"]}')
        lines.append(f'This reports the selected guide\'s requirements, not universal compatibility. [{evidence[0]["citation"]}]')
        answer = "\n".join(lines)
    elif route in {"product_lookup", "product_comparison", "product_selection"}:
        if not records:
            answer = "No documented product matches all supplied constraints. Adjust the requirements or ask a MaiStorage solutions engineer."
        else:
            lines = ["Documented product evidence:"]
            for record in records:
                cite = labels[record["evidence_id"]]
                specs = record["specs"]
                lines.append(f'- **{record["name"]}** ({record["family"]}): interface {specs.get("interface", "not stated")}; capacity {specs.get("capacity", "not stated")}; power {specs.get("power", "not stated")}. [{cite}]')
            if route == "product_selection":
                lines.append("These are candidates from public constraints, not an official commercial recommendation.")
            answer = "\n".join(lines)
    else:
        lines = ["I found the following relevant approved evidence:"]
        for item in evidence[:3]:
            excerpt = " ".join(item["content"].split())[:500]
            location = f', page {item["page"]}' if item.get("page") else ""
            lines.append(f'- {excerpt} [{item["citation"]}] ({item["source_title"]}{location})')
        answer = "\n".join(lines)
    return {"answer": answer, "evidence": evidence, "trace": state.get("trace", []) + [{"event": "answer_generated", "mode": "deterministic"}]}


def refusal_node(state: AgentState) -> AgentState:
    return {
        "answer": "The approved MaiStorage sources do not support this request. They do not contain enough evidence to answer it. Ask about MaiStorage products, company information, or aiDAPTIV+ documentation. Pricing, inventory, warranty approval, confidential information, and unpublished compatibility require an official MaiStorage contact.",
        "evidence": [],
        "citation_status": "not_required",
        "trace": state.get("trace", []) + [{"event": "refused", "reason_code": state.get("reason_code", "INSUFFICIENT_EVIDENCE")}],
    }


def verify_node(state: AgentState) -> AgentState:
    valid = {item["citation"] for item in state.get("evidence", [])}
    used = set(re.findall(r"\[(S\d+)\]", state.get("answer", "")))
    status = "passed" if used and used.issubset(valid) else "not_required" if not used and not valid else "failed"
    answer = state["answer"] if status != "failed" else "The generated response failed citation verification and was withheld."
    return {"answer": answer, "citation_status": status, "trace": state.get("trace", []) + [{"event": "citations_verified", "status": status}]}


def route_edge(state: AgentState) -> str:
    return state["route"]


def grade_edge(state: AgentState) -> str:
    if state["adequate"]:
        return "answer"
    return "rewrite" if state.get("attempts", 0) < 2 else "refuse"


builder = StateGraph(AgentState)
builder.add_node("route", route_node)
builder.add_node("product", product_tool)
builder.add_node("environment", environment_tool)
builder.add_node("documents", document_tool)
builder.add_node("grade", grade_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("answer", answer_node)
builder.add_node("refuse", refusal_node)
builder.add_node("verify", verify_node)
builder.add_edge(START, "route")
builder.add_conditional_edges("route", route_edge, {
    "product_lookup": "product", "product_comparison": "product", "product_selection": "product",
    "aidaptiv_environment": "environment", "company_information": "documents", "document_search": "documents",
    "unsupported": "refuse",
})
builder.add_edge("product", "grade")
builder.add_edge("environment", "grade")
builder.add_edge("documents", "grade")
builder.add_conditional_edges("grade", grade_edge, {"answer": "answer", "rewrite": "rewrite", "refuse": "refuse"})
builder.add_edge("rewrite", "documents")
builder.add_edge("answer", "verify")
builder.add_edge("verify", END)
builder.add_edge("refuse", END)
GRAPH = builder.compile()


def active_corpus_version() -> str:
    with connect() as connection:
        row = connection.execute("SELECT version::text FROM active_corpus WHERE singleton").fetchone()
    if not row:
        raise RuntimeError("No active corpus. Run python -m app.index first.")
    return row["version"]


def save_run(state: AgentState) -> None:
    with connect() as connection:
        connection.execute(
            "INSERT INTO agent_run(id, question, route, answer, citation_status, trace, evidence, corpus_version) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                uuid.UUID(state["run_id"]), state["question"], state["route"], state["answer"],
                state.get("citation_status", "unknown"), Jsonb(state.get("trace", [])),
                Jsonb(state.get("evidence", [])), uuid.UUID(active_corpus_version()),
            ),
        )


def ask(question: str, history: list[str] | None = None) -> AgentState:
    initial: AgentState = {
        "run_id": str(uuid.uuid4()), "question": question,
        "working_question": contextualize(question, history), "attempts": 0, "trace": [],
    }
    result = GRAPH.invoke(initial)
    save_run(result)
    return result
