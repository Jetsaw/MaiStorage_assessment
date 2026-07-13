from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LOGO_DATA = base64.b64encode((Path(__file__).with_name("assets") / "maistorage-logo.png").read_bytes()).decode()


def session_id() -> str:
    candidate = st.query_params.get("session", "")
    try:
        value = str(uuid.UUID(candidate))
    except (ValueError, TypeError, AttributeError):
        value = str(uuid.uuid4())
        st.query_params["session"] = value
    return value


def api_json(path: str, payload: dict | None = None):
    request = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def load_history(current_session: str) -> list[dict]:
    return api_json(f"/api/v1/chat/{current_session}")["messages"]


def stream_events(current_session: str, message: str, provider: str, telemetry: dict):
    request = urllib.request.Request(
        f"{API_BASE_URL}/api/v1/chat/stream",
        data=json.dumps({"session_id": current_session, "message": message, "provider": provider}).encode(),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    event, data_lines = None, []
    with urllib.request.urlopen(request, timeout=60) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif not line and event:
                data = json.loads("\n".join(data_lines))
                telemetry.setdefault("events", []).append(event)
                if event == "meta":
                    telemetry.update(data)
                elif event == "token":
                    telemetry.setdefault("ttft_ms", round((time.perf_counter() - started) * 1000, 2))
                    yield data["text"]
                elif event == "done":
                    telemetry.update(data)
                elif event == "error":
                    telemetry["error_code"] = data.get("code")
                    raise RuntimeError(data["message"])
                event, data_lines = None, []


st.set_page_config(page_title="MaiStorage Solutions Copilot", layout="wide")
st.markdown("""
<style>
    :root {
        --ms-bg: #fbfaf8;
        --ms-surface: #ffffff;
        --ms-text: #202633;
        --ms-muted: #68707d;
        --ms-line: #e7e3de;
        --ms-accent: #c92f34;
        --ms-accent-soft: #fff1ef;
        --ms-success: #117a34;
    }
    html, body, [class*="st-"] { font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--ms-bg); color: var(--ms-text); }
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
    [data-testid="stHeader"] { background: transparent; height: 0; }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="stMainBlockContainer"] { max-width: none; padding: 0 1.5rem 1.5rem; }
    .app-topbar {
        height: 72px; display: flex; align-items: center; justify-content: space-between;
        border-bottom: 1px solid var(--ms-line); margin: 0 -1.5rem; padding: 0 1.5rem;
        background: rgba(255,255,255,.72);
    }
    .brand-lockup { display: flex; align-items: center; gap: 1.5rem; color: var(--ms-text); }
    .brand-logo-frame { width: 205px; height: 36px; overflow: hidden; }
    .brand-logo { display: block; width: 272px; height: auto; transform: translate(-16px, -6px); }
    .brand-divider { width: 1px; height: 28px; background: #aaa49e; }
    .product-name { font-size: 1.05rem; font-weight: 620; }
    .prototype-label { color: var(--ms-muted); font-size: .78rem; }
    div[data-baseweb="tab-list"], div[role="tablist"] { gap: .4rem; border-bottom: 1px solid var(--ms-line); }
    button[data-baseweb="tab"], [data-testid="stTab"] { min-height: 54px; padding: 0 .85rem; color: #343a46; font-weight: 520; }
    button[data-baseweb="tab"][aria-selected="true"], [data-testid="stTab"][aria-selected="true"] { color: var(--ms-accent); }
    div[data-baseweb="tab-highlight"] { background: var(--ms-accent); height: 2px; }
    div[data-testid="stHorizontalBlock"]:has(.workspace-column-marker) { gap: 0; align-items: stretch; }
    div[data-testid="stColumn"]:has(.chat-marker) {
        padding: 1.6rem clamp(1.5rem, 4vw, 4.5rem) 2rem; min-height: calc(100vh - 128px);
    }
    div[data-testid="stColumn"]:has(.evidence-marker) {
        border-left: 1px solid var(--ms-line); padding: 1.7rem 1.5rem 1.25rem;
        min-height: calc(100vh - 128px);
    }
    .chat-heading { margin: .1rem 0 1.5rem; }
    .chat-heading h1 { font-size: clamp(1.55rem, 2.2vw, 2rem); margin: 0 0 .35rem; letter-spacing: -.04em; }
    .chat-heading p { color: var(--ms-muted); margin: 0; font-size: .9rem; }
    .empty-chat { text-align: center; max-width: 570px; margin: clamp(3.5rem, 12vh, 7rem) auto 1.8rem; }
    .empty-chat h1 { font-size: clamp(1.8rem, 3vw, 2.4rem); letter-spacing: -.045em; margin: 0 0 .7rem; }
    .empty-chat p { color: var(--ms-muted); line-height: 1.65; margin: 0 auto; max-width: 44ch; }
    .conversation-end-spacer { height: 8rem; }
    div[data-testid="stColumn"]:has(.chat-marker) button[kind="secondary"] {
        min-height: 62px; border-color: var(--ms-line); background: var(--ms-surface);
        color: var(--ms-text); text-align: left; justify-content: flex-start; box-shadow: none;
    }
    div[data-testid="stColumn"]:has(.chat-marker) button[kind="secondary"]:hover {
        border-color: #c7c1ba; color: var(--ms-accent); background: #fffdfb;
    }
    button[kind="secondary"], button[kind="secondaryFormSubmit"] {
        min-height: 44px;
        background: var(--ms-surface) !important; color: var(--ms-text) !important;
        border-color: var(--ms-line) !important; box-shadow: none !important;
    }
    button, a { transition: color .2s, background-color .2s, border-color .2s, box-shadow .2s; }
    button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible {
        outline: 3px solid var(--ms-accent) !important; outline-offset: 2px !important;
    }
    a[aria-label="Link to heading"] {
        min-width: 44px; min-height: 44px; align-items: center; justify-content: center;
    }
    [data-testid="stChatMessage"] { background: transparent; border: 0; margin-bottom: .8rem; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        width: fit-content; min-width: 9rem; max-width: 72%; margin-left: auto;
        text-align: left; background: var(--ms-accent-soft); border-radius: 14px 14px 4px 14px;
        padding: .75rem 1rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        max-width: 48rem; margin-right: auto; padding: .35rem 0 1rem;
        border-bottom: 1px solid #efebe7;
    }
    [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { display: none; }
    [data-testid="stChatMessageContent"] { line-height: 1.65; }
    [data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }
    [data-testid="stChatInput"] {
        position: fixed; bottom: 2rem; left: calc(1.5rem + clamp(1.5rem, 4vw, 4.5rem));
        right: calc(21.43vw + .857rem + clamp(1.5rem, 4vw, 4.5rem)); z-index: 1000;
        border: 1px solid #d7d1ca; border-radius: 14px;
        background: var(--ms-surface) !important; box-shadow: 0 10px 30px rgba(32,38,51,.08);
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] div[data-baseweb="base-input"] {
        background: var(--ms-surface) !important; color: var(--ms-text) !important;
    }
    [data-testid="stChatInput"] textarea { min-height: 58px !important; }
    [data-testid="stChatInput"] button { min-width: 44px; min-height: 44px; }
    [data-testid="stChatInput"]:focus-within { border-color: var(--ms-accent); box-shadow: 0 0 0 2px rgba(240,68,68,.10); }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: var(--ms-surface) !important; border-color: var(--ms-line) !important;
        color: var(--ms-text) !important;
    }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div { min-height: 44px; }
    [data-testid="stNumberInput"] button { min-width: 44px; min-height: 44px; }
    .trust-note {
        position: fixed; bottom: .45rem; left: calc(1.5rem + clamp(1.5rem, 4vw, 4.5rem));
        z-index: 1000; color: var(--ms-muted); font-size: .74rem;
    }
    .trust-note strong { color: var(--ms-success); font-weight: 600; }
    .evidence-copy { color: var(--ms-muted); font-size: .86rem; line-height: 1.65; }
    .evidence-title { color: var(--ms-text); font-size: .94rem; font-weight: 650; margin: .15rem 0 1.15rem; }
    .evidence-empty { margin: 1rem 0 0; color: var(--ms-muted); max-width: 230px; line-height: 1.55; font-size: .82rem; }
    .evidence-empty strong { display: block; color: var(--ms-text); font-size: .9rem; margin-bottom: .35rem; }
    .citation-row { border-top: 1px solid var(--ms-line); padding: .8rem 0; font-size: .84rem; }
    [data-testid="stAlert"] { border-radius: 8px; border-width: 1px; box-shadow: none; }
    [data-testid="stDataFrame"] { border: 1px solid var(--ms-line); border-radius: 10px; overflow: hidden; }
    [data-testid="stElementToolbarButton"] button { min-width: 44px; min-height: 44px; }
    h1, h2, h3 { color: var(--ms-text); letter-spacing: -.025em; }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
    }
    @media (max-width: 900px) {
        .app-topbar { height: auto; min-height: 70px; }
        .product-name, .brand-divider, .prototype-label { display: none; }
        div[data-testid="stHorizontalBlock"]:has(.chat-heading) {
            display: grid !important; grid-template-columns: minmax(0, 1fr) 8rem; gap: .75rem;
        }
        div[data-testid="stHorizontalBlock"]:has(.chat-heading) > div[data-testid="stColumn"] {
            width: auto !important; min-width: 0 !important; flex: none !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.chat-heading) > div[data-testid="stColumn"]:first-child {
            grid-column: 1 / -1;
        }
        div[data-testid="stColumn"]:has(.evidence-marker) { display: none; }
        div[data-testid="stColumn"]:has(.chat-marker) { width: 100% !important; flex: 1 1 100% !important; padding: 1rem 0; }
        [data-testid="stChatInput"] { left: 1.5rem; right: 1.5rem; }
        .trust-note { left: 1.5rem; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { max-width: 88%; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="app-topbar">
  <div class="brand-lockup">
    <div class="brand-logo-frame"><img class="brand-logo" src="data:image/png;base64,{LOGO_DATA}" alt="MaiStorage"></div>
    <div class="brand-divider"></div>
    <div class="product-name">Streaming Chat Lab</div>
  </div>
  <div class="prototype-label">Question 2 interview prototype &middot; backend-only provider keys</div>
</div>
""", unsafe_allow_html=True)

current_session = session_id()
chat_tab, product_tab, environment_tab, source_tab, evaluation_tab = st.tabs([
    "Streaming Chat", "Product Explorer", "Environment Check", "Source Library", "Evaluation Lab"
])

with chat_tab:
    try:
        history = load_history(current_session)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        history = []
        st.warning("The chat is not ready yet. Please wait a moment and refresh the page.")

    latest_telemetry = {}
    chat_area, evidence_panel = st.columns([4.4, 1.2], gap=None)

    with chat_area:
        st.markdown('<span class="workspace-column-marker chat-marker"></span>', unsafe_allow_html=True)
        title_col, provider_col, action_col = st.columns([3.2, 1.35, .8], vertical_alignment="bottom")
        with title_col:
            st.markdown(
                '<div class="chat-heading"><h1>Streaming LLM Chat</h1>'
                '<p>Real provider deltas, PostgreSQL conversation memory, and refresh restoration.</p></div>',
                unsafe_allow_html=True,
            )
        with provider_col:
            provider = st.selectbox("Model provider", ["DeepSeek", "OpenAI"], index=0).lower()
        with action_col:
            if st.button("New chat", width="stretch"):
                st.query_params["session"] = str(uuid.uuid4())
                st.rerun()

        suggested_prompt = None
        messages = st.container()
        with messages:
            if history:
                for item in history:
                    with st.chat_message(item["role"]):
                        st.markdown(item["content"])
                        if item["role"] == "assistant" and item.get("provider"):
                            st.caption(f'{item["provider"]} | {item.get("model") or "configured model"}')
            else:
                st.markdown("""
                <div class="empty-chat">
                  <h1>Start conversation</h1>

                </div>
                """, unsafe_allow_html=True)
                prompt_left, prompt_right = st.columns(2)
                with prompt_left:
                    if st.button("Explain token streaming in two sentences", width="stretch"):
                        suggested_prompt = "Explain token streaming in two sentences."
                with prompt_right:
                    if st.button("Remember ORBIT for my next question", width="stretch"):
                        suggested_prompt = "Remember the word ORBIT for my next question."

        st.markdown('<div class="conversation-end-spacer"></div>', unsafe_allow_html=True)
        typed_question = st.chat_input("Send a message", key="chat_prompt", max_chars=4000)

        question = suggested_prompt or typed_question
        if question:
            with messages:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    try:
                        st.write_stream(stream_events(current_session, question, provider, latest_telemetry))
                        st.caption(
                            f'{latest_telemetry.get("provider", provider)} | {latest_telemetry.get("model", "configured model")} | '
                            f'TTFT {latest_telemetry.get("ttft_ms", "n/a")} ms | total {latest_telemetry.get("total_latency_ms", "n/a")} ms'
                        )
                    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as error:
                        st.error(str(error) or "The chat stream was interrupted. Please try again.")

    with evidence_panel:
        st.markdown('<span class="workspace-column-marker evidence-marker"></span>', unsafe_allow_html=True)
        st.markdown('<div class="evidence-title">Stream inspector</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="evidence-copy">Operational Log</p>',
            unsafe_allow_html=True,
        )
        previous_assistants = [item for item in history if item["role"] == "assistant"]
        previous = previous_assistants[-1] if previous_assistants else {}
        inspector_provider = latest_telemetry.get("provider") or previous.get("provider") or provider
        inspector_model = latest_telemetry.get("model") or previous.get("model") or "waiting for first response"
        event_order = " &rarr; ".join(latest_telemetry.get("events", []))
        stored_messages = len(history) + (2 if latest_telemetry.get("events", [])[-1:] == ["done"] else 0)
        st.markdown(f'<div class="citation-row"><strong>Session</strong><br>{current_session[:8]}&hellip;</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="citation-row"><strong>Provider</strong><br>{inspector_provider}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="citation-row"><strong>Model</strong><br>{inspector_model}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="citation-row"><strong>Persisted messages</strong><br>{stored_messages}</div>', unsafe_allow_html=True)
        if latest_telemetry:
            st.markdown(f'<div class="citation-row"><strong>Observed events</strong><br>{event_order or "waiting"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="citation-row"><strong>Time to first token</strong><br>{latest_telemetry.get("ttft_ms", "n/a")} ms</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="citation-row"><strong>Total latency</strong><br>{latest_telemetry.get("total_latency_ms", "n/a")} ms</div>', unsafe_allow_html=True)

with product_tab:
    st.subheader("Product catalogue")

    try:
        products = api_json("/api/v1/products")["products"]
        selected = st.multiselect("Products", [product["name"] for product in products])
        rows = [product for product in products if not selected or product["name"] in selected]
        st.dataframe(
            [{"Product": row["name"], "Family": row["family"], **row["specs"], "Source": row["source_url"]} for row in rows],
            width="stretch",
            hide_index=True,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        st.warning("The product catalogue is temporarily unavailable.")

with environment_tab:
    st.subheader("aiDAPTIV+ documented environment check")
    st.caption("Compare an environment against the selected public installation guide.")
    with st.form("environment_check", border=False):
        os_col, driver_col = st.columns(2)
        with os_col:
            operating_system = st.text_input("Operating system", "Ubuntu 24.04")
        with driver_col:
            driver = st.number_input("NVIDIA driver version", min_value=0, value=545)
        ports = st.text_input("Available ports (comma-separated)", "8899,8799,8000")
        validate_environment = st.form_submit_button("Validate documented environment")
    if validate_environment:
        try:
            available_ports = [int(port.strip()) for port in ports.split(",") if port.strip()]
            result = api_json("/api/v1/aidaptiv/validate-environment", {
                "operating_system": operating_system,
                "nvidia_driver": driver,
                "available_ports": available_ports,
            })
            st.markdown(result["answer"])
        except ValueError:
            st.error("Ports must be comma-separated numbers.")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            st.error("The environment check is temporarily unavailable.")

with source_tab:
    st.subheader("Approved public source registry")
    st.caption("Versioned webpages and manuals currently approved for retrieval.")
    try:
        st.dataframe(api_json("/api/v1/sources"), width="stretch", hide_index=True)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        st.warning("The source registry is temporarily unavailable.")

with evaluation_tab:
    st.subheader("Measured regression evidence")
    st.caption("Reproducible results for the current corpus and frozen evaluation set.")
    try:
        report = api_json("/api/v1/evaluations/latest")
        first, second, third = st.columns(3)
        first.metric("Strict accuracy", f'{report["strict_accuracy"]:.1%}')
        second.metric("Passed", f'{report["passed"]}/{report["cases"]}')
        third.metric("Corpus", report["corpus_version"][:8])
        st.dataframe(report["results"], width="stretch", hide_index=True)
        st.caption(f'Executed: {report["executed_at"]}. Results describe this versioned dataset only.')
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        st.info("Run the evaluation suite to create measured results.")
